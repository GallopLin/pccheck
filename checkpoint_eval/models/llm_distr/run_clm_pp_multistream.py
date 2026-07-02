#!/usr/bin/env python
# coding=utf-8
"""
Pipeline-parallel CLM training with MultiStream async checkpointing.

Based on run_clm_pp_pccheck.py but uses MultiStreamPipelineEngine for
async 4-stream checkpointing instead of the legacy Chk_monitor path.

Outputs "EXECUTION TIME: X sec" for benchmark compatibility.
"""

import logging
import math
import time
import os
import sys
import importlib
from pathlib import Path
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from torch.utils.data import Dataset

import datasets
import evaluate
import torch
from datasets import load_dataset
import json

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PPTrainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from deepspeed.pipe import PipelineModule
from convert_to_ds import convert
import deepspeed

check_min_version("4.31.0.dev0")
require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def _import_multistream_utils():
    """导入 pccheck multistream 工具；必要时补充路径。"""
    try:
        return importlib.import_module("checkpoint_eval.pccheck.multistream_pipeline_utils")
    except ImportError:
        code_root = Path(__file__).resolve().parents[4]
        pccheck_root = code_root / "pccheck"
        if pccheck_root.exists() and str(pccheck_root) not in sys.path:
            sys.path.insert(0, str(pccheck_root))
        return importlib.import_module("checkpoint_eval.pccheck.multistream_pipeline_utils")


def _write_step_breakdown_record(record, breakdown_dir, rank):
    os.makedirs(breakdown_dir, exist_ok=True)
    path = os.path.join(breakdown_dir, f"step_wall_breakdown_rank{rank}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _metadata_barrier_ms(checkpoint_breakdown):
    if not isinstance(checkpoint_breakdown, dict):
        return 0.0
    if "metadata_barrier_ms" in checkpoint_breakdown:
        return float(checkpoint_breakdown.get("metadata_barrier_ms") or 0.0)
    return (
        float(checkpoint_breakdown.get("metadata_ms") or 0.0)
        + float(checkpoint_breakdown.get("stage_state_ms") or 0.0)
        + float(checkpoint_breakdown.get("global_state_ms") or 0.0)
        + float(checkpoint_breakdown.get("barrier_ms") or 0.0)
    )


def _build_step_breakdown_record(
    *,
    rank,
    world_size,
    step,
    cfreq,
    sync,
    step_wall_ms,
    train_batch_ms,
    checkpoint_breakdown,
):
    checkpoint_submit_ms = 0.0
    if isinstance(checkpoint_breakdown, dict):
        checkpoint_submit_ms = float(
            checkpoint_breakdown.get("checkpoint_submit_ms") or 0.0)
    metadata_barrier_ms = _metadata_barrier_ms(checkpoint_breakdown)

    misc_ms = step_wall_ms - train_batch_ms - checkpoint_submit_ms - metadata_barrier_ms
    if misc_ms < 0:
        # Keep the stacked bar non-negative and exactly closed. Small negative
        # values come from nested timer rounding; large ones indicate that the
        # checkpoint helper measured slightly wider than the outer step timer.
        metadata_barrier_ms = max(0.0, metadata_barrier_ms + misc_ms)
        misc_ms = step_wall_ms - train_batch_ms - checkpoint_submit_ms - metadata_barrier_ms
    if misc_ms < 0:
        checkpoint_submit_ms = max(0.0, checkpoint_submit_ms + misc_ms)
        misc_ms = step_wall_ms - train_batch_ms - checkpoint_submit_ms - metadata_barrier_ms
    misc_ms = max(0.0, misc_ms)

    buckets = {
        "train_batch_ms": train_batch_ms,
        "checkpoint_submit_ms": checkpoint_submit_ms,
        "metadata_barrier_ms": metadata_barrier_ms,
        "misc_ms": misc_ms,
    }
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "step": int(step),
        "cfreq": int(cfreq),
        "sync": bool(sync),
        "selected": True,
        "step_wall_ms": float(step_wall_ms),
        "bucket_sum_ms": float(sum(buckets.values())),
        "buckets": buckets,
        "checkpoint_breakdown": checkpoint_breakdown or {},
        "note": (
            "Main buckets are foreground wall time only; async D2H/SSD work "
            "is not added to this stacked bar."
        ),
    }


class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = 1024

    def __len__(self):
        return len(self.hf_dataset)

    def truncate_and_pad_func(self, input_ids, att_mask, labels, side='right', add_eos=True):
        input_ids = input_ids[:self.seq_length]
        att_mask = att_mask[:self.seq_length]
        labels = labels[:self.seq_length]

        if add_eos:
            if input_ids.size(0) == self.seq_length:
                input_ids[-1] = self.tokenizer.eos_token_id
                labels[-1] = self.tokenizer.eos_token_id
            elif input_ids.size(0) < self.seq_length:
                eos_token_id = self.tokenizer.eos_token_id
                eos_inp = torch.empty(1, dtype=torch.long).fill_(eos_token_id)
                eos_att = torch.ones(1, dtype=torch.long)
                eos_lb = torch.empty(1, dtype=torch.long).fill_(eos_token_id)
                input_ids = torch.cat([input_ids, eos_inp], dim=0)
                att_mask = torch.cat([att_mask, eos_att], dim=0)
                labels = torch.cat([labels, eos_lb], dim=0)

        len_pad = self.seq_length - input_ids.size(0)
        if len_pad > 0:
            pad_token_id = self.tokenizer.pad_token_id
            pad_inp = torch.empty(len_pad, dtype=torch.long).fill_(pad_token_id)
            pad_att = torch.zeros(len_pad, dtype=torch.long)
            pad_lb = torch.empty(len_pad, dtype=torch.long).fill_(-100)
            if side == 'left':
                input_ids = torch.cat([pad_inp, input_ids], dim=0)
                att_mask = torch.cat([pad_att, att_mask], dim=0)
                labels = torch.cat([pad_lb, labels], dim=0)
            elif side == 'right':
                input_ids = torch.cat([input_ids, pad_inp], dim=0)
                att_mask = torch.cat([att_mask, pad_att], dim=0)
                labels = torch.cat([labels, pad_lb], dim=0)

        inputs = torch.cat(
            [input_ids.unsqueeze(-1), att_mask.unsqueeze(-1)], dim=-1)
        return inputs.clone(), labels.clone()

    def __getitem__(self, idx):
        items = self.hf_dataset[idx]
        input_ids = torch.tensor(items["input_ids"])
        att_mask = torch.tensor(items["attention_mask"])
        labels = input_ids.clone()
        res = self.truncate_and_pad_func(input_ids, att_mask, labels, add_eos=False)
        inputs, labels = res
        return inputs, labels


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)
    ds_config: str = field(default='')
    cfreq: int = field(default=0, metadata={"help": "Checkpoint frequency"})
    c_lib_path: str = field(default='', metadata={"help": "Path to libtest_ssd.so"})
    max_async: int = field(default=2, metadata={"help": "Max async checkpoints"})
    num_threads: int = field(default=2, metadata={"help": "SSD write threads"})
    bench_total_steps: int = field(default=100, metadata={"help": "Total benchmark steps"})
    num_layer_groups: int = field(default=8, metadata={"help": "Layer groups for multistream"})
    checkpoint_dir: str = field(default='./ms_checkpoints', metadata={"help": "Checkpoint directory"})
    step_breakdown: bool = field(
        default=False,
        metadata={"help": "Collect one checkpoint-step foreground wall-time breakdown"}
    )
    breakdown_step: int = field(
        default=-1,
        metadata={"help": "Step to record; -1 selects the first checkpoint step after warmup"}
    )
    breakdown_dir: str = field(
        default="",
        metadata={"help": "Directory for step wall-time breakdown JSONL output"}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with "
                "--config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    keep_linebreaks: bool = field(default=True)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}")

    set_seed(training_args.seed)

    # ------------------------------------------------------------------ data
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming)
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming)
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1])
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, **dataset_args)
            raw_datasets["train"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, **dataset_args)

    # --------------------------------------------------------------- config
    with open(model_args.ds_config, 'r') as f:
        deepspeed_config = json.load(f)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Tokenizer required.")

    # --------------------------------------------------------------- model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype))
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, from_tf=False, config=config,
            cache_dir=model_args.cache_dir, revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage)
    else:
        model = AutoModelForCausalLM.from_config(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = convert("opt", model, config, 2)

    # ----------------------------------------------------------- tokenize
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - "
                "this long input will be chunked into smaller bits "
                "before being passed to the model.")
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset")
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function, batched=True, remove_columns=column_names)

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}")
        else:
            lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    if training_args.do_train:
        train_dataset = HuggingFaceDatasetWrapper(
            lm_datasets["train"], tokenizer)

    # ------------------------------------------------ DeepSpeed initialize
    bench_total_steps = model_args.bench_total_steps
    cfreq = model_args.cfreq
    warmup = 3

    deepspeed.init_distributed(dist_backend='nccl')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    checkp_params = {
        "checkpoint_dir": model_args.checkpoint_dir,
        "lib_path": model_args.c_lib_path,
        "max_async": model_args.max_async,
        "num_layer_groups": model_args.num_layer_groups,
        "num_threads": model_args.num_threads,
    }

    model_engine, optimizer, train_loader, lr_schdlr = deepspeed.initialize(
        args=training_args, model=model,
        training_data=train_dataset,
        config=deepspeed_config,
        checkp_type='MultiStream',
        checkp_params=checkp_params,
    )

    # 某些环境下 deepspeed.initialize() 可能仍返回 PipelineEngine，缺少 multistream API。
    has_native_ms_api = hasattr(model_engine, "save_multistream_checkpoint") and hasattr(
        model_engine, "shutdown_multistream")
    ms_fallback = None
    if not has_native_ms_api:
        logger.warning(
            "Engine %s does not expose multistream APIs; using pccheck fallback.",
            type(model_engine).__name__,
        )
        ms_utils = _import_multistream_utils()
        ms_ckpt, ms_gpu_buffer, ms_param_layout = ms_utils.create_stage_checkpoint(
            model=model_engine.module,
            optimizer=model_engine.optimizer,
            checkpoint_dir=model_args.checkpoint_dir,
            lib_path=model_args.c_lib_path,
            rank=model_engine.global_rank,
            world_size=world_size,
            max_async=model_args.max_async,
            num_layer_groups=model_args.num_layer_groups,
            num_threads=model_args.num_threads,
        )
        ms_fallback = {
            "utils": ms_utils,
            "ckpt": ms_ckpt,
            "gpu_buffer": ms_gpu_buffer,
            "param_layout": ms_param_layout,
        }

    def _save_multistream_checkpoint(step: int, sync: bool = False, return_breakdown: bool = False):
        if has_native_ms_api:
            return model_engine.save_multistream_checkpoint(
                step=step,
                sync=sync,
                return_breakdown=return_breakdown,
            )

        stage_result = ms_fallback["utils"].save_stage_checkpoint(
            ms_ckpt=ms_fallback["ckpt"],
            gpu_buffer=ms_fallback["gpu_buffer"],
            param_layout=ms_fallback["param_layout"],
            model=model_engine.module,
            optimizer=model_engine.optimizer,
            checkpoint_dir=model_args.checkpoint_dir,
            rank=model_engine.global_rank,
            step=step,
            sync=sync,
            return_breakdown=return_breakdown,
        )
        if return_breakdown:
            breakdown = dict(stage_result)
            elapsed = float(breakdown.get("elapsed_sec", 0.0))
        else:
            elapsed = stage_result

        global_state_ms = 0.0
        if model_engine.global_rank == 0:
            global_state_t0 = time.perf_counter()
            ms_fallback["utils"].save_global_training_state(
                model_args.checkpoint_dir, step)
            global_state_ms = (time.perf_counter() - global_state_t0) * 1000.0

        barrier_ms = 0.0
        if torch.distributed.is_initialized():
            barrier_t0 = time.perf_counter()
            torch.distributed.barrier()
            barrier_ms = (time.perf_counter() - barrier_t0) * 1000.0

        if return_breakdown:
            metadata_barrier_ms = (
                float(breakdown.get("metadata_ms", 0.0))
                + float(breakdown.get("stage_state_ms", 0.0))
                + global_state_ms
                + barrier_ms
            )
            total_ms = float(breakdown.get("total_ms", elapsed * 1000.0))
            total_ms += global_state_ms + barrier_ms
            breakdown.update({
                "elapsed_sec": total_ms / 1000.0,
                "total_ms": total_ms,
                "global_state_ms": global_state_ms,
                "barrier_ms": barrier_ms,
                "metadata_barrier_ms": metadata_barrier_ms,
            })
            return breakdown
        return elapsed

    def _shutdown_multistream():
        if has_native_ms_api:
            model_engine.shutdown_multistream()
        elif ms_fallback is not None and ms_fallback["ckpt"] is not None:
            ms_fallback["ckpt"].shutdown()

    # ------------------------------------------------ training loop
    steps_since_checkp = 0
    checkpoints = 0
    breakdown_dir = model_args.breakdown_dir or os.path.join(
        model_args.checkpoint_dir, "step_breakdown")
    breakdown_recorded = False
    breakdown_target_step = int(model_args.breakdown_step)
    if model_args.step_breakdown and cfreq <= 0 and rank == 0:
        logger.warning("Step breakdown requested, but cfreq <= 0 so no checkpoint step exists.")

    starts = time.time()
    for step in range(bench_total_steps):
        step_wall_t0 = time.perf_counter()
        print(f"Train for step {step}")

        train_t0 = time.perf_counter()
        model_engine.train_batch()
        train_batch_ms = (time.perf_counter() - train_t0) * 1000.0

        checkpoint_breakdown = None
        is_checkpoint_step = (step == warmup) or (cfreq > 0 and steps_since_checkp == cfreq - 1)
        should_record_breakdown = False
        if model_args.step_breakdown and is_checkpoint_step and cfreq > 0:
            if breakdown_target_step >= 0:
                should_record_breakdown = (step == breakdown_target_step)
            else:
                should_record_breakdown = (step > warmup and not breakdown_recorded)

        if is_checkpoint_step:
            if cfreq > 0:
                print("save checkpoint (multistream)!!!")
                checkpoint_breakdown = _save_multistream_checkpoint(
                    step=step,
                    sync=False,
                    return_breakdown=should_record_breakdown,
                )
                steps_since_checkp = 0
                checkpoints += 1
            if step == warmup:
                print("Start clock!")
                start_training = time.time()
        else:
            steps_since_checkp += 1

        step_wall_ms = (time.perf_counter() - step_wall_t0) * 1000.0
        if should_record_breakdown and isinstance(checkpoint_breakdown, dict):
            record = _build_step_breakdown_record(
                rank=rank,
                world_size=world_size,
                step=step,
                cfreq=cfreq,
                sync=False,
                step_wall_ms=step_wall_ms,
                train_batch_ms=train_batch_ms,
                checkpoint_breakdown=checkpoint_breakdown,
            )
            _write_step_breakdown_record(record, breakdown_dir, rank)
            breakdown_recorded = True
            print(
                f"[StepBreakdown] rank {rank} recorded step {step} "
                f"to {breakdown_dir}"
            )

        print(f"Step {step} took {time.time()-starts}")
        starts = time.time()

    _shutdown_multistream()
    if model_args.step_breakdown and not breakdown_recorded:
        if breakdown_target_step >= 0:
            raise RuntimeError(
                f"No checkpoint-step breakdown was recorded for step {breakdown_target_step}. "
                "Please choose a step that triggers checkpointing."
            )
        raise RuntimeError(
            "No post-warmup checkpoint-step breakdown was recorded. "
            "Increase --bench_total_steps or pass --breakdown_step to a checkpoint step."
        )
    total_train_time = time.time() - start_training

    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, "
        f"Number of iterations: {step}, Number of checkpoints: {checkpoints}")
    print(f"EXECUTION TIME: {total_train_time} sec")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
