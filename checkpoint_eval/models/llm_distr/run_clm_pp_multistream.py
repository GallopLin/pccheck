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

    # ------------------------------------------------ training loop
    steps_since_checkp = 0
    checkpoints = 0

    starts = time.time()
    for step in range(bench_total_steps):
        print(f"Train for step {step}")
        model_engine.train_batch()

        if (step == warmup) or (cfreq > 0 and steps_since_checkp == cfreq - 1):
            if cfreq > 0:
                print("save checkpoint (multistream)!!!")
                model_engine.save_multistream_checkpoint(step=step, sync=False)
                steps_since_checkp = 0
                checkpoints += 1
            if step == warmup:
                print("Start clock!")
                start_training = time.time()
        else:
            steps_since_checkp += 1
        print(f"Step {step} took {time.time()-starts}")
        starts = time.time()

    model_engine.shutdown_multistream()
    total_train_time = time.time() - start_training

    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, "
        f"Number of iterations: {step}, Number of checkpoints: {checkpoints}")
    print(f"EXECUTION TIME: {total_train_time} sec")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
