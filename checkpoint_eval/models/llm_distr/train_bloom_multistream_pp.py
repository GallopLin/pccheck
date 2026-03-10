#!/usr/bin/env python3
"""BLOOM + DeepSpeed Pipeline Parallelism + MultiStream Checkpoint + Pipelayer Resume

单机多卡训练 BLOOM 模型，使用 multistream 异步保存检查点，
恢复时使用 pipelayer 的 MultiStreamStateLoader 进行流水线加载。

用法（4 GPU 训练）:
    deepspeed --num_gpus=4 train_bloom_multistream_pp.py \
        --deepspeed ds_config.json \
        --model_name_or_path bigscience/bloom-560m \
        --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
        --num_stages 4 --total_steps 100 --checkpoint_every 20 \
        --checkpoint_dir ./ms_bloom_checkpoints \
        --lib_path ../../pccheck/libtest_ssd.so

恢复训练（从检查点）:
    deepspeed --num_gpus=4 train_bloom_multistream_pp.py \
        --deepspeed ds_config.json \
        --model_name_or_path bigscience/bloom-560m \
        --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
        --num_stages 4 --total_steps 200 --checkpoint_every 20 \
        --checkpoint_dir ./ms_bloom_checkpoints \
        --lib_path ../../pccheck/libtest_ssd.so \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from itertools import chain

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from checkpoint_eval.deepspeed import initialize as ds_initialize
import deepspeed as ds_upstream
from deepspeed.pipe import PipelineModule

from bloom_ds import get_bloom_causal_lm_specs
from convert_to_ds import LMLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset wrapper (same as existing run_clm_pp_pccheck.py)
# ---------------------------------------------------------------------------
class CLMDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_length=1024):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"][:self.seq_length], dtype=torch.long)
        att_mask = torch.tensor(
            item["attention_mask"][:self.seq_length], dtype=torch.long)
        labels = input_ids.clone()

        pad_len = self.seq_length - input_ids.size(0)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), pad_id, dtype=torch.long)])
            att_mask = torch.cat([
                att_mask,
                torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long)])

        inputs = torch.stack([input_ids, att_mask], dim=-1)
        return inputs, labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="BLOOM PP training with multistream checkpoint")
    p.add_argument("--model_name_or_path", type=str,
                   default="bigscience/bloom-560m")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config_name", type=str,
                   default="wikitext-2-raw-v1")
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--num_stages", type=int, default=2,
                   help="Pipeline parallelism stages (= num GPUs)")
    p.add_argument("--total_steps", type=int, default=100)
    p.add_argument("--checkpoint_every", type=int, default=20,
                   help="Save checkpoint every N steps (0 = no save)")
    p.add_argument("--checkpoint_dir", type=str,
                   default="./ms_bloom_checkpoints")
    p.add_argument("--lib_path", type=str,
                   default=os.path.join(REPO_ROOT, "pccheck", "libtest_ssd.so"))
    p.add_argument("--max_async", type=int, default=2)
    p.add_argument("--num_layer_groups", type=int, default=8)
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint_dir using pipelayer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by deepspeed launcher")

    p.add_argument("--deepspeed", type=str, default=None,
                   help="DeepSpeed config JSON file")
    p.add_argument("--deepspeed_config", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Convert HuggingFace BLOOM → DeepSpeed PipelineModule
# ---------------------------------------------------------------------------
def build_bloom_pipeline(config, model_or_path, num_stages, from_pretrained=True):
    """加载 BLOOM 预训练权重并转换为 PipelineModule。"""
    if from_pretrained:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_or_path, config=config, torch_dtype=torch.float32)
    else:
        hf_model = AutoModelForCausalLM.from_config(config)

    state = hf_model.state_dict()
    import re

    res = {
        0: {
            "word_embeddings.weight":
                state["transformer.word_embeddings.weight"],
            "word_embeddings_layernorm.weight":
                state["transformer.word_embeddings_layernorm.weight"],
            "word_embeddings_layernorm.bias":
                state["transformer.word_embeddings_layernorm.bias"],
        },
    }

    ind_last = -1
    for k, v in state.items():
        if not re.search(r"^transformer\.h\.", k):
            continue
        k2 = re.sub(r"^transformer\.h\.", "", k)
        ind = int(re.search(r"^\d+", k2).group()) + 1
        k2 = re.sub(r"^\d+\.", "", k2)
        if ind not in res:
            res[ind] = {}
        res[ind][k2] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    res[ind_last] = {
        "word_embeddings.weight": state["transformer.word_embeddings.weight"],
        "word_embeddings_layernorm.weight": state["transformer.ln_f.weight"],
        "word_embeddings_layernorm.bias": state["transformer.ln_f.bias"],
    }
    if "lm_head.weight" in state:
        res[ind_last]["word_embeddings.weight"] = state["lm_head.weight"]

    layers = get_bloom_causal_lm_specs(config, res)
    pipeline_model = PipelineModule(
        layers, loss_fn=LMLoss(False), num_stages=num_stages)

    del hf_model, state
    torch.cuda.empty_cache()
    return pipeline_model


# ---------------------------------------------------------------------------
# Resume with pipelayer
# ---------------------------------------------------------------------------
def maybe_resume(model_engine, args):
    """如果 --resume 且检查点存在，使用 pipelayer 恢复。"""
    if not args.resume:
        return 0

    global_state_path = os.path.join(args.checkpoint_dir, "global_state.json")
    if not os.path.exists(global_state_path):
        if model_engine.global_rank == 0:
            print(f"[Resume] No checkpoint found at {args.checkpoint_dir}")
        return 0

    sys.path.insert(0, REPO_ROOT)
    from checkpoint_eval.pccheck.multistream_pipeline_utils import (
        resume_stage_with_pipelayer,
        load_global_training_state,
    )

    rank = model_engine.global_rank
    local_rank = model_engine.local_rank
    device = f"cuda:{local_rank}"

    stage_training_state = resume_stage_with_pipelayer(
        model=model_engine.module,
        optimizer=model_engine.optimizer,
        checkpoint_dir=args.checkpoint_dir,
        rank=rank,
        lib_path=args.lib_path,
        device=device,
    )

    global_state = load_global_training_state(args.checkpoint_dir)
    start_step = global_state.get("completed_steps", 0)

    if rank == 0:
        print(f"[Resume] Restored from step {start_step}")

    if dist.is_initialized():
        dist.barrier()

    return start_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    ds_config_path = args.deepspeed or args.deepspeed_config
    if ds_config_path is None:
        ds_config_path = os.path.join(SCRIPT_DIR, "ds_config.json")
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    ds_upstream.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"BLOOM MultiStream Pipeline Training")
        print(f"  World size: {world_size}")
        print(f"  Num stages: {args.num_stages}")
        print(f"  Model: {args.model_name_or_path}")
        print(f"  Checkpoint dir: {args.checkpoint_dir}")
        print(f"  Lib path: {args.lib_path}")
        print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print("[Data] Loading and tokenizing dataset...")

    raw_datasets = load_dataset(
        args.dataset_name, args.dataset_config_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw_datasets.map(
        tokenize_fn, batched=True,
        remove_columns=raw_datasets["train"].column_names)

    block_size = min(args.seq_length, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)
    train_dataset = CLMDatasetWrapper(lm_dataset["train"], tokenizer, block_size)

    if rank == 0:
        print(f"[Data] Train samples: {len(train_dataset)}")
        print("[Model] Converting to PipelineModule...")

    pipeline_model = build_bloom_pipeline(
        config, args.model_name_or_path, args.num_stages)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkp_params = {
        "checkpoint_dir": args.checkpoint_dir,
        "lib_path": args.lib_path,
        "max_async": args.max_async,
        "num_layer_groups": args.num_layer_groups,
        "num_threads": args.num_threads,
    }

    # 创建 args 对象传给 DeepSpeed（它需要 local_rank）
    class DSArgs:
        pass
    ds_args = DSArgs()
    ds_args.local_rank = local_rank
    ds_args.deepspeed_config = ds_config_path

    model_engine, optimizer, train_loader, lr_scheduler = ds_initialize(
        args=ds_args,
        model=pipeline_model,
        training_data=train_dataset,
        config=ds_config,
        checkp_type="MultiStream",
        checkp_params=checkp_params,
    )

    start_step = maybe_resume(model_engine, args)

    if rank == 0:
        print(f"\n[Training] Starting from step {start_step}, "
              f"running to step {args.total_steps}")

    checkpoint_times = []
    step_times = []

    for step in range(start_step, args.total_steps):
        step_start = time.time()

        model_engine.train_batch()

        step_time = time.time() - step_start
        step_times.append(step_time)

        do_checkpoint = (
            args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
            and step > start_step
        )

        if do_checkpoint:
            ckpt_start = time.time()
            elapsed = model_engine.save_multistream_checkpoint(
                step=step + 1, sync=True)
            ckpt_time = time.time() - ckpt_start
            checkpoint_times.append(ckpt_time)
            if rank == 0:
                print(f"[Step {step+1}] Checkpoint saved in "
                      f"{ckpt_time:.2f}s (ms_internal: {elapsed:.2f}s)")

        if rank == 0 and (step + 1) % 10 == 0:
            avg_step = sum(step_times[-10:]) / min(len(step_times), 10)
            print(f"[Step {step+1}] avg step time: {avg_step:.3f}s")

    model_engine.shutdown_multistream()

    if rank == 0:
        total_train = sum(step_times)
        total_ckpt = sum(checkpoint_times) if checkpoint_times else 0
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"  Steps: {args.total_steps - start_step}")
        print(f"  Total train time: {total_train:.2f}s")
        print(f"  Total checkpoint time: {total_ckpt:.2f}s")
        print(f"  Checkpoints saved: {len(checkpoint_times)}")
        if step_times:
            print(f"  Avg step time: {sum(step_times)/len(step_times):.3f}s")
        print(f"{'='*60}\n")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
