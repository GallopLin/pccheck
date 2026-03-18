import argparse
import os
import subprocess
from typing import List

import pandas as pd


home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/transformers/examples/pytorch/language-modeling"
default_out_root = f"{home_dir}/pccheck/artifact_evaluation/evaluation/throughput"
WARMUP = 3


def _run_cmd(cmd: str, log_file: str) -> int:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print(f"[RUN] {cmd}")
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"[DONE] exit={proc.returncode} log={log_file}")
    return proc.returncode


def _parse_exec_time(log_file: str) -> float:
    exec_time = None
    extra_time_ms = 0.0
    with open(log_file, "r") as f:
        for line in f:
            if "EXECUTION TIME" in line:
                tokens = line.split()
                exec_time = float(tokens[-2])
            elif "MMAP/UMAP" in line:
                tokens = line.split()
                extra_time_ms = float(tokens[-2])

    if exec_time is None:
        raise RuntimeError(f"Could not parse EXECUTION TIME from: {log_file}")

    return max(exec_time - (extra_time_ms / 1000.0), 1e-8)


def run_single_node_bloom_pccheck(
    cfreqs: List[int], iters: int, num_gpus: int, out_tag: str, pipeline_stages: int
) -> str:
    out_dir = f"{default_out_root}/{out_tag}"
    os.makedirs(out_dir, exist_ok=True)

    if pipeline_stages <= 0:
        raise ValueError("--pp-stages must be >= 1")
    if num_gpus % pipeline_stages != 0:
        raise ValueError(
            f"num_gpus ({num_gpus}) must be divisible by pp_stages ({pipeline_stages}) "
            "for pipeline+data parallel setup."
        )

    data_parallel_size = num_gpus // pipeline_stages

    # Keep batch settings consistent with DeepSpeed assertion:
    # train_batch_size == micro_batch * grad_acc * data_parallel_size
    ds_cfg = f"{script_dir}/ds_config.json"
    ds_tmp = f"{script_dir}/ds_config.bloom_tmp.json"
    with open(ds_cfg, "r") as f:
        content = f.read()
    content = content.replace('"train_batch_size": 1', f'"train_batch_size": {data_parallel_size}')
    with open(ds_tmp, "w") as f:
        f.write(content)

    rows = []

    for cf in cfreqs:
        log_file = f"{out_dir}/log_bloom7_pccheck_{cf}.txt"
        cmd = (
            f"cd {script_dir} && "
            f"NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "
            f"deepspeed --num_gpus={num_gpus} ./run_clm_pp_pccheck.py "
            f"--deepspeed ./ds_config.bloom_tmp.json --ds_config ./ds_config.bloom_tmp.json "
            f"--model_name_or_path bigscience/bloom-7b1 "
            f"--output_dir ./output_bloom7_pccheck "
            f"--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 "
            f"--do_train --per_device_train_batch_size 1 "
            
            f"--cfreq {cf} --bench_total_steps {iters} "
            f"--max_async 2 --num_threads 2 "
            f"--c_lib_path {lib_path} "
            f"--bf16 --torch_dtype bfloat16"
        )

        rc = _run_cmd(cmd, log_file)
        if rc != 0:
            rows.append({"cfreq": cf, "status": "failed", "exec_time_sec": None, "throughput_iter_per_sec": None})
            continue

        exec_time = _parse_exec_time(log_file)
        thr = (iters - WARMUP) / exec_time
        rows.append({"cfreq": cf, "status": "ok", "exec_time_sec": exec_time, "throughput_iter_per_sec": thr})

    csv_file = f"{out_dir}/bloom7_pccheck_throughput.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    print(f"[RESULT] {csv_file}")

    return csv_file


def parse_args():
    p = argparse.ArgumentParser(
        description="Run PCcheck throughput for BLOOM-7B on single-node multi-GPU and save CSV."
    )
    p.add_argument("--mode", choices=["single-node", "multi-node"], default="single-node")
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--pp-stages", type=int, default=2, help="Pipeline stages for BLOOM conversion.")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cfreqs", type=str, default="10", help="Comma-separated list, e.g. 1,10,25")
    p.add_argument("--out-tag", type=str, default="bloom_7")
    # Kept for compatibility, not used in single-node mode.
    p.add_argument("--ip1", type=str, default="")
    p.add_argument("--ip2", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "multi-node":
        raise NotImplementedError(
            "This script is now focused on single-node multi-GPU BLOOM-7B throughput for PCcheck. "
            "Use existing multi-node workflow if you need OPT-2.7B 2-node results."
        )

    cfreqs = [int(x.strip()) for x in args.cfreqs.split(",") if x.strip()]
    run_single_node_bloom_pccheck(
        cfreqs=cfreqs,
        iters=args.iters,
        num_gpus=args.num_gpus,
        out_tag=args.out_tag,
        pipeline_stages=args.pp_stages,
    )


if __name__ == "__main__":
    main()