import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


WARMUP = 3


def run_cmd(cmd: str, log_file: str) -> int:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print(f"[RUN] {cmd}")
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"[DONE] exit={proc.returncode} log={log_file}")
    return proc.returncode


def parse_exec_time(log_file: str) -> float:
    exec_time = None
    extra_time_ms = 0.0
    with open(log_file, "r") as f:
        for line in f:
            if "EXECUTION TIME" in line:
                exec_time = float(line.split()[-2])
            elif "MMAP/UMAP" in line:
                extra_time_ms = float(line.split()[-2])

    if exec_time is None:
        raise RuntimeError(f"Could not parse EXECUTION TIME from: {log_file}")

    return max(exec_time - (extra_time_ms / 1000.0), 1e-8)


def write_ds_tmp(base_cfg: str, out_cfg: str, train_batch_size: int) -> None:
    with open(base_cfg, "r") as f:
        content = f.read()
    content = content.replace('"train_batch_size": 1', f'"train_batch_size": {train_batch_size}')
    content = content.replace('"train_batch_size": 2', f'"train_batch_size": {train_batch_size}')
    content = content.replace('"train_batch_size": 4', f'"train_batch_size": {train_batch_size}')
    with open(out_cfg, "w") as f:
        f.write(content)


def plot_single_line(df: pd.DataFrame, out_png: str) -> None:
    ok = df[df["status"] == "ok"].sort_values("cfreq")
    if ok.empty:
        raise RuntimeError("No successful rows found. Skip plotting.")

    # Keep the visual language close to original.py.
    label_font_size = 36
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        ok["cfreq"].to_numpy(),
        ok["throughput_iter_per_sec"].to_numpy(),
        color="#A7B972",
        marker="s",
        linewidth=3,
        markersize=8,
        label="PCcheck",
    )

    ax.set_xlabel("Checkpoint interval(iterations)", fontsize=label_font_size)
    ax.set_ylabel("Throughput (iterations/sec)", fontsize=label_font_size)
    ax.tick_params(axis="both", labelsize=label_font_size)
    ax.legend(loc="upper left", fontsize=label_font_size - 2)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=500, pad_inches=0.1)
    print(f"[RESULT] {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run OPT-2.7B PCcheck throughput sweep and plot with the same metric logic as get_throughput_multi_node.py."
    )
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cfreqs", type=str, default="0,1,10,25,50,75,100")
    p.add_argument("--out-tag", type=str, default="opt27_pccheck_pp")
    p.add_argument(
        "--script-dir",
        type=str,
        default="/root/transformers/examples/pytorch/language-modeling",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="/root/pccheck/artifact_evaluation/evaluation/throughput",
    )
    p.add_argument(
        "--lib-path",
        type=str,
        default="/root/pccheck/checkpoint_eval/pccheck/libtest_ssd.so",
    )
    p.add_argument(
        "--pipeline-stages",
        type=int,
        default=2,
        help="OPT pipeline stages used by run_clm_pp_pccheck.py (default 2).",
    )
    args = p.parse_args()

    if args.num_gpus % args.pipeline_stages != 0:
        raise ValueError(
            f"num_gpus ({args.num_gpus}) must be divisible by pipeline_stages ({args.pipeline_stages})"
        )

    cfreqs = [int(x.strip()) for x in args.cfreqs.split(",") if x.strip()]
    out_dir = f"{args.out_root}/{args.out_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # DeepSpeed PP validates train_batch_size with data-parallel world size.
    dp_world_size = args.num_gpus // args.pipeline_stages
    ds_base = f"{args.script_dir}/ds_config.bloom_tmp.json"
    ds_tmp = f"{args.script_dir}/ds_config.opt27_tmp.json"
    write_ds_tmp(ds_base, ds_tmp, train_batch_size=dp_world_size)

    rows = []
    for cf in cfreqs:
        log_file = f"{out_dir}/log_opt27_pccheck_{cf}.txt"
        cmd = (
            f"cd {args.script_dir} && "
            f"NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "
            f"deepspeed --num_gpus={args.num_gpus} ./run_clm_pp_pccheck.py "
            f"--deepspeed ./ds_config.opt27_tmp.json --ds_config ./ds_config.opt27_tmp.json "
            f"--model_name_or_path facebook/opt-2.7b "
            f"--output_dir ./output_opt27_pccheck "
            f"--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 "
            f"--do_train --per_device_train_batch_size 1 "
            f"--cfreq {cf} --bench_total_steps {args.iters} "
            f"--max_async 2 --num_threads 2 "
            f"--c_lib_path {args.lib_path} "
            f"--bf16 --torch_dtype bfloat16 --gradient_checkpointing"
        )

        rc = run_cmd(cmd, log_file)
        if rc != 0:
            rows.append({"cfreq": cf, "status": "failed", "exec_time_sec": None, "throughput_iter_per_sec": None})
            continue

        exec_time = parse_exec_time(log_file)
        thr = (args.iters - WARMUP) / exec_time
        rows.append({"cfreq": cf, "status": "ok", "exec_time_sec": exec_time, "throughput_iter_per_sec": thr})

    df = pd.DataFrame(rows).sort_values("cfreq")
    csv_path = f"{out_dir}/opt27_pccheck_throughput.csv"
    df.to_csv(csv_path, index=False)
    print(f"[RESULT] {csv_path}")

    plot_single_line(df, f"{out_dir}/opt27_pccheck_throughput.png")


if __name__ == "__main__":
    main()
