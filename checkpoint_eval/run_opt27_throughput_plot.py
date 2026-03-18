
import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

WARMUP = 3
SUCCESS_MARKERS = ("Train for step", "BENCHMARK ENDED", "EXECUTION TIME")


def run_cmd(cmd: str, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {cmd}")
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"[DONE] exit={proc.returncode} log={log_file}")
    return proc.returncode


def parse_log(log_file: Path) -> Dict[str, Optional[float]]:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    exec_match = re.search(r"EXECUTION TIME:\s*([0-9]+(?:\.[0-9]+)?)\s*sec", text)
    mmap_match = re.search(r"MMAP/UMAP.*?([0-9]+(?:\.[0-9]+)?)\s*ms", text)
    ckpt_times = [
        float(x)
        for x in re.findall(r"Single Checkpoint time is\s*([0-9]+(?:\.[0-9]+)?)\s*sec", text)
    ]

    exec_time = float(exec_match.group(1)) if exec_match else None
    extra_time_ms = float(mmap_match.group(1)) if mmap_match else 0.0
    adj_exec = None
    if exec_time is not None:
        adj_exec = max(exec_time - (extra_time_ms / 1000.0), 1e-8)

    return {
        "has_all_success_markers": all(marker in text for marker in SUCCESS_MARKERS),
        "exec_time_sec": exec_time,
        "adj_exec_time_sec": adj_exec,
        "single_ckpt_time_median_sec": statistics.median(ckpt_times) if ckpt_times else None,
        "single_ckpt_time_avg_sec": (sum(ckpt_times) / len(ckpt_times)) if ckpt_times else None,
        "single_ckpt_samples": len(ckpt_times),
    }


def write_ds_tmp(base_cfg: Path, out_cfg: Path, train_batch_size: int, torch_adam: bool) -> None:
    cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
    cfg["train_batch_size"] = train_batch_size
    optimizer = cfg.get("optimizer", {})
    params = optimizer.get("params", {})
    params["torch_adam"] = torch_adam
    optimizer["params"] = params
    cfg["optimizer"] = optimizer
    out_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def plot_single_line(rows: List[Dict], out_png: Path, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed, skip plotting.")
        return

    ok_rows = sorted([r for r in rows if r["status"] == "ok"], key=lambda x: x["cfreq"])
    if not ok_rows:
        raise RuntimeError("No successful rows found. Skip plotting.")

    label_font_size = 36
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        [r["cfreq"] for r in ok_rows],
        [r["throughput_iter_per_sec"] for r in ok_rows],
        color="#A7B972",
        marker="s",
        linewidth=3,
        markersize=8,
        label=label,
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
        description="Run OPT-2.7B PCcheck throughput sweep and export throughput/checkpoint stats."
    )
    repo_root = Path(__file__).resolve().parents[1]
    default_script_dir = repo_root / "checkpoint_eval/models/llm_distr"
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cfreqs", type=str, default="0,1,10,25,50,75,100")
    p.add_argument("--out-tag", type=str, default="opt27_pccheck_pp")
    p.add_argument("--script-dir", type=str, default=str(default_script_dir))
    p.add_argument("--entry-script", type=str, default="run_clm_pp_pccheck.py")
    p.add_argument("--ds-config", type=str, default="ds_config.json")
    p.add_argument(
        "--out-root",
        type=str,
        default=str(repo_root / "artifact_evaluation/evaluation/throughput"),
    )
    p.add_argument(
        "--lib-path",
        type=str,
        default=str(repo_root / "checkpoint_eval/pccheck/libtest_ssd.so"),
    )
    p.add_argument("--pipeline-stages", type=int, default=2)
    p.add_argument("--model-name", type=str, default="facebook/opt-2.7b")
    p.add_argument("--max-async", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--torch-adam", action="store_true")
    p.add_argument("--disable-set-storage", action="store_true")
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Additional args appended verbatim after built-in args.",
    )
    args = p.parse_args()

    if args.num_gpus % args.pipeline_stages != 0:
        raise ValueError(
            f"num_gpus ({args.num_gpus}) must be divisible by pipeline_stages ({args.pipeline_stages})"
        )

    script_dir = Path(args.script_dir)
    ds_base = script_dir / args.ds_config
    if not ds_base.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {ds_base}")
    entry_script = script_dir / args.entry_script
    if not entry_script.exists():
        raise FileNotFoundError(f"Entry script not found: {entry_script}")

    cfreqs = [int(x.strip()) for x in args.cfreqs.split(",") if x.strip()]
    out_dir = Path(args.out_root) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_world_size = args.num_gpus // args.pipeline_stages
    ds_tmp = script_dir / f"ds_config.opt27_tmp.{args.out_tag}.json"
    write_ds_tmp(ds_base, ds_tmp, train_batch_size=dp_world_size, torch_adam=args.torch_adam)

    rows: List[Dict] = []
    for cf in cfreqs:
        log_file = out_dir / f"log_{Path(args.entry_script).stem}_{cf}.txt"
        disable_set_storage = " --disable_set_storage" if args.disable_set_storage else ""
        extra_args = f" {args.extra_args.strip()}" if args.extra_args.strip() else ""
        cmd = (
            f"cd {shlex.quote(str(script_dir))} && "
            f"NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "
            f"deepspeed --num_gpus={args.num_gpus} ./{shlex.quote(args.entry_script)} "
            f"--deepspeed {shlex.quote(ds_tmp.name)} --ds_config {shlex.quote(ds_tmp.name)} "
            f"--model_name_or_path {shlex.quote(args.model_name)} "
            f"--output_dir ./output_{shlex.quote(args.out_tag)} "
            f"--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 "
            f"--do_train --per_device_train_batch_size 1 "
            f"--pipeline_stages {args.pipeline_stages} "
            f"--cfreq {cf} --bench_total_steps {args.iters} "
            f"--max_async {args.max_async} --num_threads {args.num_threads} "
            f"--c_lib_path {shlex.quote(args.lib_path)} "
            f"--bf16 --torch_dtype bfloat16 --gradient_checkpointing"
            f"{disable_set_storage}{extra_args}"
        )

        rc = run_cmd(cmd, log_file)
        parsed = parse_log(log_file)
        status = "ok" if (rc == 0 and parsed["has_all_success_markers"]) else "failed"

        thr = None
        if status == "ok" and parsed["adj_exec_time_sec"] is not None:
            thr = (args.iters - WARMUP) / parsed["adj_exec_time_sec"]

        rows.append(
            {
                "entry_script": args.entry_script,
                "cfreq": cf,
                "status": status,
                "return_code": rc,
                "exec_time_sec": parsed["exec_time_sec"],
                "adj_exec_time_sec": parsed["adj_exec_time_sec"],
                "throughput_iter_per_sec": thr,
                "single_ckpt_time_median_sec": parsed["single_ckpt_time_median_sec"],
                "single_ckpt_time_avg_sec": parsed["single_ckpt_time_avg_sec"],
                "single_ckpt_samples": parsed["single_ckpt_samples"],
                "log_file": str(log_file),
            }
        )

    rows = sorted(rows, key=lambda x: x["cfreq"])
    csv_path = out_dir / f"{Path(args.entry_script).stem}_throughput.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[RESULT] {csv_path}")

    if any(r["status"] == "ok" for r in rows):
        png_path = out_dir / f"{Path(args.entry_script).stem}_throughput.png"
        plot_single_line(rows, png_path, label=Path(args.entry_script).stem)


if __name__ == "__main__":
    main()
