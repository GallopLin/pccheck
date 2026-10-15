#!/usr/bin/env python3
"""Single-GPU experiment runner (A800 workstation).

Every run goes through the same steps:
  1. wait until the GPU is idle and host dirty/writeback pages have settled
     (no drop_caches permission here; O_DIRECT writes avoid most page-cache
     effects, and the pre-run covariates are recorded instead)
  2. launch the HF entry point in a fresh work dir on the checkpoint SSD
  3. sample GPU memory/utilisation and host MemAvailable/Dirty/Writeback every second
  4. collect log, profile JSONL, per-step timing JSON and metadata; delete the work dir
  5. append a manifest row (JSONL)

The runner is idempotent: a (tag, rep) that already has rc==0 in the
manifest is skipped, so an interrupted batch resumes where it stopped.
Within each repetition the method order is rotated, so slow drift of the
machine does not systematically favour one method.

Usage:
  runner.py --matrix matrix.json --result-root DIR [--reps 5] [--dry-run]

matrix.json:
  {"defaults": {...}, "runs": [{"tag": "...", "method": "cascade|pccheck|dcp|checkfreq|gpm",
                                 "model": "...", "cf": 10, "steps": 60, "env": {...},
                                 "args": [...]}, ...]}
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HOME = "/home/linzhicheng"
PY = os.environ.get("BENCH_PY", f"{HOME}/anaconda3/envs/pccheck/bin/python")
PCCHECK_ROOT = os.environ.get("PCCHECK_ROOT", f"{HOME}/code/pccheck-cascade-v3-merge")
TRANSFORMERS_ROOT = os.environ.get("TRANSFORMERS_ROOT", f"{HOME}/code/transformers-cascade-v3-merge")
EXAMPLES = f"{TRANSFORMERS_ROOT}/examples/pytorch/language-modeling"
CASCADE_LIB = f"{PCCHECK_ROOT}/checkpoint_eval/pccheck/libtest_ssd.so"
PCCHECK_LIB = os.environ.get("PCCHECK_BASELINE_LIB", CASCADE_LIB)
WORK_ROOT = os.environ.get("BENCH_WORK_ROOT", "/mnt/data1/linzhicheng_bench/runs")

MODELS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-large": "openai-community/gpt2-large",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
}

ENTRY = {
    "cascade": "run_clm_multistream.py",
    "dcp": "run_clm_multistream.py",
    "pccheck": "run_clm_pccheck.py",
    "checkfreq": "run_clm_cfreq.py",
    "gpm": "run_clm_gpm.py",
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemAvailable", "Dirty", "Writeback", "Cached"):
                out[k] = int(v.split()[0])
    return out


def gpu_state():
    q = sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0")
    try:
        used, util = (int(x) for x in q.split(","))
    except ValueError:
        used, util = -1, -1
    return used, util


def wait_gpu_idle(log):
    stable = 0
    t0 = time.time()
    while stable < 3:
        used, util = gpu_state()
        stable = stable + 1 if (0 <= used <= 1024 and util <= 5) else 0
        if stable < 3:
            time.sleep(5)
        if time.time() - t0 > 3600:
            raise RuntimeError("GPU not idle for 1h")
    log(f"gpu idle after {time.time() - t0:.0f}s")


def settle_io(log, dirty_kb=65536, wb_kb=4096, timeout=900):
    os.sync()
    stable = 0
    t0 = time.time()
    while stable < 3:
        m = meminfo()
        stable = stable + 1 if (m["Dirty"] <= dirty_kb and m["Writeback"] <= wb_kb) else 0
        if stable < 3:
            time.sleep(2)
        if time.time() - t0 > timeout:
            raise RuntimeError(f"io did not settle in {timeout}s: {m}")
    log(f"io settled after {time.time() - t0:.0f}s")


class Monitor(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop = threading.Event()
        self.peak = {"gpu_mib": 0, "dirty_kb": 0, "writeback_kb": 0, "mem_avail_min_kb": 1 << 60}

    def run(self):
        with open(self.path, "w") as f:
            f.write("t,gpu_mib,gpu_util,mem_avail_kb,dirty_kb,writeback_kb\n")
            while not self.stop.is_set():
                used, util = gpu_state()
                m = meminfo()
                f.write(f"{time.time():.1f},{used},{util},{m['MemAvailable']},{m['Dirty']},{m['Writeback']}\n")
                f.flush()
                self.peak["gpu_mib"] = max(self.peak["gpu_mib"], used)
                self.peak["dirty_kb"] = max(self.peak["dirty_kb"], m["Dirty"])
                self.peak["writeback_kb"] = max(self.peak["writeback_kb"], m["Writeback"])
                self.peak["mem_avail_min_kb"] = min(self.peak["mem_avail_min_kb"], m["MemAvailable"])
                self.stop.wait(1.0)


def build_cmd(run, work):
    method = run["method"]
    model = MODELS.get(run["model"], run["model"])
    common = ["--model_name_or_path", model, "--output_dir", f"{work}/hf_output",
              "--dataset_name", run.get("dataset", "wikitext"),
              "--dataset_config_name", run.get("dataset_config", "wikitext-2-raw-v1"),
              "--do_train", "--per_device_train_batch_size", str(run.get("bs", 1)),
              "--optim", run.get("optim", "adamw_torch_fused"),
              "--cfreq", str(run["cf"]), "--bench_total_steps", str(run["steps"])]
    if method in ("cascade", "dcp"):
        extra = ["--max_async", str(run.get("max_async", 2)), "--num_threads", "2",
                 "--num_layer_groups", str(run.get("groups", 6)), "--c_lib_path", CASCADE_LIB,
                 "--multistream_checkpoint_file", f"{work}/multistream.chk",
                 "--multistream_metadata_file", f"{work}/multistream.chk.metadata.json"]
    elif method == "pccheck":
        extra = ["--max_async", str(run.get("max_async", 2)), "--num_threads", "2",
                 "--psize", "4", "--dram_ratio", "2.0", "--c_lib_path", PCCHECK_LIB]
    else:
        extra = []
    return [PY, "-u", f"{EXAMPLES}/{ENTRY[method]}"] + common + extra + list(run.get("args", []))


def run_env(run, work, prefix):
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{TRANSFORMERS_ROOT}/src:{PCCHECK_ROOT}",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONFAULTHANDLER": "1",
        "PCCHECK_BENCH_WARMUP": str(run.get("warmup", 3)),
        "PCCHECK_MS_PROFILE": "1",
        "PCCHECK_MS_PROFILE_PATH": f"{prefix}.profile.jsonl",
        "PCCHECK_MS_PROFILE_RUN_ID": os.path.basename(prefix),
        "PCCHECK_STEP_EVENTS_PATH": f"{prefix}.steps.json",
        "DCP_CHECKPOINT_DIR": f"{work}/dcp",
    })
    if run["method"] == "dcp":
        env["PCCHECK_BACKEND"] = "dcp"
    if run["method"] == "pccheck":
        env.setdefault("PCCHECK_DROP_GRAD", "1")  # 3N，与其他方法同 payload
    env.update({k: str(v) for k, v in run.get("env", {}).items()})
    return env


def file_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _sigterm(*_):
    raise KeyboardInterrupt("SIGTERM")


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()

    a.result_root = os.path.abspath(a.result_root)  # 子进程在工作目录里运行
    spec = json.load(open(a.matrix))
    defaults = spec.get("defaults", {})
    runs = [dict(defaults, **r) for r in spec["runs"]]
    reps = a.reps or int(defaults.get("reps", 5))
    os.makedirs(a.result_root, exist_ok=True)
    shutil.copy(a.matrix, os.path.join(a.result_root, "matrix.json"))
    manifest = os.path.join(a.result_root, "manifest.jsonl")
    progress = os.path.join(a.result_root, "progress.log")
    done = set()
    if os.path.exists(manifest):
        for line in open(manifest):
            r = json.loads(line)
            if r.get("rc") == 0:
                done.add((r["tag"], r["rep"]))

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(progress, "a") as f:
            f.write(line + "\n")

    provenance = {
        "pccheck_commit": sh(f"git -C {PCCHECK_ROOT} rev-parse HEAD"),
        "pccheck_dirty": bool(sh(f"git -C {PCCHECK_ROOT} status --porcelain --untracked-files=no")),
        "transformers_commit": sh(f"git -C {TRANSFORMERS_ROOT} rev-parse HEAD"),
        "transformers_dirty": bool(sh(f"git -C {TRANSFORMERS_ROOT} status --porcelain --untracked-files=no")),
        "cascade_lib_sha256": file_sha(CASCADE_LIB),
        "pccheck_lib_sha256": file_sha(PCCHECK_LIB),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0"),
    }
    log(f"provenance {json.dumps(provenance)}")

    for rep in range(1, reps + 1):
        k = (rep - 1) % max(1, len(runs))
        order = runs[k:] + runs[:k]
        for run in order:
            tag = run["tag"]
            if (tag, rep) in done:
                continue
            prefix = os.path.join(a.result_root, f"{tag}_r{rep}")
            work = os.path.join(WORK_ROOT, f"work_{os.path.basename(a.result_root)}_{tag}_r{rep}")
            cmd = build_cmd(run, work)
            if a.dry_run:
                print(" ".join(cmd))
                continue
            wait_gpu_idle(log)
            settle_io(log)
            pre = meminfo()
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(f"{work}/hf_output", exist_ok=True)
            env = run_env(run, work, prefix)
            log(f"START {tag} rep={rep}")
            mon = Monitor(f"{prefix}.monitor.csv")
            mon.start()
            t0 = time.time()
            with open(f"{prefix}.log", "w") as lf:
                lf.write(json.dumps({"cmd": cmd, "env": {k2: env[k2] for k2 in sorted(env)
                                                        if k2.startswith(("PCCHECK", "DCP", "BENCH"))},
                                     "provenance": provenance}) + "\n")
                lf.flush()
                # 子进程独立进程组：超时或 runner 被终止时整组清理，不留孤儿训练进程
                p = subprocess.Popen(cmd, cwd=work, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                try:
                    rc = p.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    rc = -999
                except BaseException:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    mon.stop.set()
                    raise
            wall = time.time() - t0
            mon.stop.set()
            mon.join()
            exec_s = None
            for line in open(f"{prefix}.log", errors="replace"):
                if line.startswith("EXECUTION TIME:"):
                    exec_s = float(line.split()[2])
            meta = f"{work}/multistream.chk.metadata.json"
            if os.path.exists(meta):
                shutil.copy(meta, f"{prefix}.metadata.json")
            disk_kib = int(sh(f"du -sk {work} | cut -f1") or 0)
            shutil.rmtree(work, ignore_errors=True)
            row = {"tag": tag, "rep": rep, "method": run["method"], "model": run["model"],
                   "cf": run["cf"], "steps": run["steps"], "env": run.get("env", {}),
                   "rc": rc, "execution_s": exec_s, "wall_s": wall, "work_disk_kib": disk_kib,
                   "pre_dirty_kb": pre["Dirty"], "pre_writeback_kb": pre["Writeback"],
                   "pre_cached_kb": pre["Cached"], "pre_mem_avail_kb": pre["MemAvailable"],
                   **{f"peak_{k2}": v for k2, v in mon.peak.items()}}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"DONE {tag} rep={rep} rc={rc} exec={exec_s} wall={wall:.0f}s")
            settle_io(log)
    log("COMPLETE")


if __name__ == "__main__":
    main()
