#!/usr/bin/env python3
"""Aggregate runner results: per-run metrics, then median / IQR / bootstrap CI per tag.

Per-run metrics
  execution_s       timed region (post warmup, includes the tail drain)
  throughput        timed steps / execution_s
  overhead_s        execution_s - median execution_s of the matching cf=0 tag
                    (same model and method family); divided by the number of
                    checkpoints in the timed region gives overhead_per_ckpt_s
  stall_per_ckpt_ms from the step timer: sum over timed steps of
                    (gpu_ms - base_step_ms) / #checkpoints, where base_step_ms is
                    the median step time of the cf=0 run.  Excess time is counted
                    whether it lands on the checkpoint step or on the steps after it
                    (e.g. a WAR wait)
  L_ms              median durability latency (checkpoint start -> durable publish)
  peaks             GPU memory, host dirty/writeback, min MemAvailable

Ratios vs a reference tag come with a bootstrap 95% CI (resampling reps of
both tags independently).

Usage: analyze.py RESULT_ROOT [--ref TAG] [--baseline-map cf0map.json] [--out summary.json]
"""
import argparse
import json
import os
import random
import statistics as st


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_profile(path):
    ev = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return ev


def median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def boot_ci_median(xs, n=4000, seed=0):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (None, None)
    rnd = random.Random(seed)
    meds = sorted(st.median(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def boot_ci_ratio(num, den, n=4000, seed=1):
    num = [x for x in num if x is not None]
    den = [x for x in den if x is not None]
    if len(num) < 2 or len(den) < 2:
        return (None, None)
    rnd = random.Random(seed)
    rs = sorted(st.median(rnd.choices(num, k=len(num))) / st.median(rnd.choices(den, k=len(den)))
                for _ in range(n))
    return rs[int(0.025 * n)], rs[int(0.975 * n) - 1]


def per_run(root, row):
    prefix = os.path.join(root, f"{row['tag']}_r{row['rep']}")
    steps = load_json(prefix + ".steps.json") or {}
    prof = load_profile(prefix + ".profile.jsonl")
    out = dict(row)
    warm = steps.get("warmup", 3)
    timed = [(s, c, g) for s, c, g in zip(steps.get("steps", []), steps.get("is_ckpt", []),
                                          steps.get("gpu_ms", [])) if g is not None and s > warm]
    out["timed_steps"] = len(timed)
    out["timed_ckpts"] = sum(1 for _, c, _ in timed if c)
    out["step_gpu_ms"] = [g for _, _, g in timed]
    out["throughput"] = (len(timed) / row["execution_s"]) if row.get("execution_s") and timed else None
    fin = [e for e in prof if e.get("event") == "checkpoint_finalize"]
    out["L_ms"] = median([e.get("durability_latency_ms", e.get("total_ms")) for e in fin])
    out["published"] = len(fin)
    out["aborted"] = sum(1 for e in prof if e.get("event") == "checkpoint_abort")
    out["guard_wait_ms_sum"] = sum(e.get("wait_ms", 0) for e in prof if e.get("event") == "update_guard")
    out["pre_wait_ms_sum"] = sum(e.get("pre_wait_ms", 0) for e in prof if e.get("event") == "group_update")
    out["stage_ms_median"] = median([e.get("stage_ms") for e in prof if e.get("event") == "begin_checkpoint"])
    cfg = [e for e in prof if e.get("event") == "config"]
    out["payload_bytes"] = cfg[0].get("payload_bytes") if cfg else None
    out["config"] = cfg[0] if cfg else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", default=None, help="tag used as ratio denominator (per model+cf)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.root, "manifest.jsonl"))]
    runs = [per_run(a.root, r) for r in rows if r.get("rc") == 0]
    failed = [r for r in rows if r.get("rc") != 0]

    # cf=0 reference per model: median execution time and median step time
    base = {}
    for r in runs:
        if r["cf"] == 0:
            b = base.setdefault(r["model"], {"exec": [], "step": []})
            b["exec"].append(r["execution_s"])
            b["step"].extend(r["step_gpu_ms"])
    base = {m: {"exec": median(v["exec"]), "step_ms": median(v["step"])} for m, v in base.items()}

    for r in runs:
        b = base.get(r["model"])
        if b and r["cf"] > 0 and r["timed_ckpts"]:
            r["overhead_s"] = r["execution_s"] - b["exec"]
            r["overhead_per_ckpt_s"] = r["overhead_s"] / r["timed_ckpts"]
            r["stall_per_ckpt_ms"] = sum(g - b["step_ms"] for g in r["step_gpu_ms"]) / r["timed_ckpts"]
        else:
            r["overhead_s"] = r["overhead_per_ckpt_s"] = r["stall_per_ckpt_ms"] = None

    tags = {}
    for r in runs:
        tags.setdefault(r["tag"], []).append(r)
    summary = {"base": base, "failed": [{k: f[k] for k in ("tag", "rep", "rc")} for f in failed],
               "tags": {}}
    metrics = ["execution_s", "throughput", "overhead_per_ckpt_s", "stall_per_ckpt_ms", "L_ms",
               "peak_gpu_mib", "peak_dirty_kb", "peak_mem_avail_min_kb", "guard_wait_ms_sum",
               "pre_wait_ms_sum", "stage_ms_median"]
    for tag, rs in sorted(tags.items()):
        t = {"n": len(rs), "method": rs[0]["method"], "model": rs[0]["model"], "cf": rs[0]["cf"],
             "env": rs[0]["env"], "payload_bytes": rs[0]["payload_bytes"],
             "published_min": min(r["published"] for r in rs),
             "aborted_max": max(r["aborted"] for r in rs)}
        for m in metrics:
            xs = [r.get(m) for r in rs]
            med = median(xs)
            if med is None:
                continue
            lo, hi = boot_ci_median(xs)
            q1, q3 = quantile(xs, 0.25), quantile(xs, 0.75)
            t[m] = {"median": med, "iqr": [q1, q3], "ci95": [lo, hi],
                    "cv": (st.pstdev([x for x in xs if x is not None]) / st.mean([x for x in xs if x is not None]))
                    if len([x for x in xs if x is not None]) > 1 and st.mean([x for x in xs if x is not None]) else None,
                    "values": xs}
        summary["tags"][tag] = t

    if a.ref:
        ref_runs = {}
        for r in runs:
            if r["tag"].startswith(a.ref):
                ref_runs.setdefault((r["model"], r["cf"]), []).append(r["execution_s"])
        for tag, t in summary["tags"].items():
            den = ref_runs.get((t["model"], t["cf"]))
            if den and "execution_s" in t:
                num = t["execution_s"]["values"]
                lo, hi = boot_ci_ratio(den, num)
                t["speedup_vs_ref"] = {"median": median(den) / median(num), "ci95": [lo, hi]}

    out = a.out or os.path.join(a.root, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    # compact table
    print(f"{'tag':40s} {'n':>2s} {'exec_s':>9s} {'CI95':>17s} {'cv%':>5s} {'ovh/ck_s':>8s} "
          f"{'stall/ck':>9s} {'L_ms':>8s} {'speedup':>14s}")
    for tag, t in summary["tags"].items():
        e = t.get("execution_s", {})
        ci = e.get("ci95", [None, None])
        sp = t.get("speedup_vs_ref", {})
        fmt = lambda v, f: (f % v) if v is not None else "-"
        print(f"{tag:40s} {t['n']:2d} {fmt(e.get('median'), '%9.2f')} "
              f"{fmt(ci[0], '%8.2f')}-{fmt(ci[1], '%-8.2f')} {fmt((e.get('cv') or 0) * 100, '%5.1f')} "
              f"{fmt(t.get('overhead_per_ckpt_s', {}).get('median'), '%8.2f')} "
              f"{fmt(t.get('stall_per_ckpt_ms', {}).get('median'), '%9.0f')} "
              f"{fmt(t.get('L_ms', {}).get('median'), '%8.0f')} "
              f"{fmt(sp.get('median'), '%5.2f')}[{fmt((sp.get('ci95') or [None])[0], '%.2f')},"
              f"{fmt((sp.get('ci95') or [None, None])[1], '%.2f')}]")
    if failed:
        print(f"FAILED runs: {summary['failed']}")


if __name__ == "__main__":
    main()
