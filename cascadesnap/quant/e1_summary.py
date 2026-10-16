#!/usr/bin/env python3
"""E1 summary: one table per run (rows = scheme/block, columns = training steps).

Usage: e1_summary.py E1_DIR [--metric rel_u]
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--metric", default="rel_u")
a = ap.parse_args()

for path in sorted(glob.glob(os.path.join(a.dir, "e1_*.jsonl"))):
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        continue
    steps = sorted({r["step"] for r in rows})
    cfgs = []
    for r in rows:
        k = (r["scheme"], r["block"])
        if k not in cfgs:
            cfgs.append(k)
    print(f"\n== {os.path.basename(path)}  metric={a.metric}  (model={rows[0]['meta']['model']}, "
          f"beta2={rows[0]['meta']['beta2']})")
    print(f"{'scheme':14s} {'block':>5s} " + " ".join(f"{'s' + str(s):>10s}" for s in steps)
          + f" {'worst kind (max rel_u)':>34s}")
    for sch, blk in cfgs:
        vals, worst = [], ""
        for s in steps:
            r = next((x for x in rows if x["step"] == s and x["scheme"] == sch and x["block"] == blk), None)
            v = r.get(a.metric) if r else None
            vals.append(f"{v:10.3g}" if isinstance(v, (int, float)) else f"{'-':>10s}")
            if r and s == steps[-1] and r.get("by_kind"):
                k, d = max(r["by_kind"].items(), key=lambda kv: kv[1]["rel_u"][1])
                worst = f"{k}:{d['rel_u'][1]:.3g}"
        print(f"{sch:14s} {blk:5d} " + " ".join(vals) + f" {worst:>34s}")
