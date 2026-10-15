#!/usr/bin/env python3
"""E2/E3: does restoring from a quantized-moment checkpoint change training?

For one seed:
  1. train from scratch to step s0, then take snapshot S (params + optimizer, in memory)
  2. every variant restarts from S and trains K more steps on exactly the same batches
       ref          no restore at all (the original process keeps going): measures run-to-run nondeterminism
       exact        restore S unchanged
       int8_guard   restore S with int8 moments (production quantizer, default block)
       int8_noguard same, old quantizer without the sqrt-domain floor
       fp8_e4m3     restore with fp8 moments
       rep_k{k}     int8_guard QDQ applied again every k steps (repeated recovery, E3)
  3. log the validation loss every --eval-every steps and the final parameter distance to `exact`

Output: one JSON line per (seed, variant).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="scratch:openai-community/gpt2")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--s0", type=int, default=2000)
ap.add_argument("--K", type=int, default=5000)
ap.add_argument("--variants", default="ref,exact,int8_guard,int8_noguard,fp8_e4m3,rep_k100,rep_k500")
ap.add_argument("--block", type=int, default=2048)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--beta2", type=float, default=0.95)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
betas = (0.9, a.beta2)
total = a.s0 + a.K
toks = qlib.load_tokens("train")
val = qlib.make_val(qlib.load_tokens("validation"), 8, a.seq, n_batches=40)
batches = qlib.Batches(toks, a.bs, a.seq, seed=a.seed)
model = qlib.make_model(a.model, seed=a.seed)
opt = qlib.make_opt(model, a.lr, betas, a.wd)
t0 = time.time()
for i in range(a.s0):
    qlib.train_step(model, opt, batches(i), qlib.lr_at(i, a.lr, a.warmup, total))
S = qlib.snapshot(model, opt)
val_s0 = qlib.evaluate(model, val)
print(f"[seed {a.seed}] s0={a.s0} val={val_s0:.4f} ({time.time() - t0:.0f}s)", flush=True)


def continue_run(variant):
    if variant != "ref":
        qlib.restore(model, opt, S)
    if variant in ("int8_guard", "int8_noguard", "fp8_e4m3"):
        qlib.apply_qdq_(opt, variant, a.block)
    rep_k = int(variant.split("rep_k")[1]) if variant.startswith("rep_k") else 0
    if rep_k:
        qlib.apply_qdq_(opt, "int8_guard", a.block)
    curve, losses = [], []
    for j in range(a.K):
        i = a.s0 + j
        if rep_k and j > 0 and j % rep_k == 0:
            qlib.apply_qdq_(opt, "int8_guard", a.block)
        loss = qlib.train_step(model, opt, batches(i), qlib.lr_at(i, a.lr, a.warmup, total))
        losses.append(loss)
        if (j + 1) % a.eval_every == 0 or j + 1 == a.K:
            tl = torch.stack(losses).float()
            curve.append({"step": i + 1, "val": qlib.evaluate(model, val),
                          "train_mean": float(tl.mean()), "train_max": float(tl.max())})
            losses = []
    return curve


results = {}
final_params = {}
order = [v for v in a.variants.split(",") if v]
if "ref" in order:  # ref must run first: it continues the original process with no restore
    order.remove("ref")
    order.insert(0, "ref")
with open(a.out, "a") as f:
    for v in order:
        tv = time.time()
        curve = continue_run(v)
        params = torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])
        final_params[v] = params.cpu()
        rec = {"seed": a.seed, "variant": v, "s0": a.s0, "K": a.K, "block": a.block,
               "beta2": a.beta2, "model": a.model, "val_s0": val_s0, "curve": curve,
               "final_val": curve[-1]["val"], "seconds": time.time() - tv}
        if "exact" in final_params and v != "exact":
            e = final_params["exact"]
            rec["param_rel_dist_to_exact"] = float((params.cpu() - e).norm() / e.norm())
        results[v] = rec
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[seed {a.seed}] {v}: final val {rec['final_val']:.4f} "
              f"dist={rec.get('param_rel_dist_to_exact')} ({rec['seconds']:.0f}s)", flush=True)
    # ref ran before exact, so fill in its distance afterwards
    if "ref" in results and "exact" in final_params:
        e = final_params["exact"]
        d = float((final_params["ref"] - e).norm() / e.norm())
        f.write(json.dumps({"seed": a.seed, "variant": "ref_distance", "param_rel_dist_to_exact": d}) + "\n")
