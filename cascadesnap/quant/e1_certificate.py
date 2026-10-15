#!/usr/bin/env python3
"""E1: static error certificate of moment quantization on real training states.

Train a model; at each step in --at, compute for every scheme x block size
  rel_u / cos_u   error of the Adam update direction u = m_hat/(sqrt(v_hat)+eps)
  rel_m / rel_v   relative L2 error of the moments
  guard_frac      fraction of elements where sqrt(v) < Delta/2 (code 0 without the guard)
  danger_frac     ... where additionally the quantized m is non-zero (u ~ m/eps blow-up)
  u_max/ut_max    largest |u| before / after QDQ
broken down by tensor kind (embedding / attention / mlp / norm_bias / head).

No resume happens here: this measures how far a restored state would be from
the exact one at each point of training.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="scratch:openai-community/gpt2")
ap.add_argument("--at", default="50,200,1000,3000")
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--beta2", type=float, default=0.95)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--total", type=int, default=10000, help="LR schedule length")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

at = sorted(int(x) for x in a.at.split(","))
betas = (0.9, a.beta2)
toks = qlib.load_tokens("train")
batches = qlib.Batches(toks, a.bs, a.seq, seed=a.seed)
model = qlib.make_model(a.model, seed=a.seed)
opt = qlib.make_opt(model, a.lr, betas, a.wd)
configs = [("int8_guard", b) for b in (256, 1024, 2048, 4096)] + \
          [("int8_noguard", 2048), ("fp8_e4m3", 2048), ("bf16", 0)]
meta = vars(a)
t0 = time.time()
with open(a.out, "w") as f:
    for step in range(1, at[-1] + 1):
        loss = qlib.train_step(model, opt, batches(step - 1), qlib.lr_at(step - 1, a.lr, a.warmup, a.total))
        if step in at:
            for scheme, block in configs:
                c = qlib.certificate(model, opt, scheme, block or 2048, step, betas, 1e-8)
                c.update({"train_loss": float(loss), "meta": meta})
                f.write(json.dumps(c) + "\n")
                f.flush()
            print(f"step {step} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
