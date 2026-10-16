#!/usr/bin/env python3
"""E1b: which moment drives the update error?  Same training as E1; at each step
certificates for (m format, v format) pairs, including fp8 with the sqrt-domain guard."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="scratch:openai-community/gpt2")
ap.add_argument("--at", default="200,1000,3000")
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--beta2", type=float, default=0.95)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--total", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
at = sorted(int(x) for x in a.at.split(","))
betas = (0.9, a.beta2)
batches = qlib.Batches(qlib.load_tokens("train"), a.bs, a.seq, seed=a.seed)
model = qlib.make_model(a.model, seed=a.seed)
opt = qlib.make_opt(model, a.lr, betas, a.wd)
pairs = [  # (m scheme, m block, v scheme, v block)
    ("int8_guard", 256, "int8_guard", 256),
    ("int8_guard", 256, "bf16", 256),
    ("bf16", 256, "int8_guard", 256),
    ("int8_guard", 256, "fp8_guard", 256),
    ("int8_guard", 2048, "fp8_guard", 256),
    ("fp8_guard", 256, "fp8_guard", 256),
    ("fp8_guard", 128, "fp8_guard", 128),
    ("fp8_guard", 1024, "fp8_guard", 1024),
    ("fp8_e4m3", 256, "fp8_e4m3", 256),
]
t0 = time.time()
with open(a.out, "w") as f:
    for step in range(1, at[-1] + 1):
        loss = qlib.train_step(model, opt, batches(step - 1), qlib.lr_at(step - 1, a.lr, a.warmup, a.total))
        if step in at:
            for sm, bm, sv, bv in pairs:
                c = qlib.certificate(model, opt, sm, bm, step, betas, 1e-8, scheme_v=sv, block_v=bv)
                c.update({"train_loss": float(loss), "meta": vars(a)})
                f.write(json.dumps(c) + "\n")
                f.flush()
            print(f"step {step} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
