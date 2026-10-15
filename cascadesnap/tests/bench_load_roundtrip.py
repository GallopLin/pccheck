"""CascadeSave → CascadeLoad 端到端 roundtrip 验证（v3 格式）。

save 阶段：小 GPT 训 4 步、每步 checkpoint（异步），旁路 torch.save 全量
ground truth（params/moments/RNG/scheduler/step/最终槽位）。
load 阶段：**全新进程**、全新模型/优化器，经 MultiStreamStateLoader 恢复后与
ground truth 逐位比对；extra blob 应用后 RNG/scheduler 逐位比对；最后做
磁盘 crc 篡改注入，断言拒载。

用法:
  python bench_load_roundtrip.py --phase save --config quant_split
  python bench_load_roundtrip.py --phase load --config quant_split
"""
import argparse
import json
import os
import sys

CONFIGS = {
    #                quant  grouping  weights_first(load)
    "fp32_split":   (False, "split",  True),
    "fp32_atomic":  (False, "atomic", False),
    "quant_split":  (True,  "split",  True),
    "quant_atomic": (True,  "atomic", False),
}

p = argparse.ArgumentParser()
p.add_argument("--phase", required=True, choices=["save", "load"])
p.add_argument("--config", required=True, choices=list(CONFIGS))
p.add_argument("--moddir", default="/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck")
p.add_argument("--ckpt-dir", default="/mnt/data1/linzhicheng_bench")
p.add_argument("--iters", type=int, default=4)
args = p.parse_args()

QUANT, GROUPING, WEIGHTS_FIRST = CONFIGS[args.config]
os.environ["PCCHECK_GROUPING"] = GROUPING
os.environ.setdefault("PCCHECK_GROUP_FLUSH", "1")

sys.path.insert(0, args.moddir)
sys.path.insert(0, "/home/linzhicheng/download/pipelayer/src")

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

LIB = os.path.join(args.moddir, "libtest_ssd.so")
CKPT = os.path.join(args.ckpt_dir, f"rt_{args.config}.chk")
TRUTH = CKPT + ".truth.pt"
V, H, L, FF = 32000, 1024, 4, 4096
BATCH, SEQ, GROUPS, WARMUP = 4, 256, 6, 2
dev = "cuda:0"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H)
        self.q, self.k, self.v, self.o = (nn.Linear(H, H) for _ in range(4))
        self.ln2 = nn.LayerNorm(H)
        self.up, self.down = nn.Linear(H, FF), nn.Linear(FF, H)

    def forward(self, x):
        h = self.ln1(x)
        att = torch.softmax(self.q(h) @ self.k(h).transpose(-1, -2) / H ** 0.5, dim=-1)
        x = x + self.o(att @ self.v(h))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}")
    return bool(ok)


# =============================== SAVE ===============================
if args.phase == "save":
    from multistream_checkpoint import (
        MultiStreamCheckpoint, build_param_layout,
        quantize_int8_blocks, dequantize_int8_blocks, QUANT_BLOCK_DEFAULT)

    torch.manual_seed(0)
    model = GPT().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"[save:{args.config}] params={n_params / 1e6:.1f}M quant={QUANT} grouping={GROUPING}")

    def fwd_bwd():
        ids = torch.randint(0, V, (BATCH, SEQ), device=dev)
        loss = F.cross_entropy(model(ids).view(-1, V), ids.view(-1))
        opt.zero_grad(set_to_none=False)
        loss.backward()
        return loss

    for _ in range(WARMUP):
        fwd_bwd(); opt.step(); sch.step()
    torch.cuda.synchronize()

    layout = build_param_layout(model, opt, quant_moments=QUANT)
    ea = sum(li["exp_avg_size"] for li in layout)
    eas = sum(li["exp_avg_sq_size"] for li in layout)
    bufs = {"param": torch.zeros(n_params, dtype=torch.float32, device=dev),
            "exp_avg": torch.zeros(ea, dtype=torch.float32, device=dev),
            "exp_avg_sq": torch.zeros(eas, dtype=torch.float32, device=dev)}
    mschk = MultiStreamCheckpoint(
        param_layout=layout, gpu_ar=bufs, total_size=n_params, num_threads=8,
        lib_path=LIB, filename=CKPT, num_streams=4, max_async=2,
        num_layer_groups=GROUPS, metadata_path=CKPT + ".metadata.json")
    wrapped = mschk.create_optimizer(opt, model)
    cur_iter = {"i": -1}
    mschk.attach_extra_state(scheduler=sch,
                             provider=lambda: {"iteration": cur_iter["i"]})

    final_slot = None
    truth = None
    for i in range(args.iters):
        cur_iter["i"] = i
        fwd_bwd()
        slot = wrapped.begin_checkpoint()
        wrapped.step_with_callback()
        sch.step()
        if i == args.iters - 1:
            # ground truth：最后一次 finalize（=blob 捕获点）之前的完整状态
            final_slot = slot
            torch.cuda.synchronize()
            truth = {
                "config": args.config, "final_slot": int(slot), "iteration": i,
                "params": {n: p.detach().float().cpu().clone()
                           for n, p in model.named_parameters()},
                "step": float(next(iter(opt.state.values()))["step"].item()),
                "scheduler": sch.state_dict(),
                "rng": {"torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate()},
            }
            for key, sqrt_domain in (("exp_avg", False), ("exp_avg_sq", True)):
                out = {}
                for n, prm in model.named_parameters():
                    live = opt.state[prm][key].detach().float().view(-1)
                    if QUANT:
                        q, sc = quantize_int8_blocks(live, QUANT_BLOCK_DEFAULT,
                                                     sqrt_domain=sqrt_domain)
                        live = dequantize_int8_blocks(q.cpu(), sc.cpu(),
                                                      QUANT_BLOCK_DEFAULT,
                                                      sqrt_domain=sqrt_domain)
                    out[n] = live.cpu().clone()
                truth[key] = out
        wrapped.finalize_checkpoint(wait=False)

    import time
    deadline = time.time() + 300
    while mschk._checkpoint_slots and time.time() < deadline:
        time.sleep(0.2)
    assert not mschk._checkpoint_slots, "drain timeout"
    torch.save(truth, TRUTH)
    print(f"[save:{args.config}] done: final_slot={final_slot} truth -> {TRUTH}")
    os._exit(0)

# =============================== LOAD ===============================
if WEIGHTS_FIRST:
    os.environ["PIPELAYER_WEIGHTS_FIRST"] = "1"
from pipelayer.checkpointing import MultiStreamStateLoader

truth = torch.load(TRUTH, map_location="cpu", weights_only=False)
torch.manual_seed(123)  # 与 save 端不同的初始化，证明恢复确实覆写了一切
model = GPT().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

loader = MultiStreamStateLoader(
    model, opt, chkpt_dir=args.ckpt_dir, lib_path=LIB,
    metadata_file=CKPT + ".metadata.json",
    parall_iter=int(truth["final_slot"]), weights_first=WEIGHTS_FIRST)
loader.wait_all_full()
torch.cuda.synchronize()

ok = True
bad_p = [n for n, p in model.named_parameters()
         if not np.array_equal(p.detach().float().cpu().numpy(),
                               truth["params"][n].numpy())]
ok &= check("params bit-exact", not bad_p, f"({len(bad_p)} mismatched)" if bad_p else "")
for key in ("exp_avg", "exp_avg_sq"):
    bad = [n for n, p in model.named_parameters()
           if not np.array_equal(opt.state[p][key].detach().float().view(-1).cpu().numpy(),
                                 truth[key][n].view(-1).numpy())]
    label = f"{key} bit-exact" + (" (vs quant-dequant truth)" if QUANT else "")
    ok &= check(label, not bad, f"({len(bad)} mismatched: {bad[:3]})" if bad else "")
step_got = float(next(iter(opt.state.values()))["step"].item())
ok &= check("optimizer step", step_got == truth["step"], f"({step_got} vs {truth['step']})")

extra = loader.load_extra_state()
user = loader.apply_extra_state(extra, scheduler=sch)
r, tr = extra and True, truth["rng"]
ok &= check("rng torch_cpu", torch.equal(torch.get_rng_state(), tr["torch_cpu"]))
got_cuda = torch.cuda.get_rng_state_all()
ok &= check("rng torch_cuda", all(torch.equal(a, b) for a, b in zip(got_cuda, tr["torch_cuda"])))
np_got, np_exp = np.random.get_state(), tr["numpy"]
ok &= check("rng numpy", np_got[0] == np_exp[0] and np.array_equal(np_got[1], np_exp[1])
            and np_got[2:] == np_exp[2:])
ok &= check("rng python", random.getstate() == tr["python"])
ok &= check("scheduler state", sch.state_dict() == truth["scheduler"],
            f"({sch.state_dict()} vs {truth['scheduler']})")
ok &= check("user dict (dataloader 契约)", user == {"iteration": truth["iteration"]}, f"({user})")

if WEIGHTS_FIRST:
    pt, ft = loader._param_ready_time, loader._full_ready_time
    lead = [ft[c] - pt[c] for c in pt if c in ft]
    ok &= check("weights-first two-phase", len(pt) == loader.num_chunks and all(v >= 0 for v in lead),
                f"(param 先于 full 就绪，平均领先 {sum(lead)/max(len(lead),1)*1000:.0f}ms)")
loader.stop()

# crc 篡改注入：翻转 extra 流 payload 一个字节 → 解析必须拒载
meta = json.load(open(CKPT + ".metadata.json"))
stride = int(meta["checkpoint_stride_floats"])
extra_off = sum(meta["stream_sizes"][:4])
byte_pos = ((int(meta["max_async"]) + 4) * 4096
            + truth["final_slot"] * stride + extra_off) * 4 + 64
with open(CKPT, "r+b") as f:
    f.seek(byte_pos)
    b = f.read(1)
    f.seek(byte_pos)
    f.write(bytes([b[0] ^ 0xFF]))
raw = np.memmap(CKPT, dtype=np.uint8, mode="r",
                offset=byte_pos - 64, shape=(int(meta["extra_state"]["floats"]) * 4,))
from multistream_checkpoint import MultiStreamCheckpoint as _MS
try:
    _MS.parse_extra_state_blob(bytes(raw))
    ok &= check("crc tamper detection", False, "(corrupted blob was accepted!)")
except ValueError:
    ok &= check("crc tamper detection", True)

print(f"\n[load:{args.config}] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
os._exit(0 if ok else 1)
