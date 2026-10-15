"""Moments int8/fp8 量化 checkpoint 的 resume loss 连续性验证。

协议：
  1) 固定 64 个数据 batch（数据序 = 迭代号的确定函数，无 RNG 依赖；模型无 dropout）。
  2) 训练 N1 步 → 打快照 S（model + optimizer state 完整克隆到 CPU）。
  3) ref     ：进程内直接继续 N2 步（不重载）。
  4) exact   ：从 S 精确恢复再训 N2 步 → 与 ref 的差 = 非确定性噪声底（backward 原子操作）。
  5) m8v8    ：恢复后对 exp_avg 做 per-block 对称 int8 QDQ、exp_avg_sq 做 sqrt 域 int8 QDQ。
  6) fp8     ：恢复后对两个 moment 做 per-block 缩放 E4M3 QDQ。
  7) w8      ：m8v8 基础上连权重也 int8 QDQ（对照：权重敏感度）。
  8) m8_p50  ：精确恢复，之后每 50 步对 moments 原位 QDQ int8（等价于反复从 int8 快照恢复的复合误差）。
判据：|loss_variant - loss_exact| 与 |loss_exact - loss_ref|（噪声底）同量级 → 连续性成立。
"""
import argparse
import copy
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--n1", type=int, default=200)
p.add_argument("--n2", type=int, default=200)
p.add_argument("--out", required=True)
args = p.parse_args()

torch.manual_seed(0)
dev = "cuda:0"
V, H, L, FF = 32000, 1024, 12, 4096
B, S = 8, 512
N_BATCH = 64
BLOCK = 4096


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H)
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.o = nn.Linear(H, H)
        self.ln2 = nn.LayerNorm(H)
        self.up = nn.Linear(H, FF)
        self.down = nn.Linear(FF, H)

    def forward(self, x):
        h = self.ln1(x)
        att = torch.softmax(self.q(h) @ self.k(h).transpose(-1, -2) / H ** 0.5, dim=-1)
        x = x + self.o(att @ self.v(h))
        h = self.ln2(x)
        return x + self.down(F.gelu(self.up(h)))


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


g = torch.Generator().manual_seed(1234)
batches = [torch.randint(0, V, (B, S), generator=g) for _ in range(N_BATCH)]


def make():
    torch.manual_seed(0)
    model = GPT().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, opt


def train(model, opt, start, n):
    losses = []
    for it in range(start, start + n):
        ids = batches[it % N_BATCH].to(dev, non_blocking=True)
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, V), ids.view(-1))
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return losses


# ---------------- QDQ 算子 ----------------
def qdq_int8(x, sqrt_domain=False):
    """per-block(4096) 对称 int8 量化-反量化；sqrt_domain 用于非负、动态范围大的 v。"""
    flat = x.detach().view(-1).float()
    y = flat.sqrt() if sqrt_domain else flat
    n = y.numel()
    pad = (BLOCK - n % BLOCK) % BLOCK
    if pad:
        y = torch.cat([y, y.new_zeros(pad)])
    yb = y.view(-1, BLOCK)
    scale = yb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0
    q = torch.clamp((yb / scale).round(), -127, 127)
    deq = (q * scale).view(-1)[:n]
    if sqrt_domain:
        deq = deq.square()
    return deq.view_as(x).to(x.dtype)


def qdq_fp8(x, sqrt_domain=False):
    """per-block 缩放 + E4M3 往返（块内最大值归一到 240，留出 E4M3 448 上限余量）。"""
    flat = x.detach().view(-1).float()
    y = flat.sqrt() if sqrt_domain else flat
    n = y.numel()
    pad = (BLOCK - n % BLOCK) % BLOCK
    if pad:
        y = torch.cat([y, y.new_zeros(pad)])
    yb = y.view(-1, BLOCK)
    scale = yb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 240.0
    deq = ((yb / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale).view(-1)[:n]
    if sqrt_domain:
        deq = deq.square()
    return deq.view_as(x).to(x.dtype)


def rel_err(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def apply_qdq(opt, fn, also_weights=None):
    errs_m, errs_v = [], []
    with torch.no_grad():
        for group in opt.param_groups:
            for prm in group["params"]:
                st = opt.state.get(prm, {})
                if "exp_avg" in st:
                    q = fn(st["exp_avg"], sqrt_domain=False)
                    errs_m.append(rel_err(q, st["exp_avg"]))
                    st["exp_avg"].copy_(q)
                if "exp_avg_sq" in st:
                    q = fn(st["exp_avg_sq"], sqrt_domain=True)
                    errs_v.append(rel_err(q, st["exp_avg_sq"]))
                    st["exp_avg_sq"].copy_(q)
                if also_weights is not None:
                    qw = also_weights(prm.data, sqrt_domain=False)
                    prm.data.copy_(qw)
    return (sum(errs_m) / max(len(errs_m), 1), sum(errs_v) / max(len(errs_v), 1))


# ---------------- 主流程 ----------------
print(f"[quant] warmup {args.n1} iters...")
model, opt = make()
t0 = time.time()
warm = train(model, opt, 0, args.n1)
print(f"[quant] warmup done ({time.time()-t0:.0f}s), loss {warm[0]:.3f} -> {warm[-1]:.3f}")

snap_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
snap_opt = copy.deepcopy(opt.state_dict())
for st in snap_opt["state"].values():
    for k, v in st.items():
        if isinstance(v, torch.Tensor):
            st[k] = v.detach().cpu().clone()

losses = {}
losses["ref"] = train(model, opt, args.n1, args.n2)
del model, opt
torch.cuda.empty_cache()

VARIANTS = {
    "exact": dict(fn=None),
    "m8v8": dict(fn=qdq_int8),
    "fp8": dict(fn=qdq_fp8),
    "w8": dict(fn=qdq_int8, weights=True),
    "m8_p50": dict(fn=None, periodic=50),
}
qerr = {}
for name, cfg in VARIANTS.items():
    model, opt = make()
    model.load_state_dict({k: v.to(dev) for k, v in snap_model.items()})
    sd = copy.deepcopy(snap_opt)
    for st in sd["state"].values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(dev)
    opt.load_state_dict(sd)
    if cfg.get("fn") is not None:
        qerr[name] = apply_qdq(opt, cfg["fn"],
                               also_weights=cfg["fn"] if cfg.get("weights") else None)
    if cfg.get("periodic"):
        cur, ls = args.n1, []
        while cur < args.n1 + args.n2:
            step = min(cfg["periodic"], args.n1 + args.n2 - cur)
            ls += train(model, opt, cur, step)
            apply_qdq(opt, qdq_int8)
            cur += step
        losses[name] = ls
    else:
        losses[name] = train(model, opt, args.n1, args.n2)
    print(f"[quant] {name}: final loss {losses[name][-1]:.4f}")
    del model, opt
    torch.cuda.empty_cache()

# ---------------- 指标 ----------------
import statistics
exact = losses["exact"]
report = {"n1": args.n1, "n2": args.n2, "warm_last": warm[-1], "losses": losses,
          "qerr": {k: list(v) for k, v in qerr.items()}}
print(f"\n{'variant':8s} {'final':>8s} {'mean|Δ|vs exact':>16s} {'max|Δ|':>8s}")
for name in ["ref", "m8v8", "fp8", "w8", "m8_p50"]:
    d = [abs(a - b) for a, b in zip(losses[name], exact)]
    report.setdefault("gap", {})[name] = {"mean": statistics.mean(d), "max": max(d)}
    print(f"{name:8s} {losses[name][-1]:8.4f} {statistics.mean(d):16.5f} {max(d):8.5f}")
print(f"(ref 行即噪声底：精确恢复 vs 不间断训练的非确定性漂移)")
for k, (em, ev) in qerr.items():
    print(f"quant rel-err {k}: exp_avg {em:.4f}, exp_avg_sq {ev:.4f}")

with open(args.out, "w") as f:
    json.dump(report, f)
print(f"[quant] saved -> {args.out}")
