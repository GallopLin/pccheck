"""Shared library for the checkpoint-quantization quality experiments (E1–E3).

- data: wikitext-103 tokenized with the GPT-2 tokenizer once and cached as a
  uint16 tensor.  Batch i is a deterministic function of (i, data_seed), so a
  resumed run sees exactly the batches the reference run would have seen
- model: GPT-2 family from config (pretraining from scratch) or a pretrained HF model
- optimizer: fused AdamW, linear warmup + cosine decay, grad clip, bf16 autocast
- certificate(): per-tensor error statistics of a moment QDQ (int8 block
  absmax with/without the sqrt-domain guard, fp8-e4m3 per-block, bf16)
  measured on the Adam update direction

The int8 QDQ uses the production quantizer
(multistream_checkpoint.quantize_int8_blocks / dequantize_int8_blocks).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.environ.get("PCCHECK_ROOT", "/home/linzhicheng/code/pccheck-cascade-v3-merge")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from checkpoint_eval.pccheck.multistream_checkpoint import (  # noqa: E402
    dequantize_int8_blocks, quantize_int8_blocks)

CACHE = os.environ.get("QLIB_CACHE", "/mnt/data1/linzhicheng_bench/qcache")
DEV = "cuda"


# ----------------------------------------------------------------------------- data
def load_tokens(split="train"):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"wt103_gpt2_{split}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids = []
    batch = []
    for t in ds["text"]:
        batch.append(t)
        if len(batch) == 4096:
            for x in tok(batch)["input_ids"]:
                ids.extend(x)
            batch = []
    if batch:
        for x in tok(batch)["input_ids"]:
            ids.extend(x)
    out = torch.tensor(ids, dtype=torch.int32)
    torch.save(out, path)
    return out


class Batches:
    """batch(i) is a pure function of (i, seed): random but reproducible windows."""

    def __init__(self, tokens, bs, seq, seed):
        self.tokens = tokens
        self.bs, self.seq, self.seed = bs, seq, seed
        self.n = tokens.numel() - seq - 1

    def __call__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        starts = torch.randint(0, self.n, (self.bs,), generator=g)
        x = torch.stack([self.tokens[s: s + self.seq + 1] for s in starts.tolist()]).long()
        return x[:, :-1].to(DEV, non_blocking=True), x[:, 1:].to(DEV, non_blocking=True)


def make_val(tokens, bs, seq, n_batches=50):
    b = Batches(tokens, bs, seq, seed=999_983)
    return [b(i) for i in range(n_batches)]


# ----------------------------------------------------------------------------- model
def make_model(name, seed):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(seed)
    if name.startswith("scratch:"):
        cfg = AutoConfig.from_pretrained(name.split(":", 1)[1])
        cfg.attn_pdrop = cfg.embd_pdrop = cfg.resid_pdrop = 0.0
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    return model.to(DEV)


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def make_opt(model, lr, betas, wd, eps=1e-8):
    return torch.optim.AdamW(param_groups(model, wd), lr=lr, betas=betas, eps=eps, fused=True)


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train_step(model, opt, batch, lr, clip=1.0):
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, val):
    model.eval()
    tot = 0.0
    for x, y in val:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return tot / len(val)


# ----------------------------------------------------------------------------- QDQ
def qdq_int8(x, block, sqrt_domain, guard=True):
    """Production int8 block quantizer.  guard=False reproduces the old
    quantizer, where a sqrt-domain value can round to code 0."""
    flat = x.detach().reshape(-1).float()
    if guard or not sqrt_domain:
        q, sc = quantize_int8_blocks(flat, block, sqrt_domain=sqrt_domain)
        return dequantize_int8_blocks(q, sc, block, sqrt_domain=sqrt_domain).view_as(x)
    y = flat.sqrt()
    n = y.numel()
    nb = (n + block - 1) // block
    yp = torch.cat([y, y.new_zeros(nb * block - n)]).view(nb, block)
    scale = yp.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127.0
    deq = (torch.clamp((yp / scale).round(), -127, 127) * scale).view(-1)[:n]
    return deq.square().view_as(x)


def qdq_fp8(x, block, sqrt_domain):
    flat = x.detach().reshape(-1).float()
    y = flat.sqrt() if sqrt_domain else flat
    n = y.numel()
    nb = (n + block - 1) // block
    yp = torch.cat([y, y.new_zeros(nb * block - n)]).view(nb, block)
    scale = yp.abs().amax(1, keepdim=True).clamp_min(1e-30) / 448.0
    deq = ((yp / scale).to(torch.float8_e4m3fn).float() * scale).view(-1)[:n]
    if sqrt_domain:
        deq = deq.square()
    return deq.view_as(x)


def qdq_bf16(x, block, sqrt_domain):
    return x.detach().float().to(torch.bfloat16).float()


SCHEMES = {
    "int8_guard": lambda x, b, s: qdq_int8(x, b, s, guard=True),
    "int8_noguard": lambda x, b, s: qdq_int8(x, b, s, guard=False),
    "fp8_e4m3": qdq_fp8,
    "bf16": qdq_bf16,
}


def tensor_kind(name, p):
    n = name.lower()
    if "wte" in n or "embed" in n or "wpe" in n:
        return "embedding"
    if p.ndim < 2:
        return "norm_bias"
    if any(k in n for k in ("attn", "q_proj", "k_proj", "v_proj", "out_proj", "c_attn")):
        return "attention"
    if any(k in n for k in ("mlp", "fc", "c_fc", "c_proj")):
        return "mlp"
    if "lm_head" in n:
        return "head"
    return "other"


@torch.no_grad()
def certificate(model, opt, scheme, block, step, betas, eps):
    """Per-tensor error statistics, aggregated by kind and over the whole model."""
    b1, b2 = betas
    bc1 = 1 - b1 ** step
    bc2 = 1 - b2 ** step
    fn = SCHEMES[scheme]
    agg = {}

    def acc(kind, key, v):
        d = agg.setdefault(kind, {})
        d.setdefault(key, []).append(v)

    tot = {"num_u": 0.0, "den_u": 0.0, "dot": 0.0, "nu2": 0.0, "nut2": 0.0, "n": 0,
           "guard": 0, "danger": 0, "u_max": 0.0, "ut_max": 0.0,
           "m_err2": 0.0, "m2": 0.0, "v_err2": 0.0, "v2": 0.0}
    for name, p in model.named_parameters():
        st = opt.state.get(p, {})
        if "exp_avg" not in st:
            continue
        m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
        mt = fn(m, block, False)
        vt = fn(v, block, True)
        u = (m / bc1) / ((v / bc2).sqrt() + eps)
        ut = (mt / bc1) / ((vt / bc2).sqrt() + eps)
        du = ut - u
        kind = tensor_kind(name, p)
        # guard trigger / danger counts: sqrt(v) below half a step of its block
        s = v.reshape(-1).sqrt()
        nb = (s.numel() + block - 1) // block
        sp = torch.cat([s, s.new_zeros(nb * block - s.numel())]).view(nb, block)
        delta = (sp.amax(1, keepdim=True) / 127.0).expand_as(sp).reshape(-1)[: s.numel()]
        low = s < delta / 2
        guard = int(low.sum())
        danger = int((low & (mt.reshape(-1) != 0)).sum())
        rel_u = float(du.norm() / u.norm().clamp_min(1e-30))
        acc(kind, "rel_u", rel_u)
        acc(kind, "guard_frac", guard / s.numel())
        tot["num_u"] += float(du.pow(2).sum())
        tot["den_u"] += float(u.pow(2).sum())
        tot["dot"] += float((u * ut).sum())
        tot["nu2"] += float(u.pow(2).sum())
        tot["nut2"] += float(ut.pow(2).sum())
        tot["n"] += s.numel()
        tot["guard"] += guard
        tot["danger"] += danger
        tot["u_max"] = max(tot["u_max"], float(u.abs().max()))
        tot["ut_max"] = max(tot["ut_max"], float(ut.abs().max()))
        tot["m_err2"] += float((mt - m).pow(2).sum())
        tot["m2"] += float(m.pow(2).sum())
        tot["v_err2"] += float((vt - v).pow(2).sum())
        tot["v2"] += float(v.pow(2).sum())
    out = {
        "scheme": scheme, "block": block, "step": step,
        "rel_u": math.sqrt(tot["num_u"] / max(tot["den_u"], 1e-30)),
        "cos_u": tot["dot"] / math.sqrt(max(tot["nu2"] * tot["nut2"], 1e-30)),
        "rel_m": math.sqrt(tot["m_err2"] / max(tot["m2"], 1e-30)),
        "rel_v": math.sqrt(tot["v_err2"] / max(tot["v2"], 1e-30)),
        "guard_frac": tot["guard"] / max(tot["n"], 1),
        "danger_frac": tot["danger"] / max(tot["n"], 1),
        "u_max": tot["u_max"], "ut_max": tot["ut_max"],
        "by_kind": {k: {kk: (sorted(vv)[len(vv) // 2], max(vv)) for kk, vv in d.items()}
                    for k, d in agg.items()},
    }
    return out


@torch.no_grad()
def apply_qdq_(opt, scheme, block):
    """In-place moment QDQ on the live optimizer state (= restoring from that checkpoint)."""
    fn = SCHEMES[scheme]
    for st in opt.state.values():
        if "exp_avg" in st:
            st["exp_avg"].copy_(fn(st["exp_avg"], block, False))
            st["exp_avg_sq"].copy_(fn(st["exp_avg_sq"], block, True))


def snapshot(model, opt):
    return ({k: v.detach().clone() for k, v in model.state_dict().items()},
            {"state": {id_: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                             for k, v in st.items()}
                       for id_, st in opt.state_dict()["state"].items()},
             "param_groups": opt.state_dict()["param_groups"]})


def restore(model, opt, snap):
    msd, osd = snap
    model.load_state_dict(msd)
    opt.load_state_dict({"state": {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                                       for kk, vv in st.items()} for k, st in osd["state"].items()},
                         "param_groups": osd["param_groups"]})
