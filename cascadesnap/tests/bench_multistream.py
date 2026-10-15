"""CascadeSave breakdown 基准：真实 GPT 类模型 + multistream 检查点。

用法:
  python bench_multistream.py --variant orig --moddir <含 multistream_checkpoint.py 的目录> \
      --ckpt /mnt/data1/.../ms.chk --out summary.json --iters 6

配合 PCCHECK_MS_PROFILE=1 / PCCHECK_MS_PROFILE_PATH 产出逐事件 JSONL。
"""
import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--moddir", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--iters", type=int, default=6)
p.add_argument("--groups", type=int, default=8)
p.add_argument("--verify", action="store_true", default=False,
               help="（默认关闭，不影响计时）drain 后 memmap 回读最新槽位，与在存状态逐位对比")
p.add_argument("--quant", action="store_true", default=False,
               help="融合 int8 量化 moment 流（stream2/3）；v3 交错布局下与 split 分组可组合")
p.add_argument("--deterministic", action="store_true", default=False,
               help="确定性算法（跨进程 byte-diff 用；embedding 反向等改用确定性实现）")
args = p.parse_args()

if args.deterministic:
    import os as _os
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, args.moddir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from multistream_checkpoint import MultiStreamCheckpoint, build_param_layout

LIB = "/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

torch.manual_seed(0)
if args.deterministic:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
dev = "cuda:0"
V, H, L, FF = 32000, 1024, 12, 4096


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


model = GPT().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
n_params = sum(t.numel() for t in model.parameters())
n_entries = len(list(model.named_parameters()))
print(f"[bench] params={n_params/1e6:.1f}M entries={n_entries} "
      f"checkpoint={n_params*4*4/1e9:.2f}GB (4 streams)")

B, S = 8, 512


def fwd_bwd():
    ids = torch.randint(0, V, (B, S), device=dev)
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, V), ids.view(-1))
    opt.zero_grad(set_to_none=False)
    loss.backward()
    return loss


# 预热 + 物化 Adam 状态
for _ in range(2):
    fwd_bwd()
    opt.step()
torch.cuda.synchronize()

# 无检查点参考迭代时间
t0 = time.perf_counter()
for _ in range(3):
    fwd_bwd()
    opt.step()
torch.cuda.synchronize()
ref_iter_s = (time.perf_counter() - t0) / 3
print(f"[bench] no-checkpoint iter = {ref_iter_s*1000:.1f} ms")

layout = build_param_layout(model, opt, quant_moments=args.quant)
master_floats = sum(li.get("master_size", li.get("grad_size", 0)) for li in layout)
ea_floats = sum(li["exp_avg_size"] for li in layout)      # quant 时为 packed
eas_floats = sum(li["exp_avg_sq_size"] for li in layout)
gpu_buffers = {"param": torch.zeros(n_params, dtype=torch.float32, device=dev),
               "exp_avg": torch.zeros(ea_floats, dtype=torch.float32, device=dev),
               "exp_avg_sq": torch.zeros(eas_floats, dtype=torch.float32, device=dev)}
if master_floats > 0:
    gpu_buffers["master"] = torch.zeros(master_floats, dtype=torch.float32, device=dev)
payload_floats = n_params + master_floats + ea_floats + eas_floats
print(f"[bench] quant={args.quant} payload={payload_floats*4/1e9:.2f}GB "
      f"(param={n_params} master={master_floats} exp_avg={ea_floats} exp_avg_sq={eas_floats})")
mschk = MultiStreamCheckpoint(
    param_layout=layout,
    gpu_ar=gpu_buffers,
    total_size=n_params,
    num_threads=8,
    lib_path=LIB,
    filename=args.ckpt,
    num_streams=4,
    max_async=2,
    num_layer_groups=args.groups,
    metadata_path=args.ckpt + ".metadata.json",
)
def _pieces(g):
    """兼容两种组格式：[lid,...] 或 [(lid,s,e),...]"""
    out = []
    for item in g:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            out.append((int(item[0]), int(item[1]), int(item[2])))
        else:
            out.append((int(item), 0, int(layout[int(item)].get("param_size", 0))))
    return out

covered_lids = {lid for g in mschk.layer_groups for (lid, _, _) in _pieces(g)}
covered = len(covered_lids)
group_bytes = [sum((e - s) for (_, s, e) in _pieces(g)) * 4 for g in mschk.layer_groups]
print(f"[bench] variant={args.variant} groups={len(mschk.layer_groups)} "
      f"covered={covered}/{n_entries} dropped={n_entries-covered}")

wrapped = mschk.create_optimizer(opt, model)

# P1：注册非分层状态 provider（迭代号等应用状态；RNG 由系统自动捕获）
bench_iter = {"i": -1}
mschk.attach_extra_state(provider=lambda: {"bench_iter": bench_iter["i"]})

iter_times = []
for i in range(args.iters):
    bench_iter["i"] = i
    t = time.perf_counter()
    fwd_bwd()
    wrapped.begin_checkpoint()
    wrapped.step_with_callback()
    wrapped.finalize_checkpoint(wait=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    iter_times.append(dt)
    print(f"[bench] iter {i}: {dt*1000:.1f} ms")

# 等全部后台检查点收尾
deadline = time.time() + 300
while mschk._checkpoint_slots and time.time() < deadline:
    time.sleep(0.2)
drained = not mschk._checkpoint_slots
print(f"[bench] drained={drained}")

verify_result = None
if args.verify and drained:
    import numpy as np
    import json as J
    meta = J.load(open(args.ckpt + ".metadata.json"))
    stride = int(meta["checkpoint_stride_floats"])
    ssz = meta["stream_sizes"]
    max_async = 2
    OFFSET_SIZE = 4096
    # 数据区起点 = (max_async+4)*4096 个 float（header 含 checkpoint*/PEER/init/
    # metadata[0..max_async] 共 max_async+4 段；旧值 +3 有元数据-数据重叠 bug 已修）
    data_base_floats = (max_async + 4) * OFFSET_SIZE
    mm = np.memmap(args.ckpt, dtype=np.float32, mode="r")
    stream_off = [0, ssz[0], ssz[0] + ssz[1], ssz[0] + ssz[1] + ssz[2]]
    extra_meta = meta.get("extra_state", {})
    EXTRA_FLOATS = int(extra_meta.get("floats", 0)) if extra_meta.get("enabled") else 0
    extra_off = sum(ssz[:4])  # extra 流紧随 4 条数据流

    named = list(model.named_parameters())
    probes = [0, len(named) // 2, len(named) - 1]  # emb / 中部小 bias / head
    layer_by_name = {li["name"]: li for li in layout}
    QUANT = bool(meta.get("quant_moments", {}).get("enabled"))
    Q_BLOCK = int(meta.get("quant_moments", {}).get("block", 0))
    REL_TOL = 0.02  # 量化 rel-err 阈值（单测实测 ~0.008，留 2.5x 余量）
    if QUANT:
        import torch as _t
        from multistream_checkpoint import dequantize_int8_blocks, _quant_unit_floats

    def slot_checks(slot):
        base = data_base_floats + slot * stride
        out = []
        for pi in probes:
            name, prm = named[pi]
            li = layer_by_name[name]
            n = prm.numel()
            # param 流始终 fp32，逐位比对
            po = li["param_offset"]
            fv = mm[base + stream_off[0] + po: base + stream_off[0] + po + n]
            lv = prm.detach().float().view(-1).cpu().numpy()
            out.append((f"param:{name}", bool(np.array_equal(fv, lv))))
            st = opt.state.get(prm, {})
            if "exp_avg" not in st:
                continue
            live = st["exp_avg"].detach().float().view(-1).cpu().numpy()
            if QUANT:
                # v3 交错布局：每 block 一个单元 [B/4 float 的 int8 负载 | 1 fp32 scale]
                eo = li["exp_avg_offset"]
                nb = li["q_nblocks"]
                U = _quant_unit_floats(Q_BLOCK)
                region = np.array(
                    mm[base + stream_off[2] + eo: base + stream_off[2] + eo + nb * U]
                ).reshape(nb, U)
                i8 = _t.from_numpy(region[:, :U - 1].copy()).view(_t.int8).view(-1)[:n]
                sc = _t.from_numpy(region[:, U - 1].copy())
                deq = dequantize_int8_blocks(i8, sc, Q_BLOCK, sqrt_domain=False).numpy()
                rel = float(np.linalg.norm(deq - live) / max(np.linalg.norm(live), 1e-30))
                out.append((f"exp_avg:{name}(rel={rel:.4f})", rel < REL_TOL))
            else:
                eo = li["exp_avg_offset"]
                fv2 = mm[base + stream_off[2] + eo: base + stream_off[2] + eo + n]
                out.append((f"exp_avg:{name}", bool(np.array_equal(fv2, live))))
        # extra 流（非分层状态 blob）：与保存时捕获的字节逐位对比 + crc/解析
        if EXTRA_FLOATS > 0:
            expected = mschk._extra_state_debug.get(slot)
            raw = np.array(mm[base + extra_off: base + extra_off + EXTRA_FLOATS]).tobytes()
            if expected is None:
                out.append(("extra:missing-capture", False))
            else:
                byte_ok = raw[:len(expected)] == expected
                try:
                    est, _ = MultiStreamCheckpoint.parse_extra_state_blob(raw)
                    parse_ok = ("rng" in est and est.get("parall_iter") == slot)
                except Exception as e:
                    parse_ok = False
                out.append((f"extra:blob(bytes={'=' if byte_ok else '≠'})",
                            bool(byte_ok and parse_ok)))
        return out

    per_slot = {s: slot_checks(s) for s in range(max_async + 1)}
    match_slot = next((s for s, cs in per_slot.items() if all(v for _, v in cs)), None)

    if match_slot is None:
        # 失配定位：对"最接近"的槽位输出每个失配探针的详细诊断
        best = max(per_slot, key=lambda s: sum(v for _, v in per_slot[s]))
        base = data_base_floats + best * stride
        print(f"[bench] verify diagnostics on best slot {best}:")
        for pi in probes:
            name, prm = named[pi]
            o = layer_by_name[name]["param_offset"]
            n = prm.numel()
            fv = np.asarray(mm[base + stream_off[0] + o: base + stream_off[0] + o + n])
            lv = prm.detach().float().view(-1).cpu().numpy()
            bad = np.nonzero(fv != lv)[0]
            if len(bad):
                b0, b1 = int(bad[0]), int(bad[-1])
                print(f"    param:{name} n_bad={len(bad)}/{n} first={b0} last={b1} "
                      f"file[{b0}:{b0+3}]={fv[b0:b0+3].tolist()} live={lv[b0:b0+3].tolist()} "
                      f"file_zeros_in_bad={float((fv[bad]==0).mean()):.3f}")
    verify_result = {"match_slot": match_slot,
                     "latest_parall_iter": int(meta.get("latest_parall_iter", -1)),
                     "per_slot": {str(s): cs for s, cs in per_slot.items()}}
    print(f"[bench] CONTENT VERIFY: {'PASS (slot %s == live state)' % match_slot if match_slot is not None else 'FAIL'}")
    for s, cs in per_slot.items():
        print(f"    slot {s}: " + " ".join(('✓' if v else '✗') + k.split(':')[0] for k, v in cs))

summary = {
    "variant": args.variant,
    "verify": verify_result,
    "params": n_params,
    "entries": n_entries,
    "covered_entries": covered,
    "dropped_entries": n_entries - covered,
    "group_layers": [len(g) for g in mschk.layer_groups],
    "group_bytes": group_bytes,
    "ref_iter_s": ref_iter_s,
    "ckpt_iter_s": iter_times,
    "save_times_s": list(mschk.save_times),
    "drained": drained,
}
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[bench] summary -> {args.out}")
os._exit(0)  # 跳过析构中的线程池清理，避免退出阶段干扰计时结果
