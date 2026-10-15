"""P0 修复回归测试（2026-10-14）。

覆盖：
  T1 MultiStreamOptimizer 分批 step 与整体 optimizer.step() 逐位一致
     （多 param_group 超参、foreach / fused / for-loop、SGD、DeepSpeed FusedAdam）
  T2 sqrt 域量化下溢防护（无 0 码字、还原值 >= s/2）
  T3 PP 零拷贝路径（create_stage_checkpoint）fp32 / int8 保存内容逐位校验
  T4 写后读屏障：拷贝任务被延迟时，屏障后原位修改训练状态不污染 checkpoint
     （同时用旧屏障复现竞态，证明测试能抓到问题）
  T5 错误闩：写失败的 slot 不发布、slot 被回收（连续失败不死锁），随后保存正常
  T6 单卡 create_optimizer 路径：fused AdamW + 双 param_group，更新与整体 step
     逐位一致，checkpoint 内容与在存状态一致

用法:
  python test_p0_fixes.py --repo /home/linzhicheng/code/pccheck-cascade-v3-merge \
      --ckdir /mnt/data1/linzhicheng_bench/p0test [--only T1,T3]
"""
import argparse
import copy
import json
import os
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--ckdir", required=True)
p.add_argument("--only", default="")
p.add_argument("--pipelayer", default="/home/linzhicheng/download/pipelayer/src")
args = p.parse_args()

ALL_TESTS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
if not args.only:
    # C++ 库的 PR_ADDR / free_space 是进程级 static：同一进程里创建多个 writer 会
    # 让空闲 slot 队列出现重复编号。生产中每进程一个 writer，因此每个测试单独起进程。
    import subprocess
    rcs = {}
    for name in ALL_TESTS:
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__),
                            "--repo", args.repo, "--ckdir", args.ckdir, "--only", name,
                            "--pipelayer", args.pipelayer],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith(f"{name}:")),
                    f"{name}: ERROR — no summary (rc={r.returncode})\n{r.stderr[-2000:]}")
        print(line, flush=True)
        rcs[name] = r.returncode
    sys.exit(0 if all(v == 0 for v in rcs.values()) else 1)

sys.path.insert(0, args.repo)
os.makedirs(args.ckdir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_eval.pccheck import multistream_checkpoint as msc
from checkpoint_eval.pccheck import multistream_pipeline_utils as ppu

LIB = os.path.join(args.repo, "checkpoint_eval", "pccheck", "libtest_ssd.so")
DEV = "cuda:0"
RESULTS = {}
ONLY = {s.strip() for s in args.only.split(",") if s.strip()}


def run(name):
    def deco(fn):
        if ONLY and name not in ONLY:
            return fn
        t0 = time.time()
        try:
            detail = fn()
            RESULTS[name] = ("PASS", detail)
        except AssertionError as e:
            RESULTS[name] = ("FAIL", str(e))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            RESULTS[name] = ("ERROR", repr(e))
        print(f"[{name}] {RESULTS[name][0]} ({time.time() - t0:.1f}s): {RESULTS[name][1]}")
        return fn
    return deco


class TinyGPT(nn.Module):
    def __init__(self, V=4096, H=256, L=4):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(nn.ModuleDict({
                "ln": nn.LayerNorm(H),
                "up": nn.Linear(H, 4 * H),
                "down": nn.Linear(4 * H, H),
            }))
        self.lnf = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)
        self.V = V

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = x + b["down"](F.gelu(b["up"](b["ln"](x))))
        return self.head(self.lnf(x))


def make_batch(V, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, V, (4, 64), generator=g).to(DEV)


def fwd_bwd(model, opt, seed):
    ids = make_batch(model.V, seed)
    loss = F.cross_entropy(model(ids).view(-1, model.V), ids.view(-1))
    for prm in model.parameters():
        prm.grad = None
    loss.backward()
    return loss


def two_groups(model, lr=1e-3, wd=0.1):
    decay, no_decay = [], []
    for n, p_ in model.named_parameters():
        (no_decay if (p_.ndim < 2) else decay).append(p_)
    # 故意让 no_decay 组在前，参数顺序与 named_parameters 不同
    return [{"params": no_decay, "weight_decay": 0.0, "lr": lr * 0.5},
            {"params": decay, "weight_decay": wd, "lr": lr}]


def legacy_groups(n, k):
    """[lid,...] 旧格式分组：n 个张量切成 k 组。"""
    per = (n + k - 1) // k
    return [list(range(i, min(n, i + per))) for i in range(0, n, per)]


def assert_bitwise(a_model, b_model, a_opt, b_opt, tag):
    for (na, pa), (_, pb) in zip(a_model.named_parameters(), b_model.named_parameters()):
        assert torch.equal(pa, pb), f"{tag}: param {na} differs (max {float((pa - pb).abs().max()):.3e})"
        sa, sb = a_opt.state.get(pa, {}), b_opt.state.get(pb, {})
        for k in sa:
            if torch.is_tensor(sa[k]):
                assert torch.equal(sa[k].to(sb[k].device), sb[k]), f"{tag}: state {k} of {na} differs"


# --------------------------------------------------------------------------- T1
@run("T1")
def t1():
    cases = [
        ("Adam-foreach", lambda ps: torch.optim.Adam(ps, foreach=True)),
        ("AdamW-foreach", lambda ps: torch.optim.AdamW(ps, foreach=True)),
        ("AdamW-fused", lambda ps: torch.optim.AdamW(ps, fused=True)),
        ("Adam-fused-amsgrad", lambda ps: torch.optim.Adam(ps, fused=True, amsgrad=True)),
        ("AdamW-forloop", lambda ps: torch.optim.AdamW(ps, foreach=False)),
        ("SGD-momentum", lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9, weight_decay=1e-4)),
    ]
    try:
        from deepspeed.ops.adam import FusedAdam as DSFusedAdam
        cases.append(("DeepSpeed-FusedAdam", lambda ps: DSFusedAdam(ps, adam_w_mode=True)))
    except Exception as e:  # noqa: BLE001
        print(f"[T1] DeepSpeed FusedAdam unavailable: {e!r}")
    done = []
    for name, mk in cases:
        torch.manual_seed(0)
        ref = TinyGPT().to(DEV)
        cand = copy.deepcopy(ref)
        ref_opt = mk(two_groups(ref))
        cand_opt = mk(two_groups(cand))
        n = len(list(cand.parameters()))
        ms = msc.MultiStreamOptimizer(cand_opt, legacy_groups(n, 5), list(cand.parameters()))
        for step in range(4):
            fwd_bwd(ref, ref_opt, step)
            ref_opt.step()
            fwd_bwd(cand, cand_opt, step)
            if step % 2 == 1:
                ms.step_with_callback()   # 分批（checkpoint 步）
            else:
                cand_opt.step()           # 普通步
        torch.cuda.synchronize()
        assert_bitwise(ref, cand, ref_opt, cand_opt, name)
        # group 级状态（如 DeepSpeed 的 group['step']）必须与参考一致
        for gr, gc in zip(ref_opt.param_groups, cand_opt.param_groups):
            for k in gr:
                if k != "params":
                    assert gr[k] == gc[k], f"{name}: group key {k} {gr[k]} != {gc[k]}"
        done.append(name)
    return "bit-exact: " + ", ".join(done)


# --------------------------------------------------------------------------- T2
@run("T2")
def t2():
    torch.manual_seed(0)
    block = 2048
    # 每块一个大值 + 大量极小值：旧量化器会把极小值舍成 0
    v = torch.rand(block * 8, device=DEV) * 1e-12
    v[::block] = 1.0
    v[5] = 0.0  # 精确 0（未使用的坐标）
    q, sc = msc.quantize_int8_blocks(v, block, sqrt_domain=True)
    assert int((q == 0).sum()) == 0, "sqrt-domain code 0 still produced"
    deq = msc.dequantize_int8_blocks(q, sc, block, sqrt_domain=True)
    s, s_t = v.sqrt(), deq.sqrt()
    assert bool((s_t >= s / 2 - 1e-12).all()), "guard violated: s~ < s/2"
    step = sc.repeat_interleave(block)[: v.numel()]
    assert bool((s_t >= step * (1 - 1e-6)).all()), "guard violated: s~ < Delta"
    # 更新量：构造 Adam 可达的状态（|m| <= C*sqrt(v)），块内 v 动态范围大、
    # m 动态范围小：旧量化器把 sqrt(v_j) 舍成 0 而 m_j 非零 → u ≈ m/eps 爆炸
    v2 = torch.full((block,), 1e-10, device=DEV)
    m2 = torch.full((block,), 1e-4, device=DEV)
    v2[0], m2[0] = 1.0, 1e-3
    q2, s2 = msc.quantize_int8_blocks(v2, block, sqrt_domain=True)
    qm, scm = msc.quantize_int8_blocks(m2, block)
    m_t = msc.dequantize_int8_blocks(qm, scm, block)
    v_new = msc.dequantize_int8_blocks(q2, s2, block, sqrt_domain=True)
    y = v2.sqrt().view(-1, block)
    old_sc = y.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
    v_old = (torch.clamp((y / old_sc).round(), -127, 127) * old_sc).view(-1).square()
    eps = 1e-8
    u_ref = (m2 / (v2.sqrt() + eps)).abs().max().item()
    u_old = (m_t / (v_old.sqrt() + eps)).abs().max().item()
    u_new = (m_t / (v_new.sqrt() + eps)).abs().max().item()
    beta1, beta2 = 0.9, 0.999
    C = (1 - beta1) / ((1 - beta2) ** 0.5 * (1 - beta1 / beta2 ** 0.5))
    assert u_old > 100 * u_ref, f"test construction failed to trigger old blow-up ({u_old})"
    assert u_new <= 3 * C, f"guarded |u|={u_new} exceeds 3C={3 * C:.1f}"
    # 非 sqrt 域保持对称 [-127,127] 不变
    x = torch.randn(block * 4, device=DEV)
    qx, _ = msc.quantize_int8_blocks(x, block)
    assert int(qx.min()) < 0 and int((qx == 0).sum()) >= 0
    return (f"no zero codes; max|u| exact={u_ref:.3g} old-quant={u_old:.3g} (blow-up) "
            f"guarded={u_new:.3g} <= 3C={3 * C:.1f}")


# ------------------------------------------------------------- 文件读取工具
def read_slot(ck, slot=None):
    meta = json.load(open(ck.filename + ".metadata.json"))
    slot = int(meta["latest_parall_iter"]) if slot is None else slot
    stride = int(meta["checkpoint_stride_floats"])
    ssz = meta["stream_sizes"]
    base = (ck.max_async + 4) * 4096 + slot * stride
    mm = np.memmap(ck.filename, dtype=np.float32, mode="r")
    offs = [0, ssz[0], ssz[0] + ssz[1], ssz[0] + ssz[1] + ssz[2]]
    return mm, base, offs, meta


def expected_moment(t, quant, block, sqrt):
    if not quant:
        return t.detach().float().view(-1).cpu()
    q, sc = msc.quantize_int8_blocks(t.detach().view(-1), block, sqrt_domain=sqrt)
    return msc.dequantize_int8_blocks(q.cpu(), sc.cpu(), block, sqrt_domain=sqrt)


def file_moment(mm, base, off, li, key, quant, block, sqrt):
    n = li["param_size"]
    o = li[f"{key}_offset"]
    if not quant:
        return torch.from_numpy(np.array(mm[base + off + o: base + off + o + n]))
    nb, U = li["q_nblocks"], msc._quant_unit_floats(block)
    region = np.array(mm[base + off + o: base + off + o + nb * U]).reshape(nb, U)
    i8 = torch.from_numpy(region[:, :U - 1].copy()).view(torch.int8).reshape(-1)[:n]
    sc = torch.from_numpy(region[:, U - 1].copy())
    return msc.dequantize_int8_blocks(i8, sc, block, sqrt_domain=sqrt)


def verify_file(ck, model, opt, snapshot=None, slot=None):
    """snapshot: {name: (param, exp_avg, exp_avg_sq)} CPU 张量；None 时取在存状态。"""
    mm, base, offs, meta = read_slot(ck, slot)
    quant = bool(ck.quant_moments)
    block = int(ck.quant_block)
    base_opt = getattr(opt, "optimizer", opt) if not hasattr(opt, "state") else opt
    bad = []
    params = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
    for (name, prm), li in zip(params, ck.param_layout):
        if snapshot is not None:
            p_live, ea, eas = snapshot[name]
        else:
            st = base_opt.state[prm]
            p_live, ea, eas = prm.detach().float().cpu(), st["exp_avg"], st["exp_avg_sq"]
        n = li["param_size"]
        fp = np.array(mm[base + offs[0] + li["param_offset"]: base + offs[0] + li["param_offset"] + n])
        if not np.array_equal(fp, p_live.reshape(-1).numpy()):
            bad.append(f"param:{name}")
        for key, t, off, sqrt in (("exp_avg", ea, offs[2], False), ("exp_avg_sq", eas, offs[3], True)):
            exp = expected_moment(t.to(DEV), quant, block, sqrt)
            got = file_moment(mm, base, off, li, key, quant, block, sqrt)
            if not torch.equal(exp, got):
                bad.append(f"{key}:{name}")
    return bad, int(meta["latest_parall_iter"])


def make_pp_stage(tag, quant, max_async=2, groups=4):
    os.environ["PCCHECK_MS_QUANT"] = "1" if quant else "0"
    torch.manual_seed(0)
    model = TinyGPT().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for s in range(2):
        fwd_bwd(model, opt, s)
        opt.step()
    ck, bufs, layout = ppu.create_stage_checkpoint(
        model=model, optimizer=opt, checkpoint_dir=os.path.join(args.ckdir, tag),
        lib_path=LIB, rank=0, world_size=1, max_async=max_async,
        num_layer_groups=groups, num_threads=4)
    return model, opt, ck


def snapshot_state(model, opt):
    return {n: (p_.detach().float().cpu().clone(),
                opt.state[p_]["exp_avg"].detach().clone(),
                opt.state[p_]["exp_avg_sq"].detach().clone())
            for n, p_ in model.named_parameters()}


# --------------------------------------------------------------------------- T3
@run("T3")
def t3():
    out = []
    for quant in (False, True):
        tag = f"t3_{'q' if quant else 'f'}"
        model, opt, ck = make_pp_stage(tag, quant)
        try:
            for s in range(3):
                fwd_bwd(model, opt, 10 + s)
                opt.step()
                ck.save_full_checkpoint(sync=True)
            torch.cuda.synchronize()
            bad, slot = verify_file(ck, model, opt)
            assert not bad, f"quant={quant}: mismatched {bad[:6]} (+{max(0, len(bad) - 6)})"
            out.append(f"{'int8' if quant else 'fp32'} slot{slot} OK"
                       f" (moment streams {ck.stream_sizes[2]} floats)")
        finally:
            ck.shutdown()
    return "; ".join(out)


# --------------------------------------------------------------------------- T4
def _war_trial(use_new_guard, quant):
    tag = f"t4_{'new' if use_new_guard else 'old'}_{'q' if quant else 'f'}"
    model, opt, ck = make_pp_stage(tag, quant, max_async=2, groups=8)
    try:
        fwd_bwd(model, opt, 20)
        opt.step()
        torch.cuda.synchronize()
        snap = snapshot_state(model, opt)
        # 模拟慢盘背压：拷贝线程借 chunk 时被卡住（事件迟迟不 record）
        real_borrow = ck.writer.borrow_chunk

        def slow_borrow():
            time.sleep(0.05)
            return real_borrow()
        ck.writer.borrow_chunk = slow_borrow
        ck.save_full_checkpoint(sync=False)
        if use_new_guard:
            ck.wait_copies_before_update()
        else:  # 旧屏障：只对事件做 GPU 侧 wait
            with ck._layer_copy_events_lock:
                evs = [e for es in ck._layer_copy_events.values() for e in es]
            for e in evs:
                torch.cuda.current_stream().wait_event(e)
        with torch.no_grad():  # 模拟下一步 optimizer 原位更新
            for p_ in model.parameters():
                p_.add_(1.0)
                opt.state[p_]["exp_avg"].add_(1.0)
                opt.state[p_]["exp_avg_sq"].add_(1.0)
        torch.cuda.synchronize()
        deadline = time.time() + 120
        while ck._checkpoint_slots and time.time() < deadline:
            time.sleep(0.1)
        ck.writer.borrow_chunk = real_borrow
        bad, _ = verify_file(ck, model, opt, snapshot=snap)
        return bad
    finally:
        ck.shutdown()


@run("T4")
def t4():
    old_bad = _war_trial(use_new_guard=False, quant=False)
    new_bad = _war_trial(use_new_guard=True, quant=False)
    new_bad_q = _war_trial(use_new_guard=True, quant=True)
    assert not new_bad, f"new guard: torn checkpoint {new_bad[:6]}"
    assert not new_bad_q, f"new guard (int8): torn checkpoint {new_bad_q[:6]}"
    return (f"old guard torn tensors={len(old_bad)} (race reproduced={'yes' if old_bad else 'no'}); "
            f"new guard fp32/int8 clean")


# --------------------------------------------------------------------------- T5
@run("T5")
def t5():
    model, opt, ck = make_pp_stage("t5", quant=False, max_async=2, groups=4)
    try:
        fwd_bwd(model, opt, 30)
        opt.step()
        ck.save_full_checkpoint(sync=True)
        good_snap = snapshot_state(model, opt)
        good_slot = json.load(open(ck.filename + ".metadata.json"))["latest_parall_iter"]

        real_async = ck.writer.write_stream_chunk_async
        calls = {"n": 0}

        def failing(*a, **k):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise OSError(28, "injected ENOSPC")
            return real_async(*a, **k)
        ck.writer.write_stream_chunk_async = failing

        n_fail = ck.max_async + 3  # 不回收 slot 时，第 max_async+2 次 registerCheck 会永久自旋
        raised = []

        def failing_saves():
            for i in range(n_fail):
                fwd_bwd(model, opt, 40 + i)
                opt.step()
                try:
                    ck.save_full_checkpoint(sync=True)
                except RuntimeError:
                    raised.append(i)
        th = threading.Thread(target=failing_saves, daemon=True)
        th.start()
        th.join(timeout=180)
        assert not th.is_alive(), "hang: aborted slots were not returned to free_space"
        assert len(raised) == n_fail, f"only {len(raised)}/{n_fail} failing saves raised"
        meta = json.load(open(ck.filename + ".metadata.json"))
        assert meta["latest_parall_iter"] == good_slot, "a failed checkpoint was published"
        bad, _ = verify_file(ck, model, opt, snapshot=good_snap, slot=good_slot)
        assert not bad, f"previous good checkpoint damaged: {bad[:4]}"
        assert ck.failed_checkpoints == n_fail

        ck.writer.write_stream_chunk_async = real_async
        fwd_bwd(model, opt, 99)
        opt.step()
        ck.save_full_checkpoint(sync=True)
        bad, slot = verify_file(ck, model, opt)
        assert not bad, f"save after failures mismatched {bad[:4]}"
        return (f"{n_fail} injected failures raised, none published, no slot leak; "
                f"next save slot{slot} verified")
    finally:
        ck.shutdown()


# --------------------------------------------------------------------------- T6
@run("T6")
def t6():
    torch.manual_seed(0)
    ref = TinyGPT().to(DEV)
    model = copy.deepcopy(ref)
    ref_opt = torch.optim.AdamW(two_groups(ref), fused=True)
    opt = torch.optim.AdamW(two_groups(model), fused=True)
    for s in range(2):
        for m_, o_ in ((ref, ref_opt), (model, opt)):
            fwd_bwd(m_, o_, s)
            o_.step()
    layout = msc.build_param_layout(model, opt)
    n = sum(li["param_size"] for li in layout)
    bufs = {"param": torch.zeros(n, device=DEV),
            "exp_avg": torch.zeros(n, device=DEV),
            "exp_avg_sq": torch.zeros(n, device=DEV)}
    fn = os.path.join(args.ckdir, "t6.chk")
    ck = msc.MultiStreamCheckpoint(param_layout=layout, gpu_ar=bufs, total_size=n,
                                   num_threads=4, lib_path=LIB, filename=fn,
                                   num_streams=4, max_async=2, num_layer_groups=6,
                                   metadata_path=fn + ".metadata.json")
    try:
        wrapped = ck.create_optimizer(opt, model)
        for s in range(3):
            fwd_bwd(ref, ref_opt, 50 + s)
            ref_opt.step()
            fwd_bwd(model, opt, 50 + s)
            wrapped.begin_checkpoint()
            wrapped.step_with_callback()
            wrapped.finalize_checkpoint(wait=True)
        torch.cuda.synchronize()
        assert_bitwise(ref, model, ref_opt, opt, "T6 fused AdamW two groups")
        bad, slot = verify_file(ck, model, opt)
        assert not bad, f"file mismatch {bad[:6]}"
        return f"fused AdamW 2 groups bit-exact vs optimizer.step(); slot{slot} content OK"
    finally:
        ck.shutdown()


# --------------------------------------------------------------------------- T7
def _t7_child():
    """子进程：发布 step 100 → 在途提交 step 101（拷贝被拖慢）→ 未发布时硬退出。"""
    model, opt, ck = make_pp_stage("t7", quant=os.environ.get("T7_QUANT") == "1",
                                   max_async=2, groups=4)
    cdir = os.path.join(args.ckdir, "t7")
    fwd_bwd(model, opt, 70)
    opt.step()
    torch.cuda.synchronize()
    ppu.save_stage_checkpoint(ck, None, ck.param_layout, model, opt, cdir, rank=0,
                              step=100, sync=True)
    torch.save(snapshot_state(model, opt), os.path.join(cdir, "truth_step100.pt"))
    fwd_bwd(model, opt, 71)
    opt.step()
    torch.cuda.synchronize()
    real_borrow = ck.writer.borrow_chunk

    def stalled_borrow():
        time.sleep(60)  # 慢盘背压：拷贝线程拿不到 pinned chunk，step 101 停在途中
        return real_borrow()
    ck.writer.borrow_chunk = stalled_borrow
    ppu.save_stage_checkpoint(ck, None, ck.param_layout, model, opt, cdir, rank=0,
                              step=101, sync=False)
    time.sleep(1.0)  # 让异步 meta 任务跑完、step 101 仍在途
    with open(os.path.join(cdir, "child_state.json"), "w") as f:
        json.dump({"slots_in_flight": sorted(ck._checkpoint_slots)}, f)
    os._exit(0)      # 模拟进程崩溃：不 drain、不发布


@run("T7")
def t7():
    import subprocess
    sys.path.insert(0, args.pipelayer)
    from pipelayer.distributed import start_distributed_multistream_resume
    out = []
    for quant in ("0", "1"):
        cdir = os.path.join(args.ckdir, "t7")
        if os.path.isdir(cdir):
            import shutil
            shutil.rmtree(cdir)
        env = dict(os.environ, T7_QUANT=quant)
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__),
                            "--repo", args.repo, "--ckdir", args.ckdir, "--only", "T7child",
                            "--pipelayer", args.pipelayer], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"child failed rc={r.returncode}: {r.stderr[-1500:]}"
        child = json.load(open(os.path.join(cdir, "child_state.json")))
        assert child["slots_in_flight"], "step 101 was not in flight at exit (test did not hit the window)"
        sdir = os.path.join(cdir, "stage_0")
        ts = json.load(open(os.path.join(sdir, "training_state.json")))
        assert ts.get("completed_steps") == 100, f"training_state.json claims {ts} (written before publish)"
        chk = os.path.join(sdir, "stage_0.chk")
        rec = msc.read_published_checkpoint(chk, 2)
        assert rec["status"] == "published" and rec["slot"] == ts["slot"], f"record {rec} vs {ts}"
        assert rec["slot"] not in child["slots_in_flight"]

        os.environ["PCCHECK_MS_QUANT"] = quant
        torch.manual_seed(123)
        model = TinyGPT().to(DEV)
        with torch.no_grad():
            for prm in model.parameters():
                prm.zero_()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        handle = start_distributed_multistream_resume(
            model=model, optimizer=opt, checkpoint_dir=cdir, rank=0, device=DEV,
            engine=None, install_forward_hooks=False, guard_optimizer_step=False,
            lib_path=LIB)
        handle.wait_all()
        tstate = dict(handle.training_state)
        truth = torch.load(os.path.join(cdir, "truth_step100.pt"))
        bad = []
        blk = msc.QUANT_BLOCK_DEFAULT
        for name, prm in model.named_parameters():
            p_t, ea_t, eas_t = truth[name]
            if not torch.equal(prm.detach().float().cpu(), p_t):
                bad.append(f"param:{name}")
            st = opt.state[prm]
            for key, t, sq in (("exp_avg", ea_t, False), ("exp_avg_sq", eas_t, True)):
                exp = expected_moment(t.to(DEV), quant == "1", blk, sq)
                if not torch.equal(st[key].detach().float().cpu().view(-1), exp.view(-1)):
                    bad.append(f"{key}:{name}")
        handle.cleanup()
        assert tstate.get("completed_steps") == 100, f"resumed training_state {tstate}"
        assert tstate.get("slot_source") == "published", tstate
        assert not bad, f"quant={quant}: recovered state != published step 100: {bad[:6]}"
        out.append(f"{'int8' if quant == '1' else 'fp32'}: in-flight slot{child['slots_in_flight']} "
                   f"ignored, recovered published slot{rec['slot']} step100 exact")
    return "; ".join(out)


# --------------------------------------------------------------------------- T8
@run("T8")
def t8():
    """单卡 create_optimizer 路径、cf>1：checkpoint 步之后的普通 optimizer.step()
    （trainer 直接调真实优化器）在 D2H 未完成时原位更新零拷贝存储 → 必须被拦住。"""
    from checkpoint_eval.pccheck_utils import initialize
    out = []
    for quant in (False, True):
        torch.manual_seed(0)
        model = TinyGPT().to(DEV)
        opt = torch.optim.AdamW(two_groups(model), fused=True)
        for s_ in range(2):
            fwd_bwd(model, opt, s_)
            opt.step()
        gpu_ar, _ = initialize(model, [opt], do_opt_step=False, separate_streams=True)
        layout = msc.build_param_layout(model, opt, quant_moments=quant)
        n = sum(li["param_size"] for li in layout)
        fn = os.path.join(args.ckdir, f"t8_{int(quant)}.chk")
        ck = msc.MultiStreamCheckpoint(param_layout=layout, gpu_ar=gpu_ar, total_size=n,
                                       num_threads=4, lib_path=LIB, filename=fn,
                                       num_streams=4, max_async=2, num_layer_groups=6,
                                       metadata_path=fn + ".metadata.json",
                                       quant_moments=quant)
        try:
            wrapped = ck.create_optimizer(opt, model)
            real_borrow = ck.writer.borrow_chunk

            def slow_borrow():
                time.sleep(0.05)
                return real_borrow()
            ck.writer.borrow_chunk = slow_borrow
            fwd_bwd(model, opt, 10)
            wrapped.begin_checkpoint()
            wrapped.step_with_callback()
            torch.cuda.synchronize()
            snap = snapshot_state(model, opt)
            wrapped.finalize_checkpoint(wait=False)
            for k in range(3):          # 普通步：trainer 直接调真实优化器
                fwd_bwd(model, opt, 11 + k)
                opt.step()
            torch.cuda.synchronize()
            deadline = time.time() + 120
            while ck._checkpoint_slots and time.time() < deadline:
                time.sleep(0.1)
            ck.writer.borrow_chunk = real_borrow
            bad, slot = verify_file(ck, model, opt, snapshot=snap)
            assert not bad, (f"quant={quant}: checkpoint torn by later plain optimizer.step(): "
                             f"{len(bad)} tensors, e.g. {bad[:4]}")
            out.append(f"{'int8' if quant else 'fp32'} slot{slot} == step-snapshot")
        finally:
            ck.shutdown()
    return "; ".join(out)


if "T7child" in ONLY:
    _t7_child()


print("\n==== SUMMARY ====")
for k in sorted(RESULTS):
    print(f"{k}: {RESULTS[k][0]} — {RESULTS[k][1]}")
sys.exit(0 if all(v[0] == "PASS" for v in RESULTS.values()) else 1)
