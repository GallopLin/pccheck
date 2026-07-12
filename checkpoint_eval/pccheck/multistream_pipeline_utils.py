"""
DeepSpeed Pipeline Parallelism + MultiStream Checkpoint 工具

提供 PP 阶段的状态打包、元数据写入、以及 pipelayer 恢复的辅助函数。
每个 PP 阶段是独立的 PyTorch Module，可直接使用 MultiStreamCheckpoint。

采用与单卡 pccheck 相同的零拷贝设计：
    1. initialize() 分配 GPU 存储（支持四块独立分配）
    2. set_storage() 用 tensor.set_() 将 params/grads/optimizer states 重映射到底层存储
    3. MultiStreamCheckpoint 直接从映射后的存储读取，无需额外拷贝
"""

import atexit
import json
import os
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .multistream_checkpoint import MultiStreamCheckpoint, build_param_layout
from ..pccheck_utils import initialize, set_storage

# P1.2：metadata/training_state 写盘下沉后台单线程（保序），主线程只做
# 廉价快照 + submit。tmpfs 对照显示每步 checkpoint 前台开销 p50 +26%，
# 主要来自 export_metadata（per-param step .item() 逐个 CUDA sync + 大 JSON
# 序列化 + 原子写盘）。PCCHECK_MS_ASYNC_META=0 回退旧同步行为。
_ASYNC_META_ENABLED = os.environ.get("PCCHECK_MS_ASYNC_META", "1") == "1"
_meta_executor: Optional[ThreadPoolExecutor] = None
_meta_last_future = None


def _get_meta_executor() -> ThreadPoolExecutor:
    global _meta_executor
    if _meta_executor is None:
        _meta_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ms-meta")
        atexit.register(drain_metadata_writes)
    return _meta_executor


def drain_metadata_writes(timeout: float = 60.0) -> None:
    """等待最后一个后台 metadata 写完成（进程退出/同步模式屏障）。"""
    f = _meta_last_future
    if f is not None:
        try:
            f.result(timeout=timeout)
        except Exception as e:  # 不吞真实错误信号，仅避免退出路径抛出
            print(f"[WARN] async metadata write failed: {e}")


@torch.no_grad()
def _snapshot_optimizer_steps(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer) -> Dict[str, int]:
    """主线程 step 快照：GPU 标量批量 cat 后单次 D2H，替代 per-param .item()
    的逐个同步；同时保证后台写出的 steps 属于本 checkpoint（非下一迭代）。"""
    pmap = {p: n for n, p in model.named_parameters()}
    names, tensors, vals = [], [], {}
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            st = optimizer.state.get(p)
            if not st or "step" not in st or p not in pmap:
                continue
            s = st["step"]
            if isinstance(s, torch.Tensor):
                if s.is_cuda:
                    names.append(pmap[p])
                    tensors.append(s.detach().reshape(1).float())
                else:
                    vals[pmap[p]] = int(s.item())
            else:
                vals[pmap[p]] = int(s)
    if tensors:
        flat = torch.cat(tensors).cpu()  # 单次同步
        for n, v in zip(names, flat.tolist()):
            vals[n] = int(v)
    return vals


@torch.no_grad()
def _ensure_grad_and_optimizer_states(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """确保grad/Adam状态在训练前已显式创建，以便完成4区连续映射。"""
    for group in optimizer.param_groups:
        amsgrad = bool(group.get("amsgrad", False))
        for p in group["params"]:
            if p is None or not p.requires_grad:
                continue

            if p.grad is None:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)

            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), device=p.device, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)


@torch.no_grad()
def _set_storage_multistream(model, optimizer_list, gpu_ar):
    """
    将模型参数、梯度和优化器状态映射到 gpu_ar 的连续内存区域。

    与 pccheck_utils.set_storage 的区别：
    - 此版本严格按 4 个连续区域映射，与 MultiStreamCheckpoint 的 4 流布局对齐：
      [0, N)    : params
      [N, 2N)   : grads
      [2N, 3N)  : exp_avg
      [3N, 4N)  : exp_avg_sq
    - pccheck_utils.set_storage 的 Region 3&4 是 exp_avg/exp_avg_sq 交错的，
      不适合 MultiStreamCheckpoint 的按流切分。
    """
    model_size = sum(p.numel() for p in model.parameters())

    # ==================== Region 1: Model Parameters [0, N) ====================
    offset = 0
    for name, ref in model.named_parameters():
        sz = ref.numel()
        my_ar = gpu_ar[offset:offset + sz]
        prev_shape = ref.size()
        temp = ref.clone()
        ref.set_(my_ar, 0, tuple(prev_shape))
        ref.copy_(temp)
        offset += sz

    # ==================== Region 2: Gradients [N, 2N) ====================
    offset = model_size
    for name, ref in model.named_parameters():
        if ref.grad is not None:
            sz = ref.grad.numel()
            my_ar = gpu_ar[offset:offset + sz]
            prev_shape = ref.grad.size()
            ref.grad.set_(my_ar, 0, tuple(prev_shape))
            offset += sz

    # ==================== Region 3: exp_avg [2N, 3N) ====================
    offset = model_size * 2
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    key = 'exp_avg' if 'exp_avg' in state else (
                        'next_m' if 'next_m' in state else None)
                    if key:
                        ea = state[key]
                        sz = ea.numel()
                        my_ar = gpu_ar[offset:offset + sz]
                        prev_shape = ea.size()
                        temp = ea.clone()
                        ea.set_(my_ar, 0, tuple(prev_shape))
                        ea.copy_(temp)
                        offset += sz

    # ==================== Region 4: exp_avg_sq [3N, 4N) ====================
    offset = model_size * 3
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    key = 'exp_avg_sq' if 'exp_avg_sq' in state else (
                        'next_v' if 'next_v' in state else None)
                    if key:
                        eas = state[key]
                        sz = eas.numel()
                        my_ar = gpu_ar[offset:offset + sz]
                        prev_shape = eas.size()
                        temp = eas.clone()
                        eas.set_(my_ar, 0, tuple(prev_shape))
                        eas.copy_(temp)
                        offset += sz

    return model_size


def _resolve_multistream_lib_path(lib_path: str) -> str:
    """解析并校验 multistream C 库路径（不改变算法，仅做兼容选择）。"""
    required_symbols = ("writer", "init_streams", "write_stream_chunk")

    def _has_required_symbols(path: str) -> bool:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            return False
        return all(hasattr(lib, s) for s in required_symbols)

    candidates = []
    if lib_path:
        candidates.append(lib_path)

    local_candidate = os.path.join(os.path.dirname(__file__), "libtest_ssd.so")
    if local_candidate not in candidates:
        candidates.append(local_candidate)

    for cand in candidates:
        if os.path.exists(cand) and _has_required_symbols(cand):
            if lib_path and cand != lib_path:
                print(f"[WARN] Provided lib_path missing required multistream symbols, fallback to: {cand}")
            return cand

    raise RuntimeError(
        "No usable multistream shared library found. "
        f"Given lib_path={lib_path}, checked candidates={candidates}. "
        "Required symbols: writer, init_streams, write_stream_chunk."
    )


def create_stage_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    lib_path: str,
    rank: int,
    world_size: int,
    max_async: int = 2,
    num_layer_groups: int = 8,
    num_threads: int = 16,
) -> Tuple[MultiStreamCheckpoint, torch.Tensor, List[Dict]]:
    """
    为当前 PP 阶段创建 MultiStreamCheckpoint，采用零拷贝设计。

    与单卡 pccheck 相同的思路：
    1. initialize() 进行四块独立分配（param/grad/exp_avg/exp_avg_sq）
    2. set_storage() 将 params/grads/optimizer states 重映射到上述四块存储
    3. 将 dict 形式缓冲区传给 MultiStreamCheckpoint，checkpoint 时零拷贝

    gpu_ar 不占用额外显存——它就是模型本身的存储。

    Returns:
        (ms_ckpt, gpu_buffers, param_layout)
    """
    resolved_lib_path = _resolve_multistream_lib_path(lib_path)

    # 1. 训练前显式初始化 grad + optimizer state，确保4区布局完整
    _ensure_grad_and_optimizer_states(model, optimizer)

    total_params = sum(p.numel() for p in model.parameters())

    # 2. 零拷贝重映射：使用 pccheck_utils 的四块独立分配路径
    #    gpu_buffers: {'param','grad','exp_avg','exp_avg_sq'}
    gpu_buffers, _ = initialize(
        model,
        [optimizer],
        do_opt_step=False,
        separate_streams=True,
    )
    set_storage(model, [optimizer], gpu_buffers)
    torch.cuda.empty_cache()

    # 3. 构建 param_layout（MultiStreamCheckpoint 需要的元数据）
    # PCCHECK_MS_QUANT=1: moments 流 int8 量化融进拷贝（payload 3N→~1.5N，
    # 慢盘环境直接砍近半写盘时间；单卡 resume-loss 1e-5 近无损已验证）。
    # 量化/分组交互（split 刀口 snap 到 block 边界、交错布局）由 v3 模块内部处理。
    quant_moments = os.environ.get("PCCHECK_MS_QUANT", "0") == "1"
    quant_kwargs = {}
    if quant_moments:
        quant_block = int(os.environ.get("PCCHECK_MS_QUANT_BLOCK", "2048"))
        quant_kwargs = {"quant_moments": True, "quant_block": quant_block}
        print(f"[multistream] PCCHECK_MS_QUANT=1: int8 moment quantization "
              f"enabled (block={quant_block})")
    param_layout = build_param_layout(model, optimizer, **quant_kwargs)
    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    os.makedirs(stage_dir, exist_ok=True)
    chk_file = os.path.join(stage_dir, f"stage_{rank}.chk")

    actual_layer_groups = min(num_layer_groups, len(param_layout))
    actual_layer_groups = max(actual_layer_groups, 1)

    # 4. 创建 MultiStreamCheckpoint，直接使用四块独立缓冲区（零拷贝）
    ms_ckpt = MultiStreamCheckpoint(
        param_layout=param_layout,
        gpu_ar=gpu_buffers,
        total_size=total_params,
        num_threads=num_threads,
        lib_path=resolved_lib_path,
        filename=chk_file,
        num_streams=4,
        max_async=max_async,
        num_layer_groups=actual_layer_groups,
        distributed=True,
        rank=rank,
        world_size=world_size,
        **quant_kwargs,
    )

    return ms_ckpt, gpu_buffers, param_layout


def save_stage_checkpoint(
    ms_ckpt: MultiStreamCheckpoint,
    gpu_buffer: torch.Tensor,
    param_layout: List[Dict],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    rank: int,
    step: int,
    sync: bool = False,
    return_breakdown: bool = False,
):
    """
    执行一次完整的 multistream 保存。

    零拷贝设计：set_storage 已将 params/grads/optimizer states 重映射到 gpu_ar，
    因此无需 pack_stage_states 拷贝，直接调用 save_full_checkpoint 即可。

    Returns:
        return_breakdown=False: 保存耗时（秒）
        return_breakdown=True: 高层前台耗时分解（毫秒）
    """
    t0 = time.perf_counter()

    # 零拷贝：gpu_ar 就是模型的存储，无需 pack_stage_states
    submit_t0 = time.perf_counter()
    parall_iter, metric = ms_ckpt.save_full_checkpoint(sync=sync)
    checkpoint_submit_ms = (time.perf_counter() - submit_t0) * 1000.0

    # export_metadata needs the model to serialize optimizer param groups/steps by name.
    metadata_t0 = time.perf_counter()
    ms_ckpt._pending_model = model
    if _ASYNC_META_ENABLED and not sync:
        # 主线程只做快照（steps 批量单 sync + param_groups 纯 CPU 结构），
        # 序列化与写盘下沉后台单线程；async 模式下 metadata_ms ≈ 快照+submit。
        name_map = {n: p for n, p in model.named_parameters()}
        steps_snap = _snapshot_optimizer_steps(model, optimizer)
        pg_snap = ms_ckpt._serialize_optimizer_param_groups(optimizer, name_map)
        lr_snap = optimizer.param_groups[0].get("lr", 0)

        def _meta_task():
            ms_ckpt.export_metadata(optimizer=None,
                                    optimizer_steps_override=steps_snap,
                                    param_groups_override=pg_snap)
            _write_multistream_info(ms_ckpt, checkpoint_dir, rank)
            _write_stage_training_state(checkpoint_dir, rank, step,
                                        optimizer=None, slot=parall_iter,
                                        lr=lr_snap)

        global _meta_last_future
        _meta_last_future = _get_meta_executor().submit(_meta_task)
        metadata_ms = (time.perf_counter() - metadata_t0) * 1000.0
        stage_state_ms = 0.0
    else:
        ms_ckpt.export_metadata(optimizer=optimizer)
        _write_multistream_info(ms_ckpt, checkpoint_dir, rank)
        metadata_ms = (time.perf_counter() - metadata_t0) * 1000.0

        state_t0 = time.perf_counter()
        _write_stage_training_state(checkpoint_dir, rank, step, optimizer,
                                    slot=parall_iter)
        stage_state_ms = (time.perf_counter() - state_t0) * 1000.0
    if sync:
        drain_metadata_writes()

    elapsed = time.perf_counter() - t0
    if return_breakdown:
        return {
            "elapsed_sec": elapsed,
            "total_ms": elapsed * 1000.0,
            "checkpoint_submit_ms": checkpoint_submit_ms,
            "metadata_ms": metadata_ms,
            "stage_state_ms": stage_state_ms,
            "parall_iter": int(parall_iter),
            "sync": bool(sync),
            "save_full_return_sec": float(metric) if isinstance(metric, (int, float)) else None,
        }
    return elapsed


def _write_multistream_info(ms_ckpt: MultiStreamCheckpoint, checkpoint_dir: str, rank: int):
    """写入 multistream_info.json，供 pipelayer MultiStreamStateLoader 发现。
    内容对一次训练静态，只写一次。"""
    if getattr(ms_ckpt, "_ms_info_written", None) == checkpoint_dir:
        return
    ms_ckpt._ms_info_written = checkpoint_dir
    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    info = {
        "lib_path": ms_ckpt.lib_path,
        "metadata_file": os.path.basename(ms_ckpt.metadata_path),
        "checkpoint_file": os.path.basename(ms_ckpt.filename),
    }
    info_path = os.path.join(stage_dir, "multistream_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _write_stage_training_state(
    checkpoint_dir: str,
    rank: int,
    step: int,
    optimizer: Optional[torch.optim.Optimizer],
    slot: Optional[int] = None,
    lr: Optional[float] = None,
):
    """写入每个阶段的训练状态。异步路径下 optimizer=None、lr 用主线程快照值。"""
    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    if lr is None:
        lr = optimizer.param_groups[0].get("lr", 0) if optimizer is not None else 0
    state = {
        "completed_steps": step,
        "rank": rank,
        "lr": lr,
    }
    if slot is not None:
        state["slot"] = int(slot)
    path = os.path.join(stage_dir, "training_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def save_global_training_state(
    checkpoint_dir: str,
    step: int,
    epoch: int = 0,
    extra: Optional[Dict] = None,
):
    """rank 0 写入全局训练状态"""
    state = {
        "completed_steps": step,
        "epoch": epoch,
        "timestamp": time.time(),
    }
    if extra:
        state.update(extra)
    path = os.path.join(checkpoint_dir, "global_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_global_training_state(checkpoint_dir: str) -> Dict:
    """读取全局训练状态"""
    path = os.path.join(checkpoint_dir, "global_state.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resume_stage_with_pipelayer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    rank: int,
    lib_path: str,
    device: str = "cuda",
    forward_prefetch_window: int = 2,
    restore_schedule_policy: str = "window_interleave",
):
    """
    使用 pipelayer 的 MultiStreamStateLoader 恢复当前 PP 阶段。

    Returns:
        training_state dict（包含 completed_steps 等信息）
    """
    handle = resume_stage_with_pipelayer_async(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        lib_path=lib_path,
        device=device,
        forward_prefetch_window=forward_prefetch_window,
        restore_schedule_policy=restore_schedule_policy,
    )
    handle.wait_all()
    training_state = dict(handle.training_state)
    handle.cleanup()
    return training_state


def resume_stage_with_pipelayer_async(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    rank: int,
    lib_path: str,
    device: str = "cuda",
    engine=None,
    iteration: Optional[int] = None,
    load_grad: bool = False,
    num_copy_workers: Optional[int] = None,
    forward_prefetch_window: int = 2,
    restore_schedule_policy: str = "window_interleave",
):
    """
    异步恢复当前 PP 阶段。

    返回 PipelayerAsyncLoadHandle。调用方可以立即进入 train_batch；
    forward hook 会按模块等待必要 chunk，第一次 optimizer step 前会等待
    全部 optimizer state 恢复完成。
    """
    from pipelayer.distributed import start_distributed_multistream_resume

    return start_distributed_multistream_resume(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        lib_path=lib_path,
        rank=rank,
        iteration=iteration,
        device=device,
        engine=engine,
        install_forward_hooks=True,
        guard_optimizer_step=True,
        load_grad=load_grad,
        num_copy_workers=num_copy_workers,
        forward_prefetch_window=forward_prefetch_window,
        restore_schedule_policy=restore_schedule_policy,
    )
