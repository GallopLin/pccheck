"""
DeepSpeed Pipeline Parallelism + MultiStream Checkpoint 工具

提供 PP 阶段的状态打包、元数据写入、以及 pipelayer 恢复的辅助函数。
每个 PP 阶段是独立的 PyTorch Module，可直接使用 MultiStreamCheckpoint。

采用与单卡 pccheck 相同的零拷贝设计：
    1. initialize() 分配 GPU 存储（支持四块独立分配）
    2. set_storage() 用 tensor.set_() 将 params/grads/optimizer states 重映射到底层存储
    3. MultiStreamCheckpoint 直接从映射后的存储读取，无需额外拷贝
"""

import json
import os
import time
import ctypes
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .multistream_checkpoint import MultiStreamCheckpoint, build_param_layout
from ..pccheck_utils import initialize, set_storage


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
    param_layout = build_param_layout(model, optimizer)
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
) -> float:
    """
    执行一次完整的 multistream 保存。

    零拷贝设计：set_storage 已将 params/grads/optimizer states 重映射到 gpu_ar，
    因此无需 pack_stage_states 拷贝，直接调用 save_full_checkpoint 即可。

    Returns:
        保存耗时（秒）
    """
    t0 = time.time()

    # 零拷贝：gpu_ar 就是模型的存储，无需 pack_stage_states
    parall_iter, metric = ms_ckpt.save_full_checkpoint(sync=sync)

    ms_ckpt.export_metadata(optimizer=optimizer)

    _write_multistream_info(ms_ckpt, checkpoint_dir, rank)
    _write_stage_training_state(checkpoint_dir, rank, step, optimizer)

    elapsed = time.time() - t0
    return elapsed


def _write_multistream_info(ms_ckpt: MultiStreamCheckpoint, checkpoint_dir: str, rank: int):
    """写入 multistream_info.json，供 pipelayer MultiStreamStateLoader 发现"""
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
    optimizer: torch.optim.Optimizer,
):
    """写入每个阶段的训练状态"""
    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    state = {
        "completed_steps": step,
        "rank": rank,
        "lr": optimizer.param_groups[0].get("lr", 0),
    }
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
):
    """
    使用 pipelayer 的 MultiStreamStateLoader 恢复当前 PP 阶段。

    Returns:
        training_state dict（包含 completed_steps 等信息）
    """
    from pipelayer.checkpointing import MultiStreamStateLoader

    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    info_path = os.path.join(stage_dir, "multistream_info.json")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    metadata_file = info["metadata_file"]
    checkpoint_file = info["checkpoint_file"]
    loader_lib_path = info.get("lib_path", lib_path)

    loader = MultiStreamStateLoader(
        model=model,
        optimizer=optimizer,
        chkpt_dir=stage_dir,
        lib_path=loader_lib_path,
        metadata_file=metadata_file,
        checkpoint_file=checkpoint_file,
        device=device,
    )

    for i in range(loader.num_chunks):
        loader.wait_for_chunk(i)
    loader.stop()

    state_path = os.path.join(stage_dir, "training_state.json")
    training_state = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            training_state = json.load(f)

    return training_state
