"""
DeepSpeed Pipeline Parallelism + MultiStream Checkpoint 工具

提供 PP 阶段的状态打包、元数据写入、以及 pipelayer 恢复的辅助函数。
每个 PP 阶段是独立的 PyTorch Module，可直接使用 MultiStreamCheckpoint。
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .multistream_checkpoint import MultiStreamCheckpoint, build_param_layout


def _ensure_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """确保优化器状态已初始化（至少执行一次 dummy step）"""
    has_state = any(len(s) > 0 for s in optimizer.state.values())
    if has_state:
        return
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def pack_stage_states(
    param_layout: List[Dict],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu_buffer: torch.Tensor,
    stream_sizes: List[int],
):
    """
    将 PP 阶段的参数/梯度/优化器状态打包到连续 GPU buffer。

    gpu_buffer 的布局与 MultiStreamCheckpoint 的 4 流对齐：
      [param | grad | exp_avg | exp_avg_sq]
    每段长度为 stream_sizes[i]。

    Args:
        param_layout: build_param_layout() 的输出
        model: 当前 PP 阶段的模型（未被 DDP 包装）
        optimizer: 该阶段的优化器
        gpu_buffer: 预分配的 float32 GPU tensor, 大小 = sum(stream_sizes)
        stream_sizes: 4 个流的大小列表
    """
    named_params = dict(model.named_parameters())

    buf_param = gpu_buffer[:stream_sizes[0]]
    buf_grad = gpu_buffer[stream_sizes[0]:stream_sizes[0] + stream_sizes[1]]
    buf_ea = gpu_buffer[stream_sizes[0] + stream_sizes[1]:
                        stream_sizes[0] + stream_sizes[1] + stream_sizes[2]]
    buf_eas = gpu_buffer[stream_sizes[0] + stream_sizes[1] + stream_sizes[2]:]

    for layer_info in param_layout:
        name = layer_info["name"]
        param = named_params[name]
        off = layer_info["param_offset"]
        sz = layer_info["param_size"]

        buf_param[off:off + sz].copy_(param.detach().view(-1).float())

        if param.grad is not None:
            buf_grad[off:off + sz].copy_(param.grad.detach().view(-1).float())
        else:
            buf_grad[off:off + sz].zero_()

        state = optimizer.state.get(param, {})

        ea = state.get("exp_avg")
        if ea is not None:
            buf_ea[off:off + sz].copy_(ea.detach().view(-1).float())
        else:
            buf_ea[off:off + sz].zero_()

        eas = state.get("exp_avg_sq")
        if eas is not None:
            buf_eas[off:off + sz].copy_(eas.detach().view(-1).float())
        else:
            buf_eas[off:off + sz].zero_()


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
    为当前 PP 阶段创建 MultiStreamCheckpoint 及其 GPU buffer。

    Returns:
        (ms_ckpt, gpu_buffer, param_layout)
    """
    _ensure_optimizer_state(model, optimizer)

    param_layout = build_param_layout(model, optimizer)
    total_params = sum(p.numel() for p in model.parameters())
    total_buffer_size = total_params * 4

    device = next(model.parameters()).device
    gpu_buffer = torch.empty(total_buffer_size, dtype=torch.float32, device=device)

    stage_dir = os.path.join(checkpoint_dir, f"stage_{rank}")
    os.makedirs(stage_dir, exist_ok=True)
    chk_file = os.path.join(stage_dir, f"stage_{rank}.chk")

    actual_layer_groups = min(num_layer_groups, len(param_layout))
    actual_layer_groups = max(actual_layer_groups, 1)

    ms_ckpt = MultiStreamCheckpoint(
        param_layout=param_layout,
        gpu_ar=gpu_buffer,
        total_size=total_params,
        num_threads=num_threads,
        lib_path=lib_path,
        filename=chk_file,
        num_streams=4,
        max_async=max_async,
        num_layer_groups=actual_layer_groups,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    return ms_ckpt, gpu_buffer, param_layout


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

    Returns:
        保存耗时（秒）
    """
    t0 = time.time()

    pack_stage_states(param_layout, model, optimizer, gpu_buffer, ms_ckpt.stream_sizes)

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
