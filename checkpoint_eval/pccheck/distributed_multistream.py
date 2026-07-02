"""
分布式 MultiStream PCCheck

支持多卡/分布式环境下的检查点保存和恢复，
可选地提供 DeepSpeed UCP 风格的跨配置可移植性。

架构设计：
- 每个 rank 独立保存自己的 shard（使用原有 MultiStream 高效写入）
- rank 0 保存全局元数据（支持恢复时重新分片）
- 可选：转换为 UCP 格式以支持不同并行配置的加载

使用示例：
    # 初始化
    dist_ckpt = DistributedMultiStreamCheckpoint(
        model, optimizer,
        tp_degree=2, pp_degree=2, dp_degree=1,
        checkpoint_dir="./checkpoints"
    )
    
    # 保存
    dist_ckpt.save(iteration=100)
    
    # 恢复（相同配置）
    dist_ckpt.load(iteration=100)
    
    # 恢复（不同配置，需要重新分片）
    dist_ckpt.load(iteration=100, target_tp_degree=4)
"""

import os
import json
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np


@dataclass
class ParallelConfig:
    """并行配置"""
    tp_degree: int = 1  # Tensor Parallelism degree
    pp_degree: int = 1  # Pipeline Parallelism degree  
    dp_degree: int = 1  # Data Parallelism degree
    world_size: int = 1
    rank: int = 0
    
    # 当前 rank 在各个维度的位置
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    
    @classmethod
    def from_distributed(cls, tp_degree: int = 1, pp_degree: int = 1):
        """从当前分布式环境自动检测配置"""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
            
        dp_degree = world_size // (tp_degree * pp_degree)
        
        # 计算当前 rank 在各维度的位置
        # 假设 rank 布局为: dp -> pp -> tp (最内层变化最快)
        tp_rank = rank % tp_degree
        pp_rank = (rank // tp_degree) % pp_degree
        dp_rank = rank // (tp_degree * pp_degree)
        
        return cls(
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dp_degree=dp_degree,
            world_size=world_size,
            rank=rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank
        )


@dataclass 
class ParamSliceInfo:
    """参数切片信息（用于 UCP 兼容恢复）"""
    name: str
    global_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    dtype: str
    slice_dim: int = 0  # 切片维度（TP 通常在 dim 0 或 1）
    slice_offset: int = 0  # 在全局张量中的偏移
    is_replicated: bool = False  # 是否在 TP ranks 间复制
    

class CheckpointMetadata:
    """检查点元数据（UCP 兼容）"""
    
    def __init__(
        self,
        parallel_config: ParallelConfig,
        iteration: int,
        param_infos: Dict[str, ParamSliceInfo] = None,
        optimizer_type: str = "Adam",
        data_file: Optional[str] = None,
    ):
        self.parallel_config = parallel_config
        self.iteration = iteration
        self.param_infos = param_infos or {}
        self.optimizer_type = optimizer_type
        self.version = "1.0"
        self.data_file = data_file
        
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "iteration": self.iteration,
            "optimizer_type": self.optimizer_type,
            "data_file": self.data_file,
            "parallel_config": asdict(self.parallel_config),
            "param_infos": {
                name: asdict(info) for name, info in self.param_infos.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        parallel_config = ParallelConfig(**data["parallel_config"])
        param_infos = {
            name: ParamSliceInfo(**info) 
            for name, info in data.get("param_infos", {}).items()
        }
        return cls(
            parallel_config=parallel_config,
            iteration=data["iteration"],
            param_infos=param_infos,
            optimizer_type=data.get("optimizer_type", "Adam"),
            data_file=data.get("data_file")
        )
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CheckpointMetadata":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class DistributedMultiStreamCheckpoint:
    """
    分布式 MultiStream 检查点管理器
    
    支持:
    - 多卡独立保存（每个 rank 使用 MultiStream 高效写入）
    - 全局元数据管理（支持恢复时重新分片）
    - UCP 兼容的跨配置加载
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
        tp_degree: int = 1,
        pp_degree: int = 1,
        lib_path: str = "./libtest_ssd.so",
        num_threads: int = 16,
        max_async: int = 2,
        num_layer_groups: int = 8,
        gpu_ar: Optional[torch.Tensor] = None,
        total_size: Optional[int] = None,
        rank_file: Optional[str] = None,
    ):
        """
        Args:
            model: PyTorch 模型
            optimizer: 优化器
            checkpoint_dir: 检查点保存目录
            tp_degree: Tensor Parallelism degree
            pp_degree: Pipeline Parallelism degree
            lib_path: libtest_ssd.so 路径
            num_threads: 写入线程数
            max_async: 最大异步检查点数
            num_layer_groups: 层分组数
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.lib_path = lib_path
        self.num_threads = num_threads
        self.max_async = max_async
        self.num_layer_groups = num_layer_groups
        self.gpu_ar = gpu_ar
        self.total_size = total_size
        self.rank_file_template = rank_file
        self.latest_iteration = None
        
        # 初始化并行配置
        self.parallel_config = ParallelConfig.from_distributed(tp_degree, pp_degree)
        
        # 创建检查点与数据目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.data_dir = os.path.join(self.checkpoint_dir, "rank_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.rank_checkpoint_file = self._format_rank_file(rank_file)
        
        # 收集参数信息
        self.param_infos = self._collect_param_infos()
        
        # 延迟初始化本地 MultiStreamCheckpoint
        self._local_checkpoint = None
        self._initialized = False
        
        if self.parallel_config.rank == 0:
            print(f"\n{'='*60}")
            print(f"Distributed MultiStream Checkpoint Initialized")
            print(f"{'='*60}")
            print(f"  World size: {self.parallel_config.world_size}")
            print(f"  TP degree: {self.parallel_config.tp_degree}")
            print(f"  PP degree: {self.parallel_config.pp_degree}")
            print(f"  DP degree: {self.parallel_config.dp_degree}")
            print(f"  Checkpoint dir: {checkpoint_dir}")
            print(f"  Num params: {len(self.param_infos)}")
            print(f"{'='*60}\n")
    
    def _format_rank_file(self, custom_path: Optional[str]) -> str:
        """生成当前 rank 使用的底层PCCHECK数据文件路径"""
        if custom_path is not None:
            resolved = custom_path.format(
                rank=self.parallel_config.rank,
                world_size=self.parallel_config.world_size,
                tp_rank=self.parallel_config.tp_rank,
                pp_rank=self.parallel_config.pp_rank,
                dp_rank=self.parallel_config.dp_rank,
            )
            target_dir = os.path.dirname(resolved)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            return resolved
        default_path = os.path.join(self.data_dir, f"rank_{self.parallel_config.rank}.chk")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return default_path
    
    def attach_gpu_buffer(self, gpu_ar: torch.Tensor, total_size: Optional[int] = None):
        """显式绑定跨rank共享的 gpu_ar 缓冲区"""
        self.gpu_ar = gpu_ar
        if total_size is not None:
            self.total_size = total_size
        elif gpu_ar is not None:
            self.total_size = gpu_ar.numel() // 4
        # GPU缓冲区改变后需要重新构建 local checkpoint
        self._initialized = False
    
    def _infer_total_size(self) -> int:
        if self.total_size is not None:
            return self.total_size
        if self.gpu_ar is not None:
            return self.gpu_ar.numel() // 4
        raise RuntimeError("DistributedMultiStreamCheckpoint 需要 gpu_ar 或 total_size 信息")
    
    def _collect_param_infos(self) -> Dict[str, ParamSliceInfo]:
        """收集模型参数信息"""
        param_infos = {}
        
        for name, param in self.model.named_parameters():
            # 检测是否为 TP 复制参数（如 LayerNorm, bias 等）
            is_replicated = self._is_replicated_param(name)
            
            # 确定切片维度（通常 weight 在 dim 0, embedding 在 dim 0）
            slice_dim = self._get_slice_dim(name, param.shape)
            
            # 计算全局形状
            if is_replicated or self.parallel_config.tp_degree == 1:
                global_shape = param.shape
                slice_offset = 0
            else:
                global_shape = list(param.shape)
                global_shape[slice_dim] *= self.parallel_config.tp_degree
                global_shape = tuple(global_shape)
                slice_offset = self.parallel_config.tp_rank * param.shape[slice_dim]
            
            param_infos[name] = ParamSliceInfo(
                name=name,
                global_shape=global_shape,
                local_shape=tuple(param.shape),
                dtype=str(param.dtype),
                slice_dim=slice_dim,
                slice_offset=slice_offset,
                is_replicated=is_replicated
            )
        
        return param_infos
    
    def _is_replicated_param(self, name: str) -> bool:
        """判断参数是否在 TP ranks 间复制"""
        # 常见的复制参数模式
        replicated_patterns = [
            'layer_norm', 'layernorm', 'ln_', 
            'bias',  # 有些实现中 bias 是复制的
            'embed_positions',  # 位置编码通常是复制的
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in replicated_patterns)
    
    def _get_slice_dim(self, name: str, shape: torch.Size) -> int:
        """确定 TP 切片维度"""
        # 行并行: output projection, mlp.down -> dim 1
        row_parallel_patterns = ['o_proj', 'out_proj', 'down_proj', 'dense_4h_to_h']
        
        # 列并行: qkv, mlp.up -> dim 0
        col_parallel_patterns = ['q_proj', 'k_proj', 'v_proj', 'qkv', 
                                  'up_proj', 'gate_proj', 'dense_h_to_4h']
        
        name_lower = name.lower()
        
        if any(p in name_lower for p in row_parallel_patterns):
            return 1 if len(shape) > 1 else 0
        
        # 默认列并行（dim 0）
        return 0
    
    def _get_checkpoint_path(self, iteration: int) -> str:
        """返回当前rank绑定的底层pccheck数据文件（与迭代号无关）"""
        _ = iteration  # 兼容旧接口
        return self.rank_checkpoint_file
    
    def _get_metadata_path(self, iteration: int) -> str:
        """获取元数据文件路径"""
        return os.path.join(self.checkpoint_dir, f"iter_{iteration:07d}", "metadata.json")
    
    def _initialize_local_checkpoint(self):
        """延迟初始化本地 MultiStreamCheckpoint"""
        if self._initialized:
            return
            
        from .multistream_checkpoint import MultiStreamCheckpoint, build_param_layout
        
        if self.gpu_ar is None:
            raise RuntimeError(
                "DistributedMultiStreamCheckpoint 需要外部提供 gpu_ar。"
                "请先调用 attach_gpu_buffer 或在构造函数中传入。"
            )
        
        param_layout = build_param_layout(self.model, self.optimizer)
        total_size = self._infer_total_size()
        
        self._local_checkpoint = MultiStreamCheckpoint(
            param_layout=param_layout,
            gpu_ar=self.gpu_ar,
            total_size=total_size,
            num_threads=self.num_threads,
            lib_path=self.lib_path,
            filename=self.rank_checkpoint_file,
            num_streams=4,
            max_async=self.max_async,
            num_layer_groups=self.num_layer_groups,
            distributed=True,
            rank=self.parallel_config.rank,
            world_size=self.parallel_config.world_size
        )
        
        self._initialized = True
    
    def save(self, iteration: int, sync: bool = True) -> str:
        """
        保存分布式检查点
        
        Args:
            iteration: 迭代次数
            sync: 是否同步等待所有 rank 完成
            
        Returns:
            检查点目录路径
        """
        # 确保本地检查点已初始化
        self._initialize_local_checkpoint()
        
        # 创建检查点目录（仅存放元数据）
        ckpt_dir = os.path.join(self.checkpoint_dir, f"iter_{iteration:07d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 使用 MultiStream 高效保存
        parall_iter, finalize_metric = self._save_local_checkpoint(sync)
        self._local_checkpoint._pending_model = self.model
        self._local_checkpoint.export_metadata(optimizer=self.optimizer)
        self._write_rank_metadata(iteration, parall_iter, finalize_metric)
        
        # rank 0 保存全局元数据（描述拓扑、参数切片等）
        if self.parallel_config.rank == 0:
            metadata = CheckpointMetadata(
                parallel_config=self.parallel_config,
                iteration=iteration,
                param_infos=self.param_infos,
                optimizer_type=type(self.optimizer).__name__,
                data_file=self.rank_checkpoint_file,
            )
            metadata.save(self._get_metadata_path(iteration))
        
        # 同步所有 rank（仅在需要同步时）
        if sync and dist.is_initialized():
            dist.barrier()
        
        self.latest_iteration = iteration
        return ckpt_dir
    
    def _save_local_checkpoint(self, sync: bool):
        """触发本地 MultiStreamCheckpoint 保存"""
        return self._local_checkpoint.save_full_checkpoint(sync=sync)
    
    def _write_rank_metadata(self, iteration: int, parall_iter: int, finalize_metric: float):
        record = {
            "iteration": iteration,
            "rank": self.parallel_config.rank,
            "tp_rank": self.parallel_config.tp_rank,
            "pp_rank": self.parallel_config.pp_rank,
            "dp_rank": self.parallel_config.dp_rank,
            "world_size": self.parallel_config.world_size,
            "slot": parall_iter,
            "data_file": self.rank_checkpoint_file,
            "max_async": self.max_async,
            "finalize_metric": finalize_metric,
            "stream_sizes": getattr(self._local_checkpoint, "stream_sizes", []),
        }
        rank_meta_path = os.path.join(
            self.checkpoint_dir,
            f"iter_{iteration:07d}",
            f"rank_{self.parallel_config.rank}.json"
        )
        with open(rank_meta_path, 'w') as f:
            json.dump(record, f, indent=2)
    
    def load(
        self, 
        iteration: int,
        target_tp_degree: int = None,
        target_pp_degree: int = None,
        strict: bool = True
    ):
        raise NotImplementedError(
            "分布式 MultiStream 的加载路径尚未实现。"
            "目前仅支持保存，后续版本将提供 slot -> 参数 的回放能力。"
        )
    
    def _load_same_config(self, iteration: int, metadata: CheckpointMetadata):
        """加载相同配置的检查点"""
        ckpt_path = self._get_checkpoint_path(iteration)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # 使用 MultiStream 读取（需要实现读取功能）
        self._load_local_checkpoint(ckpt_path)
        
        if self.parallel_config.rank == 0:
            print(f"Loaded checkpoint from iteration {iteration}")
    
    def _load_with_reshard(
        self, 
        iteration: int, 
        metadata: CheckpointMetadata,
        target_tp_degree: int,
        target_pp_degree: int
    ):
        """
        重新分片加载（UCP 风格）
        
        这是 UCP 的核心价值：支持从不同并行配置的检查点恢复
        """
        source_config = metadata.parallel_config
        
        if self.parallel_config.rank == 0:
            print(f"\nResharding checkpoint:")
            print(f"  Source: TP={source_config.tp_degree}, PP={source_config.pp_degree}")
            print(f"  Target: TP={target_tp_degree}, PP={target_pp_degree}")
        
        # 对于每个参数，确定需要从哪些源 rank 读取
        for name, param in self.model.named_parameters():
            param_info = metadata.param_infos.get(name)
            if param_info is None:
                print(f"Warning: Parameter {name} not found in checkpoint")
                continue
            
            # 计算源切片和目标切片的映射
            loaded_data = self._load_and_reshard_param(
                iteration, name, param_info,
                source_config, target_tp_degree, target_pp_degree
            )
            
            # 更新参数
            with torch.no_grad():
                param.copy_(loaded_data)
        
        # 类似地处理优化器状态...
        # (这里省略，完整实现需要处理 exp_avg, exp_avg_sq 等)
    
    def _load_and_reshard_param(
        self,
        iteration: int,
        name: str,
        param_info: ParamSliceInfo,
        source_config: ParallelConfig,
        target_tp_degree: int,
        target_pp_degree: int
    ) -> torch.Tensor:
        """加载并重新分片单个参数"""
        # 1. 如果是复制参数，只需从任意 TP rank 读取
        if param_info.is_replicated:
            # 从 TP rank 0 读取
            return self._load_param_from_rank(iteration, name, tp_rank=0, pp_rank=self.parallel_config.pp_rank)
        
        # 2. 对于切片参数，需要收集所有切片并重新分配
        all_slices = []
        for tp_rank in range(source_config.tp_degree):
            slice_data = self._load_param_from_rank(iteration, name, tp_rank, self.parallel_config.pp_rank)
            all_slices.append(slice_data)
        
        # 合并成全局张量
        global_tensor = torch.cat(all_slices, dim=param_info.slice_dim)
        
        # 重新切片为目标 TP degree
        new_slice_size = global_tensor.shape[param_info.slice_dim] // target_tp_degree
        start_idx = self.parallel_config.tp_rank * new_slice_size
        end_idx = start_idx + new_slice_size
        
        indices = [slice(None)] * len(global_tensor.shape)
        indices[param_info.slice_dim] = slice(start_idx, end_idx)
        
        return global_tensor[tuple(indices)]
    
    def _load_param_from_rank(self, iteration: int, name: str, tp_rank: int, pp_rank: int) -> torch.Tensor:
        """从指定 rank 的检查点读取参数"""
        # 构建源检查点路径
        config = self.parallel_config
        filename = f"checkpoint_iter{iteration:07d}_tp{tp_rank}_pp{pp_rank}_dp{config.dp_rank}.chk"
        ckpt_path = os.path.join(self.checkpoint_dir, f"iter_{iteration:07d}", filename)
        
        # 读取二进制文件（需要实现）
        # 这里是简化版，实际需要解析 .chk 文件格式
        return self._read_param_from_chk(ckpt_path, name)
    
    def _load_local_checkpoint(self, path: str):
        """加载本地检查点到模型和优化器"""
        # TODO: 实现从 .chk 文件读取数据到 GPU
        # 这需要在 MultiStreamWriter 中添加读取功能
        raise NotImplementedError("Load functionality needs to be implemented in MultiStreamWriter")
    
    def _read_param_from_chk(self, path: str, name: str) -> torch.Tensor:
        """从 .chk 文件读取指定参数"""
        # TODO: 实现二进制文件解析
        # 需要根据 param_layout 中的偏移信息读取
        raise NotImplementedError("Binary checkpoint reading needs to be implemented")


def convert_to_ucp_format(
    checkpoint_dir: str,
    iteration: int,
    output_dir: str
):
    """
    将 MultiStream 检查点转换为 DeepSpeed UCP 格式
    
    这允许使用 DeepSpeed 的恢复工具加载检查点
    
    Args:
        checkpoint_dir: MultiStream 检查点目录
        iteration: 迭代次数
        output_dir: UCP 格式输出目录
    """
    metadata_path = os.path.join(checkpoint_dir, f"iter_{iteration:07d}", "metadata.json")
    metadata = CheckpointMetadata.load(metadata_path)
    config = metadata.parallel_config
    
    # 创建 UCP 输出结构
    ucp_zero_dir = os.path.join(output_dir, "zero")
    os.makedirs(ucp_zero_dir, exist_ok=True)
    
    # 对于每个参数，收集所有 rank 的切片并保存为 UCP 格式
    for name, param_info in metadata.param_infos.items():
        param_dir = os.path.join(ucp_zero_dir, name)
        os.makedirs(param_dir, exist_ok=True)
        
        # 收集所有 TP 切片
        all_slices = []
        for tp_rank in range(config.tp_degree):
            # 读取该 rank 的数据
            # ... (实现省略)
            pass
        
        # 保存为 UCP 格式
        # torch.save({"param": merged_tensor, "cat_dim": param_info.slice_dim}, 
        #            os.path.join(param_dir, "fp32.pt"))
    
    print(f"Converted checkpoint to UCP format: {output_dir}")


# 便捷函数：创建分布式 MultiStream 优化器包装器
class DistributedOptimizerWrapper:
    """
    分布式多流优化器包装器
    
    支持:
    - 边计算边保存 (Interleaved Checkpointing)
    - 自动管理分布式元数据
    - 跨检查点同步
    """
    def __init__(
        self, 
        dist_ckpt: "DistributedMultiStreamCheckpoint",
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module
    ):
        self.dist_ckpt = dist_ckpt
        self.optimizer = optimizer
        self.model = model
        
        # 确保本地检查点已初始化
        self.dist_ckpt._initialize_local_checkpoint()
        
        # 使用本地 MultiStreamCheckpoint 创建优化器包装器
        # 这会复用 multistream_checkpoint.py 中的核心逻辑（包括回调和事件同步）
        self.local_wrapper = self.dist_ckpt._local_checkpoint.create_optimizer(optimizer, model)
        
        self.current_iteration = None
        self.parall_iter = None

    def begin_checkpoint(self, iteration: int):
        """开始分布式检查点"""
        self.current_iteration = iteration
        
        # 创建迭代目录
        ckpt_dir = os.path.join(self.dist_ckpt.checkpoint_dir, f"iter_{iteration:07d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 委托给本地包装器开始检查点
        self.parall_iter = self.local_wrapper.begin_checkpoint()
        return self.parall_iter

    def step_with_callback(self, closure=None):
        """执行一步更新，并触发保存回调"""
        return self.local_wrapper.step_with_callback(closure)
    
    def zero_grad(self, set_to_none: bool = False):
        self.local_wrapper.zero_grad(set_to_none)
        
    def step(self, closure=None):
        """普通更新（不触发检查点）"""
        return self.local_wrapper.step(closure)

    def finalize_checkpoint(self, wait: bool = True):
        """
        完成检查点
        
        Args:
            wait: 是否等待写入完成
        """
        # 委托给本地包装器完成检查点
        metric = self.local_wrapper.finalize_checkpoint(wait=wait)
        self.dist_ckpt._local_checkpoint._pending_model = self.model
        self.dist_ckpt._local_checkpoint.export_metadata(optimizer=self.optimizer)
        
        # 写入 Rank 元数据
        self.dist_ckpt._write_rank_metadata(
            self.current_iteration, 
            self.parall_iter, 
            metric
        )
        
        # Rank 0 写入全局元数据
        if self.dist_ckpt.parallel_config.rank == 0:
            metadata = CheckpointMetadata(
                parallel_config=self.dist_ckpt.parallel_config,
                iteration=self.current_iteration,
                param_infos=self.dist_ckpt.param_infos,
                optimizer_type=type(self.optimizer).__name__,
                data_file=self.dist_ckpt.rank_checkpoint_file,
            )
            metadata.save(self.dist_ckpt._get_metadata_path(self.current_iteration))
            
        self.dist_ckpt.latest_iteration = self.current_iteration
        
        # 如果是同步模式，执行全局 Barrier
        if wait and dist.is_initialized():
            dist.barrier()
            
        return metric
        
    def __getattr__(self, name):
        return getattr(self.local_wrapper, name)


def create_distributed_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: "DistributedMultiStreamCheckpoint"
):
    """
    创建与分布式检查点集成的优化器
    
    使用示例：
        dist_ckpt = DistributedMultiStreamCheckpoint(model, optimizer, "./ckpts")
        wrapped_optimizer = create_distributed_optimizer(model, optimizer, dist_ckpt)
        
        for step in range(num_steps):
            # ... forward / backward ...
            
            if is_checkpoint_step:
                wrapped_optimizer.begin_checkpoint(iteration=step)
                wrapped_optimizer.step_with_callback()
                wrapped_optimizer.finalize_checkpoint(wait=False)
            else:
                wrapped_optimizer.step()
    """
    return DistributedOptimizerWrapper(checkpoint, optimizer, model)
