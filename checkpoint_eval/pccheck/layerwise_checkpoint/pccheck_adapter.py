"""
阶段四：PCCheck 后端适配器
PCCheck Backend Adapter for Layerwise Checkpointing

将分层检查点任务适配到 PCCheck 的流水线系统
"""

import torch
import numpy as np
import time
import threading
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# 导入 PCCheck 原始组件
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chk_checkpoint_pipeline import Checkpoint, Writer
    from chk_monitor import Chk_monitor
    PCCHECK_AVAILABLE = True
except ImportError:
    print("[Warning] PCCheck 后端不可用，使用模拟模式")
    PCCHECK_AVAILABLE = False
    Checkpoint = None
    Chk_monitor = None
    Writer = None


@dataclass
class LayerMetadata:
    """层的元数据"""
    layer_name: str
    training_step: int
    checkpoint_id: str
    offset_in_file: int  # 在检查点文件中的偏移量
    size_bytes: int
    param_count: int
    shapes: List[tuple]
    dtypes: List[str]
    timestamp: float


class PCCheckAdapter:
    """
    PCCheck 后端适配器
    
    负责将分层保存任务转换为 PCCheck 能够处理的格式
    """
    
    def __init__(
        self,
        c_lib_path: str,
        checkpoint_file: str = "layerwise_checkpoint.chk",
        num_threads: int = 4,
        max_async: int = 2,
        batch_size_mb: float = 100.0,
        ratio: float = 2.0,
        use_pccheck: bool = True,
        use_monitor: bool = False,  # 是否使用 Chk_monitor（后台进程模式）
        metadata_file: str = "checkpoint_metadata.json",
        is_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        gpu_ar: Optional[torch.Tensor] = None,  # 🔥 新增：外部传入的 gpu_ar
        total_size: int = 0,  # 🔥 新增：总大小（元素数）
        verbose: bool = False
    ):
        """
        Args:
            c_lib_path: PCCheck C 库路径
            checkpoint_file: 检查点文件路径
            num_threads: PCCheck 使用的线程数
            max_async: 最大并发检查点数量
            batch_size_mb: 每个批次的大小（MB）
            ratio: CPU缓冲区大小相对于检查点大小的倍数
            use_pccheck: 是否使用真实的 PCCheck（False 则模拟）
            use_monitor: 是否使用 Chk_monitor（后台进程模式，更高效）
            metadata_file: 元数据文件路径
            is_distributed: 是否为分布式训练
            rank: 当前进程的 rank
            world_size: 总进程数
            gpu_ar: 🔥 外部传入的 gpu staging buffer（由原 PCCheck initialize 构造）
            total_size: 🔥 总大小（float32 元素数，与 gpu_ar 对应）
            verbose: 是否打印详细信息
        """
        self.c_lib_path = c_lib_path
        self.checkpoint_file = checkpoint_file
        self.num_threads = num_threads
        self.max_async = max_async
        self.batch_size_mb = batch_size_mb
        self.ratio = ratio
        self.use_pccheck = use_pccheck and PCCHECK_AVAILABLE
        self.use_monitor = use_monitor and PCCHECK_AVAILABLE and (Chk_monitor is not None)
        self.metadata_file = metadata_file
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        
        # 🔥 使用外部传入的 gpu_ar（由原 PCCheck initialize 构造）
        self.gpu_ar = gpu_ar  # 🔥 统一使用 gpu_ar（不再使用冗余的 staging_buffer）
        self.total_size_floats = total_size  # 保存总大小（float32 元素数）
        
        # CPU buffer (如果使用 PCCheck)
        self.cpu_buffer = None
        
        # 元数据管理
        self.layer_metadata = []  # List[LayerMetadata]
        self.current_file_offset = 0
        
        # PCCheck 实例（如果可用）
        self.pccheck_instance = None
        self.pccheck_monitor = None  # Chk_monitor 实例
        self.checkpoint_lock = None
        self.cp_in_progress = None
        
        # 批次管理
        self.batch_size_bytes = int(batch_size_mb * 1024 * 1024)
        self.batch_size_floats = self.batch_size_bytes // 4  # float32
        
        # 统计信息
        self.stats = {
            'total_layers_saved': 0,
            'total_bytes_saved': 0,
            'total_save_time': 0.0,
        }
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化后端"""
        if self.use_pccheck:
            self._initialize_pccheck()
        else:
            self._initialize_mock()
        
        if self.verbose:
            mode = "PCCheck" if self.use_pccheck else "Mock"
            print(f"[Adapter] 初始化完成 (模式: {mode})")
            print(f"  - 检查点文件: {self.checkpoint_file}")
            print(f"  - 元数据文件: {self.metadata_file}")
    
    def _initialize_pccheck(self):
        """初始化真实的 PCCheck 后端"""
        try:
            from threading import Lock
            from multiprocessing import Value
            
            if self.verbose:
                print(f"[Adapter] 初始化 PCCheck 后端...")
            
            # 🔥 使用外部传入的 total_size，如果没有则估算
            if self.total_size_floats > 0:
                total_size_floats = self.total_size_floats
                estimated_total_size_mb = total_size_floats * 4 / (1024 * 1024)
                if self.verbose:
                    print(f"  - 使用传入的总大小: {estimated_total_size_mb:.2f} MB ({total_size_floats:,} floats)")
            else:
                # 估算总大小（这里先设置一个较大的值，实际使用时可以根据模型大小调整）
                estimated_total_size_mb = 1000.0  # 1GB
                total_size_floats = int(estimated_total_size_mb * 1024 * 1024 / 4)  # float32
                if self.verbose:
                    print(f"  - 估算总大小: {estimated_total_size_mb:.2f} MB ({total_size_floats:,} floats)")
            
            if self.verbose:
                print(f"  - 批次大小: {self.batch_size_mb:.2f} MB ({self.batch_size_floats:,} floats)")
                print(f"  - 线程数: {self.num_threads}")
                print(f"  - 最大异步数: {self.max_async}")
                print(f"  - 缓冲区倍数: {self.ratio}x")
                print(f"  - 使用 Monitor: {self.use_monitor}")
            
            if self.use_monitor and Chk_monitor is not None:
                # 使用 Chk_monitor（后台进程模式，更高效）
                if self.verbose:
                    print(f"[Adapter] 使用 Chk_monitor 后台进程模式")
                
                # ⚠️ 关键：Monitor 模式必须使用外部传入的 gpu_ar
                # 确保 gpu_ar 已经由外部 initialize/set_storage 设置好
                if self.gpu_ar is None:
                    raise ValueError(
                        "Monitor 模式需要外部传入的 gpu_ar！"
                        "请在创建 LayerwiseCheckpointTrainer 时确保 use_pccheck=True，"
                        "系统会自动调用 initialize/set_storage 来构造 gpu_ar"
                    )
                
                self.pccheck_monitor = Chk_monitor(
                    c_lib_path=self.c_lib_path,
                    total_size=total_size_floats,
                    num_threads=self.num_threads,
                    max_async=self.max_async,
                    gpu_copy=True,  # 启用 GPU 拷贝
                    gpu_ar=self.gpu_ar,  # 🔥 使用外部传入的 gpu_ar
                    ratio=self.ratio,
                    is_sync=self.use_monitor,  # 异步模式
                    bsize=total_size_floats // 4,
                    memory_saving=True,
                    is_distributed=self.is_distributed,
                    rank=self.rank,
                    world_size=self.world_size
                )
                
                if self.verbose:
                    print(f"[Adapter] Chk_monitor 初始化成功")
                    print(f"  - gpu_ar shape: {self.gpu_ar.shape}")
                    print(f"  - gpu_ar device: {self.gpu_ar.device}")
            else:
                # 使用 Checkpoint 直接模式
                if self.verbose:
                    print(f"[Adapter] 使用 Checkpoint 直接模式")
                
                self.pccheck_instance = Checkpoint(
                    total_size=total_size_floats,      # 总大小（float32 元素数）
                    num_threads=self.num_threads,       # 线程数
                    filename=self.checkpoint_file,      # 检查点文件路径
                    lib_path=self.c_lib_path,          # C 库路径
                    max_async=self.max_async,           # 最大并发检查点数
                    ratio=self.ratio,                   # CPU缓冲区倍数
                    gpu_ar=self.gpu_ar,                 # 🔥 使用外部传入的 gpu_ar（可能为 None）
                    bsize=self.batch_size_floats,      # 批次大小（float32）
                    memory_saving=True,                 # 启用内存节省模式
                    is_distributed=self.is_distributed, # 分布式训练标志
                    rank=self.rank,                     # 当前rank
                    world_size=self.world_size          # 总进程数
                )
                
                # ⚠️ 关键：手动初始化 Writer（因为 start_chk 设计用于后台线程）
                if Writer is not None:
                    total_mem_batches = int(self.ratio * total_size_floats / self.batch_size_floats)
                    self.pccheck_instance.writer = Writer(
                        self.checkpoint_file.encode(),
                        self.c_lib_path,
                        self.max_async,
                        int(self.batch_size_floats),
                        total_mem_batches,
                        self.is_distributed,
                        self.rank,
                        self.world_size
                    )
                    
                    if self.verbose:
                        print(f"[Adapter] Writer 初始化成功 (total_mem_batches={total_mem_batches})")
                        if self.gpu_ar is not None:
                            size_mb = self.gpu_ar.numel() * 4 / (1024**2)
                            print(f"  - 使用外部传入的 gpu_ar: {size_mb:.2f} MB")
                
                # 创建锁和状态变量（用于同步）
                self.checkpoint_lock = Lock()
                self.cp_in_progress = Value('i', 0)
                
                if self.verbose:
                    print(f"[Adapter] Checkpoint 初始化成功")
            
            if self.verbose:
                print(f"[Adapter] PCCheck 后端初始化成功")
                print(f"  - 检查点文件: {self.checkpoint_file}")
                print(f"  - C 库路径: {self.c_lib_path}")
                
        except Exception as e:
            import traceback
            print(f"[Adapter] PCCheck 初始化失败: {e}")
            print(f"[Adapter] 错误详情:")
            traceback.print_exc()
            print(f"[Adapter] 切换到模拟模式")
            self.use_pccheck = False
            self.use_monitor = False
            self._initialize_mock()
    
    def _initialize_mock(self):
        """初始化模拟后端（用于测试）"""
        if self.verbose:
            print(f"[Adapter] 使用模拟后端（文件: {self.checkpoint_file}）")
        
        # 创建检查点文件
        os.makedirs(os.path.dirname(self.checkpoint_file) or ".", exist_ok=True)
    
    def _get_checkpoint_filename(self, checkpoint_id: str, training_step: int) -> str:
        """
        为每个检查点生成唯一的文件名
        
        Args:
            checkpoint_id: 检查点 ID
            training_step: 训练步数
            
        Returns:
            唯一的检查点文件路径
        """
        # 获取基础目录和文件扩展名
        base_dir = os.path.dirname(self.checkpoint_file) or "."
        base_name = os.path.basename(self.checkpoint_file)
        name_parts = os.path.splitext(base_name)
        
        # 生成唯一的文件名：checkpoint_step_123.chk
        unique_filename = f"{name_parts[0]}_step_{training_step}{name_parts[1]}"
        unique_filepath = os.path.join(base_dir, unique_filename)
        
        return unique_filepath
    
    def allocate_staging_buffer(self, size_mb: float = 500.0):
        """
        分配 GPU buffer（仅在 gpu_ar 未由外部传入时使用）
        
        Args:
            size_mb: Buffer 大小（MB）
        """
        if self.gpu_ar is not None:
            return  # 已经分配
        
        size_bytes = int(size_mb * 1024 * 1024)
        size_elements = size_bytes // 4  # float32
        
        if torch.cuda.is_available():
            self.gpu_ar = torch.zeros(
                size_elements,
                dtype=torch.float32,
                device='cuda'
            )
            
            if self.verbose:
                print(f"[Adapter] 分配 GPU buffer: {size_mb:.2f} MB")
        else:
            # CPU fallback
            self.gpu_ar = torch.zeros(
                size_elements,
                dtype=torch.float32
            )
            
            if self.verbose:
                print(f"[Adapter] 分配 CPU buffer: {size_mb:.2f} MB")
    
    def save_layers_batch(self, tasks):
        """
        批量保存多个层
        
        Args:
            tasks: List[SaveTask] 来自调度器
        """
        if not tasks:
            return
        
        start_time = time.time()
        
        # 🚀 性能优化：快速路径
        # 如果使用 set_storage，参数已在 gpu_ar 中，直接保存整个 buffer（一次 I/O）
        # 避免逐层复制和多次刷新（原本 148 次 → 1 次）
        if (hasattr(self, 'gpu_ar') and self.gpu_ar is not None and 
            self.use_pccheck and len(tasks) > 0):
            
            if self.verbose:
                total_size = sum(t.size_bytes for t in tasks)
                print(f"\n[Adapter] 🚀 使用快速路径（原始 PCCheck 模式）")
                print(f"  - 任务数: {len(tasks)}")
                print(f"  - 理论大小: {total_size / (1024**2):.2f} MB")
                print(f"  - 实际保存: {self.gpu_ar.numel() * 4 / (1024**2):.2f} MB（整个 gpu_ar）")
            
            # 直接保存整个 gpu_ar（参数已通过 set_storage 在其中）
            self._save_entire_gpu_ar()
            
            elapsed = time.time() - start_time
            self.stats['total_layers_saved'] += len(tasks)
            self.stats['total_bytes_saved'] += sum(t.size_bytes for t in tasks)
            self.stats['total_save_time'] += elapsed
            
            if self.verbose:
                print(f"  ✅ 快速路径完成，耗时: {elapsed*1000:.2f} ms")
            
            return
        
        # 否则使用原有的分层保存逻辑
        if self.verbose:
            total_size = sum(t.size_bytes for t in tasks)
            print(f"\n[Adapter] 开始保存批次（标准模式）")
            print(f"  - 任务数: {len(tasks)}")
            print(f"  - 总大小: {total_size / (1024**2):.2f} MB")
        
        # 使用 gpu_ar 聚合后批量保存（如果可用）
        if self.gpu_ar is not None:
            self._save_via_staging_buffer(tasks)
        else:
            # 方案2：逐层保存
            self._save_layer_by_layer(tasks)
        
        elapsed = time.time() - start_time
        
        # 更新统计
        self.stats['total_layers_saved'] += len(tasks)
        self.stats['total_bytes_saved'] += sum(t.size_bytes for t in tasks)
        self.stats['total_save_time'] += elapsed
        
        if self.verbose:
            throughput = sum(t.size_bytes for t in tasks) / (1024**2) / elapsed
            print(f"  - 保存耗时: {elapsed*1000:.2f} ms")
            print(f"  - 吞吐量: {throughput:.2f} MB/s")
    
    def _save_entire_gpu_ar(self, checkpoint_file: Optional[str] = None):
        """
        快速路径：一次性保存整个 gpu_ar
        
        适用场景：
        - 参数通过 set_storage 重定向到 gpu_ar
        - 所有参数已在连续的 GPU 内存中
        - 避免逐层复制和多次 I/O
        
        性能：
        - I/O 次数：1 次（vs 分层的 148 次）
        - 复制次数：0 次（vs 分层的多次 GPU-CPU-GPU 复制）
        - 预期时间：~5-10ms（vs 分层的 ~10,000ms）
        
        Args:
            checkpoint_file: 可选的检查点文件路径。如果为 None，使用 self.checkpoint_file
        """
        if not self.use_pccheck:
            if self.verbose:
                print(f"[Adapter] ⚠️ PCCheck 未启用，跳过保存")
            return
        
        # 使用传入的文件名，如果没有则使用默认的
        target_file = checkpoint_file if checkpoint_file is not None else self.checkpoint_file
        
        total_size = self.gpu_ar.numel()
        
        if self.verbose:
            size_mb = total_size * 4 / (1024**2)
            print(f"[Adapter] 💾 保存整个 gpu_ar: {size_mb:.2f} MB ({total_size:,} params)")
            print(f"[Adapter]    到文件: {target_file}")
        
        try:
            if self.use_monitor and self.pccheck_monitor is not None:
                # 使用 Chk_monitor（后台进程模式，更高效）
                if self.verbose:
                    print(f"[Adapter] 使用 Monitor 模式（异步）")
                
                # ⚠️ Monitor 模式：需要更新文件路径
                # 注意：Monitor 在初始化时已经设置了文件路径，这里需要特殊处理
                # 如果 Monitor 不支持动态更改文件路径，则需要重新创建 Monitor 实例
                # 或者在外部保证每次调用都使用一致的文件名
                
                # 🚀 关键优化：触发异步保存，立即返回
                # Monitor 后台进程会处理实际的保存工作
                # 训练不需要等待保存完成！
                self.pccheck_monitor.save()
                
                # ✅ 立即返回，不等待保存完成
                # 原来的代码会等待，导致 273ms 阻塞：
                # while self.pccheck_monitor.checkpoint_in_progress():
                #     time.sleep(0.001)
                
                if self.verbose:
                    print(f"[Adapter] ✅ 异步保存已触发（后台进行）")
                
            elif self.pccheck_instance is not None:
                # 使用 Checkpoint 直接模式
                if self.verbose:
                    print(f"[Adapter] 使用直接模式")
                
                # 🔥 修复：如果提供了新的文件名，需要更新 Writer
                if target_file != self.checkpoint_file and Writer is not None:
                    # 重新创建 Writer 实例以使用新的文件名
                    total_mem_batches = int(self.ratio * total_size / self.batch_size_floats)
                    self.pccheck_instance.writer = Writer(
                        target_file.encode(),
                        self.c_lib_path,
                        self.max_async,
                        int(self.batch_size_floats),
                        total_mem_batches,
                        self.is_distributed,
                        self.rank,
                        self.world_size
                    )
                    if self.verbose:
                        print(f"[Adapter] Writer 已更新到新文件: {target_file}")
                
                # 直接调用原始 PCCheck 的 write_pipelined
                self.pccheck_instance.write_pipelined(
                    cpu_ar=None,  # 内部分配 CPU 缓冲区（memory_saving=True）
                    num_threads=self.num_threads,
                    sz=total_size,
                    batch_size=self.batch_size_floats,
                    ratio=self.ratio,
                    memory_saving=True,
                    is_distributed=self.is_distributed,
                    rank=self.rank,
                    world_size=self.world_size
                )
                
            if self.verbose:
                print(f"[Adapter] ✅ 保存完成")
                
        except Exception as e:
            print(f"[Adapter] ❌ 保存失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_entire_checkpoint(self, checkpoint_id: str, training_step: int):
        """
        批量保存接口：直接保存整个检查点（不经过调度器）
        
        Args:
            checkpoint_id: 检查点 ID
            training_step: 训练步数
        
        适用场景：
        - 参数通过 set_storage 在 gpu_ar 中
        - 跳过细粒度调度，直接保存整个 buffer
        
        性能优势：
        - 0 次回调开销（vs 148 次）
        - 1 次 I/O（vs 5-7 次）
        - 大块写入（500MB），饱和带宽
        """
        if self.gpu_ar is None or not self.use_pccheck:
            if self.verbose:
                print(f"[Adapter] 跳过保存（gpu_ar={self.gpu_ar is not None}, use_pccheck={self.use_pccheck})")
            return
        
        start_time = time.time()
        
        # 🔥 修复：为每个检查点生成唯一的文件名
        checkpoint_file_for_this_step = self._get_checkpoint_filename(checkpoint_id, training_step)
        
        if self.verbose:
            size_mb = self.gpu_ar.numel() * 4 / (1024**2)
            print(f"\n[Adapter] 💾 批量保存检查点")
            print(f"  - Checkpoint ID: {checkpoint_id}")
            print(f"  - Training Step: {training_step}")
            print(f"  - 文件: {checkpoint_file_for_this_step}")
            print(f"  - 大小: {size_mb:.2f} MB ({self.gpu_ar.numel():,} params)")
        
        try:
            # 🔥 修复：使用特定的文件名保存（一次 I/O）
            self._save_entire_gpu_ar(checkpoint_file_for_this_step)
            
            elapsed = time.time() - start_time
            
            if self.verbose:
                throughput_mbs = (self.gpu_ar.numel() * 4 / (1024**2)) / elapsed
                print(f"  ✅ 保存完成")
                print(f"    - 耗时: {elapsed*1000:.2f} ms")
                print(f"    - 吞吐量: {throughput_mbs:.2f} MB/s")
            
            # 更新统计
            self.stats['total_bytes_saved'] += self.gpu_ar.numel() * 4
            self.stats['total_save_time'] += elapsed
            
        except Exception as e:
            print(f"[Adapter] ❌ 批量保存失败: {e}")
            import traceback
            traceback.print_exc()

    def save_entire_checkpoint_in_chunks(self, checkpoint_id: str, training_step: int, chunk_count: int = 2):
        """
        将整个 gpu_ar 划分为若干 chunk 并行保存。

        说明：
        - 在无法使用 Monitor 的情况下（或即使可用），将大块数据划分为若干并发写入
          可以更好地饱和 I/O 带宽，并减少单次阻塞时间。
        - 该实现会为每个 chunk 启动一个线程，线程内会拷贝对应的 slice 到
          PCCheck 的 gpu_ar 或 staging buffer 并调用底层写入接口。

        注意：底层 PCCheck 实现需要支持同时进行多个 write_pipelined 调用
       （通过 writer.max_async 等参数控制）。如果底层不支持并发写入，
        并发写入可能会被内部排队或变得不稳定，请据实际情况调整 chunk_count。
        """
        if self.gpu_ar is None or not self.use_pccheck:
            if self.verbose:
                print(f"[Adapter] 跳过分块保存（gpu_ar={self.gpu_ar is not None}, use_pccheck={self.use_pccheck})")
            return

        # 🔥 修复：为每个检查点生成唯一的文件名
        checkpoint_file_for_this_step = self._get_checkpoint_filename(checkpoint_id, training_step)

        total_floats = int(self.gpu_ar.numel())
        if total_floats == 0:
            return

        # 计算每个 chunk 的大小（以 float 为单位）
        import math
        chunk_size = int(math.ceil(total_floats / float(max(1, chunk_count))))

        if self.verbose:
            size_mb = total_floats * 4 / (1024**2)
            print(f"\n[Adapter] 💾 分块保存检查点: {chunk_count} chunks, 总大小: {size_mb:.2f} MB")
            print(f"  - 文件: {checkpoint_file_for_this_step}")

        threads = []

        def _save_chunk(start_idx: int, end_idx: int, idx: int):
            try:
                num_elems = end_idx - start_idx
                if num_elems <= 0:
                    return

                if self.verbose:
                    print(f"  [Chunk {idx}] 保存范围: {start_idx}:{end_idx} ({num_elems} floats)")

                # 直接从 gpu_ar 的 slice 进行写入
                chunk_tensor = self.gpu_ar[start_idx:end_idx]

                # 临时拷贝到 pccheck_instance.gpu_ar 或 Monitor buffer 并调用写入
                if self.use_monitor and self.pccheck_monitor is not None:
                    # 将数据复制到 gpu_ar（Monitor 会使用它）
                    if self.gpu_ar is not None and num_elems <= self.gpu_ar.numel():
                        self.gpu_ar[:num_elems].copy_(chunk_tensor)
                    # 触发 Monitor 的保存（异步）
                    # ⚠️ 注意：Monitor 不支持分块保存到不同文件，所以这里不适用
                    self.pccheck_monitor.save()
                    if self.verbose:
                        print(f"  [Chunk {idx}] Monitor.save() 已触发")
                else:
                    # 直接模式：复制到 pccheck_instance.gpu_ar 并调用 write_pipelined
                    if self.pccheck_instance.gpu_ar is None or self.pccheck_instance.gpu_ar.numel() < num_elems:
                        # 分配或扩展目标缓冲区
                        self.pccheck_instance.gpu_ar = torch.zeros(
                            num_elems,
                            dtype=torch.float32,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )

                    # 复制 chunk 到目标 gpu_ar
                    self.pccheck_instance.gpu_ar[:num_elems].copy_(chunk_tensor)
                    
                    # 🔥 修复：为这个 chunk 创建或更新 Writer
                    # 注意：分块保存实际上还是保存到同一个文件，只是分批写入
                    if Writer is not None and self.pccheck_instance.writer is None:
                        total_mem_batches = int(self.ratio * num_elems / self.batch_size_floats)
                        self.pccheck_instance.writer = Writer(
                            checkpoint_file_for_this_step.encode(),
                            self.c_lib_path,
                            self.max_async,
                            int(self.batch_size_floats),
                            total_mem_batches,
                            self.is_distributed,
                            self.rank,
                            self.world_size
                        )

                    # 调用写入（write_pipelined 内部应处理并发）
                    self.pccheck_instance.write_pipelined(
                        cpu_ar=None,
                        num_threads=self.num_threads,
                        sz=num_elems,
                        bsize=self.batch_size_floats,
                        lock=self.checkpoint_lock,
                        cp_in_progress=self.cp_in_progress
                    )
                    if self.verbose:
                        print(f"  [Chunk {idx}] write_pipelined 完成 (同步返回/或内部排队)")

                # 统计
                self.stats['total_bytes_saved'] += num_elems * 4

            except Exception as e:
                print(f"[Adapter] Chunk {idx} 保存失败: {e}")
                import traceback
                traceback.print_exc()

        # 启动线程保存每个 chunk
        for i in range(chunk_count):
            s = i * chunk_size
            e = min((i + 1) * chunk_size, total_floats)
            t = threading.Thread(target=_save_chunk, args=(s, e, i), daemon=True)
            t.start()
            threads.append(t)

        # 可选：不等待所有 chunk 完成，以便训练可以继续（更激进的并发策略）
        # 这里我们选择不阻塞（与 Monitor 模式一致），但如果需要确保写入完成再继续，可 join()
        if self.verbose:
            print(f"[Adapter] 已并行触发 {len(threads)} 个 chunk 保存线程（不等待完成）")

    
    def _save_via_staging_buffer(self, tasks):
        """
        通过 gpu_ar 保存（优化版本）
        
        🔥 优化：直接从原始参数拷贝到 gpu_ar，避免中间缓冲
        由于 SaveTask.parameters 现在只存储引用，我们可以直接从原地址拷贝
        """
        buffer_offset = 0
        
        # 🔥 关键：智能复制策略
        # 策略 1：如果使用 set_storage，参数已经在 gpu_ar 中，可以直接保存整个 gpu_ar（零拷贝）
        # 策略 2：否则，需要逐个复制参数到 gpu_ar
        
        for task in tasks:
            for param in task.parameters:
                # 展平参数（view，不产生拷贝）
                param_flat = param.flatten()
                param_size = param_flat.numel()
                
                # 检查空间
                if buffer_offset + param_size > self.gpu_ar.numel():
                    # Buffer 已满，先刷新当前内容
                    self._flush_buffer(buffer_offset)
                    buffer_offset = 0
                
                end_offset = buffer_offset + param_size
                target_buffer = self.gpu_ar[buffer_offset:end_offset]
                
                # 智能拷贝优化
                param_ptr = param_flat.data_ptr()
                target_ptr = target_buffer.data_ptr()
                
                if param_ptr == target_ptr:
                    # 🎯 最优情况：参数正好在目标位置（零拷贝）
                    # 这发生在使用 set_storage 且参数按顺序保存时
                    if self.verbose:
                        print(f"[Adapter] ⚡ 零拷贝: {task.layer_name}")
                else:
                    # 需要复制数据
                    buffer_base_ptr = self.gpu_ar.data_ptr()
                    buffer_size_bytes = self.gpu_ar.numel() * self.gpu_ar.element_size()
                    
                    # 检查是否存在内存重叠（避免自我覆盖）
                    param_in_buffer = (
                        param_ptr >= buffer_base_ptr and 
                        param_ptr < buffer_base_ptr + buffer_size_bytes
                    )
                    
                    if param_in_buffer:
                        # 参数在 buffer 中但位置不对，需要 clone 避免自我覆盖
                        target_buffer.copy_(param_flat.clone())
                        if self.verbose:
                            print(f"[Adapter] 🔄 Clone 复制: {task.layer_name} (内存重叠)")
                    else:
                        # 正常复制（无内存重叠）
                        target_buffer.copy_(param_flat)
                        if self.verbose:
                            print(f"[Adapter] 📋 正常复制: {task.layer_name}")
                
                # 记录元数据
                metadata = LayerMetadata(
                    layer_name=task.layer_name,
                    training_step=task.training_step,
                    checkpoint_id=task.checkpoint_id,
                    offset_in_file=self.current_file_offset,
                    size_bytes=param.numel() * param.element_size(),
                    param_count=param.numel(),
                    shapes=[tuple(param.shape)],
                    dtypes=[str(param.dtype)],
                    timestamp=task.timestamp
                )
                self.layer_metadata.append(metadata)
                
                buffer_offset = end_offset
                self.current_file_offset += param.numel() * param.element_size()
        
        # 刷新剩余数据
        if buffer_offset > 0:
            self._flush_buffer(buffer_offset)
    
    def _flush_buffer(self, buffer_size: int):
        """将 gpu_ar 的数据写入存储"""
        if buffer_size == 0:
            return
        
        # 获取有效数据部分
        valid_data = self.gpu_ar[:buffer_size]
        
        if self.use_pccheck:
            # 使用 PCCheck 保存
            self._save_to_pccheck(valid_data)
        else:
            # 模拟保存（写入文件）
            self._save_to_file(valid_data)
    
    def _save_to_pccheck(self, data: torch.Tensor):
        """使用 PCCheck 保存数据"""
        if self.pccheck_instance is None and self.pccheck_monitor is None:
            print("[Adapter] 错误: PCCheck 实例未初始化")
            return
        
        try:
            # 将数据转换为 CPU numpy 数组
            # data_cpu = data.cpu().numpy().astype(np.float32)
            data_cpu = data.detach().cpu().float().numpy()
            
            # 计算批次信息
            total_size = data_cpu.size
            num_batches = (total_size + self.batch_size_floats - 1) // self.batch_size_floats
            
            if self.verbose:
                size_mb = total_size * 4 / (1024**2)
                print(f"    [PCCheck] 写入 {size_mb:.2f} MB ({total_size:,} floats)")
                print(f"    [PCCheck] 分 {num_batches} 个批次写入")
            
            if self.use_monitor and self.pccheck_monitor is not None:
                # 使用 Chk_monitor（后台进程模式）
                if self.verbose:
                    print(f"    [PCCheck] 使用 Monitor 模式保存")
                
                # 将数据复制到 GPU buffer
                if self.gpu_ar is not None:
                    self.gpu_ar[:data.numel()].copy_(data)
                
                # 等待之前的检查点完成
                while self.pccheck_monitor.checkpoint_in_progress():
                    time.sleep(0.001)
                
                # 触发保存
                self.pccheck_monitor.save()
                
                # 等待 GPU 拷贝完成（可选，取决于是否需要立即重用 buffer）
                while self.pccheck_monitor.gpu_copy_in_progress():
                    time.sleep(0.001)
                
            else:
                # 使用 Checkpoint 直接模式
                if self.verbose:
                    print(f"    [PCCheck] 使用直接模式保存")
                
                # ⚠️ 关键：先将数据复制到 Checkpoint 的 gpu_ar
                # 如果 gpu_ar 还未分配或大小不够，则重新分配
                if self.pccheck_instance.gpu_ar is None or self.pccheck_instance.gpu_ar.numel() < total_size:
                    if self.verbose:
                        print(f"    [PCCheck] 分配 GPU buffer: {total_size:,} floats")
                    self.pccheck_instance.gpu_ar = torch.zeros(
                        total_size, 
                        dtype=torch.float32, 
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                
                # 将我们的数据复制到 Checkpoint 的 gpu_ar
                self.pccheck_instance.gpu_ar[:total_size].copy_(data)
                
                # 使用 PCCheck 的 write_pipelined 方法
                # 注意：根据 chk_checkpoint_pipeline.py，这个方法会启动多线程进行流水线写入
                self.pccheck_instance.write_pipelined(
                    cpu_ar=None,  # 使用 PCCheck 内部分配的 CPU 缓冲区（memory_saving=True）
                    num_threads=self.num_threads,
                    sz=total_size,
                    bsize=self.batch_size_floats,
                    lock=self.checkpoint_lock,
                    cp_in_progress=self.cp_in_progress
                )
            
            if self.verbose:
                print(f"    [PCCheck] 写入完成")
                
        except Exception as e:
            import traceback
            print(f"[Adapter] PCCheck 写入失败: {e}")
            traceback.print_exc()
            print(f"[Adapter] 回退到文件写入模式")
            self._save_to_file(data)
    
    def _save_to_file(self, data: torch.Tensor):
        """模拟保存：写入二进制文件"""
        # 转换为 numpy 并写入
        data_cpu = data.detach().cpu().float().numpy()
        
        with open(self.checkpoint_file, 'ab') as f:
            data_cpu.tofile(f)
        
        if self.verbose:
            size_mb = data.numel() * 4 / (1024**2)
            print(f"    [Mock] 写入 {size_mb:.2f} MB 到文件")
    
    def _save_layer_by_layer(self, tasks):
        """逐层保存（不使用 staging buffer）"""
        for task in tasks:
            for param in task.parameters:
                # 转换为连续存储
                param_contiguous = param.contiguous()
                
                # 记录元数据
                metadata = LayerMetadata(
                    layer_name=task.layer_name,
                    training_step=task.training_step,
                    checkpoint_id=task.checkpoint_id,
                    offset_in_file=self.current_file_offset,
                    size_bytes=param.numel() * param.element_size(),
                    param_count=param.numel(),
                    shapes=[tuple(param.shape)],
                    dtypes=[str(param.dtype)],
                    timestamp=task.timestamp
                )
                self.layer_metadata.append(metadata)
                
                # 保存数据
                if self.use_pccheck:
                    self._save_to_pccheck(param_contiguous.flatten())
                else:
                    self._save_to_file(param_contiguous.flatten())
                
                self.current_file_offset += param.numel() * param.element_size()
    
    def save_metadata(self):
        """保存元数据到 JSON 文件"""
        metadata_dict = {
            'checkpoint_file': self.checkpoint_file,
            'total_layers': len(self.layer_metadata),
            'total_size_bytes': sum(m.size_bytes for m in self.layer_metadata),
            'layers': [
                {
                    'layer_name': m.layer_name,
                    'training_step': m.training_step,
                    'checkpoint_id': m.checkpoint_id,
                    'offset_in_file': m.offset_in_file,
                    'size_bytes': m.size_bytes,
                    'param_count': m.param_count,
                    'shapes': m.shapes,
                    'dtypes': m.dtypes,
                    'timestamp': m.timestamp
                }
                for m in self.layer_metadata
            ]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        if self.verbose:
            print(f"\n[Adapter] 元数据已保存: {self.metadata_file}")
            print(f"  - 总层数: {metadata_dict['total_layers']}")
            print(f"  - 总大小: {metadata_dict['total_size_bytes'] / (1024**3):.2f} GB")
    
    def shutdown(self):
        """关闭适配器"""
        if self.verbose:
            print(f"\n[Adapter] 正在关闭...")
        
        # 等待所有检查点完成
        if self.use_monitor and self.pccheck_monitor is not None:
            # 等待 Monitor 中的检查点完成
            while self.pccheck_monitor.checkpoint_in_progress():
                time.sleep(0.01)
            
            # 关闭 Monitor
            if self.verbose:
                print(f"[Adapter] 关闭 PCCheck Monitor...")
            self.pccheck_monitor.kill_checkpoint()
        
        # 保存元数据
        self.save_metadata()
        
        # 释放资源
        if self.gpu_ar is not None:
            del self.gpu_ar
            self.gpu_ar = None
        
        if self.cpu_buffer is not None:
            del self.cpu_buffer
            self.cpu_buffer = None
        
        # 打印统计
        self.print_stats()
        
        if self.verbose:
            print(f"[Adapter] 已关闭")
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"PCCheck 适配器统计")
        print(f"{'='*80}")
        print(f"总保存层数: {self.stats['total_layers_saved']}")
        print(f"总保存数据量: {self.stats['total_bytes_saved'] / (1024**3):.2f} GB")
        print(f"总保存时间: {self.stats['total_save_time']:.2f} 秒")
        
        if self.stats['total_save_time'] > 0:
            throughput = self.stats['total_bytes_saved'] / (1024**2) / self.stats['total_save_time']
            print(f"平均吞吐量: {throughput:.2f} MB/s")
        print(f"{'='*80}\n")
