"""
阶段三：分层检查点调度器
Layerwise Checkpoint Scheduler

管理检查点保存任务队列，实现智能调度和批量聚合策略
"""

import torch
import threading
import queue
import time
import json
from typing import Dict, List, Optional, Callable, Any
from collections import OrderedDict
from dataclasses import dataclass, asdict
import os


@dataclass
class SaveTask:
    """
    保存任务数据类（优化版本）
    
    🔥 优化：不再存储深拷贝的参数张量，而是存储引用
    这样可以避免在 Scheduler 阶段的内存拷贝，直接由 PCCheckAdapter 从原地址拷贝到 staging buffer
    """
    layer_name: str
    training_step: int
    parameters: List[torch.Tensor]  # 参数张量的引用（不再深拷贝，由 Adapter 负责拷贝）
    param_count: int
    size_bytes: int
    timestamp: float
    checkpoint_id: str  # 用于标识属于哪个检查点
    
    def to_dict(self) -> Dict:
        """转换为字典（用于元数据）"""
        return {
            'layer_name': self.layer_name,
            'training_step': self.training_step,
            'param_count': self.param_count,
            'size_bytes': self.size_bytes,
            'timestamp': self.timestamp,
            'checkpoint_id': self.checkpoint_id
        }


class LayerwiseCheckpointScheduler:
    """
    分层检查点调度器
    
    核心功能：
    1. 接收来自 LayerwiseOptimizer 的保存任务
    2. 实现智能缓冲和批量聚合
    3. 管理任务队列
    4. 调度保存操作到后端（PCCheck）
    """
    
    def __init__(
        self,
        save_callback: Callable[[List[SaveTask]], None],
        buffer_size_mb: float = 100.0,
        buffer_timeout_ms: float = 100.0,
        max_queue_size: int = 1000,
        enable_async: bool = True,
        metadata_dir: str = "./checkpoint_metadata",
        verbose: bool = False
    ):
        """
        Args:
            save_callback: 实际执行保存的回调函数，接收 List[SaveTask]
            buffer_size_mb: 缓冲区大小阈值（MB）
            buffer_timeout_ms: 缓冲区超时时间（ms）
            max_queue_size: 任务队列最大长度
            enable_async: 是否启用异步保存
            metadata_dir: 元数据保存目录
            verbose: 是否打印详细信息
        """
        self.save_callback = save_callback
        self.buffer_size_mb = buffer_size_mb
        self.buffer_timeout_ms = buffer_timeout_ms
        self.max_queue_size = max_queue_size
        self.enable_async = enable_async
        self.metadata_dir = metadata_dir
        self.verbose = verbose
        
        # 任务队列
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        
        # 缓冲区
        self.buffer = []
        self.buffer_size_bytes = 0
        self.last_flush_time = time.time()
        
        # 统计信息
        self.stats = {
            'total_tasks_received': 0,
            'total_tasks_saved': 0,
            'total_bytes_saved': 0,
            'total_flushes': 0,
            'flush_by_size': 0,
            'flush_by_timeout': 0,
        }
        
        # 当前检查点追踪
        self.current_checkpoint_layers = {}  # {checkpoint_id: set(layer_names)}
        self.checkpoint_metadata = {}  # {checkpoint_id: metadata}
        
        # 创建元数据目录
        os.makedirs(metadata_dir, exist_ok=True)
        
        # 启动后台保存线程
        self.running = False
        self.save_thread = None
        if self.enable_async:
            self._start_background_saver()
        
        if self.verbose:
            print(f"[Scheduler] 初始化完成")
            print(f"  - 缓冲区大小: {buffer_size_mb} MB")
            print(f"  - 缓冲超时: {buffer_timeout_ms} ms")
            print(f"  - 异步模式: {'启用' if enable_async else '禁用'}")
    
    def _start_background_saver(self):
        """启动后台保存线程"""
        self.running = True
        self.save_thread = threading.Thread(
            target=self._background_saver_loop,
            daemon=True
        )
        self.save_thread.start()
        if self.verbose:
            print(f"[Scheduler] 后台保存线程已启动")
    
    def _background_saver_loop(self):
        """后台保存线程主循环"""
        while self.running:
            try:
                # 从队列中获取任务（带超时）
                task = self.task_queue.get(timeout=0.01)
                
                try:
                    # 添加到缓冲区
                    self._add_to_buffer(task)
                    
                    # 检查是否需要刷新
                    self._check_and_flush()
                finally:
                    # 标记任务完成（关键！否则 join() 会永远等待）
                    self.task_queue.task_done()
                
            except queue.Empty:
                # 队列为空，检查超时刷新
                self._check_timeout_flush()
            except Exception as e:
                print(f"[Scheduler] 后台保存线程错误: {e}")
                # 即使出错也要标记任务完成
                try:
                    self.task_queue.task_done()
                except:
                    pass
    
    def schedule_save(self, layer_name: str, training_step: int, layer_params: Dict):
        """
        调度一个层的保存任务
        
        这是从 LayerwiseOptimizer 的回调函数调用的入口点
        
        Args:
            layer_name: 层名称
            training_step: 训练步数
            layer_params: 层参数字典（来自 LayerwiseOptimizer）
        """
        # 创建保存任务
        task = SaveTask(
            layer_name=layer_name,
            training_step=training_step,
            parameters=layer_params['parameters'],
            param_count=layer_params['param_count'],
            size_bytes=sum(p.numel() * p.element_size() for p in layer_params['parameters']),
            timestamp=time.time(),
            checkpoint_id=f"step_{training_step}"
        )
        
        self.stats['total_tasks_received'] += 1
        
        # 追踪检查点的层
        if task.checkpoint_id not in self.current_checkpoint_layers:
            self.current_checkpoint_layers[task.checkpoint_id] = set()
        self.current_checkpoint_layers[task.checkpoint_id].add(layer_name)
        
        if self.enable_async:
            # 异步模式：放入队列
            try:
                self.task_queue.put(task, timeout=1.0)
            except queue.Full:
                print(f"[Scheduler] 警告：任务队列已满，等待...")
                self.task_queue.put(task)  # 阻塞等待
        else:
            # 同步模式：直接处理
            self._add_to_buffer(task)
            self._check_and_flush()
    
    def _add_to_buffer(self, task: SaveTask):
        """添加任务到缓冲区"""
        self.buffer.append(task)
        self.buffer_size_bytes += task.size_bytes
        
        if self.verbose:
            size_mb = task.size_bytes / (1024 * 1024)
            print(f"[Scheduler] 缓冲任务: {task.layer_name:40s} | "
                  f"步骤 {task.training_step} | {size_mb:.2f} MB")
    
    def _check_and_flush(self):
        """检查是否需要刷新缓冲区"""
        threshold_bytes = self.buffer_size_mb * 1024 * 1024
        
        if self.buffer_size_bytes >= threshold_bytes:
            self._flush_buffer(reason="size")
    
    def _check_timeout_flush(self):
        """检查超时刷新"""
        if not self.buffer:
            return
        
        elapsed_ms = (time.time() - self.last_flush_time) * 1000
        if elapsed_ms >= self.buffer_timeout_ms:
            self._flush_buffer(reason="timeout")
    
    def _flush_buffer(self, reason: str = "manual"):
        """刷新缓冲区，执行实际保存"""
        if not self.buffer:
            return
        
        num_tasks = len(self.buffer)
        total_size_mb = self.buffer_size_bytes / (1024 * 1024)
        
        if self.verbose:
            print(f"\n[Scheduler] 刷新缓冲区 (原因: {reason})")
            print(f"  - 任务数: {num_tasks}")
            print(f"  - 总大小: {total_size_mb:.2f} MB")
        
        # 调用保存回调
        start_time = time.time()
        try:
            self.save_callback(self.buffer)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if self.verbose:
                print(f"  - 保存耗时: {elapsed_ms:.2f} ms")
                print(f"  - 吞吐量: {total_size_mb / (elapsed_ms / 1000):.2f} MB/s")
        
        except Exception as e:
            print(f"[Scheduler] 保存失败: {e}")
            raise
        
        # 更新统计
        self.stats['total_tasks_saved'] += num_tasks
        self.stats['total_bytes_saved'] += self.buffer_size_bytes
        self.stats['total_flushes'] += 1
        if reason == "size":
            self.stats['flush_by_size'] += 1
        elif reason == "timeout":
            self.stats['flush_by_timeout'] += 1
        
        # 清空缓冲区
        self.buffer.clear()
        self.buffer_size_bytes = 0
        self.last_flush_time = time.time()
    
    def force_flush(self):
        """强制刷新缓冲区"""
        if self.enable_async:
            # 等待队列清空
            self.task_queue.join()
        
        self._flush_buffer(reason="force")
    
    def finalize_checkpoint(self, training_step: int, total_layers: int):
        """
        完成一个检查点的保存
        
        Args:
            training_step: 训练步数
            total_layers: 该检查点应该包含的总层数
        """
        checkpoint_id = f"step_{training_step}"
        
        if checkpoint_id in self.current_checkpoint_layers:
            saved_layers = len(self.current_checkpoint_layers[checkpoint_id])
            
            if self.verbose:
                print(f"\n[Scheduler] 检查点完成: {checkpoint_id}")
                print(f"  - 已保存层数: {saved_layers}/{total_layers}")
            
            if saved_layers != total_layers:
                print(f"[Scheduler] 警告：检查点不完整！")
    
    def shutdown(self):
        """关闭调度器"""
        if self.verbose:
            print(f"\n[Scheduler] 正在关闭...")
        
        # 停止后台线程
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        
        # 刷新剩余任务
        self.force_flush()
        
        # 打印统计
        self.print_stats()
        
        if self.verbose:
            print(f"[Scheduler] 已关闭")
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"调度器统计信息")
        print(f"{'='*80}")
        print(f"总接收任务数: {self.stats['total_tasks_received']}")
        print(f"总保存任务数: {self.stats['total_tasks_saved']}")
        print(f"总保存数据量: {self.stats['total_bytes_saved'] / (1024**3):.2f} GB")
        print(f"总刷新次数: {self.stats['total_flushes']}")
        print(f"  - 按大小触发: {self.stats['flush_by_size']}")
        print(f"  - 按超时触发: {self.stats['flush_by_timeout']}")
        
        if self.stats['total_flushes'] > 0:
            avg_tasks_per_flush = self.stats['total_tasks_saved'] / self.stats['total_flushes']
            avg_bytes_per_flush = self.stats['total_bytes_saved'] / self.stats['total_flushes']
            print(f"平均每次刷新:")
            print(f"  - 任务数: {avg_tasks_per_flush:.1f}")
            print(f"  - 数据量: {avg_bytes_per_flush / (1024**2):.2f} MB")
        print(f"{'='*80}\n")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


class PriorityScheduler(LayerwiseCheckpointScheduler):
    """
    优先级调度器
    
    根据层的大小和重要性进行优先级排序
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_priorities = {}  # {layer_name: priority}
    
    def set_layer_priority(self, layer_name: str, priority: int):
        """设置层的优先级（数值越大优先级越高）"""
        self.layer_priorities[layer_name] = priority
    
    def _add_to_buffer(self, task: SaveTask):
        """添加到缓冲区时考虑优先级"""
        # 获取优先级（默认为0）
        priority = self.layer_priorities.get(task.layer_name, 0)
        
        # 将优先级附加到任务上
        task.priority = priority
        
        super()._add_to_buffer(task)
    
    def _flush_buffer(self, reason: str = "manual"):
        """刷新时按优先级排序"""
        if self.buffer:
            # 按优先级排序（高优先级先保存）
            self.buffer.sort(key=lambda t: getattr(t, 'priority', 0), reverse=True)
        
        super()._flush_buffer(reason)
