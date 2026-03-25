"""
多流并行检查点保存

实现方案一：多流并行写入架构
4个独立数据流分别管理 param、grad、exp_avg、exp_avg_sq
"""

import torch
import numpy as np
from ctypes import *
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, List, Dict, Optional, Tuple
from collections import deque
import threading
from itertools import cycle
import os
import json

class MultiStreamOptimizer:
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        layer_groups: List[List[int]],
        param_list: List[torch.nn.Parameter],
        callback: Optional[Callable[[List[int]], None]] = None
    ):
        """
        Args:
            optimizer: 标准PyTorch优化器（如Adam, SGD）
            layer_groups: 层分组，如 [[0,1,2], [3,4,5], ...]
            param_list: 模型参数列表（与layer_id对应）
            callback: 回调函数，签名为 callback(layer_ids)，在每层组更新后调用
        """
        self.optimizer = optimizer
        self.layer_groups = layer_groups
        self.param_list = param_list
        self.callback = callback
        
        # 构建层组到参数的映射
        self.group_params = []
        for layer_ids in layer_groups:
            group_params = []
            for layer_id in layer_ids:
                if layer_id < len(param_list):
                    group_params.append(param_list[layer_id])
            self.group_params.append(group_params)
        
        # 检测优化器类型
        self.optimizer_type = type(optimizer).__name__
        
        # ✅ 跨检查点同步：更新前的等待回调
        self.pre_update_callback = None  # 签名: pre_update_callback(group_idx)
        
    def step_with_callback(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 按层组遍历并更新
        for group_idx, (layer_ids, params) in enumerate(zip(self.layer_groups, self.group_params)):
            # ✅ 更新前回调：等待上一个检查点的该层拷贝完成
            if self.pre_update_callback is not None:
                self.pre_update_callback(group_idx)
            
            # 对这组参数执行真实的优化器更新
            self._step_param_group(params)
            
            # 触发回调（在更新完成后），传递 group_idx
            if self.callback is not None:
                self.callback(group_idx, layer_ids)
        
        return loss
    
    def _step_param_group(self, params: List[torch.nn.Parameter]):
        """
        对指定的参数组执行优化器更新（使用真实的优化器逻辑）
        
        Args:
            params: 要更新的参数列表
        """
        # 在no_grad上下文中执行更新（避免就地操作错误）
        with torch.no_grad():
            # 根据优化器类型调用对应的更新逻辑
            if self.optimizer_type == 'Adam':
                self._step_adam(params)
            elif self.optimizer_type == 'AdamW':
                self._step_adamw(params)
            elif self.optimizer_type == 'SGD':
                self._step_sgd(params)
            else:
                # 对于其他优化器，使用通用方法（逐参数调用优化器的内部逻辑）
                self._step_generic(params)
    
    def _step_adam(self, params: List[torch.nn.Parameter]):
        """执行Adam更新（使用PyTorch的functional API）"""
        from torch.optim._functional import adam
        
        # 获取优化器的超参数（从第一个param_group）
        group = self.optimizer.param_groups[0]
        
        # 收集批量参数
        params_list = []
        grads_list = []
        exp_avgs_list = []
        exp_avg_sqs_list = []
        max_exp_avg_sqs_list = []
        state_steps_list = []
        
        for param in params:
            if param.grad is None:
                continue
            
            # 获取或初始化state
            state = self.optimizer.state[param]
            
            # 初始化state（如果需要）
            if len(state) == 0:
                state['step'] = torch.tensor(0.)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
            # 增加步数
            state['step'] += 1
            
            params_list.append(param)
            grads_list.append(param.grad)
            exp_avgs_list.append(state['exp_avg'])
            exp_avg_sqs_list.append(state['exp_avg_sq'])
            state_steps_list.append(state['step'])
            
        if not params_list:
            return

        # 调用PyTorch的functional API进行批量更新 (foreach=True)
        adam(
            params_list,
            grads_list,
            exp_avgs_list,
            exp_avg_sqs_list,
            max_exp_avg_sqs_list,
            state_steps_list,
            amsgrad=group.get('amsgrad', False),
            beta1=group['betas'][0],
            beta2=group['betas'][1],
            lr=group['lr'],
            weight_decay=group.get('weight_decay', 0),
            eps=group['eps'],
            maximize=group.get('maximize', False),
            foreach=True,  # 启用foreach优化
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None
        )
    
    def _step_adamw(self, params: List[torch.nn.Parameter]):
        """执行AdamW更新"""
        from torch.optim._functional import adamw
        
        group = self.optimizer.param_groups[0]
        
        # 收集批量参数
        params_list = []
        grads_list = []
        exp_avgs_list = []
        exp_avg_sqs_list = []
        max_exp_avg_sqs_list = []
        state_steps_list = []
        
        for param in params:
            if param.grad is None:
                continue
            
            state = self.optimizer.state[param]
            
            if len(state) == 0:
                state['step'] = torch.tensor(0.)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
            state['step'] += 1
            
            params_list.append(param)
            grads_list.append(param.grad)
            exp_avgs_list.append(state['exp_avg'])
            exp_avg_sqs_list.append(state['exp_avg_sq'])
            state_steps_list.append(state['step'])
            
        if not params_list:
            return

        adamw(
            params_list,
            grads_list,
            exp_avgs_list,
            exp_avg_sqs_list,
            max_exp_avg_sqs_list,
            state_steps_list,
            amsgrad=group.get('amsgrad', False),
            beta1=group['betas'][0],
            beta2=group['betas'][1],
            lr=group['lr'],
            weight_decay=group.get('weight_decay', 0),
            eps=group['eps'],
            maximize=group.get('maximize', False),
            foreach=True,  # 启用foreach优化
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None
        )
    
    def _step_sgd(self, params: List[torch.nn.Parameter]):
        """执行SGD更新"""
        from torch.optim._functional import sgd
        
        group = self.optimizer.param_groups[0]
        
        for param in params:
            if param.grad is None:
                continue
            
            state = self.optimizer.state[param]
            
            # SGD的momentum buffer
            momentum_buffer_list = []
            if 'momentum_buffer' in state:
                momentum_buffer_list.append(state['momentum_buffer'])
            else:
                momentum_buffer_list.append(None)
            
            sgd(
                [param],
                [param.grad],
                momentum_buffer_list,
                weight_decay=group.get('weight_decay', 0),
                momentum=group.get('momentum', 0),
                lr=group['lr'],
                dampening=group.get('dampening', 0),
                nesterov=group.get('nesterov', False),
                maximize=group.get('maximize', False),
                foreach=False,
                fused=False,
                grad_scale=None,
                found_inf=None
            )
            
            # 更新momentum buffer
            if momentum_buffer_list[0] is not None:
                state['momentum_buffer'] = momentum_buffer_list[0]
    
    def _step_generic(self, params: List[torch.nn.Parameter]):
        """
        通用的参数更新方法（用于不支持的优化器类型）
        通过临时修改param_groups来实现
        """
        # 保存原始的param_groups
        original_param_groups = self.optimizer.param_groups
        
        # 创建临时param_group（只包含当前要更新的参数）
        temp_param_group = {
            'params': params,
            **{k: v for k, v in original_param_groups[0].items() if k != 'params'}
        }
        
        # 临时替换param_groups
        self.optimizer.param_groups = [temp_param_group]
        
        # 执行标准的step
        self.optimizer.step()
        
        # 恢复原始的param_groups
        self.optimizer.param_groups = original_param_groups
    
    def step(self, closure=None):
        """
        标准的step方法（一次性更新所有参数，不触发回调）
        用于非检查点步骤
        """
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """清空梯度"""
        self.optimizer.zero_grad(set_to_none)
    
    @property
    def state(self):
        """访问优化器state"""
        return self.optimizer.state
    
    @property
    def param_groups(self):
        """访问param_groups"""
        return self.optimizer.param_groups
    
    def state_dict(self):
        """返回状态字典"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name):
        """代理其他方法到原始优化器"""
        return getattr(self.optimizer, name)


class MultiStreamWriter:
    """多流Writer封装"""
    
    def __init__(self, fname, lib_path, max_async, num_streams, stream_sizes, chunk_size=None, num_chunks=None, 
                 is_distributed=False, rank=0, world_size=1):
        """
        Args:
            fname: 文件名
            lib_path: C库路径
            max_async: 最大异步检查点数
            num_streams: 流的数量（通常为4）
            stream_sizes: 每个流的大小列表
            chunk_size: DRAMAlloc分配的块大小（如果为None，则使用总大小）
            num_chunks: DRAMAlloc分配的块数量（如果为None，则使用max_async + 1）
            is_distributed: 是否分布式
            rank: 当前进程rank
            world_size: 总进程数
        """
        self.lib = cdll.LoadLibrary(lib_path)
        self.num_streams = num_streams
        self.max_async = max_async

        # CPU 亲和设置：优先避开 CPU0（主训练线程常驻），其余核轮询分配
        self._affinity_cpus = self._init_affinity_cpus()
        self._affinity_cycle = cycle(self._affinity_cpus)
        self._thread_local_affinity = threading.local()
        
        # 计算总大小
        total_size = sum(stream_sizes)
        self.total_size = total_size
        
        # 确定chunk_size和num_chunks
        if chunk_size is None:
            chunk_size = total_size
        if num_chunks is None:
            num_chunks = max_async + 1
            
        self.chunk_size = chunk_size
        
        # ✅ 设置 lib.writer 的参数类型，确保正确传递 size_t 和 bool 参数
        self.lib.writer.restype = c_void_p
        self.lib.writer.argtypes = [c_char_p, c_int, c_size_t, c_size_t, c_bool, c_int, c_int]
        
        # ✅ 设置 lib.init_streams 的参数类型
        self.lib.init_streams.restype = c_int
        self.lib.init_streams.argtypes = [c_void_p, c_int, POINTER(c_size_t)]
        
        # ✅ 内存优化：使用指定的chunk_size和num_chunks初始化DRAMAlloc
        self.writer_obj = self.lib.writer(
            fname.encode('utf-8') if isinstance(fname, str) else fname, 
            max_async, 
            c_size_t(chunk_size), 
            c_size_t(num_chunks),
            is_distributed, 
            rank, 
            world_size
        )
        
        # 初始化多流
        # stream_sizes 是 float count，C++库会自动计算字节偏移(x4)
        stream_sizes_arr = (c_size_t * num_streams)(*stream_sizes)
        ret = self.lib.init_streams(self.writer_obj, num_streams, stream_sizes_arr)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize {num_streams} streams")
        
        # ✅ 注册C函数的参数和返回类型
        self.lib.registerCheck.restype = c_int
        self.lib.registerCheck.argtypes = [c_void_p]
        
        self.lib.write_stream_chunk.restype = None
        self.lib.write_stream_chunk.argtypes = [c_void_p, c_int, c_void_p, c_size_t, c_size_t, c_int, c_int]
        
        self.lib.sync_stream.restype = None
        self.lib.sync_stream.argtypes = [c_void_p, c_int, c_int]
        
        self.lib.sync_all_streams.restype = None
        self.lib.sync_all_streams.argtypes = [c_void_p, c_int]
        
        # ✅ 注册borrow_cpu_slot和return_cpu_slot的返回类型
        self.lib.borrow_cpu_slot.restype = c_void_p
        self.lib.borrow_cpu_slot.argtypes = [c_void_p, c_int]
        self.lib.return_cpu_slot.restype = None
        self.lib.return_cpu_slot.argtypes = [c_void_p, c_int]
        
        # ✅ 注册borrow_chunk和return_chunk
        if hasattr(self.lib, 'borrow_chunk'):
            self.lib.borrow_chunk.restype = c_void_p
            self.lib.borrow_chunk.argtypes = [c_void_p]
            self.lib.return_chunk.restype = None
            self.lib.return_chunk.argtypes = [c_void_p, c_void_p]
        
        print(f"✓ MultiStreamWriter initialized with {num_streams} streams")
        print(f"  DRAMAlloc: {num_chunks} chunks of size {chunk_size} floats ({chunk_size*4/1e6:.2f} MB)")

    def _init_affinity_cpus(self):
        """选择亲和CPU列表，优先避开0号核。"""
        try:
            avail = sorted(os.sched_getaffinity(0))
        except Exception:
            return [0]
        if len(avail) > 1 and 0 in avail:
            avail.remove(0)
        return avail or [0]

    def _ensure_thread_affinity(self):
        tl = self._thread_local_affinity
        if getattr(tl, "pinned", False):
            return
        try:
            cpu = next(self._affinity_cycle)
            os.sched_setaffinity(0, {cpu})
            tl.cpu = cpu
        except Exception:
            pass
        tl.pinned = True

    def _submit_with_affinity(self, executor, fn, *args, **kwargs):
        def wrapped(*a, **k):
            self._ensure_thread_affinity()
            return fn(*a, **k)
        return executor.submit(wrapped, *args, **kwargs)
    
    def write_stream_chunk(self, stream_id, data, offset_in_stream, chunk_size, 
                           parall_iter, num_threads):
        """
        写入单个流的数据块
        
        Args:
            stream_id: 流ID (0-3)
            data: numpy数组 或 ctypes指针
            offset_in_stream: 在流内的偏移 (float count)
            chunk_size: 数据块大小 (float count)
            parall_iter: 并行迭代槽位（由registerCheck返回）
            num_threads: 写入线程数
        """
        # ✅ 优化：支持直接传递 ctypes 指针，避免 numpy 转换开销
        if isinstance(data, (POINTER(c_float), c_void_p, int)):
            ct_arr = c_void_p(data) if isinstance(data, int) else data
        else:
            ct_arr = cast(np.ctypeslib.as_ctypes(data), c_void_p)
            
        # 调用 C++ 函数（argtypes 已设置，自动类型转换）
        self.lib.write_stream_chunk(
            self.writer_obj,
            stream_id,
            ct_arr,
            offset_in_stream,
            chunk_size,
            parall_iter,
            num_threads
        )
    
    def sync_stream(self, stream_id, parall_iter):
        """同步单个流到磁盘"""
        self.lib.sync_stream(self.writer_obj, stream_id, parall_iter)
    
    def sync_all_streams(self, parall_iter):
        """同步所有流到磁盘并更新检查点元数据（沿用原pccheck的槽位管理）"""
        self.lib.sync_all_streams(self.writer_obj, parall_iter)
    
    def register(self):
        """注册新的检查点槽位（沿用原pccheck的槽位管理）"""
        return self.lib.registerCheck(self.writer_obj)
    
    def borrow_cpu_slot(self, parall_iter):
        return self.lib.borrow_cpu_slot(self.writer_obj, c_int(parall_iter))
    
    def return_cpu_slot(self, parall_iter):
        self.lib.return_cpu_slot(self.writer_obj, c_int(parall_iter))
    
    def borrow_chunk(self):
        """从DRAMAlloc借用一块pinned buffer"""
        return self.lib.borrow_chunk(self.writer_obj)
    
    def return_chunk(self, chunk_ptr):
        """归还pinned buffer到DRAMAlloc"""
        self.lib.return_chunk(self.writer_obj, c_void_p(chunk_ptr))


class MultiStreamCheckpoint:
    def __init__(
        self,
        param_layout,           # 参数布局信息
        gpu_ar,                 # GPU内存数组
        total_size,             # 总参数数量
        num_threads=16,         # 写入线程数
        lib_path="./libtest_ssd.so",
        filename="pccheck_checkpoint.chk",
        num_streams=4,          # 流的数量（param, grad, exp_avg, exp_avg_sq）
        max_async=2,            # 最大异步检查点数
        num_layer_groups=8,     # 分成多少个层组
        distributed=False,
        rank=0,
        world_size=1,
        metadata_path: Optional[str] = None,
    ):
        self.param_layout = param_layout
        self.total_size = total_size
        self.num_threads = num_threads
        self.lib_path = lib_path
        self.filename = filename
        self.num_streams = num_streams
        self.max_async = max_async
        self.num_layer_groups = num_layer_groups
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.metadata_path = metadata_path
        
        # 计算每个流的大小
        self.stream_sizes = self._calculate_stream_sizes()

        # gpu_ar 可以是单个 tensor（legacy）或 dict（四块独立分配）。
        # 统一转为 self.gpu_buffers: List[Tensor]，长度 = num_streams。
        self.gpu_buffers = self._make_gpu_buffers(gpu_ar)
        self.gpu_ar = gpu_ar if not isinstance(gpu_ar, dict) else None
        print(f"\nMultiStreamCheckpoint Configuration:")
        print(f"  Total size: {total_size:,} floats ({total_size*4/1e9:.2f} GB)")
        print(f"  Num streams: {num_streams}")
        print(f"  Stream sizes:")
        for i, size in enumerate(self.stream_sizes):
            param_type = ['param', 'grad', 'exp_avg', 'exp_avg_sq'][i]
            print(f"    Stream {i} ({param_type}): {size:,} floats ({size*4/1e9:.2f} GB)")
        
        # 创建层分组
        self.layer_groups = self._create_layer_groups()
        print(f"  Layer groups: {len(self.layer_groups)} groups")

        # self.chunk_size = 16 * 1024 * 1024  # 16M floats = 64MB
        
        # 优化：根据模型大小动态调整 chunk size (问题7)
        model_size_gb = total_size * 4 / 1e9
        if model_size_gb < 0.5:
            self.chunk_size = 4 * 1024 * 1024   # 4M floats = 16MB
        elif model_size_gb < 2:
            self.chunk_size = 8 * 1024 * 1024   # 8M floats = 32MB
        elif model_size_gb < 5:
            self.chunk_size = 16 * 1024 * 1024  # 16M floats = 64MB
        else:
            self.chunk_size = 16 * 1024 * 1024  # 32M floats = 128MB
        
        # 策略2：基于实际并发需求估算
        # 关键洞察：虽然层组是串行的，但SSD写入慢，chunks会在pipeline中堆积
        # 需要为"最坏情况"预留足够的chunks
        max_stream_size = max(self.stream_sizes)
        chunks_per_stream = (max_stream_size + self.chunk_size - 1) // self.chunk_size
        
        # 并发分析：
        # - 理论上：每个checkpoint最多num_layer_groups个层组在飞行中
        # - 实际上：由于SSD写入慢，几乎所有层组的chunks都被占用直到checkpoint完成
        # - 因此需要：max_async × num_streams × chunks_per_stream
        # 
        # 但由于chunks是按需借用的，可以适当减少（假设有50%的复用率）
        estimated_concurrent_chunks = max_async * num_streams * chunks_per_stream
        self.num_chunks = int(estimated_concurrent_chunks * 0.6) + num_streams * 2
        
        # 确保不少于一个checkpoint的完整需求
        min_chunks = num_streams * chunks_per_stream + num_streams
        self.num_chunks = max(self.num_chunks, min_chunks)
        
        print(f"  DRAMAlloc chunk size: {self.chunk_size} floats ({self.chunk_size*4/1024/1024:.2f} MB)")
        print(f"  DRAMAlloc num chunks: {self.num_chunks} (chunks/stream={chunks_per_stream}, estimated_usage=60%, Total: {self.num_chunks*self.chunk_size*4/1024/1024/1024:.2f} GB)")
        
        # 初始化多流writer
        resolved_filename = filename
        if isinstance(resolved_filename, str):
            try:
                resolved_filename = resolved_filename.format(rank=rank, world_size=world_size)
            except KeyError:
                # 如果用户提供了无法解析的模板，直接使用原始字符串
                resolved_filename = filename
            fname_bytes = resolved_filename.encode()
        else:
            fname_bytes = resolved_filename

        self.filename = resolved_filename

        if self.metadata_path is None and isinstance(resolved_filename, str):
            self.metadata_path = resolved_filename + ".metadata.json"
        self.writer = MultiStreamWriter(
            fname_bytes,
            lib_path,
            max_async,
            num_streams,
            self.stream_sizes,
            chunk_size=self.chunk_size,
            num_chunks=self.num_chunks,
            is_distributed=distributed,
            rank=rank,
            world_size=world_size
        )
        
        self.cuda_streams = []
        for i in range(max_async + 1):  # max_async+1个槽位（0到max_async）
            self.cuda_streams.append([torch.cuda.Stream() for _ in range(num_streams)])
        print(f"  ✓ Created CUDA streams: {max_async + 1} slots × {num_streams} streams = {(max_async + 1) * num_streams} streams (reusable)")
        
        # 创建双线程池架构：
        # 1. copy_thread_pool: 专门处理 GPU→CPU 拷贝（等待 CUDA stream 同步）
        # 2. ssd_thread_pool: 专门处理 SSD 写入（IO 密集型，不阻塞拷贝）
        import os
        cpu_count = os.cpu_count() or 16
        
        # GPU→CPU 拷贝线程池：每个流一个线程足够（瓶颈在 PCIe 带宽）
        copy_pool_workers = num_streams * (max_async + 1)
        self.copy_thread_pool = ThreadPoolExecutor(max_workers=copy_pool_workers, thread_name_prefix="copy_worker")
        
        # SSD 写入线程池：IO 密集型，可以多一些线程
        ssd_pool_workers = min(num_streams * max_async * 2, cpu_count)
        self.ssd_thread_pool = ThreadPoolExecutor(max_workers=ssd_pool_workers, thread_name_prefix="ssd_worker")
        
        # 保留旧名称以兼容性（指向 copy_thread_pool）
        self.thread_pool = self.copy_thread_pool
        
        print(f"  ✓ Created dual thread pools:")
        print(f"      Copy pool: {copy_pool_workers} workers (for GPU→CPU)")
        print(f"      SSD pool:  {ssd_pool_workers} workers (for SSD writes)")
        
        # 统计信息
        self.save_times = []
        
        # IO统计回调
        self.io_callback = None
        self._inflight_ssd_tasks = 0
        self._inflight_ssd_lock = Lock()
        
        # ✅ 修复：槽位管理器（支持多个并行检查点）
        # 使用字典跟踪每个槽位的状态：{parall_iter: {'futures': [...], 'start_time': ...}}
        self._checkpoint_slots = {}  # 当前活跃的检查点槽位
        self._slot_lock = Lock()  # 保护槽位操作的锁
        self._max_async = max_async
        self._slot_available_event = threading.Event()  # 槽位释放通知
        self._slot_available_event.set()  # 初始状态：有可用槽位
        
        # ✅ 跨检查点同步：记录每个层组的拷贝完成事件
        # _layer_copy_events[group_idx] = [event0, event1, event2, event3] (每个流一个事件)
        # 新检查点更新该层组之前，需要等待所有 4 个事件
        self._layer_copy_events = {}  # {group_idx: [events]}
        self._layer_copy_events_lock = Lock()
        print(f"  ✓ Cross-checkpoint layer synchronization enabled")

        # 尝试导出基础 metadata（不含 optimizer state），便于加载端提前准备
        if self.metadata_path:
            try:
                self.export_metadata(self.metadata_path)
            except Exception as e:
                print(f"[WARN] export_metadata failed during init: {e}")
        
    def _calculate_max_chunk_size(self):
        """计算最大的chunk大小（用于DRAMAlloc）"""
        max_size = 0
        for group in self.layer_groups:
            # 计算该组中每个流的大小
            group_stream_sizes = [0, 0, 0, 0]
            for layer_id in group:
                if layer_id >= len(self.param_layout):
                    continue
                layer_info = self.param_layout[layer_id]
                group_stream_sizes[0] += layer_info.get('param_size', 0)
                group_stream_sizes[1] += layer_info.get('grad_size', 0)
                group_stream_sizes[2] += layer_info.get('exp_avg_size', 0)
                group_stream_sizes[3] += layer_info.get('exp_avg_sq_size', 0)
            
            # 取该组中最大的流大小
            max_size = max(max_size, max(group_stream_sizes))
            
        return max_size
        
    def _calculate_stream_sizes(self):
        """计算每个流的总大小"""
        sizes = [0, 0, 0, 0]  # param, grad, exp_avg, exp_avg_sq
        for layer_info in self.param_layout:
            sizes[0] += layer_info.get('param_size', 0)
            sizes[1] += layer_info.get('grad_size', 0)
            sizes[2] += layer_info.get('exp_avg_size', 0)
            sizes[3] += layer_info.get('exp_avg_sq_size', 0)
        return sizes

    @staticmethod
    def _detach_buf(t):
        """detach + 关闭 requires_grad，防止 autograd 节点泄漏。"""
        if t is None:
            return None
        try:
            b = t.detach()
            b.requires_grad_(False)
            return b
        except Exception:
            return t

    def _make_gpu_buffers(self, gpu_ar):
        """将 gpu_ar（tensor 或 dict）转换为 List[Tensor]，长度 = num_streams。"""
        _KEYS = ['param', 'grad', 'exp_avg', 'exp_avg_sq']
        if isinstance(gpu_ar, dict):
            bufs = []
            for i, k in enumerate(_KEYS):
                v = gpu_ar.get(k, None)
                if v is None:
                    v = gpu_ar.get(i, None)
                bufs.append(self._detach_buf(v))
            return bufs
        # legacy: 使用 stream_sizes 精确拆分（兼容 total_size 不一致的场景）
        offset = 0
        buffers = []
        for sz in self.stream_sizes:
            buffers.append(self._detach_buf(gpu_ar[offset:offset + sz]))
            offset += sz
        return buffers
    
    def _create_layer_groups(self):
        """将层分成多组"""
        total_layers = len(self.param_layout)
        layers_per_group = max(1, total_layers // self.num_layer_groups)
        
        groups = []
        for i in range(self.num_layer_groups):
            start = i * layers_per_group
            end = min((i + 1) * layers_per_group, total_layers)
            if start < total_layers:
                groups.append(list(range(start, end)))
        
        return groups

    # ====================== Metadata Export ======================
    def _serialize_optimizer_param_groups(self, optimizer, param_name_to_param):
        if optimizer is None:
            return None
        param_id_to_name = {id(param): name for name, param in param_name_to_param.items()}
        serialized = []
        for group in optimizer.param_groups:
            new_group = {}
            for key, value in group.items():
                if key == "params":
                    new_group["params"] = [param_id_to_name.get(id(p)) for p in value if id(p) in param_id_to_name]
                else:
                    if isinstance(value, torch.Tensor):
                        new_group[key] = value.item()
                    elif isinstance(value, (list, tuple)):
                        new_group[key] = [v.item() if isinstance(v, torch.Tensor) else v for v in value]
                    else:
                        new_group[key] = value
            serialized.append(new_group)
        return serialized

    def _serialize_optimizer_steps(self, optimizer, param_name_to_param):
        if optimizer is None:
            return None
        steps = {}
        for name, param in param_name_to_param.items():
            state = optimizer.state.get(param, {})
            if "step" in state:
                step_val = state["step"]
                if isinstance(step_val, torch.Tensor):
                    steps[name] = int(step_val.item())
                else:
                    steps[name] = int(step_val)
        return steps

    def export_metadata(self, output_path: Optional[str] = None, optimizer: Optional[torch.optim.Optimizer] = None,
                        chunk_groups: Optional[List[List[int]]] = None) -> str:
        """
        导出 multistream 检查点元数据，供流水加载使用。

        Args:
            output_path: 元数据文件路径；默认使用 self.metadata_path。
            optimizer: 可选，用于导出 optimizer param_groups 与 step。
            chunk_groups: 可选自定义 layer group 划分；默认使用 self.layer_groups。
        """
        if output_path is None:
            output_path = self.metadata_path
        if output_path is None:
            raise ValueError("metadata_path is not set; please pass output_path explicitly.")

        stream_names = ["param", "grad", "exp_avg", "exp_avg_sq"]
        stream_sizes = self.stream_sizes
        total_checkpoint_size = sum(stream_sizes)
        page_size = 4096
        checkpoint_bytes = total_checkpoint_size * 4
        checkpoint_padding_bytes = (page_size - (checkpoint_bytes % page_size)) % page_size
        checkpoint_stride_floats = total_checkpoint_size + (checkpoint_padding_bytes // 4)

        # 参数布局映射
        param_map = {}
        for layer_info in self.param_layout:
            name = layer_info.get("name")
            if not name:
                continue
            param_map[name] = {
                "numel": int(layer_info.get("param_size", 0)),
                "param_offset": int(layer_info.get("param_offset", 0)),
                "param_size": int(layer_info.get("param_size", 0)),
                "grad_offset": int(layer_info.get("grad_offset", 0)),
                "grad_size": int(layer_info.get("grad_size", 0)),
                "exp_avg_offset": int(layer_info.get("exp_avg_offset", 0)),
                "exp_avg_size": int(layer_info.get("exp_avg_size", 0)),
                "exp_avg_sq_offset": int(layer_info.get("exp_avg_sq_offset", 0)),
                "exp_avg_sq_size": int(layer_info.get("exp_avg_sq_size", 0)),
            }

        chunk_groups = chunk_groups or self.layer_groups
        chunks = {}
        for group_idx, layer_ids in enumerate(chunk_groups):
            params = []
            stream_start = {name: None for name in stream_names}
            stream_end = {name: None for name in stream_names}

            for layer_id in layer_ids:
                if layer_id >= len(self.param_layout):
                    continue
                info = self.param_layout[layer_id]
                name = info.get("name")
                if not name:
                    continue
                offsets = {
                    "param": int(info.get("param_offset", 0)),
                    "grad": int(info.get("grad_offset", 0)),
                    "exp_avg": int(info.get("exp_avg_offset", 0)),
                    "exp_avg_sq": int(info.get("exp_avg_sq_offset", 0)),
                }
                sizes = {
                    "param": int(info.get("param_size", 0)),
                    "grad": int(info.get("grad_size", 0)),
                    "exp_avg": int(info.get("exp_avg_size", 0)),
                    "exp_avg_sq": int(info.get("exp_avg_sq_size", 0)),
                }
                for stream in stream_names:
                    if stream_start[stream] is None:
                        stream_start[stream] = offsets[stream]
                        stream_end[stream] = offsets[stream] + sizes[stream]
                    else:
                        stream_start[stream] = min(stream_start[stream], offsets[stream])
                        stream_end[stream] = max(stream_end[stream], offsets[stream] + sizes[stream])

                params.append({
                    "name": name,
                    "numel": int(info.get("param_size", 0)),
                    "offsets": offsets,
                    "sizes": sizes,
                })

            stream_slices = {}
            for stream in stream_names:
                if stream_start[stream] is None:
                    stream_slices[stream] = {"offset": 0, "size": 0}
                else:
                    stream_slices[stream] = {
                        "offset": int(stream_start[stream]),
                        "size": int(stream_end[stream] - stream_start[stream]),
                    }

            # 计算每个参数在 chunk 内的 offset
            for param in params:
                param["offsets_in_chunk"] = {
                    stream: param["offsets"][stream] - stream_slices[stream]["offset"]
                    for stream in stream_names
                }

            chunks[str(group_idx)] = {
                "group_idx": int(group_idx),
                "layer_ids": [int(i) for i in layer_ids],
                "params": params,
                "stream_slices": stream_slices,
            }

        # optimizer metadata
        param_name_to_param = {}
        if hasattr(self, "_pending_model") and self._pending_model is not None:
            param_name_to_param = {name: p for name, p in self._pending_model.named_parameters()}
        optimizer_param_groups = self._serialize_optimizer_param_groups(optimizer, param_name_to_param) if optimizer else None
        optimizer_steps = self._serialize_optimizer_steps(optimizer, param_name_to_param) if optimizer else None

        checkpoint_file = self.filename
        if isinstance(checkpoint_file, (bytes, bytearray)):
            try:
                checkpoint_file = checkpoint_file.decode("utf-8")
            except Exception:
                checkpoint_file = None

        metadata = {
            "format": "multistream",
            "version": 1,
            "dtype": "float32",
            "checkpoint_file": checkpoint_file,
            "max_async": int(self.max_async),
            "num_streams": int(self.num_streams),
            "stream_names": stream_names,
            "stream_sizes": [int(x) for x in stream_sizes],
            "total_checkpoint_size": int(total_checkpoint_size),
            "checkpoint_padding_bytes": int(checkpoint_padding_bytes),
            "checkpoint_stride_floats": int(checkpoint_stride_floats),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "param_map": param_map,
        }

        if optimizer_param_groups is not None:
            metadata["optimizer_param_groups"] = optimizer_param_groups
        if optimizer_steps is not None:
            metadata["optimizer_steps"] = optimizer_steps

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Multistream metadata exported to {output_path}")
        return output_path

    def _update_latest_parall_iter(self, parall_iter: int) -> None:
        """
        更新 metadata 文件中的 latest_parall_iter 字段。
        
        在 checkpoint 成功落盘后调用，记录最新可用的检查点槽位，
        供加载时直接读取，避免调用 C++ 的 get_latest_parall_iter。
        """
        if not self.metadata_path or not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["latest_parall_iter"] = int(parall_iter)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to update latest_parall_iter in metadata: {e}")
    
    def save_layer_group(self, layer_group_ids, parall_iter, group_idx, sync_event=None):
        """
        保存一组层的参数（4个流并行，异步不阻塞）
        
        Args:
            layer_group_ids: 要保存的层ID列表
            parall_iter: 并行迭代槽位（由registerCheck返回）
            group_idx: 层组索引，用于跨检查点同步
            sync_event: CUDA同步事件，用于确保参数更新完成后再开始拷贝
            
        Returns:
            futures: 返回futures列表，供后续等待
        """
        # ✅ 创建 4 个拷贝完成事件，每个流一个
        copy_done_events = [torch.cuda.Event() for _ in range(4)]
        
        # 使用线程池提交任务（避免反复创建销毁线程）
        futures = []
        
        for stream_id, param_type in enumerate(['param', 'grad', 'exp_avg', 'exp_avg_sq']):
            future = self.writer._submit_with_affinity(
                self.thread_pool,
                self._save_stream_for_layers,
                stream_id, param_type, layer_group_ids, parall_iter, sync_event, 
                copy_done_events[stream_id], group_idx
            )
            futures.append(future)
        
        # ✅ 立即更新层组的拷贝完成事件列表（事件会在各自的流中被记录）
        with self._layer_copy_events_lock:
            self._layer_copy_events[group_idx] = copy_done_events
        
        # 不阻塞，直接返回futures，让调用者决定何时等待
        return futures

    def _register_layer_group_futures(self, parall_iter, futures):
        """将异步future登记到指定槽位，供finalize阶段统一等待"""
        if not futures:
            return
        with self._slot_lock:
            if parall_iter in self._checkpoint_slots:
                self._checkpoint_slots[parall_iter]['futures'].extend(futures)
    
    def _save_stream_for_layers(self, stream_id, param_type, layer_group_ids, parall_iter, sync_event=None, copy_done_event=None, group_idx=None):
        """
        保存指定层组的某一类参数
        
        优化设计：
        1. GPU→CPU 拷贝完成后立即记录 copy_done_event，允许新 checkpoint 开始
        2. SSD 写入在后台异步完成，不阻塞新的拷贝操作
        3. 移除复杂的流水线设计，简化逻辑
        
        Args:
            stream_id: 流ID
            param_type: 参数类型 (param/grad/exp_avg/exp_avg_sq)
            layer_group_ids: 层ID列表
            parall_iter: 并行迭代槽位（由registerCheck返回）
            sync_event: CUDA同步事件，用于确保参数更新完成后再开始拷贝
            copy_done_event: 拷贝完成事件，用于跨检查点同步
            group_idx: 层组索引
        """
        try:
            import ctypes
            import numpy as np
            
            stream_index = parall_iter % (self._max_async + 1)
            current_stream = self.cuda_streams[stream_index][stream_id]

            with torch.cuda.stream(current_stream):
                # 等待默认流上的参数更新完成
                if sync_event is not None:
                    sync_event.wait(current_stream)
                
                # 计算该组在当前流中的偏移
                offset_in_stream = 0
                for i in range(min(layer_group_ids[0], len(self.param_layout))):
                    layer_info = self.param_layout[i]
                    offset_in_stream += layer_info.get(f'{param_type}_size', 0)
                
                # 预先计算需要处理的层信息
                layers_to_process = []
                total_elements = 0
                
                for layer_id in layer_group_ids:
                    if layer_id >= len(self.param_layout):
                        continue
                    
                    layer_info = self.param_layout[layer_id]
                    gpu_offset = layer_info.get(f'{param_type}_offset', 0)
                    size = layer_info.get(f'{param_type}_size', 0)
                    
                    gpu_buffer = self.gpu_buffers[stream_id]
                    if size > 0 and gpu_buffer is not None and gpu_offset + size <= len(gpu_buffer):
                        layers_to_process.append((gpu_offset, size))
                        total_elements += size
                
                if not layers_to_process:
                    # 即使没有数据，也要记录事件（保持一致性）
                    if copy_done_event is not None:
                        copy_done_event.record(current_stream)
                    return

                # ========== 阶段2：异步写入 SSD ==========
                # ✅ 供“按 chunk 流式提交”使用：避免 pinned chunk 池不足导致 borrow_chunk 阻塞/死锁（H6）
                def ssd_write_task(chunks, stream_id, offset_in_stream, parall_iter, num_threads, writer):
                    try:
                        total_chunk_elems = 0
                        try:
                            total_chunk_elems = int(sum(c[1] for c in chunks))
                        except Exception:
                            total_chunk_elems = 0
                        for chunk_ptr, chunk_size, offset, ready_event in chunks:
                            if ready_event is not None:
                                ready_event.synchronize()
                            float_ptr = cast(chunk_ptr, POINTER(c_float))
                            writer.write_stream_chunk(
                                stream_id,
                                float_ptr,
                                offset_in_stream + offset,
                                chunk_size,
                                parall_iter,
                                num_threads,
                            )
                    except Exception as e:
                        print(f"[ERROR] SSD write failed for stream {stream_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        for chunk_ptr, _, _, _ in chunks:
                            writer.return_chunk(chunk_ptr)
                        with self._inflight_ssd_lock:
                            self._inflight_ssd_tasks -= 1
                            inflight_after = int(self._inflight_ssd_tasks)

                def _submit_ssd(chunks):
                    with self._inflight_ssd_lock:
                        self._inflight_ssd_tasks += 1
                        inflight_before_submit = int(self._inflight_ssd_tasks)
                    fut = self.writer._submit_with_affinity(
                        self.ssd_thread_pool,
                        ssd_write_task,
                        chunks,
                        stream_id,
                        offset_in_stream,
                        parall_iter,
                        self.num_threads,
                        self.writer,
                    )
                    self._register_layer_group_futures(parall_iter, [fut])

                # ========== 阶段1：GPU→CPU 拷贝（按 chunk 流式提交 SSD 写入） ==========
                current_offset = 0
                current_layer_idx = 0
                offset_in_current_layer = 0

                while current_offset < total_elements:
                    elements_to_process = min(self.chunk_size, total_elements - current_offset)

                    chunk_ptr = self.writer.borrow_chunk()
                    
                    if chunk_ptr is None or chunk_ptr == 0:
                        raise RuntimeError(f"borrow_chunk() returned invalid pointer: {chunk_ptr}")

                    chunk_array = np.ctypeslib.as_array((ctypes.c_float * self.chunk_size).from_address(chunk_ptr))
                    # ✅ 修复段错误：使用 detach() 彻底断开 autograd 图
                    # 
                    # 问题根因：
                    # 1. torch.from_numpy() 创建的 tensor 与 numpy 数组共享底层内存
                    # 2. 对 tensor 进行 slice 会创建 SliceBackward0 autograd 节点
                    # 3. 当 chunk 被归还到 DRAMAlloc 池后，autograd 节点仍持有对该内存的引用
                    # 4. 在 Python GC 清理时，autograd::deleteNode() 会访问已失效的内存 → 段错误
                    # 
                    # 解决方案：
                    # - 使用 detach() 断开所有 autograd 连接
                    # - torch.no_grad() 只能阻止新梯度被记录，无法阻止已有的 storage 引用
                    chunk_tensor = torch.from_numpy(chunk_array).detach()
                    cpu_buffer_slice = chunk_tensor[:elements_to_process].detach()

                    # ✅ H5: 禁用 autograd，避免 CopySlices/SliceBackward 在后台线程构建
                    dest_offset = 0
                    with torch.no_grad():
                        while dest_offset < elements_to_process:
                            gpu_offset, size = layers_to_process[current_layer_idx]
                            remaining_in_layer = size - offset_in_current_layer
                            needed = elements_to_process - dest_offset
                            to_copy = min(remaining_in_layer, needed)

                            src_start = gpu_offset + offset_in_current_layer
                            src_view = self.gpu_buffers[stream_id][src_start : src_start + to_copy].detach()
                            dest_view = cpu_buffer_slice[dest_offset : dest_offset + to_copy]
                            dest_view.copy_(src_view, non_blocking=True)

                            dest_offset += to_copy
                            offset_in_current_layer += to_copy

                            if offset_in_current_layer == size:
                                current_layer_idx += 1
                                offset_in_current_layer = 0

                    chunk_offset = current_offset
                    current_offset += elements_to_process

                    # 记录chunk就绪事件；SSD线程在事件完成后再写盘，避免阻塞拷贝线程
                    chunk_ready_event = torch.cuda.Event()
                    chunk_ready_event.record(current_stream)
                    _submit_ssd([(chunk_ptr, elements_to_process, chunk_offset, chunk_ready_event)])

                # ✅ 所有 GPU→CPU 拷贝命令已完成/已提交，记录跨检查点同步事件
                if copy_done_event is not None:
                    copy_done_event.record(current_stream)
                
        except Exception as e:
            print(f"[ERROR] Stream {stream_id} ({param_type}) failed: {e}")
            import traceback
            traceback.print_exc()
    
    def set_io_callback(self, callback):
        """
        设置IO统计回调函数
        
        Args:
            callback: 回调函数，签名为 callback(save_time_sec, throughput_gbps)
        """
        self.io_callback = callback
    
    def create_optimizer(
        self, 
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module
    ) -> MultiStreamOptimizer:
        """
        创建MultiStreamOptimizer包装器，支持边更新边保存
        
        Args:
            optimizer: 标准PyTorch优化器（如Adam, SGD）
            model: PyTorch模型
            
        Returns:
            MultiStreamOptimizer实例
        """
        # 获取模型参数列表
        param_list = list(model.parameters())
        
        # ✅ 修复：保存优化器和模型信息，在begin_checkpoint时创建实际的MultiStreamOptimizer
        self._pending_optimizer = optimizer
        self._pending_model = model
        self._pending_param_list = param_list

        if self.metadata_path:
            try:
                self.export_metadata(self.metadata_path, optimizer=optimizer)
            except Exception as e:
                print(f"[WARN] export_metadata failed in create_optimizer: {e}")
        
        # 返回一个包装器，支持在begin_checkpoint后获取实际的MultiStreamOptimizer
        class OptimizerWrapper:
            def __init__(self, checkpoint_obj, real_optimizer):
                self._checkpoint = checkpoint_obj
                self._real_optimizer = real_optimizer
                self._ms_optimizer = None  # 在begin_checkpoint时创建
                self._current_parall_iter = None
            
            def step(self, closure=None):
                """
                普通的优化器step，不执行checkpoint。
                
                ✅ 性能优化：在非checkpoint迭代时，直接使用原生优化器，
                避免分层组更新和CUDA流同步的开销。
                
                Args:
                    closure: 可选的闭包函数
                    
                Returns:
                    loss: 如果提供了closure则返回loss，否则返回None
                """
                return self._real_optimizer.step(closure)
            
            def step_with_checkpoint(self, closure=None, wait=False):
                """
                执行带checkpoint的step（一体化接口）。
                
                自动处理 begin_checkpoint → step_with_callback → finalize_checkpoint 流程。
                
                Args:
                    closure: 可选的闭包函数
                    wait: 是否等待checkpoint完成（True=同步，False=异步）
                    
                Returns:
                    loss: 如果提供了closure则返回loss，否则返回None
                """
                self.begin_checkpoint()
                loss = self.step_with_callback(closure)
                self.finalize_checkpoint(wait=wait)
                return loss
            
            def begin_checkpoint(self):
                """开始检查点，创建实际的MultiStreamOptimizer"""
                parall_iter = self._checkpoint.begin_checkpoint()
                self._current_parall_iter = parall_iter
                
                # ✅ 跨检查点同步：只在检查点开始时获取需要等待的事件快照
                # 这样可以避免在同一个检查点内等待自己刚刚创建的事件
                with self._checkpoint._layer_copy_events_lock:
                    # 复制当前的事件映射（这些是上一个检查点的事件）
                    prev_checkpoint_events = dict(self._checkpoint._layer_copy_events)
                
                def pre_update_callback(group_idx):
                    prev_events = prev_checkpoint_events.get(group_idx)
                    
                    if prev_events is not None:
                        # 优化：使用 CUDA stream wait 替代 CPU synchronize
                        # 让当前 GPU 流等待，而不是阻塞 CPU
                        # 只等待param流（性能最优）
                        torch.cuda.current_stream().wait_event(prev_events[0])
                        
                        # 选项2：等待param + exp_avg + exp_avg_sq（更安全，慢25%）
                        # torch.cuda.current_stream().wait_event(prev_events[0])  # param
                        # torch.cuda.current_stream().wait_event(prev_events[2])  # exp_avg
                        # torch.cuda.current_stream().wait_event(prev_events[3])  # exp_avg_sq
                        
                        # 选项3：等待所有4个流（最安全，但grad不需要）
                        # for event in prev_events:
                        #     torch.cuda.current_stream().wait_event(event)
                
                # 优化：使用专门的拷贝流，避免阻塞默认流 (问题6)
                copy_stream = torch.cuda.Stream()
                
                # 检测零拷贝：第一个参数是否位于 gpu_buffers[0]（param buffer）内
                is_zero_copy_enabled = False
                if (hasattr(self._checkpoint, '_pending_param_list')
                        and self._checkpoint._pending_param_list
                        and hasattr(self._checkpoint, 'gpu_buffers')
                        and self._checkpoint.gpu_buffers
                        and self._checkpoint.gpu_buffers[0] is not None):
                    first_param = self._checkpoint._pending_param_list[0]
                    buf = self._checkpoint.gpu_buffers[0]
                    buf_ptr = buf.data_ptr()
                    buf_end = buf_ptr + buf.numel() * buf.element_size()
                    if buf_ptr <= first_param.data_ptr() < buf_end:
                        is_zero_copy_enabled = True
                        print("  \u2713 Zero-copy mode detected: skipping D2D copies in save_callback")

                # 创建回调函数，捕获parall_iter
                def save_callback(group_idx, layer_ids):
                    if is_zero_copy_enabled:
                        # 🔥 零拷贝优化：参数已经在 gpu_ar 上，无需拷贝
                        # 直接在默认流上记录事件，标记参数更新已完成
                        sync_event = torch.cuda.Event()
                        sync_event.record(torch.cuda.current_stream())
                    else:
                        # 常规模式：在拷贝流上执行 D2D 拷贝
                        with torch.cuda.stream(copy_stream):
                            # 等待默认流上的参数更新完成
                            copy_stream.wait_stream(torch.cuda.current_stream())
                            
                            # ✅ 修复段错误：必须在 no_grad 上下文中执行拷贝
                            # 并且使用 detach() 断开所有 autograd 连接
                            # 
                            # 问题根因：
                            # 1. param.view(-1).float() 会创建 autograd 节点
                            # 2. gpu_ar[...] slice 操作也会创建 SliceBackward0 节点
                            # 3. 这些节点持有对底层 storage 的引用
                            # 4. 在程序退出时 GC 清理这些节点会访问已失效的内存 → 段错误
                            with torch.no_grad():
                                bufs = self._checkpoint.gpu_buffers
                                for layer_id in layer_ids:
                                    if layer_id >= len(self._checkpoint.param_layout):
                                        continue
                                    li = self._checkpoint.param_layout[layer_id]
                                    param = self._checkpoint._pending_param_list[layer_id]

                                    off = li["param_offset"]
                                    sz = li["param_size"]
                                    bufs[0][off : off + sz].copy_(param.detach().view(-1).float(), non_blocking=True)

                                    goff = li["grad_offset"]
                                    gsz = li["grad_size"]
                                    if param.grad is None:
                                        bufs[1][goff : goff + gsz].zero_()
                                    else:
                                        bufs[1][goff : goff + gsz].copy_(param.grad.detach().view(-1).float(), non_blocking=True)

                                    state = self._real_optimizer.state.get(param, {})

                                    ea = state.get("exp_avg")
                                    ea_off = li["exp_avg_offset"]
                                    ea_sz = li["exp_avg_size"]
                                    if ea is None:
                                        bufs[2][ea_off : ea_off + ea_sz].zero_()
                                    else:
                                        bufs[2][ea_off : ea_off + ea_sz].copy_(ea.detach().view(-1).float(), non_blocking=True)

                                    eas = state.get("exp_avg_sq")
                                    eas_off = li["exp_avg_sq_offset"]
                                    eas_sz = li["exp_avg_sq_size"]
                                    if eas is None:
                                        bufs[3][eas_off : eas_off + eas_sz].zero_()
                                    else:
                                        bufs[3][eas_off : eas_off + eas_sz].copy_(eas.detach().view(-1).float(), non_blocking=True)

                            # ✅ 在拷贝流上记录事件，标记该层组的参数拷贝已完成
                            sync_event = torch.cuda.Event()
                            sync_event.record(copy_stream)
                        
                    # 提交异步保存任务，传递同步事件和group_idx
                    futures = self._checkpoint.save_layer_group(layer_ids, parall_iter, group_idx, sync_event)
                    
                    # 将futures添加到对应槽位的记录中
                    self._checkpoint._register_layer_group_futures(parall_iter, futures)
                
                # 创建实际的MultiStreamOptimizer
                self._ms_optimizer = MultiStreamOptimizer(
                    optimizer=self._real_optimizer,
                    layer_groups=self._checkpoint.layer_groups,
                    param_list=self._checkpoint._pending_param_list,
                    callback=save_callback
                )
                # ✅ 设置更新前回调
                self._ms_optimizer.pre_update_callback = pre_update_callback
                
                return parall_iter
            
            def step_with_callback(self, closure=None):
                """执行带回调的step（必须在begin_checkpoint后调用）"""
                if self._ms_optimizer is None:
                    raise RuntimeError("Must call begin_checkpoint() before step_with_callback()")
                return self._ms_optimizer.step_with_callback(closure)
            
            def finalize_checkpoint(self, wait=True):
                """完成检查点（必须在begin_checkpoint后调用）"""
                if self._current_parall_iter is None:
                    raise RuntimeError("Must call begin_checkpoint() before finalize_checkpoint()")
                parall_iter = self._current_parall_iter
                self._current_parall_iter = None
                self._ms_optimizer = None
                return self._checkpoint.finalize_checkpoint(parall_iter, wait)
            
            def zero_grad(self, set_to_none=True):
                """清零梯度，直接代理到原生优化器"""
                return self._real_optimizer.zero_grad(set_to_none=set_to_none)
            
            @property
            def param_groups(self):
                """返回原生优化器的参数组"""
                return self._real_optimizer.param_groups
            
            @property
            def state(self):
                """返回原生优化器的状态"""
                return self._real_optimizer.state
            
            def state_dict(self):
                """返回原生优化器的状态字典"""
                return self._real_optimizer.state_dict()
            
            def load_state_dict(self, state_dict):
                """加载状态字典到原生优化器"""
                return self._real_optimizer.load_state_dict(state_dict)
            
            def __getattr__(self, name):
                # 代理其他方法到原始优化器
                # 注意：不再代理到 _ms_optimizer，因为它只在 checkpoint 期间存在
                return getattr(self._real_optimizer, name)
        
        return OptimizerWrapper(self, optimizer)
    
    def save_full_checkpoint(self, sync: bool = True, stream: Optional[torch.cuda.Stream] = None) -> Tuple[int, float]:
        """
        手动触发一次完整的多流保存，常用于快照模式或分布式rank独立保存。
        
        Args:
            sync: True 表示等待写入完成；False 表示后台异步。
            stream: 可选的CUDA流，在该流完成后再开始拷贝；默认使用当前流。
        
        Returns:
            Tuple[int, float]: (parall_iter, finalize_result)
        """
        parall_iter = self.begin_checkpoint()
        working_stream = stream or torch.cuda.current_stream()
        
        for group_idx, layer_ids in enumerate(self.layer_groups):
            sync_event = torch.cuda.Event()
            sync_event.record(working_stream)
            futures = self.save_layer_group(layer_ids, parall_iter, group_idx, sync_event)
            self._register_layer_group_futures(parall_iter, futures)
        
        finalize_result = self.finalize_checkpoint(parall_iter, wait=sync)
        return parall_iter, finalize_result
    
    def begin_checkpoint(self):
        """
        开始检查点保存流程
        
        在调用optimizer.step_with_callback()之前调用
        
        ✅ 修复：支持多个并行检查点
        - 当并行检查点数超过max_async时，会等待先前的检查点完成
        - 每个检查点使用独立的槽位和CUDA流
        
        Returns:
            parall_iter: 检查点槽位ID，用于后续操作
        """
        # ✅ 修复：使用 Event 等待可用槽位，避免 time.sleep 轮询
        while True:
            self._slot_available_event.wait()  # 阻塞直到收到槽位释放通知
            with self._slot_lock:
                if len(self._checkpoint_slots) < self._max_async:
                    # 有可用槽位，注册获取
                    parall_iter = self.writer.register()
                    # 如果槽位已满，清除事件，让下次 begin_checkpoint 等待
                    if len(self._checkpoint_slots) + 1 >= self._max_async:
                        self._slot_available_event.clear()
                    break
        
        # ✅ 内存优化：不再预先借用巨大的pinned buffer
        # 改为在save_layer_group中按需借用小块buffer (chunk)
        
        # ✅ 修复：为每个检查点创建独立的记录
        with self._slot_lock:
            self._checkpoint_slots[parall_iter] = {
                'futures': [],
                'start_time': time.time(),
                # 'cpu_buffer': None  # 不再需要全局buffer
            }
        
        print(f"\n{'='*60}")
        print(f"Begin checkpoint (parall_iter: {parall_iter}, active: {len(self._checkpoint_slots)}/{self._max_async})")
        print(f"  Using on-demand chunk allocation (chunk size: {self.chunk_size*4/1e6:.2f} MB)")
        print(f"{'='*60}")
        
        return parall_iter
    
    def finalize_checkpoint(self, parall_iter, wait=True):
        """
        完成检查点保存流程
        
        Args:
            parall_iter: 检查点槽位ID（由begin_checkpoint返回）
            wait: 是否等待所有异步保存完成（True=同步模式，False=异步模式）
        
        在optimizer.step_with_callback()之后调用
        """
        # ✅ 修复：使用传入的parall_iter，从槽位记录中获取信息
        with self._slot_lock:
            if parall_iter not in self._checkpoint_slots:
                print(f"Warning: Checkpoint {parall_iter} not found in active slots")
                return None
            slot_info = self._checkpoint_slots[parall_iter]
            futures = slot_info['futures']
            start_time = slot_info['start_time']

        with self._inflight_ssd_lock:
            inflight_ssd = int(self._inflight_ssd_tasks)
            
        if wait:
            # 同步模式：等待所有保存完成
            # 等待所有异步保存任务完成
            print(f"\n  Waiting for all layer groups to complete (parall_iter: {parall_iter})...")
            for future in futures:
                future.result()
            
            # 同步所有流到磁盘
            print(f"\n  Syncing all streams to disk (parall_iter: {parall_iter})...")
            sync_start = time.time()
            self.writer.sync_all_streams(parall_iter)
            sync_time = time.time() - sync_start
            
            total_time = time.time() - start_time
            throughput = (self.total_size * 4 / 1e9) / total_time
            
            self.save_times.append(total_time)
            
            print(f"\n{'='*60}")
            print(f"✓ Checkpoint finalized (SYNC)!")
            print(f"  Slot (parall_iter): {parall_iter}")
            print(f"  Finalize time: {total_time:.2f}s")
            print(f"  Sync time: {sync_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} GB/s")
            print(f"{'='*60}\n")
            
            # ✅ 更新 metadata 中的 latest_parall_iter
            self._update_latest_parall_iter(parall_iter)
            
            # ✅ 修复：释放槽位
            with self._slot_lock:
                # 可能在 drain() 与后台 async finalize 并发时被其他线程抢先删除
                self._checkpoint_slots.pop(parall_iter, None)
            self._slot_available_event.set()  # 通知等待的 begin_checkpoint
            
            return total_time
        else:
            # 异步模式：立即返回，后台继续保存
            def async_finalize():
                try:  
                    # 等待所有异步保存任务完成（GPU→CPU拷贝）
                    for future in futures:
                        future.result()
                    
                    # 同步所有流到磁盘
                    sync_streams_start = time.time()
                    self.writer.sync_all_streams(parall_iter)
                    
                    sync_streams_time = time.time() - sync_streams_start
                    
                    total_time = time.time() - start_time
                    throughput = (self.total_size * 4 / 1e9) / total_time
                    
                    self.save_times.append(total_time)
                    
                    print(f"✓ Checkpoint {parall_iter} completed in background: {total_time:.2f}s, {throughput:.2f} GB/s")
                    
                    # ✅ 更新 metadata 中的 latest_parall_iter
                    self._update_latest_parall_iter(parall_iter)
                    
                    # 调用IO统计回调
                    if self.io_callback is not None:
                        self.io_callback(total_time, throughput)
                    
                    # ✅ 修复：释放槽位（sync_all_streams完成后）
                    with self._slot_lock:
                        if parall_iter in self._checkpoint_slots:
                            del self._checkpoint_slots[parall_iter]
                    self._slot_available_event.set()  # 通知等待的 begin_checkpoint
                except Exception as e:
                    print(f"✗ Background checkpoint {parall_iter} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # ✅ 修复：即使失败也要确保槽位被释放
                    with self._slot_lock:
                        self._checkpoint_slots.pop(parall_iter, None)
                    self._slot_available_event.set()  # 通知等待的 begin_checkpoint
            
            # 提交到线程池
            self.thread_pool.submit(async_finalize)
            
            # 立即返回（不阻塞）
            submit_time = time.time() - start_time
            print(f"✓ Checkpoint {parall_iter} submitted to background (Async submit time: {submit_time*1000:.2f}ms)")
            
            return submit_time

    def drain(self):
        """等待所有尚未完成的异步 checkpoint 完成并落盘。

        目的：
        - 让 benchmark 的时间统计包含尾部异步写入/flush，避免吞吐被高估或出现非单调异常。
        - 避免后台线程池继续占用 CPU/SSD 带宽，干扰后续 step 或后续实验。
        """
        print("Draining multistream checkpoint: waiting for all background tasks...")
        
        # 等待所有SSD写入任务完成
        max_wait_time = 60
        wait_start = time.time()
        
        while True:
            if (not hasattr(self, '_inflight_ssd_lock')) or (not hasattr(self, '_inflight_ssd_tasks')):
                # 可能在 __init__ 尚未完整结束时触发 shutdown/dtor，直接返回即可。
                return

            with self._inflight_ssd_lock:
                inflight = self._inflight_ssd_tasks
            
            if inflight == 0:
                print(f"✓ All background tasks completed")
                break
            
            if time.time() - wait_start > max_wait_time:
                print(f"⚠ Warning: drain timeout after {max_wait_time}s, {inflight} tasks still running")
                break
    
    def shutdown(self):
        """关闭线程池，释放资源"""
        # 在关闭线程池之前，先确保没有未完成的异步 checkpoint。
        # 否则 ThreadPoolExecutor.shutdown(wait=True) 只保证任务执行完，
        # 但我们的槽位状态/统计可能还没走完 finalize 的释放逻辑。
        try:
            self.drain()
        except Exception as e:
            print(f"[WARN] MultiStreamCheckpoint.drain() failed during shutdown: {e}")
        if hasattr(self, 'copy_thread_pool'):
            self.copy_thread_pool.shutdown(wait=True)
            print("✓ Copy thread pool shutdown")
        if hasattr(self, 'ssd_thread_pool'):
            self.ssd_thread_pool.shutdown(wait=True)
            print("✓ SSD thread pool shutdown")
    
    def __del__(self):
        """析构时自动关闭线程池"""
        self.shutdown()


def build_param_layout(model, optimizer):
    """
    构建参数布局信息。

    所有偏移均为局部偏移（0-based），即在每种数据类型的缓冲区内的偏移。
    这使得布局可以同时兼容单块 gpu_ar 拆分的 4 个 view 和四块独立 tensor。

    Returns:
        List[Dict]: 每层的参数信息
    """
    layout = []
    total_param_size = sum(p.numel() for p in model.parameters())

    layer_id = 0
    current_offset = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        size = param.numel()

        layer_info = {
            'layer_id': layer_id,
            'name': name,
            'param_offset': current_offset,
            'param_size': size,
            'grad_offset': current_offset,
            'grad_size': size,
            'exp_avg_offset': current_offset,
            'exp_avg_size': size,
            'exp_avg_sq_offset': current_offset,
            'exp_avg_sq_size': size,
        }

        layout.append(layer_info)
        current_offset += size
        layer_id += 1

    print(f"\nBuilt parameter layout (local offsets):")
    print(f"  Total layers: {len(layout)}")
    print(f"  Total parameters: {total_param_size:,} ({total_param_size*4/1e9:.2f} GB)")
    print(f"  Total size (4 copies): {total_param_size*4:,} ({total_param_size*16/1e9:.2f} GB)")

    return layout
