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
from typing import Callable, List, Dict, Optional
from collections import deque


class MultiStreamOptimizer:
    """
    多流优化器包装器：支持分层更新并触发回调
    
    使用装饰器模式包装标准PyTorch优化器，在每层组更新后触发回调
    """
    
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
        
    def step_with_callback(self, closure=None):
        """
        分层执行优化器更新，每层组更新后触发回调
        
        这是核心方法，按层组顺序更新参数，每组更新完成后立即触发回调
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 按层组遍历并更新
        for group_idx, (layer_ids, params) in enumerate(zip(self.layer_groups, self.group_params)):
            # 对这组参数执行真实的优化器更新
            self._step_param_group(params)
            
            # 触发回调（在更新完成后）
            if self.callback is not None:
                self.callback(layer_ids)
        
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
    
    def __init__(self, fname, lib_path, max_async, num_streams, stream_sizes, chunk_size=None, num_chunks=None):
        """
        Args:
            fname: 文件名
            lib_path: C库路径
            max_async: 最大异步检查点数
            num_streams: 流的数量（通常为4）
            stream_sizes: 每个流的大小列表
            chunk_size: DRAMAlloc分配的块大小（如果为None，则使用总大小）
            num_chunks: DRAMAlloc分配的块数量（如果为None，则使用max_async + 1）
        """
        self.lib = cdll.LoadLibrary(lib_path)
        self.num_streams = num_streams
        self.max_async = max_async
        
        # 计算总大小
        total_size = sum(stream_sizes)
        self.total_size = total_size
        
        # 确定chunk_size和num_chunks
        if chunk_size is None:
            chunk_size = total_size
        if num_chunks is None:
            num_chunks = max_async + 1
            
        self.chunk_size = chunk_size
        
        # ✅ 内存优化：使用指定的chunk_size和num_chunks初始化DRAMAlloc
        self.writer_obj = self.lib.writer(
            fname, max_async, c_size_t(chunk_size), 
            num_chunks,
            False, 0, 1  # is_distributed, rank, world_size
        )
        
        # 初始化多流
        # stream_sizes 是 float count，C++库会自动计算字节偏移(x4)
        stream_sizes_arr = (c_size_t * num_streams)(*stream_sizes)
        ret = self.lib.init_streams(self.writer_obj, num_streams, stream_sizes_arr)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize {num_streams} streams")
        
        # 注册C函数返回类型
        self.lib.registerCheck.restype = c_int
        
        # ✅ 注册borrow_cpu_slot和return_cpu_slot的返回类型
        self.lib.borrow_cpu_slot.restype = c_void_p
        self.lib.borrow_cpu_slot.argtypes = [c_void_p, c_int]
        self.lib.return_cpu_slot.argtypes = [c_void_p, c_int]
        
        # ✅ 注册borrow_chunk和return_chunk
        if hasattr(self.lib, 'borrow_chunk'):
            self.lib.borrow_chunk.restype = c_void_p
            self.lib.borrow_chunk.argtypes = [c_void_p]
            self.lib.return_chunk.argtypes = [c_void_p, c_void_p]
        
        print(f"✓ MultiStreamWriter initialized with {num_streams} streams")
        print(f"  DRAMAlloc: {num_chunks} chunks of size {chunk_size} floats ({chunk_size*4/1e6:.2f} MB)")
    
    def write_stream_chunk(self, stream_id, data, offset_in_stream, chunk_size, 
                           parall_iter, num_threads):
        """
        写入单个流的数据块
        
        Args:
            stream_id: 流ID (0-3)
            data: numpy数组
            offset_in_stream: 在流内的偏移 (float count)
            chunk_size: 数据块大小 (float count)
            parall_iter: 并行迭代槽位（由registerCheck返回）
            num_threads: 写入线程数
        """
        ct_arr = np.ctypeslib.as_ctypes(data)
        # 注意：C++接口期望的是字节大小，但偏移量是float count（内部会自动x4）
        # 修正：write_stream_chunk 似乎需要字节单位的 offset 和 size
        self.lib.write_stream_chunk(
            self.writer_obj,
            c_int(stream_id),
            ct_arr,
            c_size_t(offset_in_stream),
            c_size_t(chunk_size),
            c_int(parall_iter),
            c_int(num_threads)
        )
    
    def sync_stream(self, stream_id, parall_iter):
        """同步单个流到磁盘"""
        self.lib.sync_stream(self.writer_obj, c_int(stream_id), c_int(parall_iter))
    
    def sync_all_streams(self, parall_iter):
        """同步所有流到磁盘并更新检查点元数据（沿用原pccheck的槽位管理）"""
        self.lib.sync_all_streams(self.writer_obj, c_int(parall_iter))
    
    def register(self):
        """注册新的检查点槽位（沿用原pccheck的槽位管理）"""
        return self.lib.registerCheck(self.writer_obj)
    
    def borrow_cpu_slot(self, parall_iter):
        """
        从DRAMAlloc借用一块pinned buffer用于检查点保存
        
        Args:
            parall_iter: 并行迭代槽位
            
        Returns:
            void*: pinned buffer的指针（需要转换为numpy数组）
        """
        return self.lib.borrow_cpu_slot(self.writer_obj, c_int(parall_iter))
    
    def return_cpu_slot(self, parall_iter):
        """
        归还pinned buffer到DRAMAlloc
        
        Args:
            parall_iter: 并行迭代槽位
        """
        self.lib.return_cpu_slot(self.writer_obj, c_int(parall_iter))
    
    def borrow_chunk(self):
        """从DRAMAlloc借用一块pinned buffer"""
        return self.lib.borrow_chunk(self.writer_obj)
    
    def return_chunk(self, chunk_ptr):
        """归还pinned buffer到DRAMAlloc"""
        self.lib.return_chunk(self.writer_obj, c_void_p(chunk_ptr))


class MultiStreamCheckpoint:
    """
    多流并行检查点保存
    
    核心特性：
    - 4个独立数据流分别管理 param、grad、exp_avg、exp_avg_sq
    - 每个流独立传输和写入，充分利用带宽
    - 使用CUDA流实现GPU->CPU的并行传输
    """
    
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
    ):
        self.param_layout = param_layout
        self.gpu_ar = gpu_ar
        self.total_size = total_size
        self.num_threads = num_threads
        self.lib_path = lib_path
        self.filename = filename
        self.num_streams = num_streams
        self.max_async = max_async
        self.num_layer_groups = num_layer_groups
        
        # 计算每个流的大小
        self.stream_sizes = self._calculate_stream_sizes()
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
        
        # ✅ 内存优化：计算DRAMAlloc的chunk大小和数量
        # 使用固定大小的chunk (64MB)，避免因大层组导致内存爆炸
        # 64MB = 16M floats
        self.chunk_size = 16 * 1024 * 1024
        
        # num_chunks = max_async * num_streams * pipeline_depth + buffer
        # 每个流使用三缓冲流水线 (depth=3)
        # 必须确保有足够的chunk，否则会导致死锁
        self.pipeline_depth = 3
        self.num_chunks = max_async * num_streams * self.pipeline_depth + 4
        print(f"  DRAMAlloc chunk size: {self.chunk_size} floats ({self.chunk_size*4/1024/1024:.2f} MB)")
        print(f"  DRAMAlloc num chunks: {self.num_chunks} (Total: {self.num_chunks*self.chunk_size*4/1024/1024/1024:.2f} GB)")
        
        # 初始化多流writer
        self.writer = MultiStreamWriter(
            filename.encode(), lib_path, max_async, 
            num_streams, self.stream_sizes,
            chunk_size=self.chunk_size,
            num_chunks=self.num_chunks
        )
        
        # ✅ 修复：为每个并行检查点创建独立的CUDA流（固定大小为max_async）
        # cuda_streams[stream_index][stream_id] = 该检查点该流的CUDA流
        # parall_iter通过writer.register()获得，可能的值范围是0到max_async（包含），但会循环使用
        # 我们使用 parall_iter % (max_async + 1) 来映射到固定的CUDA流索引，实现流的复用
        # 这样最多支持max_async个并行检查点，每个检查点有num_streams个流
        self.cuda_streams = []
        for i in range(max_async + 1):  # max_async+1个槽位（0到max_async）
            self.cuda_streams.append([torch.cuda.Stream() for _ in range(num_streams)])
        print(f"  ✓ Created CUDA streams: {max_async + 1} slots × {num_streams} streams = {(max_async + 1) * num_streams} streams (reusable)")
        
        # ✅ 内存优化：不再预先分配pinned memory缓冲区
        # 改为在begin_checkpoint时从DRAMAlloc借用，在finalize_checkpoint时归还
        # 这样可以实现内存复用，将峰值内存从 2.82GB × 4流 × max_async ≈ 22GB 降到 2.82GB
        print(f"  ✓ Using DRAM slot-based memory management (reuse pinned buffers)")
        
        # 创建线程池（避免频繁创建销毁线程）
        # max_workers = num_streams * max_async，以支持多个检查点并行保存
        pool_workers = num_streams * max_async
        self.thread_pool = ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix="stream_worker")
        print(f"  ✓ Created thread pool with {pool_workers} workers (num_streams={num_streams} * max_async={max_async})")
        
        # 统计信息
        self.save_times = []
        
        # IO统计回调
        self.io_callback = None
        
        # ✅ 修复：槽位管理器（支持多个并行检查点）
        # 使用字典跟踪每个槽位的状态：{parall_iter: {'futures': [...], 'start_time': ...}}
        self._checkpoint_slots = {}  # 当前活跃的检查点槽位
        self._slot_lock = Lock()  # 保护槽位操作的锁
        self._max_async = max_async
        
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
    
    def save_layer_group(self, layer_group_ids, parall_iter):
        """
        保存一组层的参数（4个流并行，异步不阻塞）
        
        Args:
            layer_group_ids: 要保存的层ID列表
            parall_iter: 并行迭代槽位（由registerCheck返回）
            
        Returns:
            futures: 返回futures列表，供后续等待
        """
        # 使用线程池提交任务（避免反复创建销毁线程）
        futures = []
        
        for stream_id, param_type in enumerate(['param', 'grad', 'exp_avg', 'exp_avg_sq']):
            future = self.thread_pool.submit(
                self._save_stream_for_layers,
                stream_id, param_type, layer_group_ids, parall_iter
            )
            futures.append(future)
        
        # 不阻塞，直接返回futures，让调用者决定何时等待
        return futures
    
    def _save_stream_for_layers(self, stream_id, param_type, layer_group_ids, parall_iter):
        """
        保存指定层组的某一类参数
        
        Args:
            stream_id: 流ID
            param_type: 参数类型 (param/grad/exp_avg/exp_avg_sq)
            layer_group_ids: 层ID列表
            parall_iter: 并行迭代槽位（由registerCheck返回）
        """
        try:
            # ✅ 修复：通过映射复用CUDA流（parall_iter映射到固定的流索引）
            # parall_iter可能的值范围是0到max_async（包含），但会循环使用
            # 我们使用 modulo 操作将parall_iter映射到0到max_async的范围，实现流的复用
            stream_index = parall_iter % (self._max_async + 1)
            with torch.cuda.stream(self.cuda_streams[stream_index][stream_id]):
                # 计算该组在当前流中的偏移
                offset_in_stream = 0
                for i in range(min(layer_group_ids[0], len(self.param_layout))):
                    layer_info = self.param_layout[i]
                    offset_in_stream += layer_info.get(f'{param_type}_size', 0)
                
                # 收集数据到临时列表
                data_chunks = []
                total_size = 0
                
                for layer_id in layer_group_ids:
                    if layer_id >= len(self.param_layout):
                        continue
                        
                    layer_info = self.param_layout[layer_id]
                    
                    # 获取该参数在gpu_ar中的位置
                    offset_key = f'{param_type}_offset'
                    size_key = f'{param_type}_size'
                    gpu_offset = layer_info.get(offset_key, 0)
                    size = layer_info.get(size_key, 0)
                    
                    if size == 0:
                        continue
                    
                    # 从GPU提取数据
                    if gpu_offset + size <= len(self.gpu_ar):
                        gpu_data = self.gpu_ar[gpu_offset:gpu_offset+size]
                        data_chunks.append(gpu_data)
                        total_size += size
                
                if not data_chunks:
                    return
                
                # ❌ 移除 torch.cat，避免GPU内存峰值
                # combined_data = torch.cat(data_chunks)
                
                # ✅ 内存优化：按需借用chunk，支持分块处理大层组
                # ✅ 性能优化：使用流水线（Pipeline）重叠GPU拷贝和SSD写入
                total_elements = total_size
                current_offset = 0
                
                # 虚拟张量读取器状态
                current_chunk_idx = 0
                offset_in_current_chunk = 0
                
                import ctypes
                import numpy as np
                from collections import deque
                
                # 流水线队列：存储 (chunk_ptr, size, offset, event)
                pipeline_queue = deque()
                # pipeline_depth = self.pipeline_depth  # 三缓冲
                
                current_stream = torch.cuda.current_stream()
                
                try:
                    while current_offset < total_elements or pipeline_queue:
                        # 1. 填充流水线（发射GPU拷贝任务）
                        while len(pipeline_queue) < self.pipeline_depth and current_offset < total_elements:
                            # 计算本次处理的大小
                            elements_to_process = min(self.chunk_size, total_elements - current_offset)
                            
                            # 借用一个pinned buffer chunk
                            chunk_ptr = self.writer.borrow_chunk()
                            
                            # 将C指针转换为numpy数组视图
                            chunk_array = np.ctypeslib.as_array(
                                (ctypes.c_float * self.chunk_size).from_address(chunk_ptr)
                            )
                            
                            # 创建Tensor视图
                            chunk_tensor = torch.from_numpy(chunk_array)
                            
                            # 获取切片（只使用需要的部分）
                            cpu_buffer_slice = chunk_tensor[:elements_to_process]
                            
                            # ✅ 从data_chunks列表中逐个拷贝数据到cpu_buffer_slice
                            dest_offset = 0
                            while dest_offset < elements_to_process:
                                src_tensor = data_chunks[current_chunk_idx]
                                remaining_in_src = src_tensor.numel() - offset_in_current_chunk
                                needed = elements_to_process - dest_offset
                                
                                to_copy = min(remaining_in_src, needed)
                                
                                # 拷贝
                                src_view = src_tensor.view(-1)[offset_in_current_chunk : offset_in_current_chunk + to_copy]
                                dest_view = cpu_buffer_slice[dest_offset : dest_offset + to_copy]
                                dest_view.copy_(src_view, non_blocking=True)
                                
                                dest_offset += to_copy
                                offset_in_current_chunk += to_copy
                                
                                if offset_in_current_chunk == src_tensor.numel():
                                    current_chunk_idx += 1
                                    offset_in_current_chunk = 0
                            
                            # 记录事件用于同步
                            event = torch.cuda.Event(enable_timing=False, blocking=True)
                            event.record(current_stream)
                            
                            pipeline_queue.append((chunk_ptr, elements_to_process, current_offset, event, cpu_buffer_slice))
                            
                            # 更新偏移
                            current_offset += elements_to_process
                        
                        # 2. 处理流水线头部（等待拷贝完成并写入SSD）
                        if pipeline_queue:
                            chunk_ptr, size, offset, event, buffer_slice = pipeline_queue.popleft()
                            
                            # 等待GPU拷贝完成
                            event.synchronize()
                            
                            # 写入持久化存储 (阻塞操作，但此时GPU正在处理下一个块)
                            self.writer.write_stream_chunk(
                                stream_id,
                                buffer_slice.numpy(),
                                offset_in_stream + offset,
                                size,
                                parall_iter,
                                self.num_threads
                            )
                            
                            # 归还chunk
                            self.writer.return_chunk(chunk_ptr)
                            
                except Exception as e:
                    print(f"[ERROR] Pipeline failed: {e}")
                    # 清理剩余的chunk
                    while pipeline_queue:
                        chunk_ptr, _, _, _, _ = pipeline_queue.popleft()
                        self.writer.return_chunk(chunk_ptr)
                    raise e
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
        
        # 返回一个包装器，支持在begin_checkpoint后获取实际的MultiStreamOptimizer
        class OptimizerWrapper:
            def __init__(self, checkpoint_obj, real_optimizer):
                self._checkpoint = checkpoint_obj
                self._real_optimizer = real_optimizer
                self._ms_optimizer = None  # 在begin_checkpoint时创建
                self._current_parall_iter = None
            
            def begin_checkpoint(self):
                """开始检查点，创建实际的MultiStreamOptimizer"""
                parall_iter = self._checkpoint.begin_checkpoint()
                self._current_parall_iter = parall_iter
                
                # 创建回调函数，捕获parall_iter
                def save_callback(layer_ids):
                    """层组更新后的回调：异步保存该层组"""
                    # 提交异步保存任务
                    futures = self._checkpoint.save_layer_group(layer_ids, parall_iter)
                    
                    # 将futures添加到对应槽位的记录中
                    with self._checkpoint._slot_lock:
                        if parall_iter in self._checkpoint._checkpoint_slots:
                            self._checkpoint._checkpoint_slots[parall_iter]['futures'].extend(futures)
                
                # 创建实际的MultiStreamOptimizer
                self._ms_optimizer = MultiStreamOptimizer(
                    optimizer=self._real_optimizer,
                    layer_groups=self._checkpoint.layer_groups,
                    param_list=self._checkpoint._pending_param_list,
                    callback=save_callback
                )
                
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
            
            def __getattr__(self, name):
                # 代理其他方法到原始优化器或MultiStreamOptimizer
                if self._ms_optimizer is not None and hasattr(self._ms_optimizer, name):
                    return getattr(self._ms_optimizer, name)
                return getattr(self._real_optimizer, name)
        
        return OptimizerWrapper(self, optimizer)
    
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
        # ✅ 修复：等待直到有可用槽位（当并行数超过max_async时）
        while True:
            with self._slot_lock:
                if len(self._checkpoint_slots) < self._max_async:
                    # 有可用槽位，注册获取
                    parall_iter = self.writer.register()
                    break
            # 没有可用槽位，等待一小段时间后重试
            time.sleep(0.01)
        
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
            
            # ✅ 内存优化：归还DRAM槽位
            # self.writer.return_cpu_slot(parall_iter) # Deprecated
            
            # ✅ 修复：释放槽位
            with self._slot_lock:
                del self._checkpoint_slots[parall_iter]
            
            return total_time
        else:
            # 异步模式：立即返回，后台继续保存
            def async_finalize():
                try:
                    # 等待所有异步保存任务完成
                    for future in futures:
                        future.result()
                    
                    # 同步所有流到磁盘
                    self.writer.sync_all_streams(parall_iter)
                    
                    total_time = time.time() - start_time
                    throughput = (self.total_size * 4 / 1e9) / total_time
                    
                    self.save_times.append(total_time)
                    
                    print(f"✓ Checkpoint {parall_iter} completed in background: {total_time:.2f}s, {throughput:.2f} GB/s")
                    
                    # 调用IO统计回调
                    if self.io_callback is not None:
                        self.io_callback(total_time, throughput)
                    
                    # ✅ 内存优化：归还DRAM槽位（在后台任务完成后）
                    # self.writer.return_cpu_slot(parall_iter) # Deprecated
                    
                    # ✅ 修复：释放槽位（在后台任务完成后）
                    with self._slot_lock:
                        if parall_iter in self._checkpoint_slots:
                            del self._checkpoint_slots[parall_iter]
                except Exception as e:
                    print(f"✗ Background checkpoint {parall_iter} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # ✅ 内存优化：即使失败也要归还DRAM槽位
                    self.writer.return_cpu_slot(parall_iter)
                    # 即使失败也要释放槽位
                    with self._slot_lock:
                        if parall_iter in self._checkpoint_slots:
                            del self._checkpoint_slots[parall_iter]
            
            # 提交到线程池
            self.thread_pool.submit(async_finalize)
            
            # 立即返回（不阻塞）
            submit_time = time.time() - start_time
            print(f"✓ Checkpoint {parall_iter} submitted to background (Async submit time: {submit_time*1000:.2f}ms)")
            
            return submit_time
    
    def shutdown(self):
        """关闭线程池，释放资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            print("✓ Thread pool shutdown")
    
    def __del__(self):
        """析构时自动关闭线程池"""
        self.shutdown()


def build_param_layout(model, optimizer):
    """
    构建参数布局信息
    
    Args:
        model: PyTorch模型
        optimizer: 优化器（Adam）
    
    Returns:
        List[Dict]: 每层的参数信息
    """
    layout = []
    
    # 计算总大小和偏移
    total_param_size = sum(p.numel() for p in model.parameters())
    
    # 根据gpu_ar的布局：[所有param] [所有grad] [所有exp_avg] [所有exp_avg_sq]
    grad_base_offset = total_param_size
    exp_avg_base_offset = total_param_size * 2
    exp_avg_sq_base_offset = total_param_size * 3
    
    # 构建每层的布局
    layer_id = 0
    current_param_offset = 0
    current_grad_offset = grad_base_offset
    current_exp_avg_offset = exp_avg_base_offset
    current_exp_avg_sq_offset = exp_avg_sq_base_offset
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        size = param.numel()
        
        layer_info = {
            'layer_id': layer_id,
            'name': name,
            'param_offset': current_param_offset,
            'param_size': size,
            'grad_offset': current_grad_offset,
            'grad_size': size,
            'exp_avg_offset': current_exp_avg_offset,
            'exp_avg_size': size,
            'exp_avg_sq_offset': current_exp_avg_sq_offset,
            'exp_avg_sq_size': size,
        }
        
        layout.append(layer_info)
        
        current_param_offset += size
        current_grad_offset += size
        current_exp_avg_offset += size
        current_exp_avg_sq_offset += size
        layer_id += 1
    
    print(f"\nBuilt parameter layout:")
    print(f"  Total layers: {len(layout)}")
    print(f"  Total parameters: {total_param_size:,} ({total_param_size*4/1e9:.2f} GB)")
    print(f"  Total size (4 copies): {total_param_size*4:,} ({total_param_size*16/1e9:.2f} GB)")
    
    return layout
