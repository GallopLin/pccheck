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
import io
import random
import struct
import time
import zlib
from typing import Callable, List, Dict, Optional, Tuple
from collections import deque
import threading
from itertools import cycle
import os
import json


_MS_PROFILE_LOCK = Lock()

QUANT_BLOCK_DEFAULT = 2048  # moment int8 量化的默认 per-block 大小（1024 对齐的整数倍）


def _ms_profile_enabled() -> bool:
    return os.environ.get("PCCHECK_MS_PROFILE", "0") == "1"


def _ms_profile_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _ms_profile_event(event: str, **fields) -> None:
    """Append one JSONL profiling event when PCCHECK_MS_PROFILE=1."""
    if not _ms_profile_enabled():
        return

    output_path = os.environ.get("PCCHECK_MS_PROFILE_PATH")
    if not output_path:
        return

    record = {
        "ts": time.time(),
        "event": event,
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0),
        "run_id": os.environ.get("PCCHECK_MS_PROFILE_RUN_ID", ""),
    }
    record.update({key: _ms_profile_value(value) for key, value in fields.items()})

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with _MS_PROFILE_LOCK:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Profiling must never perturb checkpoint correctness.
        return

class MultiStreamOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        layer_groups,
        param_list: List[torch.nn.Parameter],
        callback: Optional[Callable[[int, list], None]] = None,
        batches=None,
        cover=None,
        param_layout=None,
    ):
        """
        Args:
            optimizer: 标准PyTorch优化器（如Adam, SGD）
            layer_groups: piece 格式分组 [[(lid,s,e),...],...]；兼容旧格式 [[lid,...],...]
            param_list: 模型参数列表（与 layer_id 对应）
            callback: 签名 callback(group_idx, pieces)，组内全部字节稳定后调用
            batches: 更新批次 [(lids, gidxs), ...]；None 时按"每组独立批"从分组推导
            cover: {lid: [gidx]} 覆盖表（防撕裂等待用）；None 时从分组推导
            param_layout: 旧格式归一化需要的布局（可选）
        """
        self.optimizer = optimizer
        self.param_list = param_list
        self.callback = callback

        layout = param_layout or [{'param_size': p.numel()} for p in param_list]
        self.pieces = [_norm_pieces(g, layout) for g in layer_groups]
        self.layer_groups = self.pieces  # 对外别名

        if cover is None:
            cover = {}
            for g, grp in enumerate(self.pieces):
                for (lid, _, _) in grp:
                    cover.setdefault(lid, []).append(g)
        self.cover = cover

        if batches is None:
            batches = []
            prev_tail = -1
            for g, grp in enumerate(self.pieces):
                if not grp:
                    continue
                tail = grp[-1][0]
                if batches and batches[-1][0] and batches[-1][0][-1] == tail:
                    batches[-1] = (batches[-1][0], batches[-1][1] + (g,))
                else:
                    batches.append((tuple(range(prev_tail + 1, tail + 1)), (g,)))
                    prev_tail = tail
        self.batches = batches

        # 每批要更新的参数（tensor 粒度，原子更新）
        self.batch_params = [
            [param_list[lid] for lid in lids if lid < len(param_list)]
            for (lids, _) in self.batches
        ]

        # 检测优化器类型
        self.optimizer_type = type(optimizer).__name__

        # ✅ 跨检查点同步：更新前的等待回调
        # 签名: pre_update_callback(group_idx_list) —— 上一检查点中覆盖本批
        # tensor 的组集合（张量内切分后一个 tensor 可能横跨多个组）。
        self.pre_update_callback = None

    def step_with_callback(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 按更新批次遍历：批 = (上一尾 tensor, 本尾 tensor]，批完成 → 触发以尾为界的组
        for batch_idx, ((lids, gidxs), params) in enumerate(zip(self.batches, self.batch_params)):
            t_pre = time.perf_counter()
            # ✅ 更新前回调：等待上一检查点所有覆盖本批 tensor 的组的拷贝完成
            if self.pre_update_callback is not None:
                wait_groups = sorted({g for lid in lids for g in self.cover.get(lid, ())})
                self.pre_update_callback(wait_groups)
            t_update = time.perf_counter()

            # 对这批参数执行真实的优化器更新（tensor 原子）
            self._step_param_group(params)
            t_cb = time.perf_counter()

            # 批内尾 tensor 更新完成 → 其字节全部稳定的组触发保存
            if self.callback is not None:
                for g in gidxs:
                    self.callback(g, self.pieces[g])
            if _ms_profile_enabled():
                _ms_profile_event(
                    "group_update",
                    group_idx=int(gidxs[0]) if gidxs else -1,
                    num_groups_triggered=int(len(gidxs)),
                    num_layers=int(len(lids)),
                    pre_wait_ms=(t_update - t_pre) * 1000.0,
                    update_enqueue_ms=(t_cb - t_update) * 1000.0,
                    save_trigger_ms=(time.perf_counter() - t_cb) * 1000.0,
                )

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

            # 注意：不要在此手动递增 step——torch 的 functional adam 内部会对
            # state_steps 自增一次（_foreach_add_(state_steps, 1)），此前的手动
            # +1 导致 checkpoint 迭代 step 双计、bias-correction 跑偏
            # （roundtrip 测试对账 step 时发现）。
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

            # 同 _step_adam：functional adamw 内部自增 state_steps，勿手动 +1
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
        """选择工作线程的亲和CPU列表。

        策略：
        1. 读取当前线程可用的 CPU 集合
        2. 移除 CPU 0（留给主训练线程）
        3. 如果当前线程亲和性过窄（例如被限制为 {0}），
           通过 NUMA 拓扑信息发现 GPU 所在 NUMA node 的其他核心。
           注意：Linux 下线程可以通过 sched_setaffinity 设置到进程
           亲和性范围之外的 CPU，无需修改进程级亲和性。

        布局设计（以 NUMA node0 = 0-23,96-119 为例）：
          CPU 0:      主训练线程（前向/反向/优化器）
          CPU 1-7:    copy_workers（GPU→CPU PCIe 拷贝，靠近 GPU）
          CPU 8-15:   ssd_workers（SSD I/O 密集型）
        """
        try:
            avail = sorted(os.sched_getaffinity(0))
        except Exception:
            avail = [0]

        # 移除 CPU 0（留给主训练线程）
        non_zero = [c for c in avail if c != 0]

        if len(non_zero) >= 2:
            # 当前线程已有足够的非零核心可用
            return non_zero

        # 亲和性过窄（通常是主线程绑定到 CPU 0），
        # 通过 NUMA 拓扑发现同节点的其他核心
        try:
            numa_path = '/sys/devices/system/node/node0/cpulist'
            if os.path.exists(numa_path):
                with open(numa_path) as f:
                    cpulist_str = f.read().strip()
                # 解析 "0-23,96-119" 格式
                numa_cpus = set()
                for part in cpulist_str.split(','):
                    if '-' in part:
                        lo, hi = part.split('-')
                        numa_cpus.update(range(int(lo), int(hi) + 1))
                    else:
                        numa_cpus.add(int(part))
                non_zero = sorted(c for c in numa_cpus if c != 0)
                if non_zero:
                    print(f"  [affinity] Worker CPUs from NUMA node0: "
                          f"{non_zero[:16]}{'...' if len(non_zero) > 16 else ''} "
                          f"({len(non_zero)} cores)")
                    return non_zero
        except Exception:
            pass

        # 兜底：使用 CPU 1-15
        fallback = list(range(1, 16))
        print(f"  [affinity] Using fallback worker CPUs: {fallback}")
        return fallback

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
        num_streams=4,          # 流的数量（param, master, exp_avg, exp_avg_sq）
        max_async=2,            # 最大异步检查点数
        num_layer_groups=8,     # 分成多少个层组
        distributed=False,
        rank=0,
        world_size=1,
        metadata_path: Optional[str] = None,
        master_accessor=None,   # 可选：param -> fp32 master tensor 的自定义解析器
        quant_moments=False,    # A 方案：stream2/3 融合 int8 量化（需 atomic 分组）
        quant_block=QUANT_BLOCK_DEFAULT,
    ):
        self.master_accessor = master_accessor
        # 量化标志：param_layout 若已带量化元数据则据此推断，否则用显式参数
        self.quant_moments = bool(quant_moments or (param_layout and param_layout[0].get('quant')))
        self.quant_block = int(param_layout[0].get('quant_block') or quant_block) if param_layout else quant_block
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

        # P1：非分层状态（RNG/LR scheduler/grad scaler/dataloader 位置/迭代号）
        # 作为数据流之后的 "extra" 流，随 slot 一起走 merged msync + CAS 发布
        # ⇒ "published slot 蕴含 blob 一致且耐久" 是结构性保证。
        # PCCHECK_EXTRA_FLOATS=0 可关闭；默认 1M floats = 4MB。
        self.extra_stream_floats = max(0, int(os.environ.get(
            "PCCHECK_EXTRA_FLOATS", str(1 << 20))))
        self.extra_stream_id = num_streams  # 紧随数据流
        self._extra_state_providers = {}
        self._extra_state_debug = {}       # parall_iter -> 最近写入的 blob 字节（供校验）

        # gpu_ar 可以是单个 tensor（legacy）或 dict（四块独立分配）。
        # 统一转为 self.gpu_buffers: List[Tensor]，长度 = num_streams。
        self.gpu_buffers = self._make_gpu_buffers(gpu_ar)
        self.gpu_ar = gpu_ar if not isinstance(gpu_ar, dict) else None
        print(f"\nMultiStreamCheckpoint Configuration:")
        print(f"  Total size: {total_size:,} floats ({total_size*4/1e9:.2f} GB)")
        print(f"  Num streams: {num_streams}")
        print(f"  Stream sizes:")
        for i, size in enumerate(self.stream_sizes):
            param_type = ['param', 'master', 'exp_avg', 'exp_avg_sq'][i]
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
        # extra 流追加到数据流之后：C++ init_streams 对流数通用，merged msync
        # 覆盖 total_checkpoint_size（含 extra）⇒ 无需任何 C++ 改动。
        writer_stream_sizes = list(self.stream_sizes)
        if self.extra_stream_floats > 0:
            writer_stream_sizes.append(self.extra_stream_floats)
        self.writer = MultiStreamWriter(
            fname_bytes,
            lib_path,
            max_async,
            len(writer_stream_sizes),
            writer_stream_sizes,
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
        cpu_count = os.cpu_count() or 16
        
        # GPU→CPU 拷贝线程池：每个 stream/async slot 一个 worker。
        # OPT-1.3B 单卡实测中继续加倍会和 SSD workers 竞争 CPU/I/O 调度资源。
        copy_pool_workers = max(1, num_streams * (max_async + 1))
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
        # metadata JSON 的读-改-写串行化（export_metadata / _update_latest_parall_iter
        # 分别在主线程与后台 finalize 线程并发访问同一文件）
        self._metadata_io_lock = Lock()
        self._profile_checkpoint_seq = 0
        self._profile_parall_to_checkpoint_id = {}
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
            # 计算该组中每个流的大小（piece 粒度，统一经 piece_extent 映射）
            group_stream_sizes = [0, 0, 0, 0]
            for (layer_id, ps, pe) in _norm_pieces(group, self.param_layout):
                if layer_id >= len(self.param_layout):
                    continue
                layer_info = self.param_layout[layer_id]
                for si, stream in enumerate(('param', 'master', 'exp_avg', 'exp_avg_sq')):
                    ext = piece_extent(layer_info, stream, ps, pe)
                    if ext is not None:
                        group_stream_sizes[si] += ext[1]

            # 取该组中最大的流大小
            max_size = max(max_size, max(group_stream_sizes))

        return max_size
        
    def _calculate_stream_sizes(self):
        """计算每个流的总大小（stream1 = master weight；无独立 master 时为 0）"""
        sizes = [0, 0, 0, 0]  # param, master, exp_avg, exp_avg_sq
        for layer_info in self.param_layout:
            sizes[0] += layer_info.get('param_size', 0)
            sizes[1] += layer_info.get('master_size', layer_info.get('grad_size', 0))
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
        _KEYS = ['param', 'master', 'exp_avg', 'exp_avg_sq']
        if isinstance(gpu_ar, dict):
            # stream1 兼容旧键名 'grad'；master 流为空时允许省略该 buffer
            def _pick(k):
                if k == 'master':
                    return gpu_ar.get('master', gpu_ar.get('grad'))
                return gpu_ar[k]
            return [self._detach_buf(_pick(k)) for k in _KEYS]
        # legacy: 使用 stream_sizes 精确拆分（兼容 total_size 不一致的场景）
        offset = 0
        buffers = []
        for sz in self.stream_sizes:
            buffers.append(self._detach_buf(gpu_ar[offset:offset + sz]))
            offset += sz
        return buffers
    
    def _create_layer_groups(self, ratios=None):
        """等分位字节区间分组（允许张量内切分，见 plan_groups）。

        分组是纯调度计划：落盘地址由 flat 布局决定、与分组无关，因此分组
        可随压缩比遥测（ratios）每次 checkpoint 重算而不影响 slot 布局兼容。
        """
        mode = os.environ.get('PCCHECK_GROUPING', 'split').strip().lower()
        if mode not in ('split', 'atomic'):
            mode = 'split'
        # v3 交错量化布局下 split 与 quant 可组合：刀口 snap 到量化 block 边界
        qb = int(getattr(self, 'quant_block', 0)) if getattr(self, 'quant_moments', False) else 0
        if qb and mode == 'split':
            print(f"  [quant] split+quant: cuts snapped to tensor-local block boundaries (block={qb})")
        self.group_plan = plan_groups(
            self.param_layout, self.num_layer_groups,
            ratios=ratios, align_elems=1024, mode=mode, quant_block=qb)
        pieces = self.group_plan["pieces"]

        sizes = [li.get('param_size', 0) for li in self.param_layout]
        group_bytes = [sum((e - s) for (_, s, e) in grp) * 4 for grp in pieces]
        total_bytes = max(sum(group_bytes), 1)
        n_split = sum(1 for grp in pieces for (lid, s, e) in grp
                      if s != 0 or e != sizes[lid])
        print(f"  Layer-group [{mode}] bytes (MB): "
              + ", ".join(f"{b/1e6:.1f}" for b in group_bytes)
              + f" | imbalance max/mean = "
                f"{max(group_bytes) * len(group_bytes) / total_bytes:.3f}"
              + f" | intra-tensor splits = {n_split}")

        return pieces

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
                        chunk_groups: Optional[List[List[int]]] = None,
                        optimizer_steps_override: Optional[Dict[str, int]] = None,
                        param_groups_override: Optional[List[Dict]] = None) -> str:
        """
        导出 multistream 检查点元数据，供流水加载使用。

        Args:
            output_path: 元数据文件路径；默认使用 self.metadata_path。
            optimizer: 可选，用于导出 optimizer param_groups 与 step。
            chunk_groups: 可选自定义 layer group 划分；默认使用 self.layer_groups。
            optimizer_steps_override / param_groups_override: 可选，主线程在
                checkpoint 提交点取好的快照。提供后本函数不再读 optimizer 活状态，
                可安全下沉到后台线程执行（否则并发的下一迭代 update 会污染
                steps/lr，导致 metadata 与该 checkpoint 语义错位一步）。
        """
        if output_path is None:
            output_path = self.metadata_path
        if output_path is None:
            raise ValueError("metadata_path is not set; please pass output_path explicitly.")

        stream_names = ["param", "master", "exp_avg", "exp_avg_sq"]  # 数据流（chunk 粒度）
        extra_floats = int(getattr(self, "extra_stream_floats", 0) or 0)
        # 顶层流表包含 extra 流：stride/偏移与 C++ init_streams 的实际布局一致
        full_stream_names = stream_names + (["extra"] if extra_floats > 0 else [])
        full_stream_sizes = [int(x) for x in self.stream_sizes] + \
                            ([extra_floats] if extra_floats > 0 else [])
        total_checkpoint_size = sum(full_stream_sizes)
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
                "master_offset": int(layer_info.get("master_offset", layer_info.get("grad_offset", 0))),
                "master_size": int(layer_info.get("master_size", layer_info.get("grad_size", 0))),
                "exp_avg_offset": int(layer_info.get("exp_avg_offset", 0)),
                "exp_avg_size": int(layer_info.get("exp_avg_size", 0)),
                "exp_avg_sq_offset": int(layer_info.get("exp_avg_sq_offset", 0)),
                "exp_avg_sq_size": int(layer_info.get("exp_avg_sq_size", 0)),
            }

        chunk_groups = chunk_groups or self.layer_groups
        chunks = {}
        for group_idx, group in enumerate(chunk_groups):
            params = []
            stream_start = {name: None for name in stream_names}
            stream_end = {name: None for name in stream_names}

            for (layer_id, ps, pe) in _norm_pieces(group, self.param_layout):
                if layer_id >= len(self.param_layout):
                    continue
                info = self.param_layout[layer_id]
                name = info.get("name")
                if not name:
                    continue
                q = bool(info.get("quant"))
                # 本 piece 在各流内的偏移与长度：统一经 piece_extent 映射
                # （master 流为空则 size=0；量化 moment 流为 block 交错单元）
                offsets, sizes = {}, {}
                for stream in stream_names:
                    ext = piece_extent(info, stream, ps, pe)
                    offsets[stream] = int(ext[0]) if ext is not None else 0
                    sizes[stream] = int(ext[1]) if ext is not None else 0
                for stream in stream_names:
                    if sizes[stream] <= 0:
                        continue
                    if stream_start[stream] is None:
                        stream_start[stream] = offsets[stream]
                        stream_end[stream] = offsets[stream] + sizes[stream]
                    else:
                        stream_start[stream] = min(stream_start[stream], offsets[stream])
                        stream_end[stream] = max(stream_end[stream], offsets[stream] + sizes[stream])

                pentry = {
                    "name": name,
                    "numel": int(info.get("param_size", 0)),
                    "elem_range": [int(ps), int(pe)],
                    "offsets": offsets,
                    "sizes": sizes,
                }
                if q:
                    pentry["quant"] = {
                        "format": "interleaved-block",  # v3: 每 block [int8*B | fp32 scale]
                        "block": int(info.get("quant_block", 0)),
                        "numel": int(info.get("q_numel", 0)),
                        "nblocks": int(info.get("q_nblocks", 0)),
                        "unit_floats": int(info.get("q_unit_floats", 0)),
                        "sqrt_domain": {"exp_avg": False, "exp_avg_sq": True},
                    }
                params.append(pentry)

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

            norm = _norm_pieces(group, self.param_layout)
            chunks[str(group_idx)] = {
                "group_idx": int(group_idx),
                "layer_ids": sorted({int(lid) for (lid, _, _) in norm}),
                "pieces": [[int(lid), int(s), int(e)] for (lid, s, e) in norm],
                "params": params,
                "stream_slices": stream_slices,
            }

        # optimizer metadata（override 命中则不读 optimizer 活状态，见 docstring）
        param_name_to_param = {}
        if hasattr(self, "_pending_model") and self._pending_model is not None:
            param_name_to_param = {name: p for name, p in self._pending_model.named_parameters()}
        if param_groups_override is not None:
            optimizer_param_groups = param_groups_override
        else:
            optimizer_param_groups = self._serialize_optimizer_param_groups(optimizer, param_name_to_param) if optimizer else None
        if optimizer_steps_override is not None:
            optimizer_steps = optimizer_steps_override
        else:
            optimizer_steps = self._serialize_optimizer_steps(optimizer, param_name_to_param) if optimizer else None

        checkpoint_file = self.filename
        if isinstance(checkpoint_file, (bytes, bytearray)):
            try:
                checkpoint_file = checkpoint_file.decode("utf-8")
            except Exception:
                checkpoint_file = None

        metadata = {
            "format": "multistream",
            "version": 3,  # v3: 交错量化布局（split+quant 可组合）+ extra 非分层状态流
                           # （v2: stream1 语义从 grad 改为 master weight）
            # 分组是纯调度：落盘地址由 layout 决定；cuts 记录本计划的切割点（全局元素坐标）
            "grouping": {
                "align_elems": 1024,
                "cuts": [int(c) for c in getattr(self, 'group_plan', {}).get('cuts', [])],
            },
            "quant_moments": {
                "enabled": bool(getattr(self, 'quant_moments', False)),
                "block": int(getattr(self, 'quant_block', 0)),
                "format": "per-block symmetric int8, interleaved [int8*B|fp32 scale] "
                          "units; exp_avg_sq in sqrt domain",
            },
            "extra_state": {
                "enabled": extra_floats > 0,
                "stream_id": int(getattr(self, "extra_stream_id", len(stream_names))),
                "floats": extra_floats,
                "format": "CSXS v1: <4sIQI magic,version,payload_len,crc32> + torch.save dict",
            },
            "dtype": "float32",
            "checkpoint_file": checkpoint_file,
            "max_async": int(self.max_async),
            "num_streams": len(full_stream_names),
            "stream_names": full_stream_names,
            "stream_sizes": full_stream_sizes,
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
        # 原子写（tmp + os.replace）+ 进程内锁：export（主线程逐 checkpoint 调用）与
        # _update_latest_parall_iter（后台 finalize 线程）并发读写同一 JSON，
        # 直接覆写会让并发读者（含跨进程的 pipelayer 恢复端）读到半截文件。
        with self._metadata_io_lock:
            tmp_path = output_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_path)

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
            # 与 export_metadata 共享同一把锁做读-改-写串行化；原子替换防半截读。
            with self._metadata_io_lock:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata["latest_parall_iter"] = int(parall_iter)
                # 兼容旧 metadata：optimizer_steps 可能在初始化导出后一直为 0，
                # 这会在恢复时导致 AdamW bias_correction 分母为 0。
                steps = metadata.get("optimizer_steps")
                if isinstance(steps, dict) and steps:
                    try:
                        if all(int(v) <= 0 for v in steps.values()):
                            metadata["optimizer_steps"] = {k: int(parall_iter) for k in steps.keys()}
                    except Exception:
                        pass
                tmp_path = self.metadata_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.metadata_path)
        except Exception as e:
            print(f"[WARN] Failed to update latest_parall_iter in metadata: {e}")

    # ================== 非分层状态（extra state）blob ==================
    # blob 格式（CSXS v1）：
    #   header  <4sIQI> = magic b"CSXS" | version(u32) | payload_len(u64) | crc32(u32)
    #   payload torch.save 序列化的 dict
    # 写入 extra 流（slot 内数据流之后），与张量数据同 slot 原子发布。
    EXTRA_MAGIC = b"CSXS"
    EXTRA_VERSION = 1
    _EXTRA_HEADER = "<4sIQI"

    def attach_extra_state(self, scheduler=None, scaler=None, provider=None):
        """注册非分层状态来源；每次 finalize_checkpoint 在主训练线程自动捕获。

        Args:
            scheduler: LR scheduler（保存 state_dict()）
            scaler:    torch.cuda.amp.GradScaler（保存 state_dict()）
            provider:  可调用，返回 dict——应用自定义状态（dataloader 位置、
                       epoch/迭代号等）。系统保证原子持久化，状态语义由应用定义。
        """
        if scheduler is not None:
            self._extra_state_providers["scheduler"] = scheduler
        if scaler is not None:
            self._extra_state_providers["scaler"] = scaler
        if provider is not None:
            self._extra_state_providers["provider"] = provider

    def _capture_extra_state(self, parall_iter):
        """在主训练线程捕获非分层状态并序列化为 blob 字节；失败/超容量返回 None。

        必须在 step_with_callback 返回后、训练线程继续前进前调用——异步 finalize
        跑起来时 RNG 已被后续迭代污染，捕获时机不可下沉到后台线程。
        """
        if self.extra_stream_floats <= 0:
            return None
        try:
            state = {
                "parall_iter": int(parall_iter),
                "seq": int(self._profile_checkpoint_seq),
                "wall_time": time.time(),
                "rng": {
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else []),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }
            sch = self._extra_state_providers.get("scheduler")
            if sch is not None:
                state["lr_scheduler"] = sch.state_dict()
            sc = self._extra_state_providers.get("scaler")
            if sc is not None:
                state["grad_scaler"] = sc.state_dict()
            fn = self._extra_state_providers.get("provider")
            if fn is not None:
                state["user"] = fn()
            # 真实 optimizer step（metadata 的 optimizer_steps 存在"用槽号回填"的
            # 历史 kludge，不可靠；恢复端优先使用 blob 中的该值）
            opt = getattr(self, "_pending_optimizer", None)
            if opt is not None:
                for st in opt.state.values():
                    sv = st.get("step")
                    if sv is not None:
                        state["optimizer_step"] = float(
                            sv.item() if isinstance(sv, torch.Tensor) else sv)
                        break
            buf = io.BytesIO()
            torch.save(state, buf)
            payload = buf.getvalue()
            header = struct.pack(self._EXTRA_HEADER, self.EXTRA_MAGIC,
                                 self.EXTRA_VERSION, len(payload),
                                 zlib.crc32(payload) & 0xFFFFFFFF)
            blob = header + payload
            cap = self.extra_stream_floats * 4
            if len(blob) > cap:
                print(f"[WARN] extra-state blob {len(blob)}B > extra stream capacity "
                      f"{cap}B; skipped (increase PCCHECK_EXTRA_FLOATS)")
                return None
            if len(blob) % 4:
                blob += b"\x00" * (4 - len(blob) % 4)
            return blob
        except Exception as e:
            print(f"[WARN] extra-state capture failed: {e}")
            return None

    def _write_extra_state(self, parall_iter, blob):
        """把 blob 写入本 slot 的 extra 流（mmap memcpy，须在 sync_all_streams 之前）。"""
        if not blob:
            return
        arr = np.frombuffer(blob, dtype=np.float32).copy()  # as_ctypes 需要可写数组
        self.writer.write_stream_chunk(
            self.extra_stream_id, arr, 0, len(arr), parall_iter, 1)
        self._extra_state_debug[parall_iter] = blob

    @classmethod
    def parse_extra_state_blob(cls, raw_bytes):
        """解析 extra 流原始字节 → (state_dict, payload_len)；校验 magic/crc。"""
        hsize = struct.calcsize(cls._EXTRA_HEADER)
        magic, version, plen, crc = struct.unpack(cls._EXTRA_HEADER, raw_bytes[:hsize])
        if magic != cls.EXTRA_MAGIC:
            raise ValueError(f"extra-state blob bad magic: {magic!r}")
        payload = raw_bytes[hsize:hsize + plen]
        if len(payload) != plen:
            raise ValueError(f"extra-state blob truncated: {len(payload)} < {plen}")
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if actual != crc:
            raise ValueError(f"extra-state blob crc mismatch: {actual:#x} != {crc:#x}")
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        return state, hsize + plen

    def save_layer_group(self, group_pieces, parall_iter, group_idx, sync_event=None):
        """
        保存一个组（4个流并行，异步不阻塞）

        Args:
            group_pieces: piece 列表 [(lid, elem_start, elem_end), ...]；兼容旧 [lid, ...]
            parall_iter: 并行迭代槽位（由registerCheck返回）
            group_idx: 层组索引，用于跨检查点同步
            sync_event: CUDA同步事件，用于确保参数更新完成后再开始拷贝

        Returns:
            futures: 返回futures列表，供后续等待
        """
        group_pieces = _norm_pieces(group_pieces, self.param_layout)
        # ✅ 创建 4 个拷贝完成事件，每个流一个
        copy_done_events = [torch.cuda.Event() for _ in range(4)]
        
        # 使用线程池提交任务（避免反复创建销毁线程）
        futures = []
        
        for stream_id, param_type in enumerate(['param', 'master', 'exp_avg', 'exp_avg_sq']):
            future = self.writer._submit_with_affinity(
                self.copy_thread_pool,
                self._save_stream_for_layers,
                stream_id, param_type, group_pieces, parall_iter, sync_event,
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
    
    def _save_stream_for_layers(self, stream_id, param_type, group_pieces, parall_iter, sync_event=None, copy_done_event=None, group_idx=None):
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
            checkpoint_id = self._profile_parall_to_checkpoint_id.get(parall_iter, -1)

            with torch.cuda.stream(current_stream):
                # 等待默认流上的参数更新完成
                if sync_event is not None:
                    sync_event.wait(current_stream)
                
                # piece 语义：每段的流内区间统一经 piece_extent 映射（layout 静态、
                # 分组动态；量化 moment 流为 block 交错单元），并校验组内连续。
                layers_to_process = []
                total_elements = 0
                offset_in_stream = None
                gpu_buffer = self.gpu_buffers[stream_id]

                for (layer_id, ps, pe) in group_pieces:
                    if layer_id >= len(self.param_layout) or gpu_buffer is None:
                        continue
                    if pe <= ps:
                        continue
                    ext = piece_extent(self.param_layout[layer_id], param_type, ps, pe)
                    if ext is None:
                        continue
                    gpu_offset, size = ext
                    if size <= 0 or gpu_offset + size > len(gpu_buffer):
                        continue
                    if offset_in_stream is None:
                        offset_in_stream = gpu_offset
                    else:
                        expected = offset_in_stream + total_elements
                        assert gpu_offset == expected, (
                            f"group {group_idx} stream {stream_id} pieces not contiguous: "
                            f"expect offset {expected}, got {gpu_offset}")
                    layers_to_process.append((gpu_offset, size))
                    total_elements += size
                if offset_in_stream is None:
                    offset_in_stream = 0
                
                if not layers_to_process:
                    # 即使没有数据，也要记录事件（保持一致性）
                    if copy_done_event is not None:
                        copy_done_event.record(current_stream)
                    return

                # ========== 阶段2：异步写入 SSD ==========
                # ✅ 供“按 chunk 流式提交”使用：避免 pinned chunk 池不足导致 borrow_chunk 阻塞/死锁（H6）
                def ssd_write_task(chunks, stream_id, offset_in_stream, parall_iter, num_threads, writer):
                    task_start = time.perf_counter()
                    task_bytes = 0
                    try:
                        for chunk_ptr, chunk_size, offset, d2h_event, d2h_meta in chunks:
                            # ✅ 事件化流水：写盘前仅等待本 chunk 的 D2H 完成事件。
                            # cudaEventSynchronize 线程安全；等待下沉到 SSD 线程后，
                            # 拷贝线程可背靠背入队后续 chunk 的 D2H。
                            d2h_wait_ms = 0.0
                            if d2h_event is not None:
                                wait_start = time.perf_counter()
                                d2h_event.synchronize()
                                d2h_wait_ms = (time.perf_counter() - wait_start) * 1000.0
                            if d2h_meta is not None:
                                d2h_cuda_ms = None
                                se = d2h_meta.get("start_ev")
                                ee = d2h_meta.get("end_ev")
                                if se is not None and ee is not None:
                                    try:
                                        d2h_cuda_ms = float(se.elapsed_time(ee))
                                    except Exception:
                                        d2h_cuda_ms = None
                                _ms_profile_event(
                                    "d2h_chunk",
                                    checkpoint_id=int(checkpoint_id),
                                    parall_iter=int(parall_iter),
                                    stream_id=int(stream_id),
                                    group_idx=int(group_idx) if group_idx is not None else -1,
                                    chunk_offset_floats=int(d2h_meta.get("chunk_offset", offset)),
                                    chunk_floats=int(chunk_size),
                                    bytes=int(chunk_size) * 4,
                                    borrow_wait_ms=d2h_meta.get("borrow_ms"),
                                    d2h_enqueue_cpu_ms=d2h_meta.get("enqueue_ms"),
                                    d2h_cuda_ms=d2h_cuda_ms,
                                    d2h_event_wait_ms=d2h_wait_ms,
                                )
                            float_ptr = cast(chunk_ptr, POINTER(c_float))
                            write_start = time.perf_counter()
                            writer.write_stream_chunk(
                                stream_id,
                                float_ptr,
                                offset_in_stream + offset,
                                chunk_size,
                                parall_iter,
                                num_threads,
                            )
                            write_ms = (time.perf_counter() - write_start) * 1000.0
                            chunk_bytes = int(chunk_size) * 4
                            task_bytes += chunk_bytes
                            _ms_profile_event(
                                "ssd_write_chunk",
                                checkpoint_id=int(checkpoint_id),
                                parall_iter=int(parall_iter),
                                stream_id=int(stream_id),
                                group_idx=int(group_idx) if group_idx is not None else -1,
                                offset_floats=int(offset_in_stream + offset),
                                chunk_floats=int(chunk_size),
                                bytes=chunk_bytes,
                                writer_call_ms=write_ms,
                            )
                    except Exception as e:
                        print(f"[ERROR] SSD write failed for stream {stream_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        for chunk_ptr, _, _, _, _ in chunks:
                            writer.return_chunk(chunk_ptr)
                        with self._inflight_ssd_lock:
                            self._inflight_ssd_tasks -= 1
                            inflight_after = int(self._inflight_ssd_tasks)
                        _ms_profile_event(
                            "ssd_task",
                            checkpoint_id=int(checkpoint_id),
                            parall_iter=int(parall_iter),
                            stream_id=int(stream_id),
                            group_idx=int(group_idx) if group_idx is not None else -1,
                            chunks=int(len(chunks)),
                            bytes=int(task_bytes),
                            total_ms=(time.perf_counter() - task_start) * 1000.0,
                            inflight_after=inflight_after,
                        )

                def _submit_ssd(chunks):
                    with self._inflight_ssd_lock:
                        self._inflight_ssd_tasks += 1
                        inflight_before_submit = int(self._inflight_ssd_tasks)
                    submit_start = time.perf_counter()
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
                    _ms_profile_event(
                        "ssd_submit",
                        checkpoint_id=int(checkpoint_id),
                        parall_iter=int(parall_iter),
                        stream_id=int(stream_id),
                        group_idx=int(group_idx) if group_idx is not None else -1,
                        chunks=int(len(chunks)),
                        bytes=int(sum(c[1] for c in chunks) * 4),
                        submit_ms=(time.perf_counter() - submit_start) * 1000.0,
                        inflight_before_submit=inflight_before_submit,
                    )

                # ========== 阶段1：GPU→CPU 拷贝（按 chunk 流式提交 SSD 写入） ==========
                current_offset = 0
                current_layer_idx = 0
                offset_in_current_layer = 0

                while current_offset < total_elements:
                    elements_to_process = min(self.chunk_size, total_elements - current_offset)

                    borrow_start = time.perf_counter()
                    chunk_ptr = self.writer.borrow_chunk()
                    borrow_ms = (time.perf_counter() - borrow_start) * 1000.0
                    
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
                    d2h_start_event = None
                    d2h_end_event = None
                    if _ms_profile_enabled():
                        d2h_start_event = torch.cuda.Event(enable_timing=True)
                        d2h_end_event = torch.cuda.Event(enable_timing=True)
                        d2h_start_event.record(current_stream)
                    enqueue_start = time.perf_counter()
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
                    if d2h_end_event is not None:
                        d2h_end_event.record(current_stream)
                    d2h_enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0

                    chunk_offset = current_offset
                    current_offset += elements_to_process

                    # ✅ 事件化流水：不再对整条流做阻塞 synchronize（旧实现导致
                    # 拷贝线程每 chunk 睡一次、pinned chunk 空闲也无人消费）。
                    # 记录该 chunk 的 D2H 完成事件后立即继续下一 chunk；
                    # SSD 线程在写盘前等待该事件，保证数据就绪。
                    d2h_done_event = torch.cuda.Event()
                    d2h_done_event.record(current_stream)
                    d2h_meta = None
                    if _ms_profile_enabled():
                        d2h_meta = {
                            "chunk_offset": chunk_offset,
                            "borrow_ms": borrow_ms,
                            "enqueue_ms": d2h_enqueue_ms,
                            "start_ev": d2h_start_event,
                            "end_ev": d2h_end_event,
                        }
                    _submit_ssd([(chunk_ptr, elements_to_process, chunk_offset,
                                  d2h_done_event, d2h_meta)])

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
                
                def pre_update_callback(group_idxs):
                    # 一致性要求：更新本批 tensor 前，必须等待上一检查点所有
                    # 覆盖这些 tensor 的组（张量内切分后可能不止一个）的全部
                    # 4 个状态流 D2H 完成，否则本次 update/D2D 会在上一检查点
                    # 仍从 gpu_buffers 读取时覆写源数据，导致动量流撕裂。
                    # wait_event 为 GPU 流侧等待，不阻塞 CPU。
                    for group_idx in group_idxs:
                        prev_events = prev_checkpoint_events.get(group_idx)
                        if prev_events is not None:
                            for event in prev_events:
                                torch.cuda.current_stream().wait_event(event)
                
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
                def save_callback(group_idx, group_pieces):
                    if is_zero_copy_enabled:
                        # 🔥 零拷贝优化：参数已经在 gpu_ar 上，无需拷贝
                        # 直接在默认流上记录事件，标记参数更新已完成
                        sync_event = torch.cuda.Event()
                        sync_event.record(torch.cuda.current_stream())
                    else:
                        # 常规模式：在拷贝流上执行 D2D 拷贝
                        # ⚠️ 必须在进入 copy_stream 上下文之前捕获生产者流：
                        # 旧代码在 with 块内调用 current_stream()，取到的是
                        # copy_stream 自身，自等待为空操作 → D2D 与默认流上的
                        # update kernel / pre-update 屏障之间没有顺序约束，
                        # 会读到更新前的旧值（逐位内容校验实测抓到该竞态）。
                        producer_stream = torch.cuda.current_stream()
                        with torch.cuda.stream(copy_stream):
                            # 等待生产者流（默认流）上的参数更新完成
                            copy_stream.wait_stream(producer_stream)
                            
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
                                for (layer_id, ps, pe) in group_pieces:
                                    if layer_id >= len(self._checkpoint.param_layout):
                                        continue
                                    li = self._checkpoint.param_layout[layer_id]
                                    param = self._checkpoint._pending_param_list[layer_id]
                                    seg = pe - ps
                                    if seg <= 0:
                                        continue

                                    off = li["param_offset"] + ps
                                    bufs[0][off : off + seg].copy_(
                                        param.detach().view(-1)[ps:pe].float(), non_blocking=True)

                                    state = self._real_optimizer.state.get(param, {})

                                    # stream1 = 独立 fp32 master weight（必须数据；
                                    # 纯 fp32 训练时 master_size=0，此流不写任何字节。
                                    # 梯度不再保存：resume 后可由反向重算，非必须。）
                                    msz = li.get("master_size", 0)
                                    if msz > 0 and bufs[1] is not None:
                                        moff = li.get("master_offset", 0) + ps
                                        m = _resolve_master_weight(
                                            param, self._real_optimizer,
                                            getattr(self._checkpoint, "master_accessor", None))
                                        if m is None:
                                            bufs[1][moff : moff + seg].zero_()
                                        else:
                                            bufs[1][moff : moff + seg].copy_(
                                                m.detach().view(-1)[ps:pe].float(), non_blocking=True)

                                    q_moments = getattr(self._checkpoint, 'quant_moments', False)
                                    q_block = getattr(self._checkpoint, 'quant_block', QUANT_BLOCK_DEFAULT)

                                    ea = state.get("exp_avg")
                                    if q_moments:
                                        # 融合量化（v3 交错布局）：piece 粒度，切口 block 对齐
                                        # ⇒ 分段量化与整张量量化逐位一致
                                        if ea is None:
                                            off, sz = piece_extent(li, 'exp_avg', ps, pe)
                                            bufs[2][off : off + sz].zero_()
                                        else:
                                            _pack_moment_int8(bufs[2], li["exp_avg_offset"],
                                                              ea.detach().view(-1), q_block,
                                                              sqrt=False, ps=ps, pe=pe)
                                    else:
                                        ea_off = li["exp_avg_offset"] + ps
                                        if ea is None:
                                            bufs[2][ea_off : ea_off + seg].zero_()
                                        else:
                                            bufs[2][ea_off : ea_off + seg].copy_(
                                                ea.detach().view(-1)[ps:pe].float(), non_blocking=True)

                                    eas = state.get("exp_avg_sq")
                                    if q_moments:
                                        if eas is None:
                                            off, sz = piece_extent(li, 'exp_avg_sq', ps, pe)
                                            bufs[3][off : off + sz].zero_()
                                        else:
                                            _pack_moment_int8(bufs[3], li["exp_avg_sq_offset"],
                                                              eas.detach().view(-1), q_block,
                                                              sqrt=True, ps=ps, pe=pe)
                                    else:
                                        eas_off = li["exp_avg_sq_offset"] + ps
                                        if eas is None:
                                            bufs[3][eas_off : eas_off + seg].zero_()
                                        else:
                                            bufs[3][eas_off : eas_off + seg].copy_(
                                                eas.detach().view(-1)[ps:pe].float(), non_blocking=True)

                            # ✅ 在拷贝流上记录事件，标记该层组的参数拷贝已完成
                            sync_event = torch.cuda.Event()
                            sync_event.record(copy_stream)
                        
                    # 提交异步保存任务，传递同步事件和group_idx
                    futures = self._checkpoint.save_layer_group(group_pieces, parall_iter, group_idx, sync_event)
                    
                    # 将futures添加到对应槽位的记录中
                    self._checkpoint._register_layer_group_futures(parall_iter, futures)
                
                # 创建实际的MultiStreamOptimizer（批次/覆盖表来自分组计划）
                plan = getattr(self._checkpoint, 'group_plan', None) or {}
                self._ms_optimizer = MultiStreamOptimizer(
                    optimizer=self._real_optimizer,
                    layer_groups=self._checkpoint.layer_groups,
                    param_list=self._checkpoint._pending_param_list,
                    callback=save_callback,
                    batches=plan.get('batches'),
                    cover=plan.get('cover'),
                    param_layout=self._checkpoint.param_layout,
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
        wait_start = time.perf_counter()
        while True:
            self._slot_available_event.wait()  # 阻塞直到收到槽位释放通知
            with self._slot_lock:
                if len(self._checkpoint_slots) < self._max_async:
                    # 有可用槽位，注册获取
                    parall_iter = self.writer.register()
                    checkpoint_id = self._profile_checkpoint_seq
                    self._profile_checkpoint_seq += 1
                    self._profile_parall_to_checkpoint_id[parall_iter] = checkpoint_id
                    # 如果槽位已满，清除事件，让下次 begin_checkpoint 等待
                    if len(self._checkpoint_slots) + 1 >= self._max_async:
                        self._slot_available_event.clear()
                    break
        _ms_profile_event(
            "begin_checkpoint",
            checkpoint_id=int(checkpoint_id),
            parall_iter=int(parall_iter),
            active_slots=int(len(self._checkpoint_slots) + 1),
            max_async=int(self._max_async),
            slot_wait_ms=(time.perf_counter() - wait_start) * 1000.0,
        )
        
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
        # P1：非分层状态必须在此刻（主训练线程、优化器更新刚完成）捕获；
        # 序列化后的字节交给后台在 sync_all_streams 之前写入 extra 流。
        extra_blob = self._capture_extra_state(parall_iter)

        # ✅ 修复：使用传入的parall_iter，从槽位记录中获取信息
        with self._slot_lock:
            if parall_iter not in self._checkpoint_slots:
                print(f"Warning: Checkpoint {parall_iter} not found in active slots")
                return None
            slot_info = self._checkpoint_slots[parall_iter]
            futures = slot_info['futures']
            start_time = slot_info['start_time']
            checkpoint_id = self._profile_parall_to_checkpoint_id.get(parall_iter, -1)

        with self._inflight_ssd_lock:
            inflight_ssd = int(self._inflight_ssd_tasks)
            
        if wait:
            # 同步模式：等待所有保存完成
            # 等待所有异步保存任务完成
            print(f"\n  Waiting for all layer groups to complete (parall_iter: {parall_iter})...")
            for future in futures:
                future.result()

            # extra 流写入必须先于 sync_all_streams（merged msync + CAS 是提交点）
            self._write_extra_state(parall_iter, extra_blob)

            # 同步所有流到磁盘
            print(f"\n  Syncing all streams to disk (parall_iter: {parall_iter})...")
            sync_start = time.time()
            self.writer.sync_all_streams(parall_iter)
            sync_time = time.time() - sync_start
            
            total_time = time.time() - start_time
            throughput = (self.total_size * 4 / 1e9) / total_time
            
            self.save_times.append(total_time)
            _ms_profile_event(
                "checkpoint_finalize",
                checkpoint_id=int(checkpoint_id),
                parall_iter=int(parall_iter),
                wait=True,
                total_ms=float(total_time) * 1000.0,
                sync_ms=float(sync_time) * 1000.0,
                throughput_gbps=float(throughput),
            )
            
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
                self._profile_parall_to_checkpoint_id.pop(parall_iter, None)
            self._slot_available_event.set()  # 通知等待的 begin_checkpoint
            
            return total_time
        else:
            # 异步模式：立即返回，后台继续保存
            def async_finalize():
                try:
                    # 等待所有异步保存任务完成（GPU→CPU拷贝）
                    for future in futures:
                        future.result()

                    # extra 流写入必须先于 sync_all_streams（提交点）；
                    # blob 字节已在主线程捕获，此处仅 memcpy 到 mmap。
                    self._write_extra_state(parall_iter, extra_blob)

                    # 同步所有流到磁盘
                    sync_streams_start = time.time()
                    self.writer.sync_all_streams(parall_iter)
                    
                    sync_streams_time = time.time() - sync_streams_start
                    
                    total_time = time.time() - start_time
                    throughput = (self.total_size * 4 / 1e9) / total_time
                    
                    self.save_times.append(total_time)
                    _ms_profile_event(
                        "checkpoint_finalize",
                        checkpoint_id=int(checkpoint_id),
                        parall_iter=int(parall_iter),
                        wait=False,
                        total_ms=float(total_time) * 1000.0,
                        sync_ms=float(sync_streams_time) * 1000.0,
                        throughput_gbps=float(throughput),
                    )
                    
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
                        self._profile_parall_to_checkpoint_id.pop(parall_iter, None)
                    self._slot_available_event.set()  # 通知等待的 begin_checkpoint
                except Exception as e:
                    print(f"✗ Background checkpoint {parall_iter} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # ✅ 修复：即使失败也要确保槽位被释放
                    with self._slot_lock:
                        self._checkpoint_slots.pop(parall_iter, None)
                        self._profile_parall_to_checkpoint_id.pop(parall_iter, None)
                    self._slot_available_event.set()  # 通知等待的 begin_checkpoint
            
            # 提交到兼容线程池：finalize 与 copy path 共享 copy_thread_pool。
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


def _resolve_master_weight(param, optimizer, master_accessor=None):
    """返回 param 对应的独立 fp32 master weight tensor；不存在则返回 None。

    检查点必须数据里，权威权重是 fp32 master。两种情形：
    - 纯 fp32 训练：param 本身就是 master，无独立副本 → 返回 None，
      master 流为空（param 流已保存权威权重，不再重复写一份）。
    - 真混精训练（bf16/fp16 工作副本 + 独立 fp32 master）：按常见约定探测；
      框架私有布局（如 DeepSpeed flat groups）请通过 master_accessor 注入。
    """
    if master_accessor is not None:
        try:
            return master_accessor(param, optimizer)
        except Exception:
            return None
    state = optimizer.state.get(param, {}) if optimizer is not None else {}
    for key in ('master_param', 'fp32_param', 'master_weight'):
        m = state.get(key)
        if isinstance(m, torch.Tensor):
            return m
    for attr in ('main_param', '_fp32_master', 'ds_fp32_param'):
        m = getattr(param, attr, None)
        if isinstance(m, torch.Tensor):
            return m
    return None


def _quant_unit_floats(block):
    """交错量化布局的每-block 单元大小（float 数）：block 个 int8 负载 + 1 个 fp32 scale。"""
    assert block % 4 == 0, "quant_block must be a multiple of 4"
    return block // 4 + 1


def _packed_moment_floats(numel, block):
    """一个 moment 张量量化后在 float32 流内占用的 float 数（block 交错布局）。

    布局 v3：每 block 一个单元 [block/4 个 float 的 int8 负载 | 1 个 fp32 scale]，
    张量 = ceil(numel/block) 个单元顺排。相比旧的 [全部payload|全部scale] 两段式，
    交错使任意 block 对齐的张量内切口在流内映射为单个连续区间——
    split 分组与量化因此可组合。

    Returns: (total_floats, nblocks, unit_floats)
    """
    nb = (numel + block - 1) // block
    unit = _quant_unit_floats(block)
    return nb * unit, nb, unit


def piece_extent(layer_info, stream, ps, pe):
    """piece [ps, pe)（张量局部元素坐标）在指定流内的 (offset, size)（float 单位）。

    布局映射的唯一权威：保存路径、metadata 导出、恢复端都必须经由本函数，
    防止偏移算术在多处分叉。返回 None 表示该流对本层无字节（master 流为空）。
    量化 moment 流按交错单元映射，要求 ps（及 pe，除张量尾外）block 对齐。
    """
    if stream == 'param':
        return layer_info['param_offset'] + ps, pe - ps
    if stream == 'master':
        msz = layer_info.get('master_size', layer_info.get('grad_size', 0))
        if msz <= 0:
            return None
        base = layer_info.get('master_offset', layer_info.get('grad_offset', 0))
        return base + ps, pe - ps
    base = layer_info[f'{stream}_offset']
    if not layer_info.get('quant'):
        return base + ps, pe - ps
    block = int(layer_info['quant_block'])
    unit = _quant_unit_floats(block)
    numel = int(layer_info.get('q_numel', layer_info.get('param_size', 0)))
    assert ps % block == 0 and (pe == numel or pe % block == 0), (
        f"quant piece [{ps},{pe}) not aligned to block {block} (numel={numel})")
    ub_s = ps // block
    ub_e = (pe + block - 1) // block
    return base + ub_s * unit, (ub_e - ub_s) * unit


def quantize_int8_blocks(x, block, sqrt_domain=False):
    """per-block 对称 int8 量化。返回 (int8[N], scale_fp32[nblocks])。
    sqrt_domain=True 用于非负、动态范围大的 exp_avg_sq（在 sqrt 域量化，恢复端平方还原）。"""
    flat = x.detach().reshape(-1).float()
    n = flat.numel()
    y = flat.sqrt() if sqrt_domain else flat
    nb = (n + block - 1) // block
    pad = nb * block - n
    if pad:
        y = torch.cat([y, y.new_zeros(pad)])
    yb = y.view(nb, block)
    scale = yb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0
    q = torch.clamp((yb / scale).round(), -127, 127).to(torch.int8).view(-1)[:n]
    return q.contiguous(), scale.view(-1).to(torch.float32).contiguous()


def dequantize_int8_blocks(q_i8, scale, block, sqrt_domain=False):
    """int8 + per-block scale 还原为 fp32（供恢复端/校验使用）。"""
    n = q_i8.numel()
    nb = scale.numel()
    pad = nb * block - n
    qf = q_i8.to(torch.float32)
    if pad:
        qf = torch.cat([qf, qf.new_zeros(pad)])
    deq = (qf.view(nb, block) * scale.view(nb, 1)).view(-1)[:n]
    if sqrt_domain:
        deq = deq.square()
    return deq


def _pack_moment_int8(buf, base_float, moment_flat, block, sqrt, ps=0, pe=None):
    """把 moment 张量的 [ps, pe) 片段量化写入 buf（fp32 视图）的交错单元区。

    交错布局 v3：每 block 一个单元 [block/4 个 float 的 int8 负载 | 1 个 fp32 scale]。
    要求 ps/pe block 对齐（pe==numel 的尾块除外）。block 是张量内局部坐标，
    切口只落在 block 边界 ⇒ 分段量化的 block 划分与整张量量化完全相同，
    产出逐字节一致——split+quant 与 atomic+quant 的 checkpoint 映像可逐位对比。
    """
    n = moment_flat.numel()
    if pe is None:
        pe = n
    assert ps % block == 0 and (pe == n or pe % block == 0), (
        f"_pack_moment_int8 piece [{ps},{pe}) not aligned to block {block} (numel={n})")
    unit = _quant_unit_floats(block)
    q, sc = quantize_int8_blocks(moment_flat[ps:pe], block, sqrt_domain=sqrt)
    seg = pe - ps
    nb = (seg + block - 1) // block
    pad = nb * block - seg
    if pad:  # 尾块 padding 清零，保证回读确定性
        q = torch.cat([q, q.new_zeros(pad)])
    ub0 = ps // block
    region = buf[base_float + ub0 * unit: base_float + (ub0 + nb) * unit].view(nb, unit)
    # int8 负载：strided fp32 视图最后一维连续，可安全 view 为 int8 后融合拷贝
    region[:, :block // 4].view(torch.int8).copy_(q.view(nb, block), non_blocking=True)
    region[:, block // 4].copy_(sc, non_blocking=True)


def build_param_layout(model, optimizer, master_accessor=None,
                       quant_moments=False, quant_block=QUANT_BLOCK_DEFAULT):
    """
    构建参数布局信息。

    所有偏移均为局部偏移（0-based），即在每种数据类型的缓冲区内的偏移。
    这使得布局可以同时兼容单块 gpu_ar 拆分的 4 个 view 和四块独立 tensor。

    流语义（layout_version=2）：
      stream0 param    — 模型参数（fp32 训练时即权威权重）
      stream1 master   — 独立 fp32 master weight（无混精副本时 size=0，不占字节）
      stream2/3        — Adam exp_avg / exp_avg_sq
    梯度不再保存：resume 后由反向传播重算，非必须数据。
    grad_offset/grad_size 作为 master_* 的别名保留，兼容旧读取方。

    quant_moments=True（A 方案）：stream2/3 以 packed int8+per-block scale 存储，
    exp_avg_*_size 变为 packed float 数、offset 为 packed 累加值。量化要求整张量
    成组（atomic 分组），由调用方保证。

    Returns:
        List[Dict]: 每层的参数信息
    """
    layout = []
    total_param_size = sum(p.numel() for p in model.parameters())

    layer_id = 0
    current_offset = 0
    master_offset = 0
    ea_offset = 0   # exp_avg 流累加（quant 时为 packed）
    eas_offset = 0  # exp_avg_sq 流累加

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        size = param.numel()
        master = _resolve_master_weight(param, optimizer, master_accessor)
        master_size = master.numel() if master is not None else 0

        if quant_moments:
            ea_floats, ea_nb, ea_unit = _packed_moment_floats(size, quant_block)
            eas_floats = ea_floats  # 同 numel → 同 packed 布局
        else:
            ea_floats = eas_floats = size
            ea_nb = ea_unit = 0

        layer_info = {
            'layer_id': layer_id,
            'name': name,
            'param_offset': current_offset,
            'param_size': size,
            'master_offset': master_offset,
            'master_size': master_size,
            # 兼容别名（旧键名，语义已是 master）
            'grad_offset': master_offset,
            'grad_size': master_size,
            'exp_avg_offset': ea_offset,
            'exp_avg_size': ea_floats,
            'exp_avg_sq_offset': eas_offset,
            'exp_avg_sq_size': eas_floats,
            # 量化元数据（quant 时有效；v3 交错布局）
            'quant': bool(quant_moments),
            'quant_block': int(quant_block) if quant_moments else 0,
            'q_numel': int(size),
            'q_nblocks': int(ea_nb),
            'q_unit_floats': int(ea_unit),
        }

        layout.append(layer_info)
        current_offset += size
        master_offset += master_size
        ea_offset += ea_floats
        eas_offset += eas_floats
        layer_id += 1

    moment_bytes = (ea_offset + eas_offset) * 4
    total_bytes = (total_param_size + master_offset) * 4 + moment_bytes
    print(f"\nBuilt parameter layout (local offsets, layout_version=2"
          f"{', quant_moments int8' if quant_moments else ''}):")
    print(f"  Total layers: {len(layout)}")
    print(f"  Total parameters: {total_param_size:,} ({total_param_size*4/1e9:.2f} GB)")
    print(f"  Master-weight stream: {master_offset:,} floats "
          f"({'independent fp32 master' if master_offset else 'empty — params are fp32 masters'})")
    if quant_moments:
        print(f"  Moment streams (int8 packed): {ea_offset:,}+{eas_offset:,} floats "
              f"(vs {total_param_size*2:,} fp32 → {(1 - (ea_offset+eas_offset)/(total_param_size*2))*100:.1f}% smaller)")
    print(f"  Checkpoint payload: {total_bytes/1e9:.2f} GB "
          f"(param + master + exp_avg + exp_avg_sq; grads excluded)")

    return layout


def _norm_pieces(group, param_layout):
    """把旧格式组 [lid, ...] 归一化为 piece 格式 [(lid, elem_start, elem_end), ...]。"""
    pieces = []
    for item in group:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            pieces.append((int(item[0]), int(item[1]), int(item[2])))
        else:
            lid = int(item)
            if lid < len(param_layout):
                pieces.append((lid, 0, int(param_layout[lid].get('param_size', 0))))
    return pieces


def plan_groups(param_layout, num_groups, ratios=None, align_elems=1024, mode='split',
                quant_block=0):
    """等分位字节区间分组：layout-schedule 解耦的调度计划。

    mode='split'（默认）：允许张量内切分，刀口对齐 align_elems，可逼近完美均衡。
    mode='atomic'：刀口只落在张量边界（不切开任何张量），复用完全相同的
      piece/触发/等待机制——因此 atomic vs split 是同一 race-fixed 代码路径的
      纯分组对照实验（隔离"切分收益"）。
    quant_block>0（split+quant 组合）：量化 block 是张量内局部坐标，刀口必须
      snap 到所在张量的局部 block 边界（或张量边界），才能保证每个 piece 独立
      量化且与整张量量化逐位一致。此时全局 align_elems 对齐让位——4KB 对齐只是
      sync_file_range 的性能偏好（C++ 端会自行页对齐），不是正确性约束。

    在 flat 元素坐标上切 K-1 刀。均衡域 = ratios 加权后的体量（例如压缩后
    字节的预测比例；ratios=None 时退化为原始元素均衡）。刀口对齐
    align_elems（默认 1024 floats = 4KB，服务 sync_file_range 与恢复对齐）。
    分组只决定调度（何时触发保存/等待哪些事件），不改变任何落盘地址。

    Returns dict:
      pieces:  List[List[(lid, s, e)]]      每组覆盖的张量片段（组内连续）
      tails:   Dict[lid, List[gidx]]        以该 tensor 为尾的组 → 触发表
      cover:   Dict[lid, List[gidx]]        覆盖该 tensor 的组 → 防撕裂等待表
      batches: List[(lids, gidxs)]          更新批次 →（更新这些 tensor 后）触发这些组
      cuts:    List[int]                    全局元素坐标刀口（存入 metadata 供恢复端复原）
    """
    sizes = [int(li.get('param_size', 0)) for li in param_layout]
    n = len(sizes)
    total = sum(sizes)
    if n == 0 or total == 0:
        return {"pieces": [], "tails": {}, "cover": {}, "batches": [], "cuts": []}

    starts = []
    acc = 0
    for s in sizes:
        starts.append(acc)
        acc += s

    w = [float(s) * (float(ratios[i]) if ratios is not None else 1.0)
         for i, s in enumerate(sizes)]
    wtotal = sum(w)
    if wtotal <= 0:
        w = [float(s) for s in sizes]
        wtotal = float(total)
    wcum = []
    acc = 0.0
    for x in w:
        wcum.append(acc)
        acc += x

    align = max(1, int(align_elems))
    K = max(1, min(int(num_groups), total // align if total >= align else 1))

    cuts = []
    prev = 0
    li = 0
    for j in range(1, K):
        target = wtotal * j / K
        while li + 1 < n and wcum[li] + w[li] < target:
            li += 1
        if mode == 'atomic':
            # 只在张量边界下刀：选权重域上更接近 target 的边界（保证整张量成组）
            lo_b, hi_b = starts[li], starts[li] + sizes[li]
            d_lo = abs(target - wcum[li])
            d_hi = abs((wcum[li] + w[li]) - target)
            pos = lo_b if d_lo <= d_hi else hi_b
            if pos <= prev:                      # 该边界已用 → 取下一张量边界
                pos = hi_b
        elif quant_block > 0:
            # split+quant：snap 到所在张量的局部 block 边界（张量边界恒合法）
            frac = 0.0 if w[li] <= 0 else (target - wcum[li]) / w[li]
            raw = starts[li] + frac * sizes[li]
            lo, hi = starts[li], starts[li] + sizes[li]
            local = int(round((raw - lo) / quant_block)) * quant_block
            pos = lo + min(max(local, 0), sizes[li])
            if pos <= prev:
                # 向上取下一个合法刀口：> prev 的最小局部 block 边界，越过张量尾则取尾
                k = (max(prev, lo) - lo) // quant_block + 1
                pos = min(lo + k * quant_block, hi)
        else:
            frac = 0.0 if w[li] <= 0 else (target - wcum[li]) / w[li]
            pos = starts[li] + frac * sizes[li]
            pos = int(round(pos / align)) * align
            pos = max(pos, prev + align)             # 严格递增
            pos = min(pos, total - (K - j) * align)  # 为后续组留出空间
        if pos <= prev or pos >= total:
            continue                             # 极端小模型/边界重合：允许并组
        cuts.append(pos)
        prev = pos

    bounds = [0] + cuts + [total]
    pieces = []
    idx = 0
    for g in range(len(bounds) - 1):
        lo, hi = bounds[g], bounds[g + 1]
        while idx < n and starts[idx] + sizes[idx] <= lo:
            idx += 1
        grp = []
        i = idx
        while i < n and starts[i] < hi:
            s = max(lo, starts[i]) - starts[i]
            e = min(hi, starts[i] + sizes[i]) - starts[i]
            if e > s:
                grp.append((i, int(s), int(e)))
            i += 1
        pieces.append(grp)

    tails, cover = {}, {}
    for g, grp in enumerate(pieces):
        for (lid, _, _) in grp:
            cover.setdefault(lid, []).append(g)
        if grp:
            tails.setdefault(grp[-1][0], []).append(g)

    # 更新批次：批 = (上一尾 tensor, 本组尾 tensor]，同尾的多个组合并到同一批触发
    batches = []
    prev_tail = -1
    g = 0
    while g < len(pieces):
        if not pieces[g]:
            g += 1
            continue
        tail = pieces[g][-1][0]
        gs = [g]
        while g + 1 < len(pieces) and pieces[g + 1] and pieces[g + 1][-1][0] == tail:
            g += 1
            gs.append(g)
        batches.append((tuple(range(prev_tail + 1, tail + 1)), tuple(gs)))
        prev_tail = tail
        g += 1
    if prev_tail + 1 < n:
        batches.append((tuple(range(prev_tail + 1, n)), tuple()))

    covered = sum(e - s for grp in pieces for (_, s, e) in grp)
    assert covered == total, f"plan_groups covers {covered} of {total} elems"
    flat = [(lid, s, e) for grp in pieces for (lid, s, e) in grp]
    for a, b in zip(flat, flat[1:]):
        assert (b[0] == a[0] and b[1] == a[2]) or (b[0] == a[0] + 1 and b[1] == 0 and a[2] == sizes[a[0]]) or \
               (b[0] > a[0] + 1 and b[1] == 0), "pieces must be contiguous in layout order"

    return {"pieces": pieces, "tails": tails, "cover": cover,
            "batches": batches, "cuts": cuts}
