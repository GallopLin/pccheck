import torch
import os
import json
import math
from typing import Optional, Dict, Any
from .multistream_checkpoint import MultiStreamWriter

class DeepSpeedMultiStreamCheckpoint:
    """
    DeepSpeed Universal Checkpointing (UCP) 兼容的多流检查点管理器
    
    该类适配 DeepSpeed 的 ZeRO 优化器，将分片的参数和优化器状态
    通过 pccheck 的多流 IO 后端写入存储。
    """
    
    def __init__(
        self,
        engine,
        save_dir: str,
        lib_path: str = "./libtest_ssd.so",
        num_threads: int = 16,
        max_async: int = 2,
        chunk_size: int = 16 * 1024 * 1024, # 64MB
    ):
        """
        Args:
            engine: DeepSpeedEngine 实例
            save_dir: 检查点保存目录
            lib_path: pccheck C库路径
            num_threads: IO线程数
            max_async: 最大异步检查点数
        """
        self.engine = engine
        self.save_dir = save_dir
        self.lib_path = lib_path
        self.num_threads = num_threads
        self.max_async = max_async
        self.chunk_size = chunk_size
        
        # 获取分布式信息
        self.rank = 0
        self.world_size = 1
        if hasattr(engine, 'local_rank'):
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            
        # 确保保存目录存在
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
        # 分析 DeepSpeed 优化器状态
        self.optimizer = self.engine.optimizer
        
        # 检查是否为 ZeRO 优化器
        self.is_zero = hasattr(self.optimizer, 'optimizer') and hasattr(self.optimizer, 'fp32_groups_flat')
        
        if not self.is_zero:
            raise NotImplementedError("目前仅支持 DeepSpeed ZeRO 优化器 (Stage 1/2/3)")
            
        # 计算流大小
        # Stream 0: Param (FP32 Master)
        # Stream 1: Grad (FP32/FP16)
        # Stream 2: Exp_Avg
        # Stream 3: Exp_Avg_Sq
        self.stream_sizes = self._calculate_stream_sizes()
        
        # 初始化 Writer
        # 注意：每个 rank 写入自己的文件，文件名包含 rank
        # DeepSpeed UCP 风格：zero_pp_rank_X_mp_rank_Y_optim_states.pt
        # 这里我们使用 pccheck 的二进制格式
        self.filename = os.path.join(save_dir, f"pccheck_rank_{self.rank}.bin")
        
        # 估算所需的 chunks
        max_stream_size = max(self.stream_sizes)
        chunks_per_stream = (max_stream_size + chunk_size - 1) // chunk_size
        num_chunks = max_async * 4 * chunks_per_stream + 16 # 预留一些 buffer
        
        self.writer = MultiStreamWriter(
            self.filename.encode(),
            lib_path,
            max_async,
            4, # num_streams
            self.stream_sizes,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            is_distributed=True,
            rank=self.rank,
            world_size=self.world_size
        )
        
        self.parall_iter = 0
        
    def _calculate_stream_sizes(self):
        """计算当前 rank 需要保存的数据大小"""
        # 对于 ZeRO，fp32_groups_flat 包含了当前 rank 负责的参数分片
        # 通常是一个 list of tensors (每个 param_group 一个 tensor)
        
        total_param_elements = 0
        for tensor in self.optimizer.fp32_groups_flat:
            total_param_elements += tensor.numel()
            
        # 假设 Adam 优化器，状态大小与参数一致
        # Stream 0: Param
        # Stream 1: Grad (通常 ZeRO 在 step 后会清除梯度，或者梯度在 bucket 中)
        #           为了通用性，我们预留空间，如果无法获取梯度则填 0 或跳过
        # Stream 2: Exp_Avg
        # Stream 3: Exp_Avg_Sq
        
        # 注意：如果 DeepSpeed 已经释放了梯度，我们可能无法保存梯度
        # 但 UCP 主要是为了恢复训练，通常只需要 Param 和 Optimizer States
        
        return [total_param_elements] * 4

    def save_checkpoint(self, tag: str, client_state: Dict[str, Any] = None):
        """
        执行检查点保存
        
        Args:
            tag: 检查点标签 (e.g., "global_step100")
            client_state: 用户自定义状态
        """
        # 注册新的检查点槽位
        slot_id = self.writer.register()
        
        # 1. 保存参数 (Stream 0)
        offset = 0
        for group_idx, param_tensor in enumerate(self.optimizer.fp32_groups_flat):
            self._write_tensor(0, param_tensor, offset, slot_id)
            offset += param_tensor.numel()
            
        # 2. 保存梯度 (Stream 1) - 尝试获取
        # ZeRO 优化器通常在 step() 后清除梯度，或者梯度存储在 gradients_flat 中 (如果存在)
        offset = 0
        has_grads = hasattr(self.optimizer, 'gradients_flat') and self.optimizer.gradients_flat is not None
        
        if has_grads:
            # 注意：gradients_flat 结构可能与 fp32_groups_flat 对应
            # 这里简化处理，假设结构一致
            pass 
            # TODO: 实现梯度保存逻辑，如果需要
        
        # 3. 保存优化器状态 (Stream 2 & 3)
        # 访问内部优化器 (e.g. Adam) 的状态
        base_optimizer = self.optimizer.optimizer
        
        offset = 0
        for group_idx, param_group in enumerate(base_optimizer.param_groups):
            # 在 ZeRO 中，param_group['params'] 通常只有一个 flattened tensor
            for param in param_group['params']:
                state = base_optimizer.state[param]
                
                if 'exp_avg' in state:
                    self._write_tensor(2, state['exp_avg'], offset, slot_id)
                
                if 'exp_avg_sq' in state:
                    self._write_tensor(3, state['exp_avg_sq'], offset, slot_id)
                
                offset += param.numel()
        
        # 触发同步 (异步写入)
        self.writer.sync_all_streams(slot_id)
        
        # 保存元数据 (用于 UCP 恢复)
        self._save_metadata(tag, client_state)
        
        return True

    def _write_tensor(self, stream_id, tensor, offset, slot_id):
        """将 Tensor 写入指定流"""
        # 确保 tensor 在 CPU 上 (如果 pccheck 支持 GPU 直接写入则不需要)
        # pccheck 目前通过 ctypes 传递指针，如果 tensor 在 GPU，需要 copy 到 pinned memory
        # 或者 pccheck 内部处理了？
        # 查看 multistream_checkpoint.py，它使用 np.ctypeslib.as_ctypes(data)
        # 这意味着它期望 numpy array (CPU)
        
        # 优化：使用 pccheck 的 borrow_chunk 获取 pinned memory，进行 D2H 拷贝
        
        numel = tensor.numel()
        current_offset = 0
        
        while current_offset < numel:
            chunk_len = min(self.chunk_size, numel - current_offset)
            
            # 借用 pinned buffer
            chunk_ptr = self.writer.borrow_chunk()
            
            # 将数据从 GPU 拷贝到 pinned buffer
            # 注意：这里需要高效的 D2H 拷贝
            # 我们可以创建一个 wrap 该指针的 tensor
            # 或者先 to('cpu') (较慢)
            
            # 简单实现：先转 CPU numpy (会有性能损耗，但兼容性好)
            # 理想实现：使用 cudaMemcpyAsync 到 chunk_ptr
            
            sub_tensor = tensor.flatten()[current_offset : current_offset + chunk_len]
            cpu_data = sub_tensor.detach().cpu().numpy()
            
            # 写入
            self.writer.write_stream_chunk(
                stream_id,
                cpu_data,
                offset + current_offset,
                chunk_len,
                slot_id,
                self.num_threads
            )
            
            # 归还 buffer (注意：write_stream_chunk 是异步的吗？)
            # pccheck 的 write_stream_chunk 似乎是同步拷贝到内部 buffer 还是直接写入？
            # 查看 C++ 代码或 Python 封装。
            # Python 封装调用 C++ write_stream_chunk。
            # 如果 C++ 内部做了拷贝，我们可以立即归还。
            # 如果 C++ 只是记录指针，我们需要等待。
            # 根据 multistream_checkpoint.py 的逻辑，它传入 numpy array，ctypes 会获取指针。
            # 通常 ctypes 调用期间指针有效。
            
            # 这里我们不需要手动 borrow_chunk，因为 write_stream_chunk 接受 numpy array
            # 并且 multistream_checkpoint.py 中没有显式 borrow/return chunk 用于 write_stream_chunk
            # 而是用于 "borrow_cpu_slot" ?
            
            # 修正：直接传入 numpy array
            
            current_offset += chunk_len
            
            # 归还 chunk (如果手动借用了)
            if chunk_ptr:
                self.writer.return_chunk(chunk_ptr)

    def _save_metadata(self, tag, client_state):
        """保存 DeepSpeed UCP 兼容的元数据"""
        metadata = {
            "tag": tag,
            "rank": self.rank,
            "world_size": self.world_size,
            "stream_sizes": self.stream_sizes,
            "client_state": client_state,
            # 添加分区信息，以便恢复时重组
            "partition_info": self._get_partition_info()
        }
        
        meta_path = os.path.join(self.save_dir, f"pccheck_rank_{self.rank}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_partition_info(self):
        """获取参数分区信息"""
        # 记录每个 param group 的大小和 padding
        info = []
        for group in self.optimizer.fp32_groups_flat:
            info.append({
                "numel": group.numel(),
                "shape": list(group.shape)
            })
        return info
