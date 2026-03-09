import os
from platform import node
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse

import ctypes
from checkpoint_eval.pccheck.multistream_checkpoint import MultiStreamCheckpoint, build_param_layout
from checkpoint_eval.pccheck_utils import initialize, set_storage

parser = argparse.ArgumentParser(description="MultiStream PCCheck microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")
parser.add_argument(
    "--num-threads", default=2, type=int, help="Number of CPU threads writing at SSD"
)
parser.add_argument(
    "--max-async", default=2, type=int, help="Maximum async checkpoints"
)
parser.add_argument(
    "--num-layer-groups", default=1, type=int, help="Number of layer groups (use 1 for fair microbenchmark comparison)"
)
parser.add_argument(
    "--c_lib_path",
    default="",
    type=str,
    required=True,
    help="path to the libtest_ssd.so library",
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output"
)


def run(args):
    # ===== --size 参数表示完整检查点大小（包括 params + grads + exp_avg + exp_avg_sq）=====
    # 对于 Adam 优化器，完整检查点 = 4 × model_size
    # 因此 model_size = size / 4
    checkpoint_size_mb = args.size
    model_size_floats = int(checkpoint_size_mb * 1000000 / 4 / 4)  # size MB / 4 bytes / 4 copies
    print(f"Target checkpoint size: {checkpoint_size_mb} MB")
    print(f"Model size: {model_size_floats} floats ({model_size_floats * 4 / 1e6:.2f} MB)")
    
    # ===== 模型层数配置 =====
    # - num_layer_groups=1: 单层模型，用于公平的 microbenchmark 对比（与 pccheck 一致）
    # - num_layer_groups>1: 多层模型，测试 MultiStream 流水线能力
    # 注意：MultiStream 的真正优势在于"边训练边保存"的异步场景，
    # 而非纯同步保存。在同步保存场景下，多层流水线的开销可能超过收益。
    num_layers = args.num_layer_groups
    layer_size = model_size_floats // num_layers
    
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 创建多层模型，每层大小相等
            for i in range(num_layers):
                # 最后一层处理余数
                if i == num_layers - 1:
                    size = model_size_floats - layer_size * (num_layers - 1)
                else:
                    size = layer_size
                setattr(self, f'layer_{i}', torch.nn.Parameter(torch.ones(size, dtype=torch.float32)))
    
    model = TestModel()
    model.cuda()
    
    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    num_model_layers = len(list(model.parameters()))
    print(f"Created model with {num_model_layers} layers, {total_params} total parameters")
    
    # 创建优化器并初始化状态
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化 gpu_ar（4 × model_size）
    gpu_ar, total_size = initialize(model, [optimizer], do_opt_step=True, separate_streams=True)
    
    # 设置存储映射
    set_storage(model, [optimizer], gpu_ar)
    
    torch.cuda.empty_cache()
    
    print(f"Checkpoint size: {total_size} floats = {total_size*4/1e6:.2f} MB (4x model_size)")
    
    # 静默 multistream_checkpoint 的内部打印
    if not args.verbose:
        import builtins
        _real_print = builtins.print
        def quiet_print(*pargs, **kwargs):
            # 只打印关键信息
            msg = str(pargs[0]) if pargs else ""
            if any(kw in msg for kw in ["Start ITER", "CHECKPOINT", "AVERAGE", "Target", "Model size", "Created model", "Checkpoint size", "Built param_layout"]):
                _real_print(*pargs, **kwargs)
        builtins.print = quiet_print
    
    # 使用 build_param_layout 构建完整的参数布局
    param_layout = build_param_layout(model, optimizer)
    
    print(f"Built param_layout with {len(param_layout)} entries")
    
    # Create multistream checkpoint
    chk = MultiStreamCheckpoint(
        param_layout=param_layout,
        gpu_ar=gpu_ar,
        total_size=total_size,
        num_streams=4,
        num_threads=args.num_threads,
        num_layer_groups=args.num_layer_groups,
        lib_path=args.c_lib_path,
        filename=f"checkpoint_multistream_{args.size}mb.chk",
        max_async=args.max_async
    )

    # ✅ 修复：Microbenchmark 只测试 checkpoint 保存时间，不包含 optimizer 更新
    # 使用 save_full_checkpoint() 而不是 step_with_callback()
    # 这样与 test_pccheck.py、test_cfreq.py、test_gpm.py 保持公平对比
    
    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)
        print(f"-------------------------- Start ITER {it}")

        start_time = time.time()
        
        # ✅ 关键修复：使用 save_full_checkpoint 只做保存，不做 optimizer 更新
        # sync=True 表示同步等待保存完成
        chk.save_full_checkpoint(sync=True)

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"----------------- CHECKPOINT {it} TOOK {duration} ms")

    # Shutdown checkpoint
    if hasattr(chk, 'shutdown'):
        chk.shutdown()

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.sched_setaffinity(0, {0})
    args = parser.parse_args()
    run(args)
