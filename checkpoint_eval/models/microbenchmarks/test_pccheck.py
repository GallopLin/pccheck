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
from checkpoint_eval.pccheck.chk_monitor import Chk_monitor
from checkpoint_eval.pccheck_utils import initialize, get_total_size, set_storage

parser = argparse.ArgumentParser(description="CheckFreq microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")
parser.add_argument(
    "--num-threads", default=1, type=int, help="Number of CPU threads writing at NVM"
)
parser.add_argument(
    "--c_lib_path",
    default="",
    type=str,
    required=True,
    help="path to the libtest.so library",
)


def run(args):
    # ===== --size 参数表示完整检查点大小（包括 params + grads + exp_avg + exp_avg_sq）=====
    # 对于 Adam 优化器，完整检查点 = 4 × model_size
    # 因此 model_size = size / 4
    checkpoint_size_mb = args.size
    model_size_floats = int(checkpoint_size_mb * 1000000 / 4 / 4)  # size MB / 4 bytes / 4 copies
    print(f"Target checkpoint size: {checkpoint_size_mb} MB")
    print(f"Model size: {model_size_floats} floats ({model_size_floats * 4 / 1e6:.2f} MB)")

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            tensor = torch.ones(model_size_floats, dtype=torch.float32)
            self.a = torch.nn.Parameter(tensor)

    model = TestModel()
    model.cuda()
    
    # ===== 保存完整训练状态：4 × model_size =====
    # 与 MultiStream 保持一致：params + grads + exp_avg + exp_avg_sq
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化优化器状态（执行一次 step）
    gpu_ar, total_size = initialize(model, [optimizer], do_opt_step=True)
    
    # 设置存储映射
    set_storage(model, [optimizer], gpu_ar)
    
    torch.cuda.empty_cache()
    
    print(f"Checkpoint size: {total_size} floats = {total_size*4/1e6:.2f} MB (4x model_size)")
    
    chk_monitor = Chk_monitor(
        args.c_lib_path,
        total_size,
        args.num_threads,
        1,
        True,
        gpu_ar=gpu_ar,
        is_sync=True,
        bsize=total_size,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        memory_saving=True,
    )

    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)
        print(f"-------------------------- Start ITER {it}")

        start_time = time.time()
        chk_monitor.save()
        while chk_monitor.checkpoint_in_progress():
            continue

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"----------------- CHECKPOINT {it} TOOK {duration} ms")

    if chk_monitor:
        chk_monitor.kill_checkpoint()

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.sched_setaffinity(0, {0})
    args = parser.parse_args()
    run(args)
