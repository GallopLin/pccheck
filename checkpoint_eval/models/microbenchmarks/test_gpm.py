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

from checkpoint_eval.gpm.gpm_manager import GPMCheckpoint
from checkpoint_eval.pccheck_utils import initialize, get_total_size

parser = argparse.ArgumentParser(description="CheckFreq microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")


def run(args):
    # ===== --size 参数表示完整检查点大小（包括 params + grads + exp_avg + exp_avg_sq）=====
    # GPM 直接保存模型参数到 mmap，所以直接使用 size MB
    checkpoint_size_mb = args.size
    total_floats = int(checkpoint_size_mb * 1000000 / 4)  # size MB / 4 bytes per float
    print(f"Target checkpoint size: {checkpoint_size_mb} MB")
    print(f"Checkpoint size: {total_floats} floats ({total_floats * 4 / 1e6:.2f} MB)")

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            tensor = torch.ones(total_floats, dtype=torch.float32)
            self.a = torch.nn.Parameter(tensor)

    model = TestModel()
    model.cuda()

    total_size = get_total_size(model, [])
    print(f"Model total_size: {total_size} floats = {total_size*4/1e6:.2f} MB")

    chk = GPMCheckpoint(
        f"checkpoint_gpm.chk", model=model, datasize=int(total_size * 4)  # use float
    )

    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)

        start_time = time.time()
        chk._checkpoint(iter_chk=it)

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"CHECKPOINT {it} TOOK {duration} ms")

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")
    chk._finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    run(args)
