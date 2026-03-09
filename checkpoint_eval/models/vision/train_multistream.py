"""
VGG16 training with MultiStream PCCheck checkpointing

Usage:
    python train_multistream.py --dataset imagenet --batchsize 32 --arch vgg16 \
        --cfreq 10 --bench_total_steps 200 --max-async 2 --num-threads 1 \
        --num_layer_groups 8
"""

import os
import sys
import time
import argparse
import torchvision.models as models
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# Import MultiStream checkpointing
sys.path.append(os.path.join(os.path.dirname(__file__), '../../pccheck'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from multistream_checkpoint import MultiStreamCheckpoint, build_param_layout
from pccheck_utils import initialize, set_storage

home_dir = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help='dataset')
parser.add_argument("--batchsize", type=int, required=True, help='batch size')
parser.add_argument("--bench_total_steps", type=int, default=200, help='number of steps')
parser.add_argument("--arch", type=str, default='vgg16', help='model architecture')
parser.add_argument("--cfreq", type=int, default=10, help='checkpoint frequency')
parser.add_argument("--max-async", type=int, default=2, help='max async checkpoints')
parser.add_argument("--num-threads", type=int, default=1, help='number of threads for writing')
parser.add_argument("--num_layer_groups", type=int, default=8, help='number of layer groups for multistream')
parser.add_argument("--c_lib_path", type=str, default=None, help='path to libtest_ssd.so')


def train():
    args = parser.parse_args()
    
    # Setup paths
    if args.c_lib_path:
        lib_path = args.c_lib_path
    else:
        lib_path = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    
    checkpoint_dir = f"{home_dir}/code/pccheck/pccheck_multistream_checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/multistream_{args.arch}.chk"
    
    # Model setup
    if args.arch == 'vgg16':
        model = models.vgg16()
    elif args.arch == 'resnet50':
        model = models.resnet50()
    elif args.arch == 'resnet18':
        model = models.resnet18()
    else:
        model = models.vgg16()
    
    model = model.cuda()
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    base_optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data setup (synthetic data for benchmarking)
    batch_size = args.batchsize
    input_size = (3, 224, 224)  # ImageNet size
    num_classes = 1000
    
    print(f"\n=== MultiStream VGG16 Training ===")
    print(f"Model: {args.arch}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint frequency: {args.cfreq}")
    print(f"Max async checkpoints: {args.max_async}")
    print(f"Num layer groups: {args.num_layer_groups}")
    print(f"Lib path: {lib_path}")
    
    # Initialize multistream checkpoint
    ms_optimizer = None
    ms_checkpoint = None
    
    if args.cfreq > 0:
        # Initialize gpu_ar and map model/optimizer storage (zero-copy, same as pccheck)
        gpu_ar, total_size = initialize(model, [base_optimizer])
        set_storage(model, [base_optimizer], gpu_ar)
        torch.cuda.empty_cache()

        # Build parameter layout
        param_layout = build_param_layout(model, base_optimizer)
        
        # Create MultiStreamCheckpoint
        ms_checkpoint = MultiStreamCheckpoint(
            param_layout=param_layout,
            gpu_ar=gpu_ar,
            total_size=total_size,
            num_threads=args.num_threads,
            lib_path=lib_path,
            filename=checkpoint_path,  # 传入字符串，不要 encode
            num_streams=4,
            max_async=args.max_async,
            num_layer_groups=args.num_layer_groups,
        )
        
        # 使用 create_optimizer 创建包装器，它会自动处理检查点和跨检查点同步
        ms_optimizer = ms_checkpoint.create_optimizer(base_optimizer, model)
        
        print(f"MultiStreamCheckpoint initialized with {len(ms_checkpoint.layer_groups)} layer groups")
    
    # Training loop
    warmup = 10
    batch_idx = 0
    steps_since_checkp = 0
    checkpoints = 0
    start_train_time = None
    
    while batch_idx < args.bench_total_steps:
        start_iter = time.time()
        
        # Generate synthetic data
        data = torch.randn(batch_size, *input_size).cuda()
        target = torch.randint(0, num_classes, (batch_size,)).cuda()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        if ms_optimizer:
            ms_optimizer.zero_grad()
        else:
            base_optimizer.zero_grad()
        
        loss.backward()
        
        # Optimizer step
        if args.cfreq > 0 and ms_optimizer:
            # Check if this is a checkpoint step
            is_checkpoint_step = (batch_idx == warmup) or \
                                 ((steps_since_checkp == args.cfreq - 1) and batch_idx >= warmup)
            
            if is_checkpoint_step and batch_idx >= warmup:
                # 使用 optimizer wrapper 的 API
                # begin_checkpoint 会创建内部的 MultiStreamOptimizer 并设置回调
                ms_optimizer.begin_checkpoint()
                
                # Use step_with_callback for per-layer-group saves
                ms_optimizer.step_with_callback()
                
                # Finalize checkpoint (non-blocking)
                ms_optimizer.finalize_checkpoint(wait=False)
                
                steps_since_checkp = 0
                checkpoints += 1
            else:
                # Normal step without checkpointing
                ms_optimizer.step()
                steps_since_checkp += 1
        else:
            base_optimizer.step()
            steps_since_checkp += 1
        
        # Start timing after warmup
        if batch_idx == warmup:
            print(f"Start clock!")
            start_train_time = time.time()
        
        batch_idx += 1
        print(f"Step {batch_idx} took {time.time()-start_iter:.4f}s")
    
    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time if start_train_time else 0

    print(f"\n-- BENCHMARK ENDED: Total time: {total_train_time:.4f} sec, "
          f"Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}")
    print(f"EXECUTION TIME: {total_train_time} sec")
    print(f"THROUGHPUT IS {(args.bench_total_steps-warmup)/total_train_time if total_train_time > 0 else 0:.4f}")

    # Wait for all pending checkpoints and cleanup
    # NOTE: shutdown() is called AFTER printing results to ensure timing data is
    # preserved even if the C library crashes during cleanup (known malloc issue).
    if ms_checkpoint:
        try:
            ms_checkpoint.shutdown()
        except Exception as e:
            print(f"Warning: ms_checkpoint.shutdown() raised: {e}")


if __name__ == "__main__":
    args = parser.parse_args()
    os.sched_setaffinity(0, {0})  # Pin to CPU 0
    train()
