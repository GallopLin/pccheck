#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCCheck 改进效果对比实验
Benchmark Comparison: Traditional vs Layerwise Checkpoint

对比三种检查点方法：
1. 传统 PyTorch 检查点 (torch.save)
2. 原始 PCCheck
3. 改进的分层 PCCheck (Layerwise)

测量指标：
- 检查点保存时间
- 训练吞吐量 (samples/sec)
- 内存峰值
- 总训练时间
- I/O 开销占比
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("[Warning] GPUtil not available, GPU memory monitoring disabled")
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pccheck'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pccheck', 'layerwise_checkpoint'))

from checkpoint_eval.pccheck.layerwise_checkpoint.complete_integration import LayerwiseCheckpointTrainer
from checkpoint_eval.pccheck.chk_monitor import Chk_monitor
from checkpoint_eval.pccheck_utils import initialize, get_total_size, set_storage


class BenchmarkMetrics:
    """性能指标收集器"""
    
    def __init__(self, name: str):
        self.name = name
        self.checkpoint_times = []
        self.training_step_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.total_time = 0.0
        self.throughput = 0.0
        self.num_samples = 0
        
    def add_checkpoint_time(self, time_ms: float):
        self.checkpoint_times.append(time_ms)
    
    def add_step_time(self, time_ms: float):
        self.training_step_times.append(time_ms)
    
    def record_memory(self):
        """记录内存使用情况"""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024**3)  # GB
        
        # GPU 内存
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_memory_usage.append(gpus[0].memoryUsed / 1024)  # GB
            except:
                pass
    
    def compute_statistics(self):
        """计算统计数据"""
        return {
            'name': self.name,
            'checkpoint': {
                'mean_ms': np.mean(self.checkpoint_times) if self.checkpoint_times else 0,
                'std_ms': np.std(self.checkpoint_times) if self.checkpoint_times else 0,
                'min_ms': np.min(self.checkpoint_times) if self.checkpoint_times else 0,
                'max_ms': np.max(self.checkpoint_times) if self.checkpoint_times else 0,
                'total_ms': np.sum(self.checkpoint_times) if self.checkpoint_times else 0,
                'count': len(self.checkpoint_times),
            },
            'training_step': {
                'mean_ms': np.mean(self.training_step_times) if self.training_step_times else 0,
                'std_ms': np.std(self.training_step_times) if self.training_step_times else 0,
            },
            'memory': {
                'peak_cpu_gb': max(self.memory_usage) if self.memory_usage else 0,
                'peak_gpu_gb': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
                'mean_cpu_gb': np.mean(self.memory_usage) if self.memory_usage else 0,
            },
            'throughput': {
                'samples_per_sec': self.throughput,
                'total_samples': self.num_samples,
            },
            'total_time_sec': self.total_time,
            'checkpoint_overhead_percent': (np.sum(self.checkpoint_times) / 1000 / self.total_time * 100) if self.total_time > 0 else 0,
        }
    
    def print_summary(self):
        """打印摘要"""
        stats = self.compute_statistics()
        print(f"\n{'='*80}")
        print(f"📊 {stats['name']} - 性能摘要")
        print(f"{'='*80}")
        print(f"⏱️  总训练时间: {stats['total_time_sec']:.2f} 秒")
        print(f"🚀 吞吐量: {stats['throughput']['samples_per_sec']:.2f} samples/sec")
        print(f"💾 检查点保存:")
        print(f"   - 平均时间: {stats['checkpoint']['mean_ms']:.2f} ms")
        print(f"   - 总时间: {stats['checkpoint']['total_ms']/1000:.2f} 秒")
        print(f"   - 开销占比: {stats['checkpoint_overhead_percent']:.2f}%")
        print(f"   - 保存次数: {stats['checkpoint']['count']}")
        print(f"📈 训练步:")
        print(f"   - 平均时间: {stats['training_step']['mean_ms']:.2f} ms")
        print(f"💻 内存:")
        print(f"   - CPU 峰值: {stats['memory']['peak_cpu_gb']:.2f} GB")
        if stats['memory']['peak_gpu_gb'] > 0:
            print(f"   - GPU 峰值: {stats['memory']['peak_gpu_gb']:.2f} GB")
        print(f"{'='*80}\n")


class TestModel(nn.Module):
    """测试模型 - 可配置大小的 Transformer-like 模型"""
    
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_len)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        x = self.fc(x)
        return x


def create_synthetic_dataset(num_samples=1000, seq_len=128, vocab_size=10000):
    """创建合成数据集"""
    print(f"📦 创建合成数据集: {num_samples} samples, seq_len={seq_len}")
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(X, y)


def benchmark_traditional_checkpoint(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    checkpoint_freq: int,
    checkpoint_dir: str,
    num_steps: int = 100
) -> BenchmarkMetrics:
    """测试传统 PyTorch 检查点"""
    
    print(f"\n{'='*80}")
    print(f"🔵 开始测试: 传统 PyTorch 检查点")
    print(f"{'='*80}")
    
    metrics = BenchmarkMetrics("Traditional PyTorch Checkpoint")
    model.train()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    step = 0
    total_samples = 0
    start_time = time.time()
    
    data_iter = iter(train_loader)
    
    while step < num_steps:
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
        
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        
        # 训练步
        step_start = time.time()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        
        step_time = (time.time() - step_start) * 1000
        metrics.add_step_time(step_time)
        
        total_samples += batch_size
        step += 1
        
        # 检查点保存
        if step % checkpoint_freq == 0:
            chk_start = time.time()
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            
            chk_time = (time.time() - chk_start) * 1000
            metrics.add_checkpoint_time(chk_time)
            # 减少输出频率
            if step % (checkpoint_freq * 5) == 0:  # 每5次检查点输出一次
                print(f"  ✓ Step {step}: 保存检查点 ({chk_time:.2f} ms)")
        
        # 记录内存
        if step % 10 == 0:
            metrics.record_memory()
    
    total_time = time.time() - start_time
    metrics.total_time = total_time
    metrics.num_samples = total_samples
    metrics.throughput = total_samples / total_time
    
    return metrics


def benchmark_original_pccheck(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    checkpoint_freq: int,
    checkpoint_file: str,
    num_threads: int,
    max_async: int,
    num_steps: int = 100
) -> BenchmarkMetrics:
    """测试原始 PCCheck"""
    
    print(f"\n{'='*80}")
    print(f"🟢 开始测试: 原始 PCCheck")
    print(f"{'='*80}")
    
    metrics = BenchmarkMetrics("Original PCCheck")
    model.train()
    
    # 初始化 PCCheck Monitor
    print(f"📝 初始化 PCCheck Monitor:")
    
    # 使用 initialize 函数来准备 GPU 数组
    gpu_ar, total_size = initialize(model, [optimizer], do_opt_step=False)
    
    print(f"   - 模型大小: {total_size/1e6:.2f}M 参数")
    print(f"   - Threads: {num_threads}, Max async: {max_async}")
    
    # 设置存储
    set_storage(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()
    
    # 创建 Chk_monitor
    c_lib_path = "/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    gpu_copy = True if device == 'cuda' else False
    
    monitor = Chk_monitor(
        c_lib_path,
        total_size,
        num_threads,
        max_async,
        gpu_copy,
        gpu_ar=gpu_ar,
        bsize=total_size // 4,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        memory_saving=True,
        is_distributed=False,
        rank=0,
        world_size=1
    )
    
    step = 0
    total_samples = 0
    start_time = time.time()
    
    data_iter = iter(train_loader)
    
    while step < num_steps:
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
        
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        
        # 训练步
        step_start = time.time()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        
        step_time = (time.time() - step_start) * 1000
        metrics.add_step_time(step_time)
        
        total_samples += batch_size
        step += 1
        
        # 检查点保存
        if step % checkpoint_freq == 0:
            chk_start = time.time()
            
            # 更新 checkpoint_dict 中的状态
            monitor.checkpoint_dict['model'] = model.state_dict()
            monitor.checkpoint_dict['optimizer'] = optimizer.state_dict()
            
            # 使用 PCCheck Monitor 保存
            monitor.save()
            
            chk_time = (time.time() - chk_start) * 1000
            metrics.add_checkpoint_time(chk_time)
            # 减少输出频率
            if step % (checkpoint_freq * 5) == 0:  # 每5次检查点输出一次
                print(f"  ✓ Step {step}: PCCheck 保存 ({chk_time:.2f} ms)")
        
        # 记录内存
        if step % 10 == 0:
            metrics.record_memory()
    
    total_time = time.time() - start_time
    metrics.total_time = total_time
    metrics.num_samples = total_samples
    metrics.throughput = total_samples / total_time
    
    # 关闭 monitor
    monitor.kill_checkpoint()
    
    return metrics


def benchmark_layerwise_pccheck(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    device: str,
    checkpoint_dir: str,
    num_threads: int,
    max_async: int,
    buffer_size_mb: float,
    batch_size_mb: float,
    use_monitor: bool,
    checkpoint_freq: int,
    num_steps: int = 100
) -> BenchmarkMetrics:
    """测试改进的分层 PCCheck"""
    
    print(f"\n{'='*80}")
    print(f"🟣 开始测试: 改进的分层 PCCheck")
    print(f"{'='*80}")
    
    metrics = BenchmarkMetrics("Layerwise PCCheck (Improved)")
    
    # 创建分层检查点训练器
    print(f"📝 初始化分层训练器:")
    print(f"   - Threads: {num_threads}")
    print(f"   - Max async: {max_async}")
    print(f"   - Buffer size: {buffer_size_mb} MB")
    print(f"   - Batch size: {batch_size_mb} MB")
    print(f"   - Use monitor: {use_monitor}")
    
    c_lib_path = "/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    
    # 使用pccheck的init创建gpu空间
    trainer = LayerwiseCheckpointTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir=checkpoint_dir,
        buffer_size_mb=buffer_size_mb,
        use_pccheck=True,
        use_monitor=use_monitor,
        num_threads=num_threads,
        max_async=max_async,
        batch_size_mb=batch_size_mb,
        ratio=2.0,
        c_lib_path=c_lib_path,
        device=device,
        verbose=False
    )
    
    model.train()
    
    step = 0
    total_samples = 0
    start_time = time.time()
    
    data_iter = iter(train_loader)
    
    # 用于测量检查点时间的变量
    last_checkpoint_step = 0
    checkpoint_start_time = None
    
    while step < num_steps:
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
        
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        
        # 🔥 判断是否需要保存检查点
        need_checkpoint = (step % checkpoint_freq == 0 and step > 0)
        
        # 训练步 (包含分层检查点保存)
        step_start = time.time()
        
        # 检查点保存开始时刻
        if need_checkpoint:
            checkpoint_start_time = time.time()
            last_checkpoint_step = step
        
        # 🔥 使用 trainer 的 train_step，只在需要时启用检查点回调
        loss_value = trainer.train_step(
            data, target, criterion, 
            enable_checkpoint=need_checkpoint  # 🔥 关键：控制是否触发回调
        )
        
        # 如果是检查点步骤，完成检查点
        if need_checkpoint:
            trainer.finalize_checkpoint()
            chk_time = (time.time() - checkpoint_start_time) * 1000
            metrics.add_checkpoint_time(chk_time)
            # 减少输出频率，只在某些步骤输出
            if step % (checkpoint_freq * 5) == 0:  # 每5次检查点输出一次
                print(f"  ✓ Step {step}: 分层保存完成 ({chk_time:.2f} ms)")
        
        step_time = (time.time() - step_start) * 1000
        metrics.add_step_time(step_time)
        
        total_samples += batch_size
        step += 1
        
        # 记录内存
        if step % 10 == 0:
            metrics.record_memory()
    
    total_time = time.time() - start_time
    metrics.total_time = total_time
    metrics.num_samples = total_samples
    metrics.throughput = total_samples / total_time
    
    # 关闭训练器
    trainer.shutdown()
    
    return metrics


def compare_methods(results: Dict[str, BenchmarkMetrics], output_file: str):
    """对比不同方法的结果"""
    
    print(f"\n{'='*80}")
    print(f"📊 实验结果对比")
    print(f"{'='*80}\n")
    
    # 收集统计数据
    all_stats = {}
    for name, metrics in results.items():
        all_stats[name] = metrics.compute_statistics()
    
    # 打印对比表格
    print(f"{'指标':<30} {'传统':<20} {'原始PCCheck':<20} {'分层PCCheck':<20}")
    print(f"{'-'*90}")
    
    # 基准方法
    baseline_name = "Traditional PyTorch Checkpoint"
    baseline = all_stats.get(baseline_name)
    
    # 吞吐量对比
    print(f"\n🚀 吞吐量 (samples/sec):")
    for name, stats in all_stats.items():
        throughput = stats['throughput']['samples_per_sec']
        if baseline and name != baseline_name:
            speedup = throughput / baseline['throughput']['samples_per_sec']
            print(f"  {name:<30}: {throughput:>10.2f}  (speedup: {speedup:.2f}x)")
        else:
            print(f"  {name:<30}: {throughput:>10.2f}  (baseline)")
    
    # 检查点开销对比
    print(f"\n💾 检查点开销:")
    for name, stats in all_stats.items():
        overhead = stats['checkpoint_overhead_percent']
        mean_time = stats['checkpoint']['mean_ms']
        if baseline and name != baseline_name:
            reduction = (1 - overhead / baseline['checkpoint_overhead_percent']) * 100
            print(f"  {name:<30}: {overhead:>6.2f}%  (平均 {mean_time:>7.2f}ms, 降低 {reduction:>5.1f}%)")
        else:
            print(f"  {name:<30}: {overhead:>6.2f}%  (平均 {mean_time:>7.2f}ms, baseline)")
    
    # 内存使用对比
    print(f"\n💻 峰值内存 (GB):")
    for name, stats in all_stats.items():
        cpu_mem = stats['memory']['peak_cpu_gb']
        gpu_mem = stats['memory']['peak_gpu_gb']
        print(f"  {name:<30}: CPU {cpu_mem:>6.2f}, GPU {gpu_mem:>6.2f}")
    
    # 保存结果到文件
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': all_stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='PCCheck 改进效果对比实验')
    
    # 模型配置
    parser.add_argument('--vocab-size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--d-model', type=int, default=512, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=6, help='Transformer 层数')
    parser.add_argument('--dim-feedforward', type=int, default=2048, help='前馈网络维度')
    
    # 训练配置
    parser.add_argument('--num-samples', type=int, default=1000, help='训练样本数')
    parser.add_argument('--seq-len', type=int, default=128, help='序列长度')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--num-steps', type=int, default=100, help='训练步数')
    parser.add_argument('--checkpoint-freq', type=int, default=10, help='检查点保存频率')
    
    # PCCheck 配置
    parser.add_argument('--num-threads', type=int, default=8, help='PCCheck 线程数')
    parser.add_argument('--max-async', type=int, default=4, help='最大并发检查点数')
    parser.add_argument('--buffer-size-mb', type=float, default=50.0, help='缓冲区大小 (MB)')
    parser.add_argument('--batch-size-mb', type=float, default=100.0, help='PCCheck 批次大小 (MB)')
    parser.add_argument('--use-monitor', action='store_true', help='使用 Monitor 模式')
    
    # 实验配置
    parser.add_argument('--methods', nargs='+', default=['traditional', 'original', 'layerwise'],
                        choices=['traditional', 'original', 'layerwise'],
                        help='要测试的方法')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='结果输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print(f"🔬 PCCheck 改进效果对比实验")
    print(f"{'='*80}")
    print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  设备: {args.device}")
    print(f"📝 配置:")
    print(f"   - 模型: Transformer (d={args.d_model}, layers={args.num_layers})")
    print(f"   - 数据: {args.num_samples} samples, seq_len={args.seq_len}")
    print(f"   - 训练: {args.num_steps} steps, batch_size={args.batch_size}")
    print(f"   - 检查点频率: 每 {args.checkpoint_freq} 步")
    print(f"   - 测试方法: {', '.join(args.methods)}")
    print(f"{'='*80}\n")
    
    # 创建数据集
    dataset = create_synthetic_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 存储结果
    results = {}
    
    # 测试传统方法
    if 'traditional' in args.methods:
        model = TestModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward
        ).to(args.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        checkpoint_dir = os.path.join(args.output_dir, f'traditional_{timestamp}')
        
        metrics = benchmark_traditional_checkpoint(
            model, train_loader, criterion, optimizer,
            args.device, args.checkpoint_freq, checkpoint_dir, args.num_steps
        )
        metrics.print_summary()
        results['Traditional PyTorch Checkpoint'] = metrics
        
        # 清理
        del model, optimizer
        torch.cuda.empty_cache()
    
    # 测试原始 PCCheck
    if 'original' in args.methods:
        model = TestModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward
        ).to(args.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        checkpoint_file = os.path.join(args.output_dir, f'original_{timestamp}.chk')
        
        metrics = benchmark_original_pccheck(
            model, train_loader, criterion, optimizer,
            args.device, args.checkpoint_freq, checkpoint_file,
            args.num_threads, args.max_async, args.num_steps
        )
        metrics.print_summary()
        results['Original PCCheck'] = metrics
        
        # 清理
        del model, optimizer
        torch.cuda.empty_cache()
    
    # 测试分层 PCCheck
    if 'layerwise' in args.methods:
        model = TestModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward
        ).to(args.device)
        
        checkpoint_dir = os.path.join(args.output_dir, f'layerwise_{timestamp}')
        
        metrics = benchmark_layerwise_pccheck(
            model, train_loader, criterion,
            args.device, checkpoint_dir,
            args.num_threads, args.max_async,
            args.buffer_size_mb, args.batch_size_mb,
            True, args.checkpoint_freq, args.num_steps
        )
        metrics.print_summary()
        results['Layerwise PCCheck (Improved)'] = metrics
        
        # 清理
        del model
        torch.cuda.empty_cache()
    
    # 对比结果
    if len(results) > 1:
        output_file = os.path.join(args.output_dir, f'comparison_{timestamp}.json')
        compare_methods(results, output_file)
    
    print(f"\n✅ 实验完成！")


if __name__ == "__main__":
    main()
