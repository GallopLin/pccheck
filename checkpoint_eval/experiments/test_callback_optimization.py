#!/usr/bin/env python3
"""
测试条件回调优化的效果
验证：
1. 不需要checkpoint时，不触发回调
2. 需要checkpoint时，正常触发回调
3. 性能提升明显
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pccheck.layerwise_checkpoint.complete_integration import LayerwiseCheckpointTrainer


class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=128, hidden_size=256, num_layers=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 10))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def test_callback_behavior():
    """测试1：验证回调行为"""
    print("="*80)
    print("测试1：验证条件回调机制")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    
    # 计数器
    callback_count = [0]
    
    # 创建 trainer
    trainer = LayerwiseCheckpointTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir='/tmp/test_callback_opt',
        use_pccheck=False,  # 使用模拟模式
        verbose=False
    )
    
    # 重写回调计数
    original_callback = trainer._optimizer_callback
    def counting_callback(*args, **kwargs):
        callback_count[0] += 1
        return original_callback(*args, **kwargs)
    trainer.optimizer.callback = counting_callback
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n步骤1: 运行10步训练，每5步保存一次检查点")
    print("-"*80)
    
    for step in range(1, 11):
        # 生成假数据
        inputs = torch.randn(4, 128).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        # 判断是否需要检查点
        need_checkpoint = (step % 5 == 0)
        
        # 训练
        loss = trainer.train_step(
            inputs, labels, criterion,
            enable_checkpoint=need_checkpoint
        )
        
        print(f"  Step {step:2d}: enable_checkpoint={need_checkpoint}, "
              f"loss={loss:.4f}")
        
        if need_checkpoint:
            trainer.finalize_checkpoint()
    
    print(f"\n✓ 总共触发回调次数: {callback_count[0]}")
    
    # 获取层数
    num_layers = len(trainer.update_order)
    print(f"✓ 模型总层数: {num_layers}")
    
    expected_callbacks = 2 * num_layers  # Step 5 和 Step 10
    print(f"✓ 预期回调次数: {expected_callbacks} (2次检查点 × {num_layers}层)")
    
    if callback_count[0] == expected_callbacks:
        print("\n✅ 测试通过！回调只在需要时触发。")
    else:
        print(f"\n❌ 测试失败！预期 {expected_callbacks} 次，实际 {callback_count[0]} 次")
    
    trainer.shutdown()
    
    return callback_count[0] == expected_callbacks


def test_performance_improvement():
    """测试2：对比性能提升"""
    print("\n" + "="*80)
    print("测试2：性能对比测试")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试配置
    num_steps = 100
    checkpoint_freq = 10  # 每10步保存一次
    
    print(f"\n配置: {num_steps}步训练, 每{checkpoint_freq}步保存检查点")
    print("-"*80)
    
    # ========== 测试1：优化后的版本（条件回调） ==========
    print("\n[1] 优化后版本 (条件回调)...")
    
    model1 = SimpleModel().to(device)
    trainer1 = LayerwiseCheckpointTrainer(
        model=model1,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir='/tmp/test_callback_opt1',
        use_pccheck=False,
        verbose=False
    )
    
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for step in range(1, num_steps + 1):
        inputs = torch.randn(4, 128).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        need_checkpoint = (step % checkpoint_freq == 0)
        loss = trainer1.train_step(
            inputs, labels, criterion,
            enable_checkpoint=need_checkpoint
        )
        
        if need_checkpoint:
            trainer1.finalize_checkpoint()
    
    optimized_time = time.time() - start_time
    trainer1.shutdown()
    
    print(f"  完成时间: {optimized_time:.3f}秒")
    
    # ========== 测试2：未优化版本（每步都回调） ==========
    print("\n[2] 未优化版本 (每步都回调)...")
    
    model2 = SimpleModel().to(device)
    trainer2 = LayerwiseCheckpointTrainer(
        model=model2,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir='/tmp/test_callback_opt2',
        use_pccheck=False,
        verbose=False
    )
    
    # 🔥 强制使用 auto 模式（每步都回调）
    trainer2.optimizer.set_checkpoint_mode('auto')
    
    start_time = time.time()
    for step in range(1, num_steps + 1):
        inputs = torch.randn(4, 128).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        # enable_checkpoint 参数在 auto 模式下被忽略
        loss = trainer2.train_step(
            inputs, labels, criterion,
            enable_checkpoint=False  # 即使设为False，auto模式仍会触发
        )
        
        if step % checkpoint_freq == 0:
            trainer2.finalize_checkpoint()
    
    unoptimized_time = time.time() - start_time
    trainer2.shutdown()
    
    print(f"  完成时间: {unoptimized_time:.3f}秒")
    
    # ========== 结果对比 ==========
    print("\n" + "="*80)
    print("性能对比结果")
    print("="*80)
    print(f"优化后版本:   {optimized_time:.3f}秒")
    print(f"未优化版本:   {unoptimized_time:.3f}秒")
    print(f"加速比:       {unoptimized_time/optimized_time:.2f}x")
    print(f"时间节省:     {(unoptimized_time - optimized_time):.3f}秒 "
          f"({(1 - optimized_time/unoptimized_time)*100:.1f}%)")
    
    if optimized_time < unoptimized_time:
        improvement = (unoptimized_time - optimized_time) / unoptimized_time * 100
        print(f"\n✅ 优化成功！性能提升 {improvement:.1f}%")
        return True
    else:
        print(f"\n⚠️  未见明显性能提升")
        return False


def test_checkpoint_correctness():
    """测试3：验证检查点正确性"""
    print("\n" + "="*80)
    print("测试3：验证检查点保存正确性")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    
    trainer = LayerwiseCheckpointTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir='/tmp/test_callback_correctness',
        use_pccheck=False,
        verbose=False
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n运行5步训练，每2步保存检查点...")
    
    for step in range(1, 6):
        inputs = torch.randn(4, 128).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        need_checkpoint = (step % 2 == 0)
        loss = trainer.train_step(
            inputs, labels, criterion,
            enable_checkpoint=need_checkpoint
        )
        
        if need_checkpoint:
            trainer.finalize_checkpoint()
            print(f"  Step {step}: 保存检查点")
    
    # 检查元数据
    saved_checkpoints = list(trainer.metadata_manager.checkpoints.keys())
    print(f"\n保存的检查点: {saved_checkpoints}")
    
    expected_checkpoints = ['step_2', 'step_4']
    if saved_checkpoints == expected_checkpoints:
        print("✅ 检查点保存正确！")
        result = True
    else:
        print(f"❌ 检查点不正确！预期 {expected_checkpoints}")
        result = False
    
    trainer.shutdown()
    return result


if __name__ == '__main__':
    print("\n" + "="*80)
    print("条件回调优化测试")
    print("="*80)
    
    # 运行所有测试
    test1_pass = test_callback_behavior()
    test2_pass = test_performance_improvement()
    test3_pass = test_checkpoint_correctness()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"测试1 (回调行为):     {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"测试2 (性能提升):     {'✅ 通过' if test2_pass else '⚠️  未见提升'}")
    print(f"测试3 (检查点正确性): {'✅ 通过' if test3_pass else '❌ 失败'}")
    
    if test1_pass and test3_pass:
        print("\n🎉 优化实现成功！")
    else:
        print("\n⚠️  部分测试未通过，需要检查")
