"""
完整集成示例：五个阶段的协同工作
Complete Integration Example: All Five Stages Working Together

展示如何将所有阶段整合到一个完整的训练流程中

更新日志 (v2.0):
- ✅ 完整的 PCCheck 集成支持
- ✅ 支持 Chk_monitor 后台进程模式（更高效）
- ✅ 新增参数：num_threads, max_async, batch_size_mb, ratio
- ✅ 分布式训练支持（is_distributed, rank, world_size）
- ✅ 三种工作模式：Mock / Checkpoint 直接 / Monitor 后台进程

使用方式：
1. Mock 模式（测试）：use_pccheck=False
2. Checkpoin    trainer = LayerwiseCheckpointTrainer(
        model, 
        optimizer,
        use_pccheck=True,         # 🔥 启用真实 PCCheck
        use_monitor=False,        # 📌 直接模式（或改为 True 使用 Monitor）
        num_threads=8,            # ⚡ 8 个写入线程
        max_async=4,              # 📦 最多 4 个并发检查点（足够容纳多次保存）
        batch_size_mb=100.0,      # 💾 每批 100MB
        ratio=2.0,                # 🔧 2倍 CPU 缓冲区
        checkpoint_dir="./layerwise_checkpoints"
    )ck=True, use_monitor=False
3. Monitor 模式（推荐）：use_pccheck=True, use_monitor=True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_dependency_graph import DependencyGraphBuilder
from layerwise_optimizer import LayerwiseOptimizer
from layerwise_scheduler import LayerwiseCheckpointScheduler
from pccheck_adapter import PCCheckAdapter
from checkpoint_metadata import CheckpointMetadataManager, ModelRecovery
from pccheck_utils import initialize, set_storage


# ============================================================================
# 定义测试模型
# ============================================================================

class SimpleCNN(nn.Module):
    """简单的 CNN 模型用于测试"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# 完整的训练系统集成类
# ============================================================================

class LayerwiseCheckpointTrainer:
    """
    集成了分层检查点的训练器
    
    整合所有五个阶段：
    1. 依赖分析 (DependencyGraphBuilder)
    2. 分层优化器 (LayerwiseOptimizer)
    3. 调度器 (LayerwiseCheckpointScheduler)
    4. PCCheck 适配器 (PCCheckAdapter)
    5. 元数据管理 (CheckpointMetadataManager)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_class,
        optimizer_kwargs: dict,
        checkpoint_dir: str = "./layerwise_checkpoints",
        buffer_size_mb: float = 50.0,
        checkpoint_chunk_count: int = 3,
        use_pccheck: bool = False,
        use_monitor: bool = False,
        num_threads: int = 8,
        max_async: int = 2,
        batch_size_mb: float = 100.0,
        ratio: float = 2.0,
        c_lib_path: str = None,
        is_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        """
        Args:
            model: PyTorch 模型
            optimizer_class: 优化器类（如 torch.optim.Adam）
            optimizer_kwargs: 优化器参数（如 {'lr': 0.001}）
            checkpoint_dir: 检查点保存目录
            buffer_size_mb: 缓冲区大小（MB）
            use_pccheck: 是否使用真实的 PCCheck
            use_monitor: 是否使用 Chk_monitor（后台进程模式，更高效）
            num_threads: PCCheck 使用的线程数
            max_async: 最大并发检查点数量
            batch_size_mb: PCCheck 批次大小（MB）
            ratio: CPU 缓冲区大小相对于检查点的倍数
            c_lib_path: PCCheck C 库路径
            is_distributed: 是否为分布式训练
            rank: 当前进程的 rank（分布式训练用）
            world_size: 总进程数（分布式训练用）
            device: 训练设备
            verbose: 是否打印详细信息
        """
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        
        # 🔥 新增：批量保存模式（当使用 gpu_ar 时，跳过细粒度调度）
        self.use_batch_checkpoint = use_pccheck  # 如果使用 PCCheck，默认启用批量模式
        # 将整个 gpu_ar 分为多少个 chunk 并行保存（1=不分块）
        self.checkpoint_chunk_count = max(1, int(checkpoint_chunk_count))
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if verbose:
            print("="*100)
            print("初始化分层检查点训练系统")
            print("="*100)
        
        # ====================================================================
        # 阶段一：构建依赖图
        # ====================================================================
        if verbose:
            print("\n[阶段 1/5] 构建参数更新依赖图...")
        self.dependency_builder = DependencyGraphBuilder(model, verbose=False)
        self.dependency_builder.build_dependency_graph()
        self.update_order = self.dependency_builder.get_update_order()
        self.layer_info = self.dependency_builder.layer_info
        
        if verbose:
            print(f"  ✓ 检测到 {len(self.update_order)} 个可训练层")
            print(f"  ✓ 总参数数: {sum(info['param_count'] for info in self.layer_info.values()):,}")
        
        # ====================================================================
        # 阶段五：创建元数据管理器（需要先创建，后面要用）
        # ====================================================================
        if verbose:
            print("\n[阶段 5/5] 初始化元数据管理器...")
        metadata_dir = os.path.join(checkpoint_dir, "metadata")
        self.metadata_manager = CheckpointMetadataManager(
            metadata_dir=metadata_dir,
            verbose=False
        )
        if verbose:
            print(f"  ✓ 元数据目录: {metadata_dir}")
        
        # ====================================================================
        # 阶段二：创建分层优化器（需要先创建，后面 initialize 要用）
        # ====================================================================
        if verbose:
            print("\n[阶段 2/5] 创建分层优化器...")
            
        base_optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # 🔥 批量模式：禁用逐层回调，减少开销
        enable_callback = not self.use_batch_checkpoint
        
        self.optimizer = LayerwiseOptimizer(
            optimizer=base_optimizer,
            model=model,
            update_order=self.update_order,
            layer_info=self.layer_info,
            callback=self._optimizer_callback if enable_callback else None,  # 🔥 批量模式下不设置回调
            enable_timing=enable_callback,  # 批量模式下不需要计时
            verbose=False
        )
        
        if verbose and self.use_batch_checkpoint:
            print(f"  ✓ 使用批量保存模式（跳过逐层回调）")

        gpu_ar = None
        total_size = 0
        
        if use_pccheck and torch.cuda.is_available():
            if verbose:
                print("\n[PCCheck] 计算模型和优化器的总大小...")
            
            try:
                # 🔥 修复：使用 do_opt_step=True 来正确初始化优化器状态
                # 这确保 gpu_ar 包含足够的空间用于：模型参数 + 优化器状态（+ 梯度）
                print(f"initialize start (do_opt_step=True for correct buffer allocation)")
                gpu_ar, total_size = initialize(model, [base_optimizer], do_opt_step=True)
                
                if verbose:
                    print(f"   - 总大小: {total_size/1e6:.2f}M 参数")
                    print(f"   - Threads: {num_threads}, Max async: {max_async}")
                
                # 🔥 调试：验证缓冲区大小是否足够
                model_params_size = sum(p.numel() for p in model.parameters())
                grad_size = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
                opt_state_size = total_size - model_params_size
                
                if verbose:
                    print(f"   - 模型参数: {model_params_size/1e6:.2f}M")
                    print(f"   - 梯度空间: {grad_size/1e6:.2f}M")
                    print(f"   - 优化器状态: {opt_state_size/1e6:.2f}M")
                    print(f"   - GPU 缓冲区: {len(gpu_ar)/1e6:.2f}M (应 >= 参数+梯度)")
                
                # 断言：确保缓冲区足够大
                required_size = model_params_size + grad_size
                assert len(gpu_ar) >= required_size, \
                    f"GPU 缓冲区不足！需要 {required_size/1e6:.2f}M，实际 {len(gpu_ar)/1e6:.2f}M"
                
                # 设置存储（将模型参数和梯度映射到 gpu_ar）
                set_storage(model, [base_optimizer], gpu_ar)
                
                if verbose:
                    print(f"   ✓ set_storage 完成，参数已重定向到统一缓冲区")
                    # 验证参数确实被重定向
                    for name, p in list(model.named_parameters())[:2]:
                        print(f"     {name}: data_ptr={p.data_ptr()}, device={p.device}")
                
                print(f"initialize end")
                
            except Exception as e:
                print(f"  ⚠️ gpu_ar 分配失败: {e}")
                import traceback
                traceback.print_exc()
                gpu_ar = None
                total_size = 0
                use_pccheck = False  # 禁用 PCCheck
        
        # ====================================================================
        # 阶段四：创建 PCCheck 适配器（传入 gpu_ar）
        # ====================================================================
        if verbose:
            print("\n[阶段 4/5] 初始化 PCCheck 适配器...")
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.chk")
        self.pccheck_adapter = PCCheckAdapter(
            c_lib_path=c_lib_path or "./libtest_ssd.so",
            checkpoint_file=checkpoint_file,
            num_threads=num_threads,
            max_async=max_async,
            batch_size_mb=batch_size_mb,
            ratio=ratio,
            use_pccheck=use_pccheck,
            use_monitor=use_monitor,
            metadata_file=os.path.join(metadata_dir, "adapter_metadata.json"),
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            gpu_ar=gpu_ar,  # 🔥 传入原 PCCheck 构造的 gpu_ar
            total_size=total_size,  # 🔥 传入总大小
            verbose=False
        )
        
        # 如果没有使用 gpu_ar，则分配 staging buffer
        if gpu_ar is None:
            self.pccheck_adapter.allocate_staging_buffer(size_mb=buffer_size_mb * 2)
            
        if verbose:
            print(f"  ✓ 检查点文件: {checkpoint_file}")
            mode_str = "PCCheck Monitor" if (use_pccheck and use_monitor) else ("PCCheck" if use_pccheck else "Mock")
            print(f"  ✓ 模式: {mode_str}")
            if use_pccheck:
                print(f"  ✓ 线程数: {num_threads}")
                print(f"  ✓ 最大异步数: {max_async}")
                print(f"  ✓ 批次大小: {batch_size_mb} MB")
                if use_monitor:
                    print(f"  ⚡ Monitor 模式：异步保存已启用（预期 ~2-5ms 开销）")
                else:
                    print(f"  ⚠️  直接模式：同步保存（预期 ~273ms 开销）")
                if gpu_ar is not None:
                    print(f"  ✓ 使用原 PCCheck gpu_ar: {total_size * 4 / (1024**2):.2f} MB")
        
        # ====================================================================
        # 阶段三：创建调度器
        # ====================================================================
        if verbose:
            print("\n[阶段 3/5] 初始化检查点调度器...")
        self.scheduler = LayerwiseCheckpointScheduler(
            save_callback=self._save_callback,
            buffer_size_mb=buffer_size_mb,
            buffer_timeout_ms=100.0,
            enable_async=True,
            metadata_dir=metadata_dir,
            verbose=False
        )
        if verbose:
            print(f"  ✓ 缓冲区大小: {buffer_size_mb} MB")
            print(f"  ✓ 异步模式: 已启用")
        
        # ====================================================================
        # 优化器配置
        # ====================================================================
        # 🔥 新增：默认使用手动模式（只在需要时触发回调，节省开销）
        self.optimizer.set_checkpoint_mode('manual')
        
        if verbose:
            print(f"\n[优化器配置]")
            print(f"  ✓ 优化器: {optimizer_class.__name__}")
            print(f"  ✓ 参数: {optimizer_kwargs}")
            print(f"  ✓ 检查点模式: manual (仅在需要时触发回调)")
        
        # 训练统计
        self.current_training_step = 0
        self.current_checkpoint_id = None
        
        if verbose:
            print("\n" + "="*100)
            print("✓ 所有组件初始化完成！")
            print("="*100)
    
    def _optimizer_callback(self, layer_name: str, step: int, layer_params: dict):
        """
        从 LayerwiseOptimizer 接收回调（阶段二 → 阶段三）
        """
        # 将任务传递给调度器
        self.scheduler.schedule_save(layer_name, step, layer_params)
        
        # 同时记录到元数据管理器
        if self.current_checkpoint_id:
            # 计算偏移量（简化版，实际需要累积）
            offset = self.pccheck_adapter.current_file_offset
            size_bytes = sum(p.numel() * p.element_size() for p in layer_params['parameters'])
            
            try:
                self.metadata_manager.add_layer(
                    checkpoint_id=self.current_checkpoint_id,
                    layer_name=layer_name,
                    offset=offset,
                    size_bytes=size_bytes,
                    param_count=layer_params['param_count'],
                    shapes=layer_params['shapes'],
                    dtypes=[str(dt) for dt in layer_params['dtypes']]
                )
            except KeyError:
                # 如果检查点未注册，先注册
                self.metadata_manager.register_checkpoint(
                    checkpoint_id=self.current_checkpoint_id,
                    training_step=step,
                    checkpoint_file=self.pccheck_adapter.checkpoint_file
                )
                # 再次尝试添加层
                self.metadata_manager.add_layer(
                    checkpoint_id=self.current_checkpoint_id,
                    layer_name=layer_name,
                    offset=offset,
                    size_bytes=size_bytes,
                    param_count=layer_params['param_count'],
                    shapes=layer_params['shapes'],
                    dtypes=[str(dt) for dt in layer_params['dtypes']]
                )
    
    def _save_callback(self, tasks):
        """
        从调度器接收批量保存任务（阶段三 → 阶段四）
        """
        # 将任务传递给 PCCheck 适配器
        self.pccheck_adapter.save_layers_batch(tasks)
    
    def train_step(self, inputs, labels, criterion, enable_checkpoint: bool = False):
        """
        执行一步训练
        
        Args:
            inputs: 输入数据
            labels: 标签
            criterion: 损失函数
            enable_checkpoint: 🔥 新增：是否启用检查点保存（默认False，不触发回调）
        
        Returns:
            loss: 损失值 (scalar)
        """
        self.current_training_step += 1
        self.current_checkpoint_id = f"step_{self.current_training_step}"
        
        # 🔥 设置检查点标志（控制优化器是否触发回调）
        if not self.use_batch_checkpoint:
            # 细粒度模式：启用回调
            self.optimizer.enable_checkpointing(enable_checkpoint)
        
        # 前向传播
        outputs = self.model(inputs)
        
        # 处理不同的输出格式
        # 如果是 Transformer 输出 (batch, seq_len, vocab_size)，需要重塑
        if len(outputs.shape) == 3:
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            loss = criterion(outputs, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 🔥 更新参数
        self.optimizer.step()
        
        # 🔥 批量模式：在 step 完成后一次性保存
        if enable_checkpoint and self.use_batch_checkpoint:
            self._save_checkpoint_batch()
        
        return loss.item()
    
    def _save_checkpoint_batch(self):
        """
        批量保存模式：直接保存整个 gpu_ar
        
        适用于：参数通过 set_storage 在 gpu_ar 中
        优势：
        - 0 次回调开销（vs 148 次）
        - 1 次 I/O（vs 5-7 次）
        - 大块写入（500MB vs 100MB），饱和带宽
        """
        if self.pccheck_adapter.gpu_ar is None:
            if self.verbose:
                print("[Trainer] 警告：批量模式但 gpu_ar 为空，跳过保存")
            return
        
        # 直接调用适配器的批量保存接口
        if getattr(self, 'checkpoint_chunk_count', 1) > 1:
            # 分块并行保存
            self.pccheck_adapter.save_entire_checkpoint_in_chunks(
                checkpoint_id=self.current_checkpoint_id,
                training_step=self.current_training_step,
                chunk_count=self.checkpoint_chunk_count
            )
        else:
            self.pccheck_adapter.save_entire_checkpoint(
                checkpoint_id=self.current_checkpoint_id,
                training_step=self.current_training_step
            )
    
    def finalize_checkpoint(self):
        """完成当前检查点的保存"""
        if self.current_checkpoint_id:
            # 🔥 批量模式：保存已在 train_step 中完成，这里只需保存元数据
            if not self.use_batch_checkpoint:
                # 细粒度模式：强制刷新调度器
                self.scheduler.force_flush()
            
            # 保存元数据
            self.metadata_manager.save_metadata(self.current_checkpoint_id)
            
            # 验证检查点
            if not self.use_batch_checkpoint:
                total_layers = len(self.update_order)
                self.scheduler.finalize_checkpoint(self.current_training_step, total_layers)
    
    def shutdown(self):
        """关闭训练系统"""
        if self.verbose:
            print("\n" + "="*100)
            print("关闭分层检查点训练系统")
            print("="*100)
        
        # 关闭各个组件
        if self.verbose:
            print("\n[1/4] 关闭调度器（并完成最后的检查点）...")
        # 强制刷新调度器以处理所有剩余任务
        self.scheduler.force_flush()
        # 保存最后的元数据
        if self.current_checkpoint_id:
            self.metadata_manager.save_metadata(self.current_checkpoint_id)
        # 现在关闭调度器
        self.scheduler.shutdown()
        
        if self.verbose:
            print("\n[2/4] 关闭 PCCheck 适配器...")
        self.pccheck_adapter.shutdown()
        
        if self.verbose:
            print("\n[3/4] 保存所有元数据...")
        self.metadata_manager.save_metadata()
        
        if self.verbose:
            print("\n[4/4] 打印优化器统计...")
            self.optimizer.print_timing_stats()
        
        if self.verbose:
            print("\n" + "="*100)
            print("✓ 系统已关闭")
            print("="*100)


# ============================================================================
# 主函数：完整的训练和恢复演示
# ============================================================================

def main():
    """完整演示：训练 + 保存 + 恢复"""
    
    print("\n" + "="*100)
    print("分层检查点系统 - 完整演示")
    print("="*100)
    
    # ========================================================================
    # Part 1: 训练并保存检查点
    # ========================================================================
    print("\n" + "="*100)
    print("Part 1: 训练并保存分层检查点")
    print("="*100)
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 创建训练数据
    num_samples = 100
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    # 创建训练器
    trainer = LayerwiseCheckpointTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        checkpoint_dir="./demo_checkpoints",
        buffer_size_mb=10.0,
        use_pccheck=True,          # 启用 PCCheck
        use_monitor=True,         # 是否使用 Monitor 模式（True 更高效）
        num_threads=8,             # 8 个并行线程
        max_async=4,               # 最多 4 个并发检查点
        batch_size_mb=100.0,       # 每批 100MB
        ratio=2.0,                 # CPU 缓冲区是检查点的 2 倍
        device='cuda',             # 使用 GPU
        c_lib_path="/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so",
        verbose=True
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\n" + "-"*100)
    print("开始训练...")
    print("-"*100)
    
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 将数据移到设备
            inputs = inputs.to(trainer.device)
            labels = labels.to(trainer.device)
            
            # 训练一步（会自动保存检查点）
            loss, outputs = trainer.train_step(inputs, labels, criterion)
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            epoch_loss += loss
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                print(f"  Batch [{batch_idx + 1:2d}/{len(train_loader):2d}] | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        # Epoch 结束
        avg_loss = epoch_loss / len(train_loader)
        acc = 100.0 * correct / total
        print(f"  → Epoch {epoch + 1} 完成 | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    
    # 关闭训练器
    trainer.shutdown()
    
    # ========================================================================
    # Part 2: 从检查点恢复模型
    # ========================================================================
    print("\n" + "="*100)
    print("Part 2: 从检查点恢复模型")
    print("="*100)
    
    # 创建新模型（随机初始化）
    new_model = SimpleCNN(num_classes=10)
    print("\n创建新模型（随机初始化）")
    
    # 创建恢复器
    recovery = ModelRecovery(
        metadata_manager=trainer.metadata_manager,
        verbose=True
    )
    
    # 列出可用的检查点
    print("\n" + "-"*100)
    print("可用的检查点:")
    available_checkpoints = recovery.list_available_checkpoints()
    for chk_id, step in available_checkpoints:
        info = recovery.get_checkpoint_info(chk_id)
        if info:
            print(f"\n  {chk_id}:")
            print(f"    - 训练步数: {info['training_step']}")
            print(f"    - 层数: {info['total_layers']}")
            print(f"    - 大小: {info['total_size_gb']:.4f} GB")
            print(f"    - 创建时间: {info['created_at']}")
    
    # 恢复最新的检查点
    if available_checkpoints:
        latest_checkpoint = available_checkpoints[-1][0]
        print(f"\n" + "-"*100)
        print(f"恢复检查点: {latest_checkpoint}")
        print("-"*100)
        
        try:
            stats = recovery.load_checkpoint(
                model=new_model,
                checkpoint_id=latest_checkpoint,
                device='cpu',
                strict=False  # 宽松模式，允许部分恢复
            )
            
            print(f"\n恢复统计:")
            print(f"  - 总层数: {stats['total_layers']}")
            print(f"  - 已加载: {stats['loaded_layers']}")
            print(f"  - 缺失层: {len(stats['missing_layers'])}")
            print(f"  - 额外层: {len(stats['unexpected_layers'])}")
            
            if stats['missing_layers']:
                print(f"  - 缺失的层: {stats['missing_layers'][:5]}...")
            
        except Exception as e:
            print(f"\n恢复失败: {e}")
            print("注意：这可能是因为元数据与实际数据文件不完全匹配")
            print("在生产环境中，需要更完善的错误处理")
    

if __name__ == "__main__":
    main()
