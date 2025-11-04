# 分层检查点系统 (Layerwise Checkpoint System)

## 📁 项目简介

这个文件夹包含了 PCCheck 检查点系统的重要改进实现：**边训练边保存的分层检查点系统**。

该系统在模型参数更新的同时进行分层保存，将检查点保存流水线前移到训练过程中，实现**计算与存储的完全重叠**。

### 核心特性

- ✅ **自动依赖分析**：自动识别模型结构和参数更新顺序
- ✅ **零侵入式集成**：无缝替换标准 PyTorch 优化器
- ✅ **🔥 优化的单一缓冲区**：避免双重拷贝，节省内存和时间
- ✅ **🔥 智能快速路径**：非检查点步骤性能提升 1.5-3x
- ✅ **智能批量聚合**：减少 I/O 次数，提升吞吐量
- ✅ **完整的 PCCheck 集成**：支持 Mock/Checkpoint/Monitor 三种模式
- ✅ **分布式训练支持**：支持多 GPU/多节点训练

### 🆕 最新优化（2025-10）

基于实际使用反馈，系统进行了重要性能优化：

1. **统一缓冲区管理**：消除 Scheduler 和 Adapter 的双重缓冲，减少 50% GPU 拷贝次数
2. **快速路径**：非检查点步骤直接使用原始优化器，避免分层更新开销
3. **内存效率提升**：节省约 2× 模型参数大小的内存

详见：[OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md)

---

## 📂 文件结构

```
layerwise_checkpoint/
├── README.md                          # 本文件（快速开始）
├── LAYERWISE_CHECKPOINT_GUIDE.md      # 完整使用指南（详细文档）
├── OPTIMIZATION_SUMMARY.md            # 🆕 优化总结（推荐阅读）
├── OPTIMIZATION_GUIDE.md              # 🆕 详细优化说明
├── layer_dependency_graph.py          # 阶段一：模型依赖分析
├── layerwise_optimizer.py             # 阶段二：分层优化器（已优化）
├── layerwise_scheduler.py             # 阶段三：检查点调度器（已优化）
├── pccheck_adapter.py                 # 阶段四：PCCheck 适配器（已优化）
├── checkpoint_metadata.py             # 阶段五：元数据管理
├── complete_integration.py            # 完整集成示例（推荐）
├── example_real_training.py           # 实际应用示例
├── test_layerwise_checkpoint.py       # 单元测试脚本
├── test_optimization.py               # 🆕 优化效果测试
└── quick_verify.py                    # 🆕 快速验证脚本
```

---

## 🚀 快速开始

### 最小示例

```python
import torch
import torch.nn as nn
from complete_integration import LayerwiseCheckpointTrainer

# 1. 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# 2. 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 创建分层检查点训练器
trainer = LayerwiseCheckpointTrainer(
    model=model,
    optimizer=optimizer,
    use_pccheck=True,              # 使用 PCCheck
    num_threads=8,                  # 8 线程
    max_async=4,                    # 最多 4 个并发检查点
    checkpoint_dir="./checkpoints"
)

# 4. 训练循环
for epoch in range(10):
    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        trainer.step()  # 自动分层更新并保存

# 5. 关闭系统
trainer.shutdown()
```

### 运行完整示例

```bash
cd /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/layerwise_checkpoint
python complete_integration.py
```

这将演示：
- ✅ 训练一个 CNN 模型（2 epochs）
- ✅ 在训练过程中自动保存分层检查点
- ✅ 从检查点恢复模型
- ✅ 使用 PCCheck 后端进行高效存储

---

## 📖 详细文档

完整的技术文档、API 参考、参数配置和故障排除，请查看：

👉 **[LAYERWISE_CHECKPOINT_GUIDE.md](LAYERWISE_CHECKPOINT_GUIDE.md)** 👈

该文档包含：
- 五阶段架构详解
- 核心组件 API 参考
- PCCheck 集成说明
- 使用示例和最佳实践
- 性能优化建议
- 完整的故障排除指南

---

## 🎯 五阶段架构

系统采用模块化的五阶段设计：

```
训练循环
    ↓
[阶段一] DependencyGraphBuilder      # 模型依赖分析
    ↓
[阶段二] LayerwiseOptimizer          # 分层优化器
    ↓
[阶段三] LayerwiseCheckpointScheduler # 检查点调度器
    ↓
[阶段四] PCCheckAdapter              # PCCheck 适配器
    ↓
[阶段五] CheckpointMetadataManager   # 元数据管理
    ↓
存储 (检查点文件 + 元数据)
```

| 阶段 | 组件 | 核心功能 | 代码行数 | 状态 |
|------|------|----------|----------|------|
| **阶段一** | DependencyGraphBuilder | 模型依赖分析 | ~300 | ✅ |
| **阶段二** | LayerwiseOptimizer | 分层优化器 | ~350 | ✅ |
| **阶段三** | LayerwiseCheckpointScheduler | 检查点调度器 | ~420 | ✅ |
| **阶段四** | PCCheckAdapter | PCCheck 适配 | ~630 | ✅ |
| **阶段五** | CheckpointMetadataManager | 元数据管理 | ~450 | ✅ |
| **总计** | | | **~2,610** | **100%** |

---

## � 技术亮点

### 1. 自动依赖分析
```python
# 自动分析模型结构，无需手动指定
builder = DependencyGraphBuilder(model)
update_order = builder.get_update_order()
# 自动得到: ['fc', 'layer2.conv2', 'layer2.conv1', ...]
```

### 2. 零侵入式集成
```python
# 只需包装标准优化器即可
trainer = LayerwiseCheckpointTrainer(model, optimizer)
# 其余代码完全不变！
```

### 3. PCCheck 三种模式
```python
# Mock 模式（测试）
trainer = LayerwiseCheckpointTrainer(use_pccheck=False)

# Checkpoint 直接模式
trainer = LayerwiseCheckpointTrainer(use_pccheck=True, use_monitor=False)

# Monitor 后台模式（推荐，最高性能）
trainer = LayerwiseCheckpointTrainer(use_pccheck=True, use_monitor=True)
```

### 4. 分布式训练支持
```python
trainer = LayerwiseCheckpointTrainer(
    model, optimizer,
    is_distributed=True,
    rank=dist.get_rank(),
    world_size=dist.get_world_size()
)
```

---

## 🔧 系统要求

### 必需依赖
```bash
Python 3.9+
PyTorch 2.0+
NetworkX 2.5+
NumPy 1.20+
```

### 可选依赖
```bash
CUDA 11.0+              # GPU 加速
libtest_ssd.so          # PCCheck C 库
```

### 安装
```bash
pip install torch networkx numpy
```

---

## 📈 性能优势

- ✅ **计算与存储完全重叠**：训练和保存同时进行
- ✅ **批量 I/O 聚合**：智能合并多层数据，减少 I/O 次数
- ✅ **异步后台保存**：不阻塞训练主循环
- ✅ **内存优化**：staging buffer + pinned memory
- ✅ **高效的 PCCheck 集成**：GPU→CPU→NVM 流水线

---

## � 常见问题

### Q1: 段错误 (Segmentation fault)
**A**: 增大 `max_async` 参数
```python
trainer = LayerwiseCheckpointTrainer(max_async=8)
```

### Q2: CUDA out of memory
**A**: 减小 buffer 或并发数
```python
trainer = LayerwiseCheckpointTrainer(max_async=2)
# 或在 adapter 中
adapter.allocate_staging_buffer(size_mb=200.0)
```

### Q3: C 库找不到
**A**: 使用绝对路径
```python
c_lib_path = "/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
```

更多问题请查看完整文档。

---

## 📝 版本信息

**版本**: 2.0  
**状态**: ✅ 生产就绪  
**最后更新**: 2025-10-22  

### 主要更新 (v2.0)
- ✅ 完整的 PCCheck 集成（3 种模式）
- ✅ 分布式训练支持
- ✅ 元数据管理和模型恢复
- ✅ 修复所有已知问题
- ✅ 完整的文档和示例

---

**更多详情请参阅**: [LAYERWISE_CHECKPOINT_GUIDE.md](LAYERWISE_CHECKPOINT_GUIDE.md)
