# PCCheck 改进工作实验测试指南

## 📌 概述

本目录包含完整的实验框架，用于评估和对比 PCCheck 分层检查点系统的改进效果。

## 🎯 改进要点

你的 PCCheck 改进工作主要包括：

1. **分层参数更新检查点** - 在参数更新时立即保存，而非等待整个训练步完成
2. **异步流水线处理** - 使用缓冲区和后台线程实现训练和保存的重叠
3. **智能调度器** - 批量聚合小层，减少 I/O 次数
4. **PCCheck 后端集成** - 利用 PCCheck 的高效 NVM 写入能力
5. **完善的元数据管理** - 支持模型恢复和断点续训

## 🚀 快速开始

### 步骤 1: 快速验证（推荐首先运行）

```bash
cd /home/linzhicheng/code/pccheck/checkpoint_eval/experiments
chmod +x start.sh
./start.sh
# 选择选项 1 - 快速验证测试
```

这将运行一个 2-3 分钟的小型测试，验证：
- ✅ 系统正常工作
- ✅ PCCheck 库正确加载
- ✅ 基本性能提升可见

### 步骤 2: 完整性能测试

```bash
./start.sh
# 选择选项 2 - 完整性能测试
```

或者直接运行：

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

这将运行约 30-60 分钟的完整测试套件，包括：
1. 小型模型测试
2. 中型模型测试
3. 大型模型测试
4. 不同检查点频率对比
5. Monitor 模式对比

### 步骤 3: 查看结果

```bash
# 查看生成的报告
cat benchmark_results_*/summary_report.md

# 或使用你喜欢的编辑器
code benchmark_results_*/summary_report.md
```

## 📊 实验内容

### 1. 三种方法对比

| 方法 | 描述 | 特点 |
|------|------|------|
| **传统 PyTorch** | `torch.save` | 阻塞式保存，简单但慢 |
| **原始 PCCheck** | `Chk_monitor` | 异步保存整个模型 |
| **分层 PCCheck** | 改进方案 | 边训练边保存，流水线处理 |

### 2. 测量指标

- **吞吐量** (samples/sec) - 越高越好
- **检查点开销** (%) - 越低越好
- **平均检查点时间** (ms) - 越低越好
- **内存峰值** (GB) - 在合理范围内越低越好
- **加速比** (speedup) - 分层方法相对传统方法的提升

### 3. 实验场景

```
experiments/
├── 小型模型 (d=256, 2 layers)    # 快速验证
├── 中型模型 (d=512, 6 layers)    # 标准场景
├── 大型模型 (d=1024, 12 layers)  # 高负载测试
├── 频率测试 (5, 10, 20, 50 步)   # 敏感性分析
└── 模式对比 (Direct vs Monitor)  # 实现对比
```

## 📈 预期结果

基于改进的设计，预期看到：

### 性能提升
- ✅ **1.5x - 3x** 吞吐量提升
- ✅ **40% - 70%** 检查点开销降低
- ✅ **50% - 80%** 保存时间减少

### 关键优势
1. **训练不被阻塞** - 参数更新后立即返回继续训练
2. **I/O 完全重叠** - 保存操作在后台异步进行
3. **批量聚合优化** - 减少小文件写入次数
4. **频繁保存可行** - 开销降低后可以更频繁地保存

## 🔧 自定义实验

### 调整模型大小

```bash
python benchmark_comparison.py \
    --d-model 768 \
    --num-layers 8 \
    --num-steps 200
```

### 测试不同保存频率

```bash
python benchmark_comparison.py \
    --checkpoint-freq 5  # 每 5 步保存一次
```

### 调整 PCCheck 参数

```bash
python benchmark_comparison.py \
    --num-threads 16 \      # 更多线程
    --max-async 8 \         # 更多并发检查点
    --buffer-size-mb 100    # 更大缓冲区
```

### 只测试特定方法

```bash
# 只测试分层方法
python benchmark_comparison.py --methods layerwise

# 对比传统和分层
python benchmark_comparison.py --methods traditional layerwise
```

## 📝 实验报告结构

生成的报告包含：

```markdown
# PCCheck 改进效果实验报告

## 核心发现
- 平均加速比
- 检查点开销降低
- 保存时间改善

## 详细数据
- 吞吐量对比表
- 检查点开销对比表
- 保存时间对比表
- 内存使用对比表

## 每个实验的详细分析
- 配置参数
- 性能指标
- 统计数据

## 结论和建议
```

## 🐛 常见问题

### Q1: CUDA 内存不足

**A:** 减小模型或批次大小
```bash
python benchmark_comparison.py --d-model 256 --batch-size 8
```

### Q2: PCCheck 库找不到

**A:** 检查并重新编译
```bash
cd /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck
ls -la libtest_ssd.so  # 检查是否存在
make clean && make     # 重新编译
```

### Q3: 导入错误

**A:** 设置 PYTHONPATH
```bash
export PYTHONPATH=/home/linzhicheng/code/pccheck/checkpoint_eval:$PYTHONPATH
```

### Q4: GPUtil 未安装

**A:** 安装或跳过（可选）
```bash
pip install gputil  # 或
# 脚本会自动跳过 GPU 监控
```

## 📚 相关文档

- [实验详细说明](README.md) - 详细的实验配置和参数说明
- [分层检查点指南](../pccheck/layerwise_checkpoint/LAYERWISE_CHECKPOINT_GUIDE.md) - 技术实现文档
- [系统架构](../pccheck/layerwise_checkpoint/ARCHITECTURE_MERMAID.md) - 架构设计

## 🎓 实验建议

### 第一次运行
1. ✅ 先运行快速验证 (`quick_test.py`)
2. ✅ 确认系统正常工作
3. ✅ 然后运行完整测试

### 论文/报告使用
1. ✅ 运行完整测试套件
2. ✅ 使用中型和大型模型的结果
3. ✅ 关注加速比和开销降低指标
4. ✅ 展示不同频率下的性能表现

### 演示使用
1. ✅ 使用快速测试展示基本效果
2. ✅ 准备几个关键的对比图表
3. ✅ 强调实时性和低开销

## 📞 技术支持

遇到问题时：

1. **检查日志** - 查看实验输出的详细信息
2. **查看代码注释** - 关键函数都有详细注释
3. **参考测试用例** - `quick_test.py` 是简化版示例
4. **检查文档** - README 和 GUIDE 文档有详细说明

## 🎉 下一步

实验完成后：

1. ✅ 分析结果报告
2. ✅ 识别性能瓶颈
3. ✅ 调整参数优化
4. ✅ 撰写论文/报告
5. ✅ 准备演示材料

---

**祝实验顺利！** 🚀

如有问题，请查看详细文档或检查代码注释。
