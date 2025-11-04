# PCCheck 改进效果对比实验指南

## 📋 实验概述

本实验旨在全面评估 PCCheck 分层检查点系统的改进效果，通过对比三种检查点方法来量化性能提升：

1. **传统 PyTorch 检查点** (`torch.save`)
2. **原始 PCCheck** (Chk_monitor)
3. **改进的分层 PCCheck** (Layerwise Checkpoint)

## 🎯 实验目标

- 测量检查点保存时间
- 评估训练吞吐量
- 分析内存使用情况
- 量化 I/O 开销占比
- 验证不同配置下的性能表现

## 📁 文件结构

```
experiments/
├── README.md                   # 本文件
├── benchmark_comparison.py     # 主实验脚本
├── run_benchmark.sh            # 自动化运行脚本
├── generate_report.py          # 报告生成器
└── benchmark_results/          # 实验结果输出目录（自动创建）
```

## 🚀 快速开始

### 方式一：自动化运行全部实验

```bash
cd /home/linzhicheng/code/pccheck/checkpoint_eval/experiments
chmod +x run_benchmark.sh
./run_benchmark.sh
```

这将自动运行以下实验：
1. 小型模型测试（快速验证）
2. 中型模型测试（标准配置）
3. 大型模型测试（高负载）
4. 不同检查点频率对比
5. Monitor 模式对比

### 方式二：运行单个实验

```bash
# 基础对比实验
python benchmark_comparison.py \
    --d-model 512 \
    --num-layers 6 \
    --num-steps 100 \
    --checkpoint-freq 10 \
    --methods traditional original layerwise \
    --output-dir ./results
```

## 📊 实验配置说明

### 模型配置参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--d-model` | 模型维度 | 512 | 256-1024 |
| `--num-layers` | Transformer 层数 | 6 | 2-12 |
| `--nhead` | 注意力头数 | 8 | 4-16 |
| `--vocab-size` | 词汇表大小 | 10000 | 5000-50000 |
| `--dim-feedforward` | 前馈网络维度 | 2048 | 1024-4096 |

### 训练配置参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--num-samples` | 训练样本数 | 1000 | 500-5000 |
| `--seq-len` | 序列长度 | 128 | 64-512 |
| `--batch-size` | 批次大小 | 16 | 8-32 |
| `--num-steps` | 训练步数 | 100 | 50-500 |
| `--checkpoint-freq` | 检查点频率 | 10 | 5-50 |

### PCCheck 配置参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--num-threads` | 并行线程数 | 8 | 4-16 |
| `--max-async` | 最大并发检查点 | 4 | 2-8 |
| `--buffer-size-mb` | 缓冲区大小 | 50.0 | 20-200 |
| `--batch-size-mb` | PCCheck 批次大小 | 100.0 | 50-500 |
| `--use-monitor` | 使用 Monitor 模式 | False | - |

## 📈 实验场景

### 1. 小型模型验证（快速测试）

**目的：** 快速验证系统正常工作

```bash
python benchmark_comparison.py \
    --d-model 256 \
    --num-layers 2 \
    --num-samples 500 \
    --num-steps 50 \
    --checkpoint-freq 10 \
    --methods traditional original layerwise
```

**预期时间：** 2-5 分钟

### 2. 中型模型对比（标准配置）

**目的：** 标准场景下的性能对比

```bash
python benchmark_comparison.py \
    --d-model 512 \
    --num-layers 6 \
    --num-samples 1000 \
    --num-steps 100 \
    --checkpoint-freq 10 \
    --methods traditional original layerwise
```

**预期时间：** 5-10 分钟

### 3. 大型模型测试（高负载）

**目的：** 评估大规模模型下的性能

```bash
python benchmark_comparison.py \
    --d-model 1024 \
    --num-layers 12 \
    --num-samples 1000 \
    --num-steps 100 \
    --batch-size 8 \
    --num-threads 16 \
    --methods traditional original layerwise
```

**预期时间：** 10-20 分钟

### 4. 检查点频率影响

**目的：** 分析保存频率对性能的影响

```bash
for freq in 5 10 20 50; do
    python benchmark_comparison.py \
        --checkpoint-freq $freq \
        --methods layerwise \
        --output-dir ./results/freq_$freq
done
```

### 5. Monitor 模式对比

**目的：** 评估 Monitor 后台进程的性能优势

```bash
# 直接模式
python benchmark_comparison.py \
    --methods layerwise \
    --output-dir ./results/direct

# Monitor 模式
python benchmark_comparison.py \
    --methods layerwise \
    --use-monitor \
    --output-dir ./results/monitor
```

## 📊 结果分析

### 查看实时输出

实验运行时会显示：
- ✓ 每个检查点的保存时间
- ✓ 训练步的平均时间
- ✓ 内存使用情况
- ✓ 完成后的性能摘要

### 查看详细报告

```bash
# 生成 Markdown 报告
python generate_report.py \
    --input-dir ./benchmark_results \
    --output-file ./report.md

# 查看报告
cat ./report.md
```

### 关键性能指标

报告包含以下关键指标：

1. **吞吐量 (samples/sec)**  
   - 越高越好
   - 衡量整体训练速度

2. **检查点开销 (%)**  
   - 越低越好
   - 检查点时间占总训练时间的比例

3. **平均检查点时间 (ms)**  
   - 越低越好
   - 单次检查点保存的平均时间

4. **内存峰值 (GB)**  
   - 越低越好（在不影响性能前提下）
   - CPU 和 GPU 内存使用

5. **加速比 (speedup)**  
   - 分层方法相对传统方法的速度提升

## 🎯 预期结果

基于初步测试，预期看到：

- ✅ **吞吐量提升：** 1.5x - 3x
- ✅ **检查点开销降低：** 40% - 70%
- ✅ **保存时间减少：** 50% - 80%
- ✅ **内存开销：** 增加 10% - 30%（可接受范围）

## 🐛 故障排除

### 问题 1: CUDA 内存不足

**解决方案：**
```bash
# 减小模型或批次大小
python benchmark_comparison.py --d-model 256 --batch-size 8
```

### 问题 2: PCCheck 库找不到

**解决方案：**
```bash
# 检查库文件是否存在
ls -la /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so

# 如果不存在，重新编译
cd /home/linzhicheng/code/pccheck/checkpoint_eval/pccheck
make clean && make
```

### 问题 3: 导入错误

**解决方案：**
```bash
# 设置 PYTHONPATH
export PYTHONPATH=/home/linzhicheng/code/pccheck/checkpoint_eval:$PYTHONPATH
```

### 问题 4: GPUtil 未安装

**解决方案：**
```bash
# 安装 GPUtil（可选，仅用于 GPU 监控）
pip install gputil

# 或者注释掉 GPU 监控代码
# 脚本会自动跳过 GPU 内存统计
```

## 📝 自定义实验

### 创建自定义配置

创建配置文件 `custom_config.yaml`:

```yaml
model:
  d_model: 768
  num_layers: 8
  nhead: 12

training:
  num_steps: 200
  batch_size: 16
  checkpoint_freq: 20

pccheck:
  num_threads: 12
  max_async: 6
  buffer_size_mb: 100
```

然后运行：

```bash
python benchmark_comparison.py --config custom_config.yaml
```

### 添加新的测试方法

在 `benchmark_comparison.py` 中添加新函数：

```python
def benchmark_your_method(...):
    # 实现你的方法
    pass
```

## 📚 参考资料

- [分层检查点系统文档](../pccheck/layerwise_checkpoint/README.md)
- [PCCheck 原始论文](https://arxiv.org/abs/2011.14439)
- [实现细节文档](../pccheck/layerwise_checkpoint/LAYERWISE_CHECKPOINT_GUIDE.md)

## 🤝 贡献

如果你发现问题或有改进建议：

1. 修改实验脚本
2. 添加新的测试场景
3. 改进报告生成器
4. 优化性能指标

## 📞 联系

如有问题，请查看：
- 项目文档
- 代码注释
- 测试用例

---

**最后更新:** 2025-10-27
