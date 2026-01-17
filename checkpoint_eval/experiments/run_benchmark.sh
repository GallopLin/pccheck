#!/bin/bash
# PCCheck 改进效果对比实验运行脚本

set -e

echo "=========================================="
echo "🔬 PCCheck 改进效果对比实验"
echo "=========================================="
echo ""

# 设置环境变量
export PYTHONPATH=/home/linzhicheng/code/pccheck/checkpoint_eval:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 实验输出目录
OUTPUT_DIR="./benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# ============================================================================
# 实验 2: 中型模型测试 (默认配置)
# ============================================================================
# echo "=========================================="
# echo "🧪 实验 2: 中型模型 (d=512, 6 layers)"
# echo "=========================================="

# python benchmark_comparison.py \
#     --d-model 512 \
#     --num-layers 6 \
#     --num-samples 1000 \
#     --num-steps 100 \
#     --batch-size 16 \
#     --checkpoint-freq 30 \
#     --num-threads 8 \
#     --max-async 8 \
#     --methods layerwise \
#     --output-dir $OUTPUT_DIR/medium_model \
#     --device cuda

# echo ""
# echo "✅ 实验 2 完成"
# echo ""

# ============================================================================
# 实验 3: 大型模型测试 (高负载)
# ============================================================================
echo "=========================================="
echo "🧪 实验 3: 大型模型 (~5GB checkpoint)"
echo "=========================================="
echo "模型配置: d_model=1536, num_layers=17, dim_feedforward=3072"
echo "总参数量: 352.48M"
echo "检查点大小: 5.25 GB (包含 param, grad, exp_avg, exp_avg_sq)"
echo ""

python benchmark_comparison.py \
    --d-model 1536 \
    --num-layers 18 \
    --dim-feedforward 3072 \
    --num-samples 1000 \
    --num-steps 100 \
    --batch-size 4 \
    --checkpoint-freq 30 \
    --num-threads 16 \
    --max-async 3 \
    --num-layer-groups 6 \
    --methods original multistream \
    --output-dir $OUTPUT_DIR/large_model \
    --device cuda

echo ""
echo "✅ 实验 3 完成"
echo ""
 # traditional original 
# ============================================================================
# 实验 4: 不同检查点频率对比
# ============================================================================
# echo "=========================================="
# echo "🧪 实验 4: 检查点频率影响"
# echo "=========================================="

# for freq in 5 10 20 50; do
#     echo ""
#     echo "  测试频率: 每 $freq 步保存一次"
#     python benchmark_comparison.py \
#         --d-model 512 \
#         --num-layers 6 \
#         --num-samples 1000 \
#         --num-steps 100 \
#         --batch-size 16 \
#         --checkpoint-freq $freq \
#         --num-threads 8 \
#         --max-async 8 \
#         --methods layerwise \
#         --output-dir $OUTPUT_DIR/freq_test/freq_$freq \
#         --device cuda
# done

# echo ""
# echo "✅ 实验 4 完成"
# echo ""

# ============================================================================
# 实验 5: Monitor 模式 vs 直接模式
# ============================================================================
# echo "=========================================="
# echo "🧪 实验 5: Monitor 模式对比"
# echo "=========================================="

# echo ""
# echo "  测试: 直接模式"
# python benchmark_comparison.py \
#     --d-model 512 \
#     --num-layers 6 \
#     --num-samples 1000 \
#     --num-steps 100 \
#     --batch-size 16 \
#     --checkpoint-freq 10 \
#     --num-threads 8 \
#     --max-async 8 \
#     --methods layerwise \
#     --output-dir $OUTPUT_DIR/monitor_test/direct \
#     --device cuda

# echo ""
# echo "  测试: Monitor 模式"
# python benchmark_comparison.py \
#     --d-model 512 \
#     --num-layers 6 \
#     --num-samples 1000 \
#     --num-steps 100 \
#     --batch-size 16 \
#     --checkpoint-freq 10 \
#     --num-threads 8 \
#     --max-async 8 \
#     --use-monitor \
#     --methods layerwise \
#     --output-dir $OUTPUT_DIR/monitor_test/monitor \
#     --device cuda

# echo ""
# echo "✅ 实验 5 完成"
# echo ""

# ============================================================================
# 生成汇总报告
# ============================================================================
echo "=========================================="
echo "📊 生成汇总报告"
echo "=========================================="

# python generate_report.py --input-dir $OUTPUT_DIR --output-file $OUTPUT_DIR/summary_report.md

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "=========================================="
echo ""
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📊 查看汇总报告: $OUTPUT_DIR/summary_report.md"
echo ""
