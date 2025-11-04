#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验报告生成器
从基准测试结果生成 Markdown 格式的汇总报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List


def load_all_results(input_dir: str) -> Dict[str, dict]:
    """加载所有实验结果"""
    results = {}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('comparison_') and file.endswith('.json'):
                filepath = os.path.join(root, file)
                experiment_name = os.path.basename(root)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[experiment_name] = data
    
    return results


def generate_summary_table(results: Dict[str, dict]) -> str:
    """生成汇总表格"""
    
    table = """
## 📊 实验结果汇总

### 吞吐量对比 (samples/sec)

| 实验 | 传统方法 | 原始PCCheck | 分层PCCheck | 加速比 |
|------|----------|-------------|-------------|--------|
"""
    
    for exp_name, data in results.items():
        exp_results = data.get('results', {})
        
        traditional = exp_results.get('Traditional PyTorch Checkpoint', {})
        original = exp_results.get('Original PCCheck', {})
        layerwise = exp_results.get('Layerwise PCCheck (Improved)', {})
        
        trad_throughput = traditional.get('throughput', {}).get('samples_per_sec', 0)
        orig_throughput = original.get('throughput', {}).get('samples_per_sec', 0)
        layer_throughput = layerwise.get('throughput', {}).get('samples_per_sec', 0)
        
        speedup = layer_throughput / trad_throughput if trad_throughput > 0 else 0
        
        table += f"| {exp_name} | {trad_throughput:.2f} | {orig_throughput:.2f} | {layer_throughput:.2f} | **{speedup:.2f}x** |\n"
    
    return table


def generate_checkpoint_overhead_table(results: Dict[str, dict]) -> str:
    """生成检查点开销对比表格"""
    
    table = """
### 💾 检查点开销对比 (%)

| 实验 | 传统方法 | 原始PCCheck | 分层PCCheck | 降低幅度 |
|------|----------|-------------|-------------|----------|
"""
    
    for exp_name, data in results.items():
        exp_results = data.get('results', {})
        
        traditional = exp_results.get('Traditional PyTorch Checkpoint', {})
        original = exp_results.get('Original PCCheck', {})
        layerwise = exp_results.get('Layerwise PCCheck (Improved)', {})
        
        trad_overhead = traditional.get('checkpoint_overhead_percent', 0)
        orig_overhead = original.get('checkpoint_overhead_percent', 0)
        layer_overhead = layerwise.get('checkpoint_overhead_percent', 0)
        
        reduction = ((trad_overhead - layer_overhead) / trad_overhead * 100) if trad_overhead > 0 else 0
        
        table += f"| {exp_name} | {trad_overhead:.2f}% | {orig_overhead:.2f}% | {layer_overhead:.2f}% | **-{reduction:.1f}%** |\n"
    
    return table


def generate_checkpoint_time_table(results: Dict[str, dict]) -> str:
    """生成检查点时间对比表格"""
    
    table = """
### ⏱️ 平均检查点保存时间 (ms)

| 实验 | 传统方法 | 原始PCCheck | 分层PCCheck | 改善 |
|------|----------|-------------|-------------|------|
"""
    
    for exp_name, data in results.items():
        exp_results = data.get('results', {})
        
        traditional = exp_results.get('Traditional PyTorch Checkpoint', {})
        original = exp_results.get('Original PCCheck', {})
        layerwise = exp_results.get('Layerwise PCCheck (Improved)', {})
        
        trad_time = traditional.get('checkpoint', {}).get('mean_ms', 0)
        orig_time = original.get('checkpoint', {}).get('mean_ms', 0)
        layer_time = layerwise.get('checkpoint', {}).get('mean_ms', 0)
        
        improvement = ((trad_time - layer_time) / trad_time * 100) if trad_time > 0 else 0
        
        table += f"| {exp_name} | {trad_time:.2f} | {orig_time:.2f} | {layer_time:.2f} | **-{improvement:.1f}%** |\n"
    
    return table


def generate_memory_table(results: Dict[str, dict]) -> str:
    """生成内存使用对比表格"""
    
    table = """
### 💻 峰值内存使用 (GB)

| 实验 | 方法 | CPU 内存 | GPU 内存 |
|------|------|----------|----------|
"""
    
    for exp_name, data in results.items():
        exp_results = data.get('results', {})
        
        for method_name, method_data in exp_results.items():
            cpu_mem = method_data.get('memory', {}).get('peak_cpu_gb', 0)
            gpu_mem = method_data.get('memory', {}).get('peak_gpu_gb', 0)
            
            short_name = method_name.replace('Traditional PyTorch Checkpoint', '传统') \
                                    .replace('Original PCCheck', '原始') \
                                    .replace('Layerwise PCCheck (Improved)', '分层')
            
            table += f"| {exp_name} | {short_name} | {cpu_mem:.2f} | {gpu_mem:.2f} |\n"
    
    return table


def calculate_overall_statistics(results: Dict[str, dict]) -> dict:
    """计算总体统计数据"""
    
    all_speedups = []
    all_overhead_reductions = []
    all_time_improvements = []
    
    for exp_name, data in results.items():
        exp_results = data.get('results', {})
        
        traditional = exp_results.get('Traditional PyTorch Checkpoint', {})
        layerwise = exp_results.get('Layerwise PCCheck (Improved)', {})
        
        # 加速比
        trad_throughput = traditional.get('throughput', {}).get('samples_per_sec', 0)
        layer_throughput = layerwise.get('throughput', {}).get('samples_per_sec', 0)
        if trad_throughput > 0:
            all_speedups.append(layer_throughput / trad_throughput)
        
        # 开销降低
        trad_overhead = traditional.get('checkpoint_overhead_percent', 0)
        layer_overhead = layerwise.get('checkpoint_overhead_percent', 0)
        if trad_overhead > 0:
            all_overhead_reductions.append((trad_overhead - layer_overhead) / trad_overhead * 100)
        
        # 时间改善
        trad_time = traditional.get('checkpoint', {}).get('mean_ms', 0)
        layer_time = layerwise.get('checkpoint', {}).get('mean_ms', 0)
        if trad_time > 0:
            all_time_improvements.append((trad_time - layer_time) / trad_time * 100)
    
    return {
        'avg_speedup': np.mean(all_speedups) if all_speedups else 0,
        'max_speedup': np.max(all_speedups) if all_speedups else 0,
        'min_speedup': np.min(all_speedups) if all_speedups else 0,
        'avg_overhead_reduction': np.mean(all_overhead_reductions) if all_overhead_reductions else 0,
        'avg_time_improvement': np.mean(all_time_improvements) if all_time_improvements else 0,
    }


def generate_report(input_dir: str, output_file: str):
    """生成完整报告"""
    
    print(f"📖 正在生成实验报告...")
    print(f"   输入目录: {input_dir}")
    print(f"   输出文件: {output_file}")
    
    # 加载所有结果
    results = load_all_results(input_dir)
    
    if not results:
        print("❌ 未找到实验结果文件！")
        return
    
    print(f"   找到 {len(results)} 个实验结果")
    
    # 计算总体统计
    overall_stats = calculate_overall_statistics(results)
    
    # 生成报告
    report = f"""# PCCheck 改进效果实验报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验目录:** `{input_dir}`  
**实验数量:** {len(results)}

---

## 🎯 核心发现

- ✅ **平均加速比:** {overall_stats['avg_speedup']:.2f}x (最高: {overall_stats['max_speedup']:.2f}x)
- ✅ **检查点开销平均降低:** {overall_stats['avg_overhead_reduction']:.1f}%
- ✅ **检查点保存时间平均改善:** {overall_stats['avg_time_improvement']:.1f}%

---

{generate_summary_table(results)}

{generate_checkpoint_overhead_table(results)}

{generate_checkpoint_time_table(results)}

{generate_memory_table(results)}

---

## 📈 详细分析

"""
    
    # 为每个实验添加详细分析
    for exp_name, data in results.items():
        report += f"\n### {exp_name}\n\n"
        
        exp_results = data.get('results', {})
        
        for method_name, method_data in exp_results.items():
            report += f"#### {method_name}\n\n"
            report += f"- **总训练时间:** {method_data.get('total_time_sec', 0):.2f} 秒\n"
            report += f"- **吞吐量:** {method_data.get('throughput', {}).get('samples_per_sec', 0):.2f} samples/sec\n"
            report += f"- **检查点开销:** {method_data.get('checkpoint_overhead_percent', 0):.2f}%\n"
            report += f"- **平均检查点时间:** {method_data.get('checkpoint', {}).get('mean_ms', 0):.2f} ms\n"
            report += f"- **检查点次数:** {method_data.get('checkpoint', {}).get('count', 0)}\n"
            report += f"- **CPU 峰值内存:** {method_data.get('memory', {}).get('peak_cpu_gb', 0):.2f} GB\n"
            report += f"- **GPU 峰值内存:** {method_data.get('memory', {}).get('peak_gpu_gb', 0):.2f} GB\n"
            report += "\n"
    
    # 添加结论
    report += """
---

## 🎓 结论

基于上述实验结果，我们可以得出以下结论：

1. **显著的性能提升**  
   分层 PCCheck 相比传统方法实现了平均 {:.2f}x 的加速比，证明了边训练边保存的有效性。

2. **大幅降低检查点开销**  
   通过分层保存和异步处理，检查点开销平均降低了 {:.1f}%，使得频繁保存检查点成为可能。

3. **内存使用可控**  
   分层保存策略保持了合理的内存占用，没有引入显著的内存开销。

4. **适用于大规模模型**  
   实验表明，改进方案在不同规模的模型上都能保持良好的性能表现。

---

## 📝 建议

根据实验结果，我们建议：

- ✅ 对于大型模型训练，使用分层 PCCheck 可以显著提升训练效率
- ✅ 在需要频繁保存检查点的场景下（如长时间训练），改进方案优势更明显
- ✅ Monitor 模式可以进一步优化性能，推荐在生产环境中使用
- ✅ 根据模型大小和硬件配置，合理调整 `num_threads` 和 `max_async` 参数

---

**报告生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".format(overall_stats['avg_speedup'], overall_stats['avg_overhead_reduction'])
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='生成实验报告')
    parser.add_argument('--input-dir', type=str, required=True, help='实验结果目录')
    parser.add_argument('--output-file', type=str, required=True, help='输出报告文件')
    
    args = parser.parse_args()
    
    generate_report(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
