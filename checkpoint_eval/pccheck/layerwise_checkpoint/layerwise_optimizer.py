"""
阶段二：挂钩（Hook）优化器与训练循环
Layerwise Optimizer Wrapper

在训练过程中，精确捕获每个层参数更新完成的事件，并触发回调
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Callable, Dict, List, Optional, Any
from collections import OrderedDict
import time
import copy


class LayerwiseOptimizer:
    """
    分层优化器包装器
    
    包装标准的 PyTorch 优化器，在每层参数更新后触发回调函数。
    这使得我们可以在参数更新完成后立即开始保存该层，而不需要等待所有层都更新完成。
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        update_order: List[str],
        layer_info: Dict[str, Dict],
        callback: Optional[Callable[[str, int, Dict], None]] = None,
        enable_timing: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            optimizer: PyTorch 优化器实例 (Adam, SGD, etc.)
            model: PyTorch 模型实例
            update_order: 层更新顺序列表（从 DependencyGraphBuilder 获取）
            layer_info: 层信息字典（从 DependencyGraphBuilder 获取）
            callback: 每层更新后的回调函数，签名为 callback(layer_name, step, layer_params)
            enable_timing: 是否启用性能计时
            verbose: 是否打印详细信息
        """
        self.optimizer = optimizer
        self.model = model
        self.update_order = update_order
        self.layer_info = layer_info
        self.callback = callback
        self.enable_timing = enable_timing
        self.verbose = verbose
        
        # 训练步数计数器
        self.training_step = 0
        
        # 🔥 新增：检查点控制机制
        self._checkpoint_enabled = False
        self._checkpoint_mode = 'manual'  # 'auto', 'manual', 'disabled'
        
        # 性能统计
        self.timing_stats = {
            'total_step_time': [],
            'update_time_per_layer': {layer: [] for layer in update_order},
            'callback_time_per_layer': {layer: [] for layer in update_order}
        }
        
        # 构建层名称到参数的映射
        self._build_layer_param_mapping()
        
        # 构建层名称到优化器 param_groups 的映射
        self._build_layer_param_groups()
        
    def _build_layer_param_mapping(self):
        """构建层名称到参数张量的映射"""
        self.layer_to_params = OrderedDict()
        
        for layer_name in self.update_order:
            if layer_name in self.layer_info:
                params = self.layer_info[layer_name]['parameters']
                # 存储参数的引用
                self.layer_to_params[layer_name] = params
            else:
                raise KeyError(f"Layer '{layer_name}' not found in layer_info")
        
        if self.verbose:
            print(f"构建了 {len(self.layer_to_params)} 个层的参数映射")
    
    def _build_layer_param_groups(self):
        """
        为每个层创建独立的 param_groups
        这样可以单独更新每个层的参数
        """
        self.layer_param_groups = OrderedDict()
        
        # 获取优化器的所有参数
        all_optimizer_params = set()
        for group in self.optimizer.param_groups:
            all_optimizer_params.update(id(p) for p in group['params'])
        
        # 为每个层创建 param_group
        for layer_name, params in self.layer_to_params.items():
            # 过滤出属于该层且在优化器中的参数
            layer_params_in_optimizer = [
                p for p in params 
                if id(p) in all_optimizer_params and p.requires_grad
            ]
            
            if layer_params_in_optimizer:
                # 复制原始 param_group 的配置（lr, weight_decay 等）
                # 这里假设所有参数使用相同的配置
                base_config = {
                    k: v for k, v in self.optimizer.param_groups[0].items()
                    if k != 'params'
                }
                
                self.layer_param_groups[layer_name] = {
                    'params': layer_params_in_optimizer,
                    **base_config
                }
        
        if self.verbose:
            print(f"为 {len(self.layer_param_groups)} 个层创建了独立的 param_groups")
    
    def enable_checkpointing(self, enable: bool = True):
        """
        启用或禁用检查点回调
        
        Args:
            enable: True=启用检查点回调, False=禁用（正常训练，不触发回调）
        """
        self._checkpoint_enabled = enable
    
    def set_checkpoint_mode(self, mode: str):
        """
        设置检查点模式
        
        Args:
            mode: 'auto' (每步自动触发回调), 
                  'manual' (手动控制，默认), 
                  'disabled' (完全禁用)
        """
        assert mode in ['auto', 'manual', 'disabled'], \
            f"Invalid mode: {mode}. Must be 'auto', 'manual', or 'disabled'"
        
        self._checkpoint_mode = mode
        
        # 根据模式设置默认状态
        if mode == 'auto':
            self._checkpoint_enabled = True
        elif mode == 'disabled':
            self._checkpoint_enabled = False
    
    def _should_trigger_callback(self) -> bool:
        """
        判断是否应该触发回调
        
        Returns:
            bool: True=应该触发回调, False=跳过回调
        """
        # 没有回调函数，直接返回 False
        if self.callback is None:
            return False
        
        # 完全禁用模式
        if self._checkpoint_mode == 'disabled':
            return False
        
        # 自动模式（每步都触发）
        if self._checkpoint_mode == 'auto':
            return True
        
        # 手动模式（根据 enable_checkpointing 设置）
        if self._checkpoint_mode == 'manual':
            return self._checkpoint_enabled
        
        return False
    
    def zero_grad(self, set_to_none: bool = False):
        """
        清零梯度（代理到底层优化器）
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步优化（关键方法）
        
        🔥 优化：添加快速路径
        - 如果不需要检查点回调，直接使用原始优化器的 step()，避免分层更新开销
        - 如果需要检查点，才按照更新顺序逐层更新参数并触发回调
        
        Args:
            closure: 可选的闭包函数（某些优化器如 LBFGS 需要）
        """
        step_start_time = time.time() if self.enable_timing else None
        
        self.training_step += 1
        
        # 🔥 快速路径：如果不需要触发检查点回调，直接使用原始优化器
        if not self._should_trigger_callback():
            if self.verbose:
                print(f"\n[步骤 {self.training_step}] 使用快速路径（无检查点）")
            
            # 直接调用原始优化器的 step，一次性更新所有参数
            self.optimizer.step(closure)
            
            if self.enable_timing:
                total_step_time = time.time() - step_start_time
                self.timing_stats['total_step_time'].append(total_step_time)
                if self.verbose:
                    print(f"快速路径耗时: {total_step_time*1000:.2f} ms")
            
            return
        
        # 🔥 慢速路径：需要检查点，执行分层更新
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"开始第 {self.training_step} 步优化（分层更新模式 + 检查点）")
            print(f"{'='*80}")
        
        # 按照依赖顺序逐层更新参数
        for layer_idx, layer_name in enumerate(self.update_order):
            if layer_name not in self.layer_param_groups:
                continue
            
            layer_update_start = time.time() if self.enable_timing else None
            
            # 获取该层的参数
            layer_param_group = self.layer_param_groups[layer_name]
            
            # 执行该层的参数更新
            # 这里我们手动调用优化器的更新逻辑
            self._update_layer_params(layer_param_group, closure)
            
            if self.enable_timing:
                update_time = time.time() - layer_update_start
                self.timing_stats['update_time_per_layer'][layer_name].append(update_time)
            
            if self.verbose:
                param_count = sum(p.numel() for p in layer_param_group['params'])
                print(f"  [{layer_idx+1:2d}/{len(self.update_order):2d}] "
                      f"更新 {layer_name:40s} | {param_count:12,d} 参数", end='')
            
            # 触发检查点回调
            callback_start = time.time() if self.enable_timing else None
            
            # 准备回调所需的参数信息
            layer_params_dict = self._prepare_layer_params_for_callback(layer_name)
            
            # 调用回调函数
            self.callback(layer_name, self.training_step, layer_params_dict)
            
            if self.enable_timing:
                callback_time = time.time() - callback_start
                self.timing_stats['callback_time_per_layer'][layer_name].append(callback_time)
                
                if self.verbose:
                    print(f" | 回调耗时: {callback_time*1000:.2f} ms")
            elif self.verbose:
                print()
        
        if self.enable_timing:
            total_step_time = time.time() - step_start_time
            self.timing_stats['total_step_time'].append(total_step_time)
            
            if self.verbose:
                print(f"\n总步骤耗时: {total_step_time*1000:.2f} ms")
        
        if self.verbose:
            print(f"{'='*80}\n")
    
    def _update_layer_params(self, param_group: Dict, closure: Optional[Callable] = None):
        """
        更新单个层的参数
        
        这个方法模拟优化器的单步更新，但只针对指定的参数组
        """
        # 临时替换优化器的 param_groups，只包含当前层
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [param_group]
        
        # 执行优化器的步进
        # 注意：这里直接调用优化器的 step，它会更新 param_groups 中的参数
        self.optimizer.step(closure)
        
        # 恢复原始的 param_groups
        self.optimizer.param_groups = original_param_groups
    
    def _prepare_layer_params_for_callback(self, layer_name: str) -> Dict[str, Any]:
        """
        为回调函数准备该层的参数信息
        
        🔥 优化：不再进行深拷贝，只传递引用
        深拷贝改为在 PCCheckAdapter 中完成，直接从原地址拷贝到 staging buffer
        这样可以避免一次额外的 GPU 内存拷贝
        
        Returns:
            包含层参数的字典，包括张量的引用（不再深拷贝）
        """
        params = self.layer_to_params[layer_name]
        
        # 🔥 关键修改：不再深拷贝，只传递引用
        # params_copy = [p.detach().clone() for p in params]  # 旧版本
        params_ref = [p for p in params]  # 新版本：只传递引用
        
        return {
            'layer_name': layer_name,
            'parameters': params_ref,  # 传递引用而不是拷贝
            'param_count': sum(p.numel() for p in params_ref),
            'shapes': [p.shape for p in params_ref],
            'dtypes': [p.dtype for p in params_ref],
            'devices': [p.device for p in params_ref],
            'training_step': self.training_step
        }
    
    def state_dict(self):
        """返回优化器状态（代理到底层优化器）"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载优化器状态（代理到底层优化器）"""
        self.optimizer.load_state_dict(state_dict)
    
    def get_timing_stats(self) -> Dict:
        """获取性能统计信息"""
        if not self.enable_timing:
            return {"message": "Timing is not enabled"}
        
        import numpy as np
        
        stats = {
            'total_steps': self.training_step,
            'avg_step_time_ms': np.mean(self.timing_stats['total_step_time']) * 1000,
            'layer_stats': {}
        }
        
        for layer_name in self.update_order:
            update_times = self.timing_stats['update_time_per_layer'][layer_name]
            callback_times = self.timing_stats['callback_time_per_layer'][layer_name]
            
            if update_times:
                stats['layer_stats'][layer_name] = {
                    'avg_update_time_ms': np.mean(update_times) * 1000,
                    'avg_callback_time_ms': np.mean(callback_times) * 1000 if callback_times else 0,
                    'total_time_ms': (np.mean(update_times) + np.mean(callback_times or [0])) * 1000
                }
        
        return stats
    
    def print_timing_stats(self):
        """打印性能统计信息"""
        stats = self.get_timing_stats()
        
        if 'message' in stats:
            print(stats['message'])
            return
        
        print(f"\n{'='*80}")
        print(f"性能统计报告 (基于 {stats['total_steps']} 步)")
        print(f"{'='*80}")
        print(f"平均每步总耗时: {stats['avg_step_time_ms']:.2f} ms\n")
        
        print(f"{'层名称':<40s} | {'更新耗时':<12s} | {'回调耗时':<12s} | {'总耗时':<12s}")
        print(f"{'-'*80}")
        
        for layer_name, layer_stats in stats['layer_stats'].items():
            print(f"{layer_name:<40s} | "
                  f"{layer_stats['avg_update_time_ms']:>10.2f} ms | "
                  f"{layer_stats['avg_callback_time_ms']:>10.2f} ms | "
                  f"{layer_stats['total_time_ms']:>10.2f} ms")
        
        print(f"{'='*80}\n")
