"""
阶段五：元数据管理与模型恢复
Checkpoint Metadata Management and Model Recovery

实现检查点的完整元数据管理和模型恢复功能
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    training_step: int
    total_layers: int
    total_size_bytes: int
    checkpoint_file: str
    created_at: float
    layers: Dict[str, Dict]  # {layer_name: {offset, size, shapes, dtypes, ...}}
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'training_step': self.training_step,
            'total_layers': self.total_layers,
            'total_size_bytes': self.total_size_bytes,
            'checkpoint_file': self.checkpoint_file,
            'created_at': self.created_at,
            'layers': self.layers
        }
    
    @staticmethod
    def from_dict(data: Dict):
        """从字典创建"""
        return CheckpointMetadata(
            checkpoint_id=data['checkpoint_id'],
            training_step=data['training_step'],
            total_layers=data['total_layers'],
            total_size_bytes=data['total_size_bytes'],
            checkpoint_file=data['checkpoint_file'],
            created_at=data['created_at'],
            layers=data['layers']
        )


class CheckpointMetadataManager:
    """
    检查点元数据管理器
    
    负责：
    1. 记录检查点的所有元数据
    2. 保存/加载元数据文件
    3. 检索检查点信息
    4. 验证检查点完整性
    """
    
    def __init__(
        self,
        metadata_dir: str = "./checkpoint_metadata",
        verbose: bool = False
    ):
        """
        Args:
            metadata_dir: 元数据存储目录
            verbose: 是否打印详细信息
        """
        self.metadata_dir = metadata_dir
        self.verbose = verbose
        
        # 创建目录
        os.makedirs(metadata_dir, exist_ok=True)
        
        # 当前检查点元数据缓存
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        
        # 加载已有的元数据
        self._load_existing_metadata()
        
        if self.verbose:
            print(f"[MetadataManager] 初始化完成")
            print(f"  - 元数据目录: {metadata_dir}")
            print(f"  - 已有检查点数: {len(self.checkpoints)}")
    
    def _load_existing_metadata(self):
        """加载已有的元数据文件"""
        try:
            index_file = os.path.join(self.metadata_dir, "index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    for chk_id, chk_data in data.items():
                        self.checkpoints[chk_id] = CheckpointMetadata.from_dict(chk_data)
                
                if self.verbose:
                    print(f"[MetadataManager] 加载了 {len(self.checkpoints)} 个检查点元数据")
        except Exception as e:
            if self.verbose:
                print(f"[MetadataManager] 加载元数据失败: {e}")
    
    def register_checkpoint(
        self,
        checkpoint_id: str,
        training_step: int,
        checkpoint_file: str
    ) -> CheckpointMetadata:
        """
        注册一个新的检查点
        
        Args:
            checkpoint_id: 检查点ID
            training_step: 训练步数
            checkpoint_file: 检查点文件路径
        
        Returns:
            CheckpointMetadata 对象
        """
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            training_step=training_step,
            total_layers=0,
            total_size_bytes=0,
            checkpoint_file=checkpoint_file,
            created_at=time.time(),
            layers={}
        )
        
        self.checkpoints[checkpoint_id] = metadata
        
        if self.verbose:
            print(f"[MetadataManager] 注册检查点: {checkpoint_id}")
        
        return metadata
    
    def add_layer(
        self,
        checkpoint_id: str,
        layer_name: str,
        offset: int,
        size_bytes: int,
        param_count: int,
        shapes: List[Tuple],
        dtypes: List[str]
    ):
        """
        为检查点添加层信息
        
        Args:
            checkpoint_id: 检查点ID
            layer_name: 层名称
            offset: 在检查点文件中的偏移量
            size_bytes: 数据大小（字节）
            param_count: 参数数量
            shapes: 参数形状列表
            dtypes: 数据类型列表
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"检查点 {checkpoint_id} 不存在")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        checkpoint.layers[layer_name] = {
            'offset': offset,
            'size_bytes': size_bytes,
            'param_count': param_count,
            'shapes': shapes,
            'dtypes': dtypes
        }
        
        checkpoint.total_layers += 1
        checkpoint.total_size_bytes += size_bytes
        
        if self.verbose:
            print(f"[MetadataManager] 添加层: {checkpoint_id}::{layer_name} "
                  f"({size_bytes / (1024**2):.2f} MB)")
    
    def save_metadata(self, checkpoint_id: Optional[str] = None):
        """
        保存元数据到文件
        
        Args:
            checkpoint_id: 如果指定，只保存该检查点的元数据
        """
        # 保存索引文件（所有检查点的概要）
        index_file = os.path.join(self.metadata_dir, "index.json")
        index_data = {
            chk_id: chk.to_dict()
            for chk_id, chk in self.checkpoints.items()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        if self.verbose:
            print(f"[MetadataManager] 元数据已保存: {index_file}")
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """获取检查点元数据"""
        return self.checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self) -> List[str]:
        """列出所有检查点ID"""
        return list(self.checkpoints.keys())
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """获取最新的检查点"""
        if not self.checkpoints:
            return None
        
        return max(
            self.checkpoints.values(),
            key=lambda c: c.training_step
        )
    
    def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """
        验证检查点的完整性
        
        检查：
        1. 元数据文件存在
        2. 检查点数据文件存在
        3. 文件大小匹配
        """
        if checkpoint_id not in self.checkpoints:
            print(f"[MetadataManager] 检查点 {checkpoint_id} 不存在")
            return False
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint.checkpoint_file):
            print(f"[MetadataManager] 检查点文件不存在: {checkpoint.checkpoint_file}")
            return False
        
        # 检查文件大小
        file_size = os.path.getsize(checkpoint.checkpoint_file)
        if file_size < checkpoint.total_size_bytes:
            print(f"[MetadataManager] 文件大小不匹配: "
                  f"{file_size} < {checkpoint.total_size_bytes}")
            return False
        
        if self.verbose:
            print(f"[MetadataManager] 检查点 {checkpoint_id} 验证通过")
        
        return True


class ModelRecovery:
    """
    模型恢复器
    
    从检查点文件中恢复模型参数
    """
    
    def __init__(
        self,
        metadata_manager: CheckpointMetadataManager,
        verbose: bool = False
    ):
        """
        Args:
            metadata_manager: 元数据管理器
            verbose: 是否打印详细信息
        """
        self.metadata_manager = metadata_manager
        self.verbose = verbose
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_id: str,
        device: str = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        从检查点恢复模型
        
        Args:
            model: 要恢复的模型
            checkpoint_id: 检查点ID
            device: 目标设备
            strict: 是否严格匹配所有层
        
        Returns:
            恢复的统计信息
        """
        if self.verbose:
            print(f"\n[Recovery] 开始恢复检查点: {checkpoint_id}")
        
        # 获取元数据
        metadata = self.metadata_manager.get_checkpoint(checkpoint_id)
        if metadata is None:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        # 验证检查点
        if not self.metadata_manager.verify_checkpoint(checkpoint_id):
            raise ValueError(f"检查点 {checkpoint_id} 验证失败")
        
        # 统计信息
        stats = {
            'total_layers': len(metadata.layers),
            'loaded_layers': 0,
            'missing_layers': [],
            'unexpected_layers': [],
            'total_bytes_loaded': 0
        }
        
        # 获取模型的状态字典
        model_state = model.state_dict()
        
        # 打开检查点文件
        with open(metadata.checkpoint_file, 'rb') as f:
            # 遍历元数据中的每一层
            for layer_name, layer_info in metadata.layers.items():
                # 构建该层的所有参数键（例如 conv1 -> conv1.weight, conv1.bias）
                # 从 state_dict 中找到以该层名开头的所有参数
                layer_param_keys = [key for key in model_state.keys() 
                                   if key.startswith(layer_name + '.')]
                
                if not layer_param_keys:
                    stats['unexpected_layers'].append(layer_name)
                    if strict:
                        raise KeyError(f"模型中不存在层: {layer_name}")
                    if self.verbose:
                        print(f"  [✗] 跳过未知层: {layer_name}")
                    continue
                
                # 读取层数据
                offset = layer_info['offset']
                size_bytes = layer_info['size_bytes']
                shapes = layer_info['shapes']
                dtypes = layer_info['dtypes']
                
                # 定位到正确的位置
                f.seek(offset)
                
                # 读取数据
                import numpy as np
                data_bytes = f.read(size_bytes)
                
                # 将字节数据转换为多个参数张量
                # 根据 shapes 和 dtypes 拆分数据
                current_offset = 0
                
                for i, (param_key, shape) in enumerate(zip(layer_param_keys[:len(shapes)], shapes)):
                    # 计算该参数的大小
                    param_size = 1
                    for dim in shape:
                        param_size *= dim
                    
                    # 确定数据类型（默认 float32）
                    if i < len(dtypes):
                        dtype_str = dtypes[i]
                        if 'float32' in dtype_str or 'FloatTensor' in dtype_str:
                            np_dtype = np.float32
                            element_size = 4
                        elif 'float64' in dtype_str or 'DoubleTensor' in dtype_str:
                            np_dtype = np.float64
                            element_size = 8
                        elif 'int64' in dtype_str or 'LongTensor' in dtype_str:
                            np_dtype = np.int64
                            element_size = 8
                        else:
                            np_dtype = np.float32
                            element_size = 4
                    else:
                        np_dtype = np.float32
                        element_size = 4
                    
                    # 计算字节数
                    param_bytes = param_size * element_size
                    
                    # 提取该参数的数据
                    param_data = data_bytes[current_offset:current_offset + param_bytes]
                    current_offset += param_bytes
                    
                    # 转换为 numpy 数组
                    param_np = np.frombuffer(param_data, dtype=np_dtype)
                    
                    # 重塑为原始形状
                    param_np = param_np.reshape(shape)
                    
                    # 转换为 PyTorch 张量
                    param_tensor = torch.from_numpy(param_np).to(device)
                    
                    # 加载到模型
                    if param_key in model_state:
                        model_state[param_key].copy_(param_tensor)
                        
                        if self.verbose:
                            size_mb = param_bytes / (1024**2)
                            print(f"  [✓] 加载参数: {param_key:40s} | {size_mb:.4f} MB")
                
                stats['loaded_layers'] += 1
                stats['total_bytes_loaded'] += size_bytes
        
        # 检查是否有未加载的层
        loaded_param_keys = set()
        for layer_name in metadata.layers.keys():
            layer_param_keys = [key for key in model_state.keys() 
                               if key.startswith(layer_name + '.')]
            loaded_param_keys.update(layer_param_keys[:len(metadata.layers[layer_name]['shapes'])])
        
        for key in model_state.keys():
            if key not in loaded_param_keys:
                stats['missing_layers'].append(key)
        
        if stats['missing_layers'] and strict:
            raise ValueError(f"以下层未在检查点中找到: {stats['missing_layers']}")
        
        # 打印摘要
        if self.verbose:
            print(f"\n[Recovery] 恢复完成:")
            print(f"  - 总层数: {stats['total_layers']}")
            print(f"  - 已加载: {stats['loaded_layers']}")
            print(f"  - 缺失层: {len(stats['missing_layers'])}")
            print(f"  - 额外层: {len(stats['unexpected_layers'])}")
            print(f"  - 总数据量: {stats['total_bytes_loaded'] / (1024**3):.2f} GB")
        
        return stats
    
    def list_available_checkpoints(self) -> List[Tuple[str, int]]:
        """
        列出所有可用的检查点
        
        Returns:
            [(checkpoint_id, training_step), ...]
        """
        checkpoints = []
        for chk_id, chk in self.metadata_manager.checkpoints.items():
            checkpoints.append((chk_id, chk.training_step))
        
        # 按训练步数排序
        checkpoints.sort(key=lambda x: x[1])
        
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict]:
        """获取检查点的详细信息"""
        metadata = self.metadata_manager.get_checkpoint(checkpoint_id)
        
        if metadata is None:
            return None
        
        return {
            'checkpoint_id': metadata.checkpoint_id,
            'training_step': metadata.training_step,
            'total_layers': metadata.total_layers,
            'total_size_gb': metadata.total_size_bytes / (1024**3),
            'created_at': time.ctime(metadata.created_at),
            'checkpoint_file': metadata.checkpoint_file,
            'layers': list(metadata.layers.keys())
        }
