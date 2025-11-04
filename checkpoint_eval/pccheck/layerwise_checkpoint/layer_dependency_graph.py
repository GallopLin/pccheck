"""
阶段一：模型依赖分析与计算图建模
Layer Dependency Graph Builder

自动分析神经网络结构，建立层与层之间的参数更新依赖关系
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set
from collections import OrderedDict
import networkx as nx


class DependencyGraphBuilder:
    """
    构建神经网络模型的参数更新依赖图
    
    在反向传播中，参数更新的顺序与前向传播的执行顺序相反。
    因此，后面的层参数会先更新，前面的层参数会后更新。
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False):
        """
        Args:
            model: PyTorch 模型实例
            verbose: 是否打印详细信息
        """
        self.model = model
        self.verbose = verbose
        self.dependency_graph = nx.DiGraph()
        self.layer_info = OrderedDict()
        
    def group_parameters_by_layer(self) -> Dict[str, Dict]:
        """
        将模型的参数按层（module）分组
        
        Returns:
            Dict[layer_name, {
                'module': nn.Module,
                'parameters': List[torch.Tensor],
                'param_count': int,
                'param_size_bytes': int
            }]
        """
        layer_params = OrderedDict()
        
        for name, module in self.model.named_modules():
            # 跳过容器模块（如 Sequential, ModuleList）和根模块
            if len(list(module.children())) > 0 or name == '':
                continue
                
            # 获取该模块的所有参数
            params = list(module.parameters(recurse=False))
            
            if len(params) > 0:
                param_count = sum(p.numel() for p in params)
                param_size = sum(p.numel() * p.element_size() for p in params)
                
                layer_params[name] = {
                    'module': module,
                    'parameters': params,
                    'param_count': param_count,
                    'param_size_bytes': param_size,
                    'dtype': params[0].dtype if params else None,
                    'device': params[0].device if params else None
                }
                
                if self.verbose:
                    print(f"Layer: {name:40s} | Params: {param_count:12,d} | "
                          f"Size: {param_size / (1024**2):8.2f} MB | "
                          f"Type: {type(module).__name__}")
        
        return layer_params
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        构建参数更新依赖图
        
        核心逻辑：
        1. 按前向传播顺序对层进行编号
        2. 在反向传播中，后面的层先更新，因此后层依赖于前层
        3. 构建有向边：layer_i -> layer_j 表示 layer_j 的更新必须在 layer_i 之后
        
        Returns:
            NetworkX DiGraph 对象
        """
        # 获取按层分组的参数信息
        self.layer_info = self.group_parameters_by_layer()
        
        if len(self.layer_info) == 0:
            raise ValueError("模型中没有找到可训练的参数！")
        
        # 获取层的前向传播顺序（即在模型中出现的顺序）
        layer_names = list(self.layer_info.keys())
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"检测到 {len(layer_names)} 个参数层")
            print(f"前向传播顺序: {' -> '.join(layer_names[:3])} ... {' -> '.join(layer_names[-3:])}")
            print(f"{'='*80}\n")
        
        # 为每个层创建图节点
        for idx, (layer_name, info) in enumerate(self.layer_info.items()):
            self.dependency_graph.add_node(
                layer_name,
                index=idx,
                param_count=info['param_count'],
                param_size_bytes=info['param_size_bytes'],
                module_type=type(info['module']).__name__,
                dtype=str(info['dtype']),
                device=str(info['device'])
            )
        
        # 构建依赖边
        # 在反向传播中，更新顺序是反向的：layer_n, layer_n-1, ..., layer_1
        # 因此 layer_i 的更新依赖于 layer_i+1 已经完成
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]
            
            # next_layer 更新完成后，current_layer 才能开始更新
            # 边的方向：next_layer -> current_layer
            self.dependency_graph.add_edge(
                next_layer, 
                current_layer,
                dependency_type='update_order'
            )
        
        if self.verbose:
            print(f"依赖图构建完成:")
            print(f"  节点数: {self.dependency_graph.number_of_nodes()}")
            print(f"  边数: {self.dependency_graph.number_of_edges()}")
            print(f"  是否为DAG: {nx.is_directed_acyclic_graph(self.dependency_graph)}")
        
        return self.dependency_graph
    
    def get_update_order(self) -> List[str]:
        """
        获取参数更新的顺序（逆拓扑排序）
        
        Returns:
            按更新顺序排列的层名称列表（后层先更新）
        """
        if self.dependency_graph.number_of_nodes() == 0:
            self.build_dependency_graph()
        
        # 逆拓扑排序：从没有前驱的节点开始（即最后的层）
        update_order = list(nx.topological_sort(self.dependency_graph))
        
        if self.verbose:
            print(f"\n参数更新顺序（反向传播）:")
            for idx, layer_name in enumerate(update_order):
                print(f"  {idx+1:2d}. {layer_name}")
        
        return update_order
    
    def get_layer_info(self, layer_name: str) -> Dict:
        """获取指定层的详细信息"""
        if layer_name not in self.layer_info:
            raise KeyError(f"Layer '{layer_name}' not found in model")
        return self.layer_info[layer_name]
    
    def visualize_graph(self, output_file: str = None):
        """
        可视化依赖图（需要安装 matplotlib）
        
        Args:
            output_file: 如果提供，将图保存到文件
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.dependency_graph.number_of_nodes() == 0:
                self.build_dependency_graph()
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
            
            # 根据参数量设置节点大小
            node_sizes = [
                self.dependency_graph.nodes[node]['param_count'] / 1000
                for node in self.dependency_graph.nodes()
            ]
            
            nx.draw(
                self.dependency_graph,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=node_sizes,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray'
            )
            
            plt.title("Layer Dependency Graph (Update Order)")
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"图已保存到: {output_file}")
            else:
                plt.show()
                
        except ImportError:
            print("需要安装 matplotlib 来可视化图: pip install matplotlib")


def test_dependency_graph():
    """测试函数：构建一个简单模型并分析其依赖关系"""
    
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # 构建依赖图
    builder = DependencyGraphBuilder(model, verbose=True)
    graph = builder.build_dependency_graph()
    
    # 获取更新顺序
    update_order = builder.get_update_order()
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")
    
    return builder, graph, update_order


if __name__ == "__main__":
    test_dependency_graph()
