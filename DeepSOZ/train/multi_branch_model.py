#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多分支特征融合模型

结合三种特征进行癫痫发作起始区(SOZ)定位:
1. EEGNet分支: 处理原始EEG波形
2. GAT分支: 处理脑网络连接性矩阵 (PLV, wPLI, AEC, Pearson, Granger, TE)
3. MLP分支: 处理图网络指标 (degree, clustering, betweenness等)

参考论文:
- EEGNet: Lawhern et al., 2018
- Graph Attention Network: Veličković et al., 2018
- DeepSOZ: Abou Jaoude et al., 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


# ==============================================================================
# EEGNet Branch (复用现有实现)
# ==============================================================================

class EEGNetBackbone(nn.Module):
    """
    EEGNet特征提取骨干网络
    
    Args:
        n_channels: EEG通道数 (默认19)
        n_samples: 每个样本的时间采样点数 (默认200, 1秒@200Hz)
        dropout_rate: Dropout率 (默认0.5)
        F1: 时间卷积滤波器数量 (默认8)
        D: 每个时间滤波器的深度乘数 (默认2)
        F2: 分离卷积滤波器数量 (默认16)
        kernel_length: 时间卷积核长度 (默认64)
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 200,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal + Spatial Convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.conv2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolution
        self.conv3 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 计算输出特征维度
        self._feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self) -> int:
        """计算展平后的特征维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self._forward_features(x)
            return x.numel()
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, 1, C, T) 或 (B, C, T)
        
        Returns:
            features: (B, feature_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = self._forward_features(x)
        features = features.view(features.size(0), -1)
        
        return features
    
    @property
    def output_dim(self) -> int:
        return self._feature_dim


class EEGNetBranch(nn.Module):
    """
    EEGNet分支 - 处理多窗口EEG数据
    
    支持时间注意力聚合和输出维度投影
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 200,
        n_windows: int = 30,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        temporal_aggregation: str = 'attention',
        output_dim: int = None  # 新增：统一输出维度，None表示使用原始维度
    ):
        super().__init__()
        
        self.n_windows = n_windows
        self.temporal_aggregation = temporal_aggregation
        
        # EEGNet backbone
        self.backbone = EEGNetBackbone(
            n_channels=n_channels,
            n_samples=n_samples,
            dropout_rate=dropout_rate,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length
        )
        
        self.backbone_dim = self.backbone.output_dim
        
        # 时间聚合
        if temporal_aggregation == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.backbone_dim, max(1, self.backbone_dim // 4)),
                nn.Tanh(),
                nn.Linear(max(1, self.backbone_dim // 4), 1),
            )
        
        # 输出投影层（将backbone输出投影到统一维度）
        self._output_dim = output_dim if output_dim is not None else self.backbone_dim
        if output_dim is not None and output_dim != self.backbone_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.output_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, n_windows, C, T) 多窗口EEG数据
        
        Returns:
            features: (B, output_dim)
        """
        B, n_windows, C, T = x.shape
        
        # 展开处理每个窗口
        x = x.view(B * n_windows, 1, C, T)
        features = self.backbone(x)
        features = features.view(B, n_windows, -1)  # (B, n_windows, backbone_dim)
        
        # 时间聚合
        if self.temporal_aggregation == 'mean':
            aggregated = features.mean(dim=1)
        elif self.temporal_aggregation == 'max':
            aggregated = features.max(dim=1)[0]
        elif self.temporal_aggregation == 'attention':
            attn_weights = self.temporal_attention(features)  # (B, n_windows, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (features * attn_weights).sum(dim=1)  # (B, backbone_dim)
        else:
            aggregated = features.mean(dim=1)
        
        # 投影到统一维度
        if self.output_proj is not None:
            aggregated = self.output_proj(aggregated)
        
        return aggregated
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# ==============================================================================
# Graph Attention Network Branch
# ==============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer
    
    参考: Veličković et al., 2018
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.5,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(1, n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, in_features) 节点特征
            adj: (B, N, N) 邻接矩阵
        
        Returns:
            out: (B, N, out_features * n_heads) if concat else (B, N, out_features)
        """
        B, N, _ = x.shape
        
        # 线性变换
        h = self.W(x)  # (B, N, out_features * n_heads)
        h = h.view(B, N, self.n_heads, self.out_features)  # (B, N, n_heads, out_features)
        
        # 计算注意力系数
        a_input = torch.cat([
            h.unsqueeze(2).repeat(1, 1, N, 1, 1),  # (B, N, N, n_heads, out_features)
            h.unsqueeze(1).repeat(1, N, 1, 1, 1)   # (B, N, N, n_heads, out_features)
        ], dim=-1)  # (B, N, N, n_heads, 2*out_features)
        
        e = self.leakyrelu((a_input * self.a).sum(dim=-1))  # (B, N, N, n_heads)
        
        # 使用邻接矩阵进行mask
        adj_expanded = adj.unsqueeze(-1)  # (B, N, N, 1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=2)  # (B, N, N, n_heads)
        attention = self.dropout(attention)
        
        # 聚合邻居特征
        h_prime = torch.einsum('bijk,bjkf->bikf', attention, h)  # (B, N, n_heads, out_features)
        
        if self.concat:
            return h_prime.reshape(B, N, -1)  # (B, N, n_heads * out_features)
        else:
            return h_prime.mean(dim=2)  # (B, N, out_features)


class GATBranch(nn.Module):
    """
    Graph Attention Network分支 - 处理连接性矩阵
    
    输入连接性矩阵作为邻接矩阵，节点特征为各个连接性指标的行向量
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_connectivity_types: int = 6,  # PLV, wPLI, AEC, Pearson, Granger, TE
        hidden_dim: int = 32,
        output_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        use_node_features: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_connectivity_types = n_connectivity_types
        self.use_node_features = use_node_features
        
        # 节点特征维度 = 连接性类型数 * 通道数 (每个节点与其他节点的连接)
        if use_node_features:
            # 使用连接性矩阵的行作为节点特征
            in_features = n_connectivity_types * n_channels
        else:
            # 使用one-hot编码
            in_features = n_channels
        
        # 输入投影
        self.input_proj = nn.Linear(in_features, hidden_dim * n_heads)
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        # 计算最后一层的输出维度（concat=False时取平均，所以输出是out_features）
        last_layer_out = hidden_dim  # 最后一层每个head输出hidden_dim，然后取平均
        
        for i in range(n_layers):
            in_dim = hidden_dim * n_heads
            out_dim = hidden_dim  # 每层都输出hidden_dim
            concat = i < n_layers - 1  # 只有最后一层concat=False
            
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=concat
                )
            )
        
        # 最后一层concat=False后输出维度是hidden_dim，需要投影到output_dim
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        connectivity: torch.Tensor,
        graph_metrics: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            connectivity: (B, M, C, C) 连接性矩阵, M=连接性类型数
            graph_metrics: (B, C, K) 可选的节点级别图指标
        
        Returns:
            features: (B, output_dim)
        """
        B, M, C, _ = connectivity.shape
        
        # 构建节点特征: 每个节点使用所有连接性矩阵的对应行
        # (B, M, C, C) -> (B, C, M*C)
        node_features = connectivity.permute(0, 2, 1, 3).reshape(B, C, M * C)
        
        # 构建邻接矩阵: 使用连接性矩阵的平均值
        adj = connectivity.mean(dim=1)  # (B, C, C)
        # 归一化并阈值化
        adj = F.relu(adj)  # 确保非负
        adj = adj / (adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        
        # 添加自环
        eye = torch.eye(C, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + eye
        
        # 输入投影
        x = self.input_proj(node_features)  # (B, C, hidden_dim * n_heads)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adj)
            x = F.elu(x)
        
        # 全局池化
        features = x.mean(dim=1)  # (B, hidden_dim)
        
        # 投影到output_dim
        features = self.output_proj(features)  # (B, output_dim)
        
        return features
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# ==============================================================================
# Graph Metrics MLP Branch
# ==============================================================================

class GraphMetricsBranch(nn.Module):
    """
    图指标MLP分支 - 处理节点级别图网络指标
    
    输入: 每个通道的图指标 (degree, strength, clustering, betweenness, eigenvector等)
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_graph_features: int = 5,  # 图指标数量
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_graph_features = n_graph_features
        
        input_dim = n_channels * n_graph_features
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, C, K) 图网络指标, C=通道数, K=指标数
        
        Returns:
            features: (B, output_dim)
        """
        B = x.shape[0]
        x = x.view(B, -1)  # 展平
        features = self.mlp(x)
        return features
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# ==============================================================================
# Feature Fusion Modules
# ==============================================================================

class ConcatFusion(nn.Module):
    """简单拼接融合"""
    
    def __init__(self, dims: List[int], output_dim: int, dropout: float = 0.5):
        super().__init__()
        total_dim = sum(dims)
        self.proj = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self._output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 各分支特征列表
        Returns:
            fused: (B, output_dim)
        """
        concat = torch.cat(features, dim=-1)
        return self.proj(concat)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class AttentionFusion(nn.Module):
    """注意力融合"""
    
    def __init__(self, dims: List[int], output_dim: int, dropout: float = 0.5):
        super().__init__()
        
        self.n_branches = len(dims)
        
        # 投影到相同维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in dims
        ])
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self._output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 各分支特征列表
        Returns:
            fused: (B, output_dim)
        """
        # 投影到相同维度
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)  # (B, n_branches, output_dim)
        
        # 计算注意力权重
        attn_weights = self.attention(stacked)  # (B, n_branches, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权融合
        fused = (stacked * attn_weights).sum(dim=1)  # (B, output_dim)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class GatedFusion(nn.Module):
    """门控融合 (Gated Multimodal Unit)"""
    
    def __init__(self, dims: List[int], output_dim: int, dropout: float = 0.5):
        super().__init__()
        
        self.n_branches = len(dims)
        
        # 投影到相同维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in dims
        ])
        
        # 门控网络
        total_dim = output_dim * len(dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        
        # 输出变换
        self.transform = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Tanh()
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self._output_dim = output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 各分支特征列表
        Returns:
            fused: (B, output_dim)
        """
        # 投影并拼接
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        concat = torch.cat(projected, dim=-1)  # (B, total_dim)
        
        # 门控
        gate = self.gate(concat)  # (B, output_dim)
        transform = self.transform(concat)  # (B, output_dim)
        
        fused = gate * transform
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# ==============================================================================
# Multi-Branch Fusion Model
# ==============================================================================

class MultiBranchFusionModel(nn.Module):
    """
    多分支特征融合模型
    
    分支1 - EEGNet: 处理原始时序波形
    分支2 - GAT: 处理连接性矩阵
    分支3 - MLP: 处理图网络指标
    
    Args:
        n_channels: EEG通道数
        n_samples: 每窗口采样点数
        n_windows: 窗口数
        n_classes: 分类类别数
        n_connectivity_types: 连接性矩阵类型数
        n_graph_features: 图指标数量
        fusion_feature_dim: 各分支统一输出维度（必须是16的倍数）
        fusion_type: 融合策略 ('concat', 'attention', 'gated')
        fusion_dim: 融合后特征维度
        dropout: dropout率
        eegnet_config: EEGNet配置
        gat_config: GAT配置
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 200,
        n_windows: int = 30,
        n_classes: int = 5,
        n_connectivity_types: int = 6,
        n_graph_features: int = 5,
        fusion_feature_dim: int = 64,  # 新增：各分支统一输出维度
        fusion_type: str = 'attention',
        fusion_dim: int = 128,
        dropout: float = 0.5,
        eegnet_config: Dict = None,
        gat_config: Dict = None
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fusion_type = fusion_type
        self.fusion_feature_dim = fusion_feature_dim
        
        # 默认配置
        eegnet_config = eegnet_config or {}
        gat_config = gat_config or {}
        
        # 分支1: EEGNet - 输出投影到fusion_feature_dim
        self.eegnet_branch = EEGNetBranch(
            n_channels=n_channels,
            n_samples=n_samples,
            n_windows=n_windows,
            dropout_rate=dropout,
            F1=eegnet_config.get('F1', 8),
            D=eegnet_config.get('D', 2),
            F2=eegnet_config.get('F2', 16),
            kernel_length=eegnet_config.get('kernel_length', 64),
            temporal_aggregation=eegnet_config.get('temporal_aggregation', 'attention'),
            output_dim=fusion_feature_dim  # 统一输出维度
        )
        
        # 分支2: GAT - 输出投影到fusion_feature_dim
        self.gat_branch = GATBranch(
            n_channels=n_channels,
            n_connectivity_types=n_connectivity_types,
            hidden_dim=gat_config.get('hidden_dim', 32),
            output_dim=fusion_feature_dim,  # 统一输出维度
            n_heads=gat_config.get('n_heads', 4),
            n_layers=gat_config.get('n_layers', 2),
            dropout=dropout
        )
        
        # 分支3: Graph Metrics MLP - 输出投影到fusion_feature_dim
        self.graph_metrics_branch = GraphMetricsBranch(
            n_channels=n_channels,
            n_graph_features=n_graph_features,
            hidden_dim=64,
            output_dim=fusion_feature_dim,  # 统一输出维度
            dropout=dropout
        )
        
        # 各分支输出维度（现在都是fusion_feature_dim）
        branch_dims = [
            fusion_feature_dim,
            fusion_feature_dim,
            fusion_feature_dim
        ]
        
        # 融合模块
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(branch_dims, fusion_dim, dropout)
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(branch_dims, fusion_dim, dropout)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(branch_dims, fusion_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, n_classes)
        )
    
    def forward(
        self,
        eeg_data: torch.Tensor,
        connectivity: torch.Tensor,
        graph_metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            eeg_data: (B, T, C, L) 原始EEG波形
                B=batch, T=n_windows, C=n_channels, L=n_samples
            connectivity: (B, M, C, C) 连接性矩阵
                M=连接性类型数 (PLV, wPLI, AEC, Pearson, Granger, TE)
            graph_metrics: (B, C, K) 图网络指标
                K=指标数 (degree, strength, clustering, betweenness, eigenvector)
        
        Returns:
            logits: (B, n_classes) 分类logits
        """
        # 各分支特征提取
        eeg_features = self.eegnet_branch(eeg_data)
        gat_features = self.gat_branch(connectivity)
        graph_features = self.graph_metrics_branch(graph_metrics)
        
        # 融合
        fused = self.fusion([eeg_features, gat_features, graph_features])
        
        # 分类
        logits = self.classifier(fused)
        
        return logits
    
    def get_branch_features(
        self,
        eeg_data: torch.Tensor,
        connectivity: torch.Tensor,
        graph_metrics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """获取各分支特征（用于可视化分析）"""
        eeg_features = self.eegnet_branch(eeg_data)
        gat_features = self.gat_branch(connectivity)
        graph_features = self.graph_metrics_branch(graph_metrics)
        fused = self.fusion([eeg_features, gat_features, graph_features])
        
        return {
            'eeg': eeg_features,
            'connectivity': gat_features,
            'graph_metrics': graph_features,
            'fused': fused
        }


def create_multi_branch_model(config: Dict = None) -> MultiBranchFusionModel:
    """
    创建多分支融合模型
    
    Args:
        config: 模型配置字典，包含:
            - fusion_feature_dim: 各分支统一输出维度（默认64，必须是16的倍数）
            - 其他参数...
    
    Returns:
        model: MultiBranchFusionModel实例
    """
    config = config or {}
    
    return MultiBranchFusionModel(
        n_channels=config.get('n_channels', 19),
        n_samples=config.get('n_samples', 200),
        n_windows=config.get('n_windows', 30),
        n_classes=config.get('n_classes', 5),
        n_connectivity_types=config.get('n_connectivity_types', 6),
        n_graph_features=config.get('n_graph_features', 5),
        fusion_feature_dim=config.get('fusion_feature_dim', 64),  # 统一分支输出维度
        fusion_type=config.get('fusion_type', 'attention'),
        fusion_dim=config.get('fusion_dim', 128),
        dropout=config.get('dropout', 0.5),
        eegnet_config=config.get('eegnet', {}),
        gat_config=config.get('gat', {})
    )


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试多分支融合模型...")
    
    # 测试参数
    batch_size = 4
    n_channels = 19
    n_samples = 200
    n_windows = 30
    n_connectivity_types = 6
    n_graph_features = 5
    n_classes = 5
    
    # 创建测试数据
    eeg_data = torch.randn(batch_size, n_windows, n_channels, n_samples)
    connectivity = torch.rand(batch_size, n_connectivity_types, n_channels, n_channels)
    connectivity = (connectivity + connectivity.transpose(-1, -2)) / 2  # 对称化
    graph_metrics = torch.randn(batch_size, n_channels, n_graph_features)
    
    print(f"输入形状:")
    print(f"  EEG data: {eeg_data.shape}")
    print(f"  Connectivity: {connectivity.shape}")
    print(f"  Graph metrics: {graph_metrics.shape}")
    
    # 测试各分支
    print("\n1. 测试EEGNet分支:")
    eegnet = EEGNetBranch(n_channels, n_samples, n_windows)
    out = eegnet(eeg_data)
    print(f"   输出: {out.shape}")
    
    print("\n2. 测试GAT分支:")
    gat = GATBranch(n_channels, n_connectivity_types)
    out = gat(connectivity)
    print(f"   输出: {out.shape}")
    
    print("\n3. 测试Graph Metrics分支:")
    graph_mlp = GraphMetricsBranch(n_channels, n_graph_features)
    out = graph_mlp(graph_metrics)
    print(f"   输出: {out.shape}")
    
    # 测试各种融合策略
    for fusion_type in ['concat', 'attention', 'gated']:
        print(f"\n4. 测试完整模型 ({fusion_type} fusion):")
        model = MultiBranchFusionModel(
            n_channels=n_channels,
            n_samples=n_samples,
            n_windows=n_windows,
            n_classes=n_classes,
            n_connectivity_types=n_connectivity_types,
            n_graph_features=n_graph_features,
            fusion_type=fusion_type
        )
        
        out = model(eeg_data, connectivity, graph_metrics)
        print(f"   输出: {out.shape}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    print("\n测试通过!")
