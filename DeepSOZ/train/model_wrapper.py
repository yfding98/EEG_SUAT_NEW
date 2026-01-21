#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN模型适配器

将EEG_SUAT_NEW/STGNN/stgnn_model.py中的STGNN_SOZ_Locator适配到三种分类任务:
1. onset_zone - 脑区级别分类 (5类多标签)
2. hemi - 半球级别分类 (4类单标签)  
3. channel - 通道级别分类 (19通道多标签)

模型架构:
1. 时间特征提取器（因果时间卷积）
2. 自适应图结构学习
3. 图卷积（GCN/GAT）
4. 分类头（带时间注意力）
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

# 添加STGNN路径
STGNN_PATH = Path(__file__).parent.parent.parent / "STGNN"
sys.path.insert(0, str(STGNN_PATH))

# 导入STGNN模块
try:
    from STGNN.temporal_extractor import TemporalFeatureExtractor
    from STGNN.graph_learning import AdaptiveGraphLearning
    from STGNN.graph_conv import GraphConvolutionBlock
    from STGNN.classifier_head import ClassificationHead
    from STGNN.positional_encoding import PositionalEncoding
    HAS_STGNN_MODULES = True
except ImportError as e:
    print(f"Warning: 无法导入STGNN模块: {e}")
    HAS_STGNN_MODULES = False


# ==============================================================================
# 脑区到通道的映射
# ==============================================================================

# 标准19通道顺序
STANDARD_19_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2'
]

# 脑区到通道索引的映射
REGION_TO_CHANNEL_IDX = {
    'frontal': [0, 1, 2, 3, 4, 5, 6],      # FP1, FP2, F7, F3, FZ, F4, F8
    'temporal': [7, 11, 12, 16],            # T3, T4, T5, T6
    'central': [8, 9, 10],                  # C3, CZ, C4
    'parietal': [13, 14, 15],               # P3, PZ, P4
    'occipital': [17, 18]                   # O1, O2
}

# 半球到通道索引的映射
HEMI_TO_CHANNEL_IDX = {
    'L': [0, 2, 3, 7, 8, 12, 13, 17],       # FP1, F7, F3, T3, C3, T5, P3, O1
    'R': [1, 5, 6, 10, 11, 15, 16, 18],     # FP2, F4, F8, C4, T4, P4, T6, O2
    'M': [4, 9, 14]                          # FZ, CZ, PZ
}


# ==============================================================================
# 通道到脑区/半球的聚合层
# ==============================================================================

class ChannelToRegionAggregation(nn.Module):
    """
    将通道级别 logits 聚合到脑区级别 logits
    
    注意：此模块现在作用于 logits（未经 sigmoid），
    输出也是 logits，以便与 BCEWithLogitsLoss 配合使用。
    
    输入: (B, 19) 通道级别 logits
    输出: (B, 5) 脑区级别 logits
    """
    
    def __init__(self, aggregation: str = 'max'):
        """
        Args:
            aggregation: 聚合方式 'max', 'mean', 'attention'
        """
        super().__init__()
        self.aggregation = aggregation
        
        # 构建索引掩码
        self.register_buffer('region_masks', self._build_region_masks())
        
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(19, 19),
                nn.Tanh(),
                nn.Linear(19, 5),
                nn.Softmax(dim=-1)
            )
    
    def _build_region_masks(self) -> torch.Tensor:
        """构建脑区掩码矩阵"""
        masks = torch.zeros(5, 19)
        for i, region in enumerate(['frontal', 'temporal', 'central', 'parietal', 'occipital']):
            for ch_idx in REGION_TO_CHANNEL_IDX[region]:
                masks[i, ch_idx] = 1.0
        return masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 19) 通道级别特征
        """
        B = x.shape[0]
        
        if self.aggregation == 'max':
            # Max pooling per region
            output = torch.zeros(B, 5, device=x.device, dtype=x.dtype)
            for i in range(5):
                mask = self.region_masks[i] > 0.5
                output[:, i] = x[:, mask].max(dim=-1)[0]
            return output
        
        elif self.aggregation == 'mean':
            # Mean pooling per region
            output = torch.zeros(B, 5, device=x.device, dtype=x.dtype)
            for i in range(5):
                mask = self.region_masks[i] > 0.5
                output[:, i] = x[:, mask].mean(dim=-1)
            return output
        
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            attn = self.attention(x)  # (B, 5)
            output = torch.zeros(B, 5, device=x.device, dtype=x.dtype)
            for i in range(5):
                mask = self.region_masks[i] > 0.5
                weights = F.softmax(attn[:, i:i+1].expand(-1, mask.sum()), dim=-1)
                output[:, i] = (x[:, mask] * weights).sum(dim=-1)
            return output
        
        return x


class ChannelToHemiAggregation(nn.Module):
    """
    将通道级别特征聚合到半球级别
    
    输入: (B, 19) 通道级别特征
    输出: (B, 4) 半球级别特征 [L, R, B(bilateral), U(unknown)]
    """
    
    def __init__(self, aggregation: str = 'learned'):
        super().__init__()
        self.aggregation = aggregation
        
        # 构建索引
        self.left_idx = HEMI_TO_CHANNEL_IDX['L']
        self.right_idx = HEMI_TO_CHANNEL_IDX['R']
        self.mid_idx = HEMI_TO_CHANNEL_IDX['M']
        
        if aggregation == 'learned':
            # 学习半球分类
            self.classifier = nn.Sequential(
                nn.Linear(19, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 4)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 19) 通道级别特征
        """
        if self.aggregation == 'learned':
            return self.classifier(x)
        
        B = x.shape[0]
        
        # 计算每个半球的特征
        left_feat = x[:, self.left_idx].max(dim=-1)[0]
        right_feat = x[:, self.right_idx].max(dim=-1)[0]
        mid_feat = x[:, self.mid_idx].max(dim=-1)[0]
        
        # 构建输出: [L, R, B, U]
        output = torch.stack([
            left_feat,
            right_feat,
            (left_feat + right_feat) / 2,  # Bilateral
            mid_feat  # 中线作为Unknown参考
        ], dim=-1)
        
        return output


# ==============================================================================
# STGNN模型适配器
# ==============================================================================

class STGNNAdapter(nn.Module):
    """
    STGNN模型适配器
    
    基于STGNN_SOZ_Locator架构，支持三种输出类型:
    - 'channel': 19通道多标签分类
    - 'onset_zone': 5脑区多标签分类
    - 'hemi': 4类半球分类
    """
    
    def __init__(
        self,
        output_type: str = 'channel',  # 'channel', 'onset_zone', 'hemi', 'all'
        n_channels: int = 19,
        n_bands: int = 5,
        time_steps: int = 200,  # 每窗口采样点数
        n_windows: int = 30,    # 窗口数
        temporal_hidden_dim: int = 32,
        temporal_reduced_steps: int = 64,
        graph_learning_method: str = 'combined',
        graph_hidden_dim: int = 32,
        graph_conv_type: str = 'gcn',
        graph_n_layers: int = 2,
        dropout: float = 0.6,
        use_batch_norm: bool = True,
        region_aggregation: str = 'max',
        hemi_aggregation: str = 'learned'
    ):
        super().__init__()
        
        self.output_type = output_type
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.time_steps = time_steps
        
        # 输入投影：将原始信号转换为多频带特征
        # 输入: (B, T, C, L) -> (B, C, n_bands, T*L/reduction)
        self.input_projection = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * n_bands, kernel_size=7, padding=3, groups=n_channels),
            nn.BatchNorm1d(n_channels * n_bands),
            nn.ReLU(),
            nn.Conv1d(n_channels * n_bands, n_channels * n_bands, kernel_size=5, padding=2, groups=n_bands),
            nn.BatchNorm1d(n_channels * n_bands),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(time_steps)  # 固定时间维度
        )
        self.n_bands = n_bands
        
        if HAS_STGNN_MODULES:
            # 使用STGNN模块
            self.positional_encoding = PositionalEncoding(
                d_model=n_bands,
                max_len=time_steps,
                dropout=dropout * 0.5
            )
            
            self.temporal_extractor = TemporalFeatureExtractor(
                n_bands=n_bands,
                hidden_dim=temporal_hidden_dim,
                reduced_time_steps=temporal_reduced_steps,
                dropout=dropout
            )
            
            self.graph_learning = AdaptiveGraphLearning(
                n_channels=n_channels,
                hidden_dim=temporal_hidden_dim,
                method=graph_learning_method,
                temperature=0.1,
                k_nearest=None
            )
            
            self.graph_conv = GraphConvolutionBlock(
                in_features=temporal_hidden_dim,
                hidden_features=graph_hidden_dim,
                out_features=graph_hidden_dim,
                n_layers=graph_n_layers,
                conv_type=graph_conv_type,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            
            # 通道级别分类头
            self.channel_classifier = ClassificationHead(
                in_features=graph_hidden_dim,
                n_channels=n_channels,
                hidden_dim=graph_hidden_dim,
                dropout=dropout
            )
        else:
            # 简化实现（不使用STGNN模块）
            self._build_simple_modules(
                n_channels, n_bands, time_steps, temporal_hidden_dim,
                graph_hidden_dim, dropout
            )
        
        # 脑区聚合层
        self.region_aggregation = ChannelToRegionAggregation(
            aggregation=region_aggregation
        )
        
        # 半球聚合层
        self.hemi_aggregation = ChannelToHemiAggregation(
            aggregation=hemi_aggregation
        )
    
    def _build_simple_modules(self, n_channels, n_bands, time_steps, 
                              temporal_hidden_dim, graph_hidden_dim, dropout):
        """构建简化的模块（不依赖STGNN）"""
        # 时间特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(n_bands, temporal_hidden_dim, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(temporal_hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((n_channels, 64))
        )
        
        # 图卷积（简化版）
        self.graph_fc = nn.Sequential(
            nn.Linear(temporal_hidden_dim * 64, graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        
        # 通道分类
        self.channel_fc = nn.Linear(graph_hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
               - 如果是 (B, T, C, L): T个窗口，每窗口C通道L采样点
               - 如果是 (B, C, F, T): C通道F频带T时间步
            return_all: 是否返回所有输出类型
        
        Returns:
            Dict with 'channel', 'onset_zone', 'hemi' predictions
        """
        # 处理输入维度
        if x.dim() == 4:
            B, T, C, L = x.shape
            if T == self.n_windows and C == self.n_channels:
                # (B, T, C, L) -> (B, C, L*T) -> 投影
                x = x.permute(0, 2, 1, 3).contiguous()  # (B, C, T, L)
                x = x.view(B, C, -1)  # (B, C, T*L)
                
                # 输入投影
                x = self.input_projection(x)  # (B, C*n_bands, time_steps)
                x = x.view(B, C, self.n_bands, -1)  # (B, C, n_bands, time_steps)
            elif C == self.n_channels:
                # 已经是 (B, C, F, T) 格式
                pass
            else:
                raise ValueError(f"不支持的输入形状: {x.shape}")
        else:
            raise ValueError(f"输入必须是4维张量，得到: {x.dim()}维")
        
        # 通道级别预测
        if HAS_STGNN_MODULES:
            channel_logits = self._forward_stgnn(x)
        else:
            channel_logits = self._forward_simple(x)
        
        # 构建输出
        outputs = {'channel': channel_logits}
        
        # 脑区级别预测 - 对 logits 进行聚合，输出也是 logits
        # 注意：region_aggregation 现在作用于 logits，而不是 probs
        outputs['onset_zone'] = self.region_aggregation(channel_logits)
        
        # 半球级别预测 - 保持使用 probs，因为 hemi 使用 CrossEntropyLoss
        channel_probs = torch.sigmoid(channel_logits)
        outputs['hemi'] = self.hemi_aggregation(channel_probs)
        
        if return_all or self.output_type == 'all':
            return outputs
        else:
            return outputs[self.output_type]
    
    def _forward_stgnn(self, x: torch.Tensor) -> torch.Tensor:
        """使用STGNN模块的前向传播"""
        # x: (B, C, n_bands, time_steps)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 时间特征提取
        temporal_features = self.temporal_extractor(x)  # (B, C, hidden, reduced_T)
        
        # 图学习
        adjacency, node_features = self.graph_learning(
            temporal_features, return_adjacency=True
        )
        
        # 图卷积
        B, C, features, T = temporal_features.shape
        temporal_features_reshaped = temporal_features.permute(0, 3, 1, 2).contiguous()
        temporal_features_reshaped = temporal_features_reshaped.view(B * T, C, features)
        
        adjacency_expanded = adjacency.unsqueeze(1).expand(-1, T, -1, -1)
        adjacency_expanded = adjacency_expanded.contiguous().view(B * T, C, C)
        
        graph_features = self.graph_conv(temporal_features_reshaped, adjacency_expanded)
        graph_features = graph_features.view(B, T, C, -1).permute(0, 2, 3, 1)
        
        # 分类
        channel_logits = self.channel_classifier(graph_features)
        
        return channel_logits
    
    def _forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        """简化前向传播（不使用STGNN模块）"""
        # x: (B, C, n_bands, time_steps)
        B, C, F, T = x.shape
        
        # 交换维度用于Conv2d: (B, F, C, T)
        x = x.permute(0, 2, 1, 3)
        
        # 时间卷积
        x = self.temporal_conv(x)  # (B, hidden, C, 64)
        
        # 每通道处理
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, C, hidden, 64)
        x = x.view(B, C, -1)  # (B, C, hidden*64)
        
        # 图特征（简化）
        x = self.graph_fc(x)  # (B, C, graph_hidden)
        
        # 通道分类
        channel_logits = self.channel_fc(x).squeeze(-1)  # (B, C)
        
        return channel_logits


class MultiTaskSTGNN(nn.Module):
    """
    多任务STGNN模型
    
    同时输出三种粒度的预测：
    - channel: (B, 19) 通道级别
    - onset_zone: (B, 5) 脑区级别
    - hemi: (B, 4) 半球级别
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_bands: int = 5,
        time_steps: int = 200,
        n_windows: int = 30,
        temporal_hidden_dim: int = 32,
        graph_hidden_dim: int = 32,
        dropout: float = 0.6,
        **kwargs
    ):
        super().__init__()
        
        # 基础STGNN编码器
        self.encoder = STGNNAdapter(
            output_type='all',
            n_channels=n_channels,
            n_bands=n_bands,
            time_steps=time_steps,
            n_windows=n_windows,
            temporal_hidden_dim=temporal_hidden_dim,
            graph_hidden_dim=graph_hidden_dim,
            dropout=dropout,
            **kwargs
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, L) 输入
        
        Returns:
            channel_logits: (B, 19)
            onset_zone_logits: (B, 5)
            hemi_logits: (B, 4)
        """
        outputs = self.encoder(x, return_all=True)
        return outputs['channel'], outputs['onset_zone'], outputs['hemi']


# ==============================================================================
# 工厂函数
# ==============================================================================

def create_model(
    task_type: str = 'channel',
    config: dict = None
) -> nn.Module:
    """
    创建指定任务类型的模型
    
    Args:
        task_type: 'channel', 'onset_zone', 'hemi', 'multi'
        config: 模型配置
    
    Returns:
        模型实例
    """
    config = config or {}
    
    if task_type == 'multi':
        return MultiTaskSTGNN(
            n_channels=config.get('n_channels', 19),
            n_bands=config.get('n_bands', 5),
            time_steps=config.get('time_steps', 200),
            n_windows=config.get('n_windows', 30),
            temporal_hidden_dim=config.get('temporal_hidden_dim', 32),
            graph_hidden_dim=config.get('graph_hidden_dim', 32),
            dropout=config.get('dropout', 0.6)
        )
    else:
        return STGNNAdapter(
            output_type=task_type,
            n_channels=config.get('n_channels', 19),
            n_bands=config.get('n_bands', 5),
            time_steps=config.get('time_steps', 200),
            n_windows=config.get('n_windows', 30),
            temporal_hidden_dim=config.get('temporal_hidden_dim', 32),
            graph_hidden_dim=config.get('graph_hidden_dim', 32),
            dropout=config.get('dropout', 0.6)
        )


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试STGNN适配器...")
    
    # 测试输入
    batch_size = 2
    n_windows = 30
    n_channels = 19
    window_length = 200
    
    x = torch.randn(batch_size, n_windows, n_channels, window_length)
    print(f"输入形状: {x.shape}")
    
    # 测试通道级别模型
    print("\n1. 通道级别模型:")
    model = create_model('channel')
    output = model(x)
    print(f"   输出形状: {output.shape}")
    
    # 测试脑区级别模型
    print("\n2. 脑区级别模型:")
    model = create_model('onset_zone')
    output = model(x)
    print(f"   输出形状: {output.shape}")
    
    # 测试半球级别模型
    print("\n3. 半球级别模型:")
    model = create_model('hemi')
    output = model(x)
    print(f"   输出形状: {output.shape}")
    
    # 测试多任务模型
    print("\n4. 多任务模型:")
    model = create_model('multi')
    channel, onset_zone, hemi = model(x)
    print(f"   通道输出: {channel.shape}")
    print(f"   脑区输出: {onset_zone.shape}")
    print(f"   半球输出: {hemi.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    print("\n所有测试通过!")
