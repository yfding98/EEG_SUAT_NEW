#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet模型实现

基于论文: EEGNet: A Compact Convolutional Neural Network for EEG-based 
Brain-Computer Interfaces (Lawhern et al., 2018)

架构特点:
1. Temporal Convolution - 时间卷积捕获频率信息
2. Depthwise Convolution - 深度卷积学习空间滤波
3. Separable Convolution - 分离卷积学习时空特征
4. 紧凑高效，适合EEG分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EEGNet(nn.Module):
    """
    EEGNet模型
    
    Args:
        n_channels: EEG通道数 (默认19)
        n_samples: 每个样本的时间采样点数 (默认200, 1秒@200Hz)
        n_classes: 输出类别数 (默认5, onset_zone脑区分类)
        dropout_rate: Dropout率 (默认0.5)
        F1: 时间卷积滤波器数量 (默认8)
        D: 每个时间滤波器的深度乘数 (默认2)
        F2: 分离卷积滤波器数量 (默认16, 通常F2=F1*D)
        kernel_length: 时间卷积核长度 (默认64, 采样率的一半)
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 200,
        n_classes: int = 5,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # ========== Block 1: Temporal + Spatial Convolution ==========
        # 时间卷积：学习频率滤波器
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # 深度卷积：学习空间滤波器（每个通道独立）
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
        
        # ========== Block 2: Separable Convolution ==========
        # 分离卷积：深度可分离卷积
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
        
        # ========== Classifier ==========
        # 计算展平后的特征维度
        self._feature_dim = self._calculate_feature_dim()
        
        self.classifier = nn.Linear(self._feature_dim, n_classes)
    
    def _calculate_feature_dim(self) -> int:
        """计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self._forward_features(x)
            return x.numel()
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取前向传播"""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
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
            x: 输入张量
               - (B, C, T): 单窗口数据
               - (B, 1, C, T): 已添加通道维度
               - (B, n_windows, C, T): 多窗口数据，将进行平均
        
        Returns:
            logits: (B, n_classes) 分类logits
        """
        # 处理输入维度
        if x.dim() == 3:
            # (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            # (B, n_windows, C, T) -> 对每个窗口处理后平均
            B, n_windows, C, T = x.shape
            x = x.view(B * n_windows, 1, C, T)
            features = self._forward_features(x)
            features = features.view(B, n_windows, -1)
            features = features.mean(dim=1)  # 窗口平均
            return self.classifier(features)
        
        # 标准前向传播
        features = self._forward_features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        
        return logits


class EEGNetOnsetZone(nn.Module):
    """
    用于onset_zone多标签分类的EEGNet
    
    处理多窗口输入并进行脑区分类
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 200,
        n_windows: int = 30,
        n_classes: int = 5,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        temporal_aggregation: str = 'attention'  # 'mean', 'max', 'attention'
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_windows = n_windows
        self.n_classes = n_classes
        self.temporal_aggregation = temporal_aggregation
        
        # EEGNet backbone
        self.backbone = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,  # 临时值，后面会覆盖classifier
            dropout_rate=dropout_rate,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length
        )
        
        # 获取backbone特征维度
        self.feature_dim = self.backbone._feature_dim
        
        # 替换分类器为特征输出
        self.backbone.classifier = nn.Identity()
        
        # 时间聚合
        if temporal_aggregation == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.Tanh(),
                nn.Linear(self.feature_dim // 4, 1),
            )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, n_windows, C, T) 多窗口EEG数据
        
        Returns:
            logits: (B, n_classes) 多标签分类logits
        """
        B, n_windows, C, T = x.shape
        
        # 展开处理每个窗口
        x = x.view(B * n_windows, 1, C, T)
        features = self.backbone._forward_features(x)
        features = features.view(B, n_windows, -1)  # (B, n_windows, feature_dim)
        
        # 时间聚合
        if self.temporal_aggregation == 'mean':
            aggregated = features.mean(dim=1)
        elif self.temporal_aggregation == 'max':
            aggregated = features.max(dim=1)[0]
        elif self.temporal_aggregation == 'attention':
            # 注意力加权聚合
            attn_weights = self.temporal_attention(features)  # (B, n_windows, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (features * attn_weights).sum(dim=1)  # (B, feature_dim)
        else:
            aggregated = features.mean(dim=1)
        
        # 分类
        logits = self.classifier(aggregated)
        
        return logits


def create_eegnet_model(
    task_type: str = 'onset_zone',
    config: dict = None
) -> nn.Module:
    """
    创建EEGNet模型
    
    Args:
        task_type: 'onset_zone', 'channel', 'hemi'
        config: 模型配置
    
    Returns:
        模型实例
    """
    config = config or {}
    
    n_channels = config.get('n_channels', 19)
    n_samples = config.get('n_samples', 200)
    n_windows = config.get('n_windows', 30)
    dropout_rate = config.get('dropout', 0.5)
    F1 = config.get('F1', 8)
    D = config.get('D', 2)
    F2 = config.get('F2', 16)
    kernel_length = config.get('kernel_length', 64)
    
    if task_type == 'onset_zone':
        n_classes = 5
    elif task_type == 'hemi':
        n_classes = 4
    elif task_type == 'channel':
        n_classes = n_channels
    else:
        n_classes = config.get('n_classes', 5)
    
    return EEGNetOnsetZone(
        n_channels=n_channels,
        n_samples=n_samples,
        n_windows=n_windows,
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        F1=F1,
        D=D,
        F2=F2,
        kernel_length=kernel_length,
        temporal_aggregation=config.get('temporal_aggregation', 'attention')
    )


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试EEGNet模型...")
    
    # 测试参数
    batch_size = 4
    n_channels = 19
    n_samples = 200
    n_windows = 30
    
    # 测试基础EEGNet
    print("\n1. 基础EEGNet:")
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=5)
    x = torch.randn(batch_size, 1, n_channels, n_samples)
    out = model(x)
    print(f"   输入: {x.shape}, 输出: {out.shape}")
    
    # 测试多窗口输入
    print("\n2. 多窗口输入:")
    x_multi = torch.randn(batch_size, n_windows, n_channels, n_samples)
    out = model(x_multi)
    print(f"   输入: {x_multi.shape}, 输出: {out.shape}")
    
    # 测试EEGNetOnsetZone
    print("\n3. EEGNetOnsetZone:")
    model = EEGNetOnsetZone(
        n_channels=n_channels,
        n_samples=n_samples,
        n_windows=n_windows,
        n_classes=5
    )
    out = model(x_multi)
    print(f"   输入: {x_multi.shape}, 输出: {out.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    print("\n测试通过!")
