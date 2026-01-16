"""
模型架构模块
迁移DeepSOZ项目的模型结构，包括：
- ConvBlock: 卷积块
- ctg_11_8: 主要的SOZ定位模型
- 脑区分类模型
- 通道分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """卷积块 - 包含多层1D卷积"""
    
    def __init__(self, channels, nlayers, kernel_size=3,
                 stride=1, padding=1, residual=False, batch_norm=False):
        """
        Args:
            channels: 通道数
            nlayers: 卷积层数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            residual: 是否使用残差连接
            batch_norm: 是否使用批标准化
        """
        super().__init__()
        
        self.residual = residual
        self.batch_norm = batch_norm
        
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding)
            for _ in range(nlayers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(nlayers)])
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        h = self.convs[0](x)
        if self.batch_norm:
            h = self.bns[0](h)
        
        for ii, conv in enumerate(self.convs[1:]):
            h = self.relu(h)
            h = conv(h)
            if self.batch_norm:
                h = self.bns[ii + 1](h)
        
        if self.residual:
            h = h + x
        
        h = self.relu(h)
        return h + x


class DeepSOZModel(nn.Module):
    """
    DeepSOZ主模型 (ctg_11_8)
    结合CNN特征提取、Transformer注意力和GRU时序建模
    """
    
    def __init__(self, num_channels=19, cnn_dropout=0.0, 
                 gru_dropout=0.0, transformer_dropout=0.0):
        """
        Args:
            num_channels: 输入通道数
            cnn_dropout: CNN dropout率
            gru_dropout: GRU dropout率
            transformer_dropout: Transformer dropout率
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.tgt_mask = nn.parameter.Parameter(
            torch.eye(num_channels), requires_grad=False
        )
        
        # ========== 单通道编码器 ==========
        self.nchn_c = 80
        self.ConvEmbeddingC = nn.Conv1d(1, 10, kernel_size=7, stride=1, padding=3)
        self.ConvC1 = ConvBlock(10, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC1 = nn.Conv1d(10, 20, kernel_size=1, stride=2, padding=0)
        self.ConvC2 = ConvBlock(20, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC2 = nn.Conv1d(20, 40, kernel_size=1, stride=2, padding=0)
        self.ConvC3 = ConvBlock(40, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC3 = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.ConvC4 = ConvBlock(80, 1, residual=True, kernel_size=7, padding=3)
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        
        # ========== 多通道编码器 ==========
        self.nchn_m = 80
        self.ConvEmbeddingM = nn.Conv1d(num_channels, 40, kernel_size=7, stride=1, padding=3)
        self.Conv1 = ConvBlock(40, 2, residual=True, kernel_size=7, padding=3)
        self.ProjM1 = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.Conv2 = ConvBlock(80, 2, residual=True, kernel_size=7, padding=3)
        self.ProjM2 = nn.Conv1d(80, 80, kernel_size=1, stride=2, padding=0)
        self.Conv3 = ConvBlock(80, 2, residual=True, kernel_size=7, padding=3)
        
        # ========== Transformer ==========
        self.channel_transformer = nn.Transformer(
            80, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=128, batch_first=True,
            dropout=transformer_dropout
        )
        
        # ========== GRU ==========
        self.nhidden_c = 40
        self.channel_gru = nn.GRU(
            input_size=80, hidden_size=self.nhidden_c,
            batch_first=True, bidirectional=True, num_layers=2,
            dropout=gru_dropout
        )
        
        self.nhidden_sz = 40
        self.multi_gru = nn.GRU(
            input_size=80, hidden_size=self.nhidden_sz,
            batch_first=True, bidirectional=True, num_layers=1,
            dropout=gru_dropout
        )
        
        # ========== 输出层 ==========
        self.channel_linear = nn.Linear(2 * self.nhidden_c, 2)
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)
        self.onset_linear = nn.Linear(2 * self.nhidden_sz, 1)
        self.sig = nn.Sigmoid()
    
    def _channel_encoder(self, x):
        """
        单通道编码器
        
        Args:
            x: 输入数据 [B, T, C, L]
        
        Returns:
            通道编码 [B, T, C, 80]
        """
        B, T, C, L = x.size()
        h = self.ConvEmbeddingC(x.view(B * T * C, 1, L))
        h = self.ConvC1(h)
        h = self.ProjC1(h)
        h = self.ConvC2(h)
        h = self.ProjC2(h)
        h = self.ConvC3(h)
        h = self.ProjC3(h)
        h = self.ConvC4(h)
        h = torch.mean(h.view(B, T, C, self.nchn_c, -1), dim=4)
        return h
    
    def _multichannel_encoder(self, x):
        """
        多通道编码器
        
        Args:
            x: 输入数据 [B, T, C, L]
        
        Returns:
            多通道编码 [B, T, 80]
        """
        B, T, C, L = x.size()
        h = x.view(B * T, C, L)
        h = self.ConvEmbeddingM(h)
        h = self.Conv1(h)
        h = self.ProjM1(h)
        h = self.Conv2(h)
        h = self.ProjM2(h)
        h = self.Conv3(h)
        h = torch.mean(h.view(B, T, self.nchn_m, -1), dim=3)
        return h
    
    def _attn_onset_map(self, h, a):
        """
        基于注意力的onset map
        
        Args:
            h: 通道预测 [B, T, C, 2]
            a: 时间注意力 [B, T, 1]
        
        Returns:
            onset_map [B, C]
        """
        B, T, Channels, classes = h.shape
        probs = F.softmax(h, dim=3)
        onset_map = torch.sum(
            a.view((B, T, 1)) * probs[:, :, :, 1], dim=1
        )
        return onset_map
    
    def _channel_onset_map(self, h_c):
        """
        基于通道的onset map
        
        Args:
            h_c: 通道预测 [B, T, C, 2]
        
        Returns:
            onset_map [B, C]
        """
        channel_probs = F.softmax(h_c, dim=3)
        max_channel_probs, _ = torch.max(channel_probs[:, :, :, 1], dim=2)
        attn = F.relu(max_channel_probs[:, 1:] - max_channel_probs[:, :-1])
        onset_map = torch.sum(
            attn.unsqueeze(2) * channel_probs[:, 1:, :, 1], dim=1
        )
        return onset_map
    
    def _max_channel_logits(self, h):
        """获取最大通道的logits"""
        B, T, _, _ = h.shape
        probs = F.softmax(h, dim=3)
        dev = h.get_device()
        if dev == -1:
            dev = None
        
        max_logits = torch.zeros((B, T, 2), device=h.device)
        for bb in range(B):
            max_channels = torch.argmax(probs[bb, :, :, 1], dim=1)
            for tt in range(T):
                max_logits[bb, tt, :] = h[bb, tt, max_channels[tt], :]
        
        return max_logits
    
    def forward_pass(self, x):
        """前向传播核心"""
        B, Nsz, T, C, L = x.size()
        x = x.reshape(B * Nsz, T, C, L)
        
        # CNN编码
        h_c = self.cnn_dropout(self._channel_encoder(x))  # [B, T, C, 80]
        h_m = self.cnn_dropout(self._multichannel_encoder(x))  # [B, T, 80]
        
        # Transformer
        h_c = h_c.reshape(B * Nsz * T, C, 80)
        h_c = self.channel_transformer(
            torch.cat((h_c, h_m.reshape(B * Nsz * T, 1, 80)), dim=1), h_c
        )
        h_c = h_c.view(B * Nsz, T, C, 80)
        
        # Channel GRU
        h_c = h_c.transpose(1, 2)  # [B, C, T, 80]
        h_c = h_c.reshape(B * Nsz * C, T, 80)
        self.channel_gru.flatten_parameters()
        h_c, _ = self.channel_gru(h_c)
        h_c = h_c.view(B * Nsz, C, T, 2 * self.nhidden_c)
        h_c = h_c.transpose(1, 2)
        
        # Multi GRU
        self.multi_gru.flatten_parameters()
        h_m, _ = self.multi_gru(h_m)
        
        # 输出层
        h_c = self.channel_linear(h_c)
        a = self.onset_linear(h_m)
        a = torch.softmax(a, dim=1)
        h_m = self.multi_linear(h_m)
        
        return h_c, h_m, a
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 [B, Nsz, T, C, L]
        
        Returns:
            chn_sz_logits: 通道癫痫分类logits
            h_m: 多通道序列输出
            attn_onset_map: 注意力onset map
            chn_onset_map: 通道onset map
        """
        B, Nsz, T, C, L = x.shape
        h_c, h_m, a = self.forward_pass(x)
        
        channel_sz_logits = self._max_channel_logits(h_c)
        attn_onset_map = self._attn_onset_map(h_c, a)
        chn_onset_map = self._channel_onset_map(h_c)
        
        return channel_sz_logits, h_m, attn_onset_map, chn_onset_map
    
    def predict_proba(self, x):
        """预测概率（用于推理）"""
        h_c, h_m, attn = self.forward_pass(x.unsqueeze(0))
        sz_logits = self._max_channel_logits(h_c)
        chn_sz_pred = torch.softmax(sz_logits, dim=2)
        sz_pred = torch.softmax(h_m, dim=2)
        chn_pred = torch.softmax(h_c, dim=3)
        attn_onset_map = self._attn_onset_map(h_c, attn)
        chn_onset_map = self._channel_onset_map(h_c)
        
        return (sz_pred.squeeze(0), chn_sz_pred.squeeze(0), chn_pred.squeeze(0),
                attn_onset_map.squeeze(0), chn_onset_map.squeeze(0), attn)


class ChannelClassifier(nn.Module):
    """
    通道级别分类器
    预测每个通道是否为SOZ
    """
    
    def __init__(self, num_channels=19, input_length=60000, 
                 hidden_dim=128, dropout=0.3):
        """
        Args:
            num_channels: 通道数
            input_length: 输入序列长度
            hidden_dim: 隐藏层维度
            dropout: dropout率
        """
        super().__init__()
        
        self.num_channels = num_channels
        
        # 1D CNN特征提取
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, C, L]
        
        Returns:
            logits: [B, num_channels]
        """
        # CNN
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        
        # 池化
        h = self.pool(h).squeeze(-1)
        
        # 全连接
        h = self.dropout(self.relu(self.fc1(h)))
        logits = self.fc2(h)
        
        return logits


class RegionClassifier(nn.Module):
    """
    脑区级别分类器
    预测发作起始脑区
    """
    
    def __init__(self, num_channels=19, num_regions=5, 
                 hidden_dim=128, dropout=0.3):
        """
        Args:
            num_channels: 输入通道数
            num_regions: 脑区数量
            hidden_dim: 隐藏层维度
            dropout: dropout率
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.num_regions = num_regions
        
        # CNN特征提取
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_regions)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, C, L]
        
        Returns:
            logits: [B, num_regions]
        """
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        
        h = self.pool(h).squeeze(-1)
        h = self.dropout(self.relu(self.fc1(h)))
        logits = self.fc2(h)
        
        return logits


class HybridSOZModel(nn.Module):
    """
    混合SOZ定位模型
    同时输出通道级别和脑区级别预测
    """
    
    def __init__(self, num_channels=19, num_regions=5, 
                 hidden_dim=256, dropout=0.3):
        """
        Args:
            num_channels: 输入通道数
            num_regions: 脑区数量
            hidden_dim: 隐藏层维度
            dropout: dropout率
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.num_regions = num_regions
        
        # 共享CNN特征提取器
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # 通道分类头
        self.channel_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_channels),
        )
        
        # 脑区分类头
        self.region_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_regions),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, C, L]
        
        Returns:
            channel_logits: [B, num_channels]
            region_logits: [B, num_regions]
        """
        features = self.encoder(x).squeeze(-1)
        channel_logits = self.channel_head(features)
        region_logits = self.region_head(features)
        
        return channel_logits, region_logits


class AttentionBlock(nn.Module):
    """注意力模块"""
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attended))


class TransformerSOZModel(nn.Module):
    """
    基于Transformer的SOZ定位模型
    """
    
    def __init__(self, num_channels=19, num_regions=5,
                 d_model=128, nhead=4, num_layers=2, dropout=0.3):
        """
        Args:
            num_channels: 输入通道数
            num_regions: 脑区数量
            d_model: Transformer维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dropout: dropout率
        """
        super().__init__()
        
        self.num_channels = num_channels
        
        # 通道嵌入
        self.channel_embedding = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_channels, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 时间聚合
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.channel_fc = nn.Linear(d_model, 1)
        self.region_fc = nn.Linear(d_model * num_channels, num_regions)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: 输入 [B, C, L]
        
        Returns:
            channel_logits: [B, C]
            region_logits: [B, num_regions]
        """
        B, C, L = x.shape
        
        # 时间聚合
        x = self.time_pool(x)  # [B, C, 1]
        
        # 通道嵌入
        x = self.channel_embedding(x)  # [B, C, d_model]
        
        # 添加位置编码
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)  # [B, C, d_model]
        
        # 通道预测
        channel_logits = self.channel_fc(x).squeeze(-1)  # [B, C]
        
        # 脑区预测
        region_logits = self.region_fc(x.reshape(B, -1))  # [B, num_regions]
        
        return channel_logits, region_logits


def create_model(model_type='channel', num_channels=19, num_regions=5, **kwargs):
    """
    创建模型
    
    Args:
        model_type: 模型类型 ('channel', 'region', 'hybrid', 'deepsoz', 'transformer')
        num_channels: 通道数
        num_regions: 脑区数
        **kwargs: 其他参数
    
    Returns:
        模型实例
    """
    if model_type == 'channel':
        return ChannelClassifier(num_channels=num_channels, **kwargs)
    elif model_type == 'region':
        return RegionClassifier(num_channels=num_channels, num_regions=num_regions, **kwargs)
    elif model_type == 'hybrid':
        return HybridSOZModel(num_channels=num_channels, num_regions=num_regions, **kwargs)
    elif model_type == 'deepsoz':
        return DeepSOZModel(num_channels=num_channels, **kwargs)
    elif model_type == 'transformer':
        return TransformerSOZModel(num_channels=num_channels, num_regions=num_regions, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
