#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM-TimeFilter-SOZ: 基于预训练LaBraM + TimeFilter图过滤的EEG癫痫起始区检测模型

Architecture:
    Input: X [B, 22, 20, 100]  (22 TCP导联, 20 patches, 100 samples/patch)

    1. Patch Embedding
       - Flatten [B, 440, 100]  → Linear → [B, 440, D=128]
       - 2D Position Encoding (channel_idx + patch_idx)

    2. LaBraM Backbone (optional pre-trained)
       - 12 frozen Transformer layers  (底层, 通用EEG表征)
       - K  trainable Transformer layers(顶层, 任务适配)

    3. TimeFilter Core
       a) Multi-head projection distance (H=4) → k-NN graph (α=0.15)
       b) Three domain-specific filters:
          - Filter_Temporal:     保留同导联相邻补丁边 (时序连续性)
          - Filter_Spatial:      保留同时间解剖邻近导联边 (10-20球面距离 < 5cm)
          - Filter_Pathological: 学习HFO/病理模式相关边 (可选gamma能量先验)
       c) Noisy gated routing + Top-p=0.85 动态分配

    4. Graph Convolution: 2-layer GAT (heads=4)

    5. SOZ Localization Head
       - Temporal attention pooling  (学习发作起始补丁权重)
       - Channel max-pooling          (22导联聚合)
       - BipolarToMonopolarMapper     (22 TCP → 19 monopolar)
       - Sigmoid → SOZ probability

    Output: monopolar_probs [B, 19]

Loss:
    - Primary: Focal Loss (γ=2.0)
    - Auxiliary: Domain adversarial loss (GRL + discriminator)

Reference:
    - TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration (NeurIPS 2024)
    - LaBraM: Large Brain Model for EEG (ICLR 2024)
    - DeepSOZ: Deep Learning for SOZ Localization (Abou Jaoude et al., 2020)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 常量  (与 data_preprocess/config.py 一致)
# =============================================================================

# 标准19通道 (10-20)
STANDARD_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]
STD19_IDX = {ch: i for i, ch in enumerate(STANDARD_19)}

# TCP 22通道双极导联 (顺序与 eeg_pipeline.py / config.py 对齐)
TCP_PAIRS: List[Tuple[str, str]] = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),       # 左颞链   0-3
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),       # 右颞链   4-7
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),       # 左副矢状 8-11
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),       # 右副矢状 12-15
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),       # 中央链   16-21
    ('C4', 'T4'),  ('T4', 'A2'),
]
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]
N_TCP = 22
N_STD = 19

# 10-20 电极球面坐标 (单位球, 用于空间距离计算)
# 近似 (x, y, z): x=left/right, y=anterior/posterior, z=superior/inferior
_ELECTRODE_3D: Dict[str, Tuple[float, float, float]] = {
    'FP1': (-0.31, 0.95, 0.00), 'FP2': (0.31, 0.95, 0.00),
    'F7':  (-0.81, 0.59, 0.00), 'F3':  (-0.55, 0.67, 0.50),
    'FZ':  (0.00, 0.71, 0.71),  'F4':  (0.55, 0.67, 0.50),
    'F8':  (0.81, 0.59, 0.00),
    'T3':  (-1.00, 0.00, 0.00), 'C3':  (-0.57, 0.00, 0.82),
    'CZ':  (0.00, 0.00, 1.00),  'C4':  (0.57, 0.00, 0.82),
    'T4':  (1.00, 0.00, 0.00),
    'T5':  (-0.81, -0.59, 0.00), 'P3': (-0.55, -0.67, 0.50),
    'PZ':  (0.00, -0.71, 0.71),  'P4': (0.55, -0.67, 0.50),
    'T6':  (0.81, -0.59, 0.00),
    'O1':  (-0.31, -0.95, 0.00), 'O2': (0.31, -0.95, 0.00),
    'A1':  (-1.05, 0.00, -0.30), 'A2': (1.05, 0.00, -0.30),
}


# =============================================================================
# 配置
# =============================================================================

@dataclass
class ModelConfig:
    """LaBraM-TimeFilter-SOZ 模型配置"""

    # ---- 输入 ----
    n_channels: int = 22          # TCP导联数
    n_patches: int = 20           # 每导联补丁数
    patch_len: int = 100          # 每补丁采样点数
    n_nodes: int = 440            # n_channels * n_patches

    # ---- Patch Embedding ----
    embed_dim: int = 128          # D

    # ---- LaBraM Backbone ----
    labram_checkpoint: str = ''   # 预训练权重路径 (空=随机初始化)
    n_transformer_layers: int = 14  # 总Transformer层数
    n_frozen_layers: int = 12     # 冻结底层数
    n_heads_transformer: int = 8  # Transformer注意力头数
    ff_mult: float = 4.0          # FFN扩展倍数
    transformer_dropout: float = 0.1

    # ---- TimeFilter ----
    tf_n_heads: int = 4           # 多头投影距离的头数 H
    tf_alpha: float = 0.15        # k-NN保留比例 α
    tf_n_filters: int = 3         # 过滤器数目
    spatial_dist_thresh: float = 0.55  # 球面距离阈值 (≈5cm)
    top_p: float = 0.85           # Top-p 动态路由
    noisy_gating: bool = True     # 含噪门控

    # ---- GAT ----
    gat_layers: int = 2
    gat_heads: int = 4
    gat_dropout: float = 0.1

    # ---- Localization Head ----
    head_hidden: int = 64
    head_dropout: float = 0.3
    n_output: int = 22            # 输出通道数 (22=双极直接输出, 19=单极映射)
    output_mode: str = 'bipolar'  # 'bipolar' (22ch) or 'monopolar' (19ch)

    # ---- Domain Adversarial ----
    use_domain_adversarial: bool = True
    domain_hidden: int = 64
    grl_lambda: float = 0.1       # 梯度反转强度

    # ---- Loss ----
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    domain_loss_weight: float = 0.1

    # ---- TCP pairs (允许外部覆盖) ----
    tcp_pairs: List[Tuple[str, str]] = field(default_factory=lambda: list(TCP_PAIRS))


# =============================================================================
# 1. Patch Embedding + 2D Position Encoding
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    [B, C, P, L] → [B, N, D]

    C=22 channels, P=20 patches, L=100 samples → N=440 nodes, D=128
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_len, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # 可学习的2D位置编码: channel_embed + patch_embed
        self.channel_embed = nn.Embedding(cfg.n_channels, cfg.embed_dim)
        self.patch_embed = nn.Embedding(cfg.n_patches, cfg.embed_dim)

        # 预计算展开后的 (channel_idx, patch_idx)
        ch_ids = torch.arange(cfg.n_channels).unsqueeze(1).expand(-1, cfg.n_patches).reshape(-1)
        pa_ids = torch.arange(cfg.n_patches).unsqueeze(0).expand(cfg.n_channels, -1).reshape(-1)
        self.register_buffer('ch_ids', ch_ids)   # (440,)
        self.register_buffer('pa_ids', pa_ids)    # (440,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 22, 20, 100]
        Returns:
            [B, 440, D]
        """
        B, C, P, L = x.shape
        x = x.reshape(B, C * P, L)           # [B, 440, 100]
        x = self.proj(x)                      # [B, 440, D]
        x = self.norm(x)

        # 加上2D位置编码
        pos = self.channel_embed(self.ch_ids) + self.patch_embed(self.pa_ids)  # [440, D]
        x = x + pos.unsqueeze(0)              # [B, 440, D]
        return x


# =============================================================================
# 2. LaBraM Backbone (Transformer Encoder)
# =============================================================================

class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer Block"""

    def __init__(self, dim: int, n_heads: int, ff_mult: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_mult), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class LaBraMBackbone(nn.Module):
    """
    LaBraM-style Transformer Backbone

    底层 n_frozen 层冻结参数 (通用EEG表征),
    顶层 (n_total - n_frozen) 层可训练 (任务适配).

    若提供 checkpoint_path 则加载预训练权重。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim,
                n_heads=cfg.n_heads_transformer,
                ff_mult=cfg.ff_mult,
                dropout=cfg.transformer_dropout,
            )
            for _ in range(cfg.n_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

        # 加载预训练权重 (如果有)
        if cfg.labram_checkpoint:
            self._load_checkpoint(cfg.labram_checkpoint)

        # 冻结底层
        self._freeze_bottom(cfg.n_frozen_layers)

    def _load_checkpoint(self, path: str):
        """加载LaBraM预训练权重 (兼容性映射)"""
        try:
            state = torch.load(path, map_location='cpu')
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']

            # 尝试加载 (忽略不匹配的键)
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info(
                f"LaBraM checkpoint loaded: {path}\n"
                f"  missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as e:
            logger.warning(f"无法加载LaBraM checkpoint: {e}, 使用随机初始化")

    def _freeze_bottom(self, n_freeze: int):
        """冻结底部 n_freeze 层"""
        for i, layer in enumerate(self.layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LaBraM: {len(self.layers)} layers, "
            f"frozen={n_freeze}, trainable params={n_trainable}/{n_total}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B, N, D]"""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# =============================================================================
# 3. TimeFilter Core
# =============================================================================

# ---- 3a. Multi-Head Projection Distance + k-NN ----

class MultiHeadDistanceKNN(nn.Module):
    """
    多头投影距离计算 → k-NN 图构建

    对每个头: 将节点投影到低维空间, 计算欧氏距离, 保留 top-α 近邻。
    最终: H 个头的邻接矩阵取并集/平均。
    """

    def __init__(self, dim: int, n_heads: int = 4, alpha: float = 0.15):
        super().__init__()
        self.n_heads = n_heads
        self.alpha = alpha
        head_dim = dim // n_heads
        self.projections = nn.ModuleList([
            nn.Linear(dim, head_dim, bias=False) for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            adj: [B, N, N]  soft adjacency (0~1, higher=closer)
        """
        B, N, _ = x.shape
        k = max(1, int(N * self.alpha))  # 保留的近邻数
        adjs = []

        for proj in self.projections:
            z = proj(x)                                      # [B, N, d_h]
            # 成对欧氏距离: dist[i,j] = ||z_i - z_j||^2
            dist = torch.cdist(z, z, p=2)                   # [B, N, N]
            # k-NN: 每行保留最近的k个, 其余置为inf
            _, topk_idx = dist.topk(k, dim=-1, largest=False)  # [B, N, k]
            mask = torch.zeros_like(dist)
            mask.scatter_(-1, topk_idx, 1.0)
            # 对称化
            mask = torch.maximum(mask, mask.transpose(-1, -2))
            # 距离转相似度 (高斯核)
            sim = torch.exp(-dist ** 2 / (2 * dist.mean() ** 2 + 1e-8))
            adjs.append(sim * mask)

        # 多头平均
        adj = torch.stack(adjs, dim=0).mean(dim=0)          # [B, N, N]
        return adj


# ---- 3b. Domain-Specific Filters ----

class TemporalFilter(nn.Module):
    """
    Filter_Temporal: 保留同一导联内相邻补丁的边 (时序连续性)

    对于节点 (ch_i, patch_j), 仅保留连向 (ch_i, patch_{j±1}) 的边。
    """

    def __init__(self, n_channels: int = 22, n_patches: int = 20):
        super().__init__()
        N = n_channels * n_patches
        # 预计算时序邻接模板
        mask = torch.zeros(N, N)
        for ch in range(n_channels):
            for p in range(n_patches):
                node = ch * n_patches + p
                if p > 0:
                    mask[node, ch * n_patches + (p - 1)] = 1.0
                if p < n_patches - 1:
                    mask[node, ch * n_patches + (p + 1)] = 1.0
        self.register_buffer('mask', mask)  # [N, N]

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """[B, N, N] → [B, N, N]  (仅保留时序边)"""
        return adj * self.mask.unsqueeze(0)


class SpatialFilter(nn.Module):
    """
    Filter_Spatial: 保留同一时间点解剖邻近导联的边

    基于10-20系统球面距离 < threshold,
    仅保留 (ch_i, patch_t) ↔ (ch_j, patch_t) 的边。
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_patches: int = 20,
        tcp_pairs: List[Tuple[str, str]] = None,
        dist_thresh: float = 0.55,
    ):
        super().__init__()
        pairs = tcp_pairs or list(TCP_PAIRS)
        N = n_channels * n_patches

        # 计算双极导联中点位置
        ch_pos = []
        for a, b in pairs:
            pa = np.array(_ELECTRODE_3D.get(a, (0, 0, 0)))
            pb = np.array(_ELECTRODE_3D.get(b, (0, 0, 0)))
            ch_pos.append((pa + pb) / 2.0)
        ch_pos = np.array(ch_pos)  # [22, 3]

        # 通道间距离矩阵
        from scipy.spatial.distance import cdist
        ch_dist = cdist(ch_pos, ch_pos)  # [22, 22]
        ch_adj = (ch_dist < dist_thresh).astype(np.float32)
        np.fill_diagonal(ch_adj, 0.0)  # 不含自环

        # 扩展到 [N, N]: 只有同一时间点才有空间边
        mask = np.zeros((N, N), dtype=np.float32)
        for t in range(n_patches):
            for ci in range(n_channels):
                for cj in range(n_channels):
                    if ch_adj[ci, cj] > 0:
                        ni = ci * n_patches + t
                        nj = cj * n_patches + t
                        mask[ni, nj] = 1.0
        self.register_buffer('mask', torch.from_numpy(mask))

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """[B, N, N] → [B, N, N]  (仅保留空间边)"""
        return adj * self.mask.unsqueeze(0)


class PathologicalFilter(nn.Module):
    """
    Filter_Pathological: 学习病理模式(HFO/高频振荡)相关边

    设计:
    - 对每个节点学习一个"病理敏感度"分数
    - 两个高敏感度节点之间的边被保留
    - 可接受可选的 gamma 频段能量先验
    - 实质是可学习的注意力掩码

    若有gamma能量先验 (预提取80-250Hz小波包分解):
        score = MLP(node_feat ⊕ gamma_energy)
    否则:
        score = MLP(node_feat)
    """

    def __init__(self, dim: int, n_nodes: int = 440, use_gamma_prior: bool = False):
        super().__init__()
        self.use_gamma_prior = use_gamma_prior
        in_dim = dim + (1 if use_gamma_prior else 0)

        self.score_net = nn.Sequential(
            nn.Linear(in_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        adj: torch.Tensor,
        node_feat: torch.Tensor,
        gamma_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            adj: [B, N, N]
            node_feat: [B, N, D]
            gamma_energy: [B, N, 1] optional
        Returns:
            [B, N, N]
        """
        if self.use_gamma_prior and gamma_energy is not None:
            inp = torch.cat([node_feat, gamma_energy], dim=-1)  # [B, N, D+1]
        else:
            inp = node_feat

        scores = self.score_net(inp).squeeze(-1)  # [B, N]
        # 两节点的病理分数乘积 → 边权重
        edge_weight = scores.unsqueeze(-1) * scores.unsqueeze(-2)  # [B, N, N]
        # 软阈值门控
        gate = torch.sigmoid((edge_weight - self.threshold) * 10.0)
        return adj * gate


# ---- 3c. Noisy Gated Router (Top-p) ----

class NoisyGatedRouter(nn.Module):
    """
    含噪门控路由: 动态分配每个节点通过哪些过滤器

    - 对每个节点计算 logits → softmax → 过滤器权重
    - Top-p 选择: 累积概率达到 p 后截断 (允许不同节点使用不同数量的过滤器)
    - 训练时加噪 (探索); 推理时不加噪
    """

    def __init__(self, dim: int, n_filters: int = 3, top_p: float = 0.85,
                 noisy: bool = True):
        super().__init__()
        self.n_filters = n_filters
        self.top_p = top_p
        self.noisy = noisy

        self.gate = nn.Linear(dim, n_filters)
        # 噪声参数
        self.w_noise = nn.Linear(dim, n_filters) if noisy else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]  节点特征
        Returns:
            weights: [B, N, F]  每个节点分配给F个过滤器的权重 (经Top-p截断)
        """
        logits = self.gate(x)  # [B, N, F]

        # 训练时加噪
        if self.training and self.noisy and self.w_noise is not None:
            noise_std = F.softplus(self.w_noise(x))  # [B, N, F]
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)  # [B, N, F]

        # Top-p 截断
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # 创建掩码: 累积概率超过 top_p 的位置置零 (保留第一个超过阈值的)
        mask = cumsum - sorted_probs < self.top_p   # [B, N, F]
        sorted_probs = sorted_probs * mask.float()

        # 恢复原始顺序
        weights = torch.zeros_like(probs)
        weights.scatter_(-1, sorted_idx, sorted_probs)

        # 重新归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights


# ---- 3d. TimeFilter Block ----

class TimeFilterBlock(nn.Module):
    """
    TimeFilter 核心: 多头距离 → k-NN → 三过滤器 → 门控路由 → 加权邻接

    输出: 经过过滤的图邻接矩阵
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # 多头距离 + k-NN
        self.distance_knn = MultiHeadDistanceKNN(
            dim=cfg.embed_dim, n_heads=cfg.tf_n_heads, alpha=cfg.tf_alpha,
        )

        # 三个过滤器
        self.filter_temporal = TemporalFilter(cfg.n_channels, cfg.n_patches)
        self.filter_spatial = SpatialFilter(
            cfg.n_channels, cfg.n_patches, cfg.tcp_pairs, cfg.spatial_dist_thresh,
        )
        self.filter_pathological = PathologicalFilter(
            cfg.embed_dim, cfg.n_nodes, use_gamma_prior=False,
        )
        self.filters = [self.filter_temporal, self.filter_spatial, self.filter_pathological]

        # 门控路由
        self.router = NoisyGatedRouter(
            cfg.embed_dim, cfg.tf_n_filters, cfg.top_p, cfg.noisy_gating,
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]  节点特征
            gamma_energy: [B, N, 1]  可选gamma能量先验
        Returns:
            filtered_adj: [B, N, N]  过滤后的邻接矩阵
        """
        # 1. 构建k-NN图
        adj = self.distance_knn(x)  # [B, N, N]

        # 2. 三个过滤器分别过滤
        adj_temporal = self.filter_temporal(adj)
        adj_spatial = self.filter_spatial(adj)
        adj_pathological = self.filter_pathological(adj, x, gamma_energy)
        filtered_adjs = torch.stack(
            [adj_temporal, adj_spatial, adj_pathological], dim=-1
        )  # [B, N, N, F]

        # 3. 门控路由 → 加权融合
        weights = self.router(x)  # [B, N, F]
        # 对称化权重: 边(i,j)的过滤器权重 = mean(w_i, w_j)
        w_i = weights.unsqueeze(2)  # [B, N, 1, F]
        w_j = weights.unsqueeze(1)  # [B, 1, N, F]
        edge_weights = (w_i + w_j) / 2.0  # [B, N, N, F]

        # 加权求和
        filtered_adj = (filtered_adjs * edge_weights).sum(dim=-1)  # [B, N, N]
        return filtered_adj


# =============================================================================
# 4. Graph Attention Network (GAT)
# =============================================================================

class GATLayer(nn.Module):
    """
    Graph Attention Layer (masked by adjacency)

    attention(i,j) = LeakyReLU(a^T [Wh_i || Wh_j]) * adj(i,j)
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        head_dim = out_dim // n_heads if concat else out_dim

        self.W = nn.Linear(in_dim, head_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim if concat else out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in]
            adj: [B, N, N]
        Returns:
            [B, N, D_out]
        """
        B, N, _ = x.shape
        H = self.n_heads
        head_dim = self.W.out_features // H

        Wh = self.W(x).view(B, N, H, head_dim)  # [B, N, H, d_h]

        # 注意力分数
        e_src = (Wh * self.a_src).sum(dim=-1)    # [B, N, H]
        e_dst = (Wh * self.a_dst).sum(dim=-1)    # [B, N, H]
        attn = self.leaky_relu(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # [B, N, N, H]
        )

        # 掩码: 非邻居置为 -inf
        mask = (adj > 0).unsqueeze(-1).expand_as(attn)  # [B, N, N, H]
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=2)            # softmax over j (neighbors)
        attn = torch.nan_to_num(attn, 0.0)       # 处理全 -inf 行
        attn = self.dropout(attn)

        # 聚合
        # attn: [B, N, N, H], Wh: [B, N, H, d_h]
        out = torch.einsum('bnjh,bjhd->bnhd', attn, Wh)  # [B, N, H, d_h]

        if self.concat:
            out = out.reshape(B, N, H * head_dim)
        else:
            out = out.mean(dim=2)

        return self.norm(out)


class GATBlock(nn.Module):
    """多层 GAT + 残差"""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            concat = (i < n_layers - 1)  # 最后一层不concat
            self.layers.append(GATLayer(dim, dim, n_heads, dropout, concat=concat))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x
            x = layer(x, adj)
            x = F.elu(x)
            x = self.dropout(x) + residual  # 残差
        return x


# =============================================================================
# 5. SOZ Localization Head
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化: 学习每个补丁时间点的重要性权重

    对每个导联的 P=20 个补丁, 学习注意力权重 → 加权聚合。
    关键: 允许模型自动关注发作起始时刻附近的补丁。
    """

    def __init__(self, dim: int, n_patches: int = 20):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, P, D]  (C=22 channels, P=20 patches)
        Returns:
            pooled: [B, C, D]
            attn_weights: [B, C, P]
        """
        scores = self.attn(x).squeeze(-1)    # [B, C, P]
        weights = F.softmax(scores, dim=-1)  # [B, C, P]
        pooled = (x * weights.unsqueeze(-1)).sum(dim=2)  # [B, C, D]
        return pooled, weights



# 导入极性感知的 BipolarToMonopolarMapper (详见 bipolar_to_monopolar.py)
try:
    from .bipolar_to_monopolar import BipolarToMonopolarMapper
except ImportError:
    from bipolar_to_monopolar import BipolarToMonopolarMapper


class SOZLocalizationHead(nn.Module):
    """
    SOZ定位头: 从440节点特征 → SOZ概率

    支持两种输出模式:
    - bipolar  (n_output=22): 直接输出22通道双极导联SOZ logits
    - monopolar (n_output=19): 22双极 → BipolarToMonopolar映射 → 19单极

    流程:
    1. Reshape: [B, 440, D] → [B, 22, 20, D]
    2. Temporal attention pooling: [B, 22, 20, D] → [B, 22, D]
    3. Channel max-pooling (辅助): [B, 22, 20, D] → [B, 22, D]
    4. 融合: concat → FC → [B, 22]
    5. (仅monopolar模式) BipolarToMonopolar: [B, 22] → [B, 19]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.embed_dim
        self.n_channels = cfg.n_channels
        self.n_patches = cfg.n_patches
        self.output_mode = cfg.output_mode  # 'bipolar' or 'monopolar'
        self.n_output = cfg.n_output

        self.temporal_attn = TemporalAttentionPooling(D, cfg.n_patches)

        # 融合两种池化
        self.fusion = nn.Sequential(
            nn.Linear(D * 2, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        # 仅在 monopolar 模式下使用 BipolarToMonopolar 映射
        self.bipolar_to_mono = None
        if self.output_mode == 'monopolar':
            self.bipolar_to_mono = BipolarToMonopolarMapper(
                monopolar_channels=list(STANDARD_19),
                bipolar_pairs=cfg.tcp_pairs,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, N=440, D]
        Returns:
            soz_logits: [B, n_output]  (22 bipolar or 19 monopolar)
            aux: dict with bipolar_logits, attention weights
        """
        B = x.size(0)
        D = x.size(-1)

        # Reshape to channel x patch
        x_4d = x.view(B, self.n_channels, self.n_patches, D)  # [B, 22, 20, D]

        # 时序注意力池化
        attn_pooled, attn_weights = self.temporal_attn(x_4d)   # [B, 22, D], [B, 22, 20]

        # 通道最大池化
        max_pooled = x_4d.max(dim=2).values                     # [B, 22, D]

        # 融合 → 22通道双极logits
        fused = torch.cat([attn_pooled, max_pooled], dim=-1)    # [B, 22, 2D]
        bipolar_logits = self.fusion(fused).squeeze(-1)          # [B, 22]

        aux = {
            'bipolar_logits': bipolar_logits,
            'temporal_attn_weights': attn_weights,
        }

        if self.output_mode == 'bipolar':
            # 直接输出22通道双极logits
            return bipolar_logits, aux
        else:
            # 映射到19通道单极logits
            monopolar_logits = self.bipolar_to_mono.forward_logits(bipolar_logits)  # [B, 19]
            return monopolar_logits, aux


# =============================================================================
# 6. Domain Adversarial (GRL + Discriminator)
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层: 前向不变, 反向乘以 -λ"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """
    域判别器: 判断样本来自公共(TUSZ)还是私有数据

    输入: 全局池化后的特征 [B, D]
    输出: domain logit [B, 1]  (0=public, 1=private)
    """

    def __init__(self, dim: int, hidden: int = 64, grl_lambda: float = 0.1):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]  全局特征
        Returns:
            [B, 1]  域预测 logit
        """
        x = self.grl(x)
        return self.net(x)


# =============================================================================
# 7. Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t (1-p_t)^γ log(p_t)

    适用于多标签分类 (每个通道独立二分类)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  [B, C]  logits
            targets: [B, C]  binary labels (0/1)
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SOZDetectionLoss(nn.Module):
    """
    SOZ检测组合损失

    L_total = L_focal(SOZ定位) + λ_domain * L_bce(域判别)

    Args:
        focal_gamma: Focal Loss γ参数
        focal_alpha: Focal Loss α参数
        domain_weight: 域对抗损失权重
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        domain_weight: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.domain_bce = nn.BCEWithLogitsLoss()
        self.domain_weight = domain_weight

    def forward(
        self,
        soz_logits: torch.Tensor,
        soz_targets: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss, {'focal': ..., 'domain': ..., 'total': ...}
        """
        loss_focal = self.focal(soz_logits, soz_targets)
        losses = {'focal': loss_focal}

        total = loss_focal

        if domain_logits is not None and domain_targets is not None:
            loss_domain = self.domain_bce(domain_logits, domain_targets)
            total = total + self.domain_weight * loss_domain
            losses['domain'] = loss_domain

        losses['total'] = total
        return total, losses


# =============================================================================
# 8. Main Model: LaBraM_TimeFilter_SOZ
# =============================================================================

class LaBraM_TimeFilter_SOZ(nn.Module):
    """
    LaBraM-TimeFilter-SOZ: 预训练大模型 + 图过滤的EEG癫痫起始区检测

    支持两种输出模式:
    - bipolar  (n_output=22): 直接输出22通道TCP双极导联SOZ概率
    - monopolar (n_output=19): 通过BipolarToMonopolar映射输出19通道单极SOZ概率

    Forward:
        Input:  X [B, 22, 20, 100]   TCP双极导联 × 补丁 × 采样点
                domain_labels [B, 1]  可选, 0=public 1=private
                gamma_energy [B, 440, 1]  可选, 预提取γ频段能量

        Output: {
            'soz_probs':      [B, n_output]  SOZ概率 (sigmoid)
            'soz_logits':     [B, n_output]  SOZ logits
            'bipolar_logits': [B, 22]        22 TCP通道 logits (中间层)
            'domain_logits':  [B, 1]         域判别 logits (如果启用)
            'temporal_attn':  [B, 22, 20]    时序注意力权重
            'filtered_adj':   [B, 440, 440]  过滤后邻接矩阵
        }

    Usage:
        # 22通道双极模式（默认，配合 combined_manifest.csv 的双极标签）
        cfg = ModelConfig(n_output=22, output_mode='bipolar')
        model = LaBraM_TimeFilter_SOZ(cfg)
        out = model(torch.randn(4, 22, 20, 100))
        probs = out['soz_probs']   # [4, 22]

        # 19通道单极模式（配合 eeg_pipeline.py 的单极标签）
        cfg = ModelConfig(n_output=19, output_mode='monopolar')
        model = LaBraM_TimeFilter_SOZ(cfg)
        out = model(torch.randn(4, 22, 20, 100))
        probs = out['soz_probs']   # [4, 19]
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        # ---- 1. Patch Embedding ----
        self.patch_embed = PatchEmbedding(cfg)

        # ---- 2. LaBraM Backbone ----
        self.backbone = LaBraMBackbone(cfg)

        # ---- 3. TimeFilter ----
        self.timefilter = TimeFilterBlock(cfg)

        # ---- 4. GAT ----
        self.gat = GATBlock(
            dim=cfg.embed_dim,
            n_layers=cfg.gat_layers,
            n_heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
        )

        # ---- 5. SOZ Localization Head ----
        self.soz_head = SOZLocalizationHead(cfg)

        # ---- 6. Domain Discriminator (optional) ----
        self.domain_disc = None
        if cfg.use_domain_adversarial:
            self.domain_disc = DomainDiscriminator(
                dim=cfg.embed_dim,
                hidden=cfg.domain_hidden,
                grl_lambda=cfg.grl_lambda,
            )

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化 (仅可训练参数)"""
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        gamma_energy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 22, 20, 100]  TCP双极导联数据
            domain_labels: [B, 1]  0=public, 1=private (训练域判别器时需要)
            gamma_energy: [B, 440, 1]  可选, 预提取γ频段能量

        Returns:
            Dict with keys:
                soz_probs, soz_logits, bipolar_logits,
                domain_logits, temporal_attn, filtered_adj
        """
        B = x.size(0)

        # (1) Patch Embedding
        h = self.patch_embed(x)                      # [B, 440, D]

        # (2) LaBraM Backbone
        h = self.backbone(h)                          # [B, 440, D]

        # (3) TimeFilter → 过滤后邻接矩阵
        filtered_adj = self.timefilter(h, gamma_energy)  # [B, 440, 440]

        # (4) GAT 图卷积
        h = self.gat(h, filtered_adj)                 # [B, 440, D]

        # (5) SOZ Localization Head
        soz_logits, aux = self.soz_head(h)            # [B, n_output]
        soz_probs = torch.sigmoid(soz_logits)

        outputs = {
            'soz_probs': soz_probs,                   # [B, n_output]
            'soz_logits': soz_logits,                  # [B, n_output]
            'bipolar_logits': aux['bipolar_logits'],    # [B, 22] 始终可用
            'temporal_attn': aux['temporal_attn_weights'],
            'filtered_adj': filtered_adj,
            # 向后兼容旧键名
            'monopolar_probs': soz_probs,
            'monopolar_logits': soz_logits,
        }

        # (6) Domain Discriminator
        if self.domain_disc is not None:
            global_feat = h.mean(dim=1)               # [B, D]
            domain_logits = self.domain_disc(global_feat)  # [B, 1]
            outputs['domain_logits'] = domain_logits

        return outputs

    def get_loss_fn(self) -> SOZDetectionLoss:
        """获取配套损失函数"""
        return SOZDetectionLoss(
            focal_gamma=self.cfg.focal_gamma,
            focal_alpha=self.cfg.focal_alpha,
            domain_weight=self.cfg.domain_loss_weight,
        )

    def set_grl_lambda(self, lambda_: float):
        """动态调整梯度反转强度 (通常随训练进度递增)"""
        if self.domain_disc is not None:
            self.domain_disc.grl.set_lambda(lambda_)

    def get_trainable_params(self) -> List[Dict]:
        """获取分组参数 (便于差异学习率)"""
        backbone_params = []
        head_params = []
        domain_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(p)
            elif 'domain_disc' in name:
                domain_params.append(p)
            else:
                head_params.append(p)

        groups = [
            {'params': backbone_params, 'lr_scale': 0.1, 'name': 'backbone'},
            {'params': head_params, 'lr_scale': 1.0, 'name': 'head'},
        ]
        if domain_params:
            groups.append({'params': domain_params, 'lr_scale': 1.0, 'name': 'domain'})

        return groups

    def summary(self) -> str:
        """模型摘要"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        lines = [
            "=" * 60,
            "LaBraM-TimeFilter-SOZ Model Summary",
            "=" * 60,
            f"Input:  [B, {self.cfg.n_channels}, {self.cfg.n_patches}, {self.cfg.patch_len}]",
            f"Output: soz_probs [B, {self.cfg.n_output}] ({self.cfg.output_mode} mode)",
            f"",
            f"Backbone: {self.cfg.n_transformer_layers} layers "
            f"({self.cfg.n_frozen_layers} frozen + "
            f"{self.cfg.n_transformer_layers - self.cfg.n_frozen_layers} trainable)",
            f"Embed dim: {self.cfg.embed_dim}",
            f"TimeFilter: H={self.cfg.tf_n_heads}, alpha={self.cfg.tf_alpha}, "
            f"F={self.cfg.tf_n_filters}, Top-p={self.cfg.top_p}",
            f"GAT: {self.cfg.gat_layers} layers, {self.cfg.gat_heads} heads",
            f"Domain adversarial: {self.cfg.use_domain_adversarial}",
            f"",
            f"Parameters: {total:,} total, {trainable:,} trainable, {frozen:,} frozen",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# 便捷构造函数
# =============================================================================

def build_model(
    checkpoint: str = '',
    n_frozen: int = 12,
    embed_dim: int = 128,
    use_domain_adversarial: bool = True,
    **kwargs,
) -> LaBraM_TimeFilter_SOZ:
    """快速构建模型"""
    cfg = ModelConfig(
        labram_checkpoint=checkpoint,
        n_frozen_layers=n_frozen,
        embed_dim=embed_dim,
        use_domain_adversarial=use_domain_adversarial,
        **kwargs,
    )
    model = LaBraM_TimeFilter_SOZ(cfg)
    logger.info(model.summary())
    return model


# =============================================================================
# 自测
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    for mode, n_out in [('bipolar', 22), ('monopolar', 19)]:
        print(f"\n{'='*60}")
        print(f"Testing {mode} mode (n_output={n_out})")
        print(f"{'='*60}")

        cfg = ModelConfig(
            labram_checkpoint='',
            n_transformer_layers=4,
            n_frozen_layers=2,
            n_output=n_out,
            output_mode=mode,
        )
        model = LaBraM_TimeFilter_SOZ(cfg)
        print(model.summary())

        B = 4
        X = torch.randn(B, 22, 20, 100)
        domain = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)

        # Forward (shape check)
        print(f"\nForward pass...")
        with torch.no_grad():
            out = model(X, domain_labels=domain)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")

        assert out['soz_probs'].shape == (B, n_out), \
            f"Expected soz_probs shape ({B}, {n_out}), got {out['soz_probs'].shape}"

        # Loss + Backward
        print(f"\nLoss + Backward...")
        out = model(X, domain_labels=domain)
        loss_fn = model.get_loss_fn()
        y_soz = torch.zeros(B, n_out)
        y_soz[0, [0, 2]] = 1.0
        y_soz[2, [5, min(8, n_out - 1)]] = 1.0

        total_loss, loss_dict = loss_fn(
            out['soz_logits'], y_soz,
            out.get('domain_logits'), domain,
        )
        print(f"  Total loss: {total_loss.item():.4f}")
        total_loss.backward()
        print(f"  Backward OK")

    print(f"\n[OK] All tests passed (bipolar + monopolar modes)!")

