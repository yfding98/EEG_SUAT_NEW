#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ_new 损失函数模块

忠实复现官方 szloc_loss.py 中的所有 Map Loss 变体，
并补充 Stage-1 发作检测损失。

官方损失权重（来自 szloc_train.py）：
  chn_sz_weight         = 1   通道级发作 CrossEntropy
  tot_sz_weight         = 1   全局发作  CrossEntropy
  attn_map_weight_pos   = 2   attn_onset_map 正类 L2 损失
  attn_map_weight_neg   = 1   attn_onset_map 负类 L2 损失
  attn_map_weight_margin= 1   attn_onset_map margin 损失
  chn_map_weight_pos    = 2   chn_onset_map  正类 L2 损失（与 attn 相同权重）
  chn_map_weight_neg    = 1   chn_onset_map  负类 L2 损失
  chn_map_weight_margin = 1   chn_onset_map  margin 损失

Stage-1 损失：
  CrossEntropyLoss(weight=[0.2, 0.8])   处理非发作/发作不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Map Loss 家族（与官方 szloc_loss.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

class MapLossL1Pos(nn.Module):
    """最大化最高正类预测值  → 1 - max(pred * label)"""

    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=0)
            onset_map_pred = onset_map_pred / (maxes + 1e-6)
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map, dim=1)
        return torch.mean(1 - pos_loc_max)


class MapLossL1PosSum(nn.Module):
    """最大化正类预测值之和（归一化）"""

    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale     = scale

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=0)
            onset_map_pred = onset_map_pred / (maxes + 1e-6)
        pos_loc_sum = torch.sum(onset_map_pred * onset_map, dim=1)
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            pos_loc_sum = pos_loc_sum / factor
        return torch.mean(1 - pos_loc_sum)


class MapLossL2PosSum(nn.Module):
    """
    正类 L2 损失（官方 Stage-2 主要用此）

    最小化 Σ[(label - pred*label)²] / n_pos
    """

    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale     = scale

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        pos_loc_sum = torch.sum((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            pos_loc_sum = pos_loc_sum / (factor + 1e-6)
        return torch.mean(pos_loc_sum)


class MapLossL2PosMax(nn.Module):
    """最大正类误差的 L2 损失"""

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        pos_loc_max, _ = torch.max((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        return torch.mean(pos_loc_max)


class MapLossL2Neg(nn.Module):
    """
    负类 L2 损失

    最小化 Σ[(pred * (1-label))²] / n_neg
    """

    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale     = scale

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        neg_loc_sum = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1)
        if self.scale:
            factor = torch.sum(1 - onset_map, dim=1)
            neg_loc_sum = neg_loc_sum / (factor + 1e-6)
        return torch.mean(neg_loc_sum)


class MapLossMargin(nn.Module):
    """
    Margin 损失（官方版本）

    (1 - max_pos² + max_neg²) / 2
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map,       dim=1)
        neg_loc_max, _ = torch.max(onset_map_pred * (1 - onset_map), dim=1)
        return torch.mean((1 - pos_loc_max ** 2 + neg_loc_max ** 2) / 2)


class MapLossL2(nn.Module):
    """完整 L2 Map 损失（正类 + 负类）"""

    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, onset_map_pred: torch.Tensor,
                onset_map: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        C = onset_map_pred.shape[1]
        neg = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1) / C
        pos = torch.sum((onset_map - onset_map_pred * onset_map) ** 2,  dim=1) / C
        return torch.mean(neg + pos)


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 损失
# ─────────────────────────────────────────────────────────────────────────────

class Stage1DetectionLoss(nn.Module):
    """
    Stage-1 发作检测损失

    官方：CrossEntropyLoss(weight=[0.2, 0.8])
    处理非发作窗口远多于发作窗口的不平衡问题。
    """

    def __init__(self, neg_weight: float = 0.2, pos_weight: float = 0.8):
        super().__init__()
        w = torch.tensor([neg_weight, pos_weight])
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, T, 2]   (或任何形状，最后一维=2)
        labels : [B, T]      int64  0=非发作  1=发作
        """
        device = logits.device
        # 把 weight 移到同一设备
        if self.ce.weight.device != device:
            self.ce.weight = self.ce.weight.to(device)
        return self.ce(logits.reshape(-1, 2), labels.reshape(-1).long())


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 组合损失（官方 szloc_train.py 中的设计）
# ─────────────────────────────────────────────────────────────────────────────

class Stage2SOZLoss(nn.Module):
    """
    Stage-2 SOZ 定位组合损失

    官方权重（szloc_train.py）：
      chn_sz_weight         = 1   通道级发作 CE（sz_label_idx 取发作前后各15帧）
      tot_sz_weight         = 1   全局发作 CE
      attn_map_weight_pos   = 2
      attn_map_weight_neg   = 1
      attn_map_weight_margin= 1
      chn_map_weight_pos    = 2
      chn_map_weight_neg    = 1
      chn_map_weight_margin = 1

    官方只取时间窗口中发作前15帧 + 发作后15帧（共30帧）计算发作 CE，
    避免过多非发作窗口主导梯度。
    本实现默认使用全部窗口；可通过 sz_label_idx 参数指定子集。
    """

    def __init__(
        self,
        chn_sz_weight:          float = 1.0,
        tot_sz_weight:          float = 1.0,
        attn_map_weight_pos:    float = 2.0,
        attn_map_weight_neg:    float = 1.0,
        attn_map_weight_margin: float = 1.0,
        chn_map_weight_pos:     float = 2.0,
        chn_map_weight_neg:     float = 1.0,
        chn_map_weight_margin:  float = 1.0,
    ):
        super().__init__()
        # 权重
        self.chn_sz_weight          = chn_sz_weight
        self.tot_sz_weight          = tot_sz_weight
        self.attn_map_weight_pos    = attn_map_weight_pos
        self.attn_map_weight_neg    = attn_map_weight_neg
        self.attn_map_weight_margin = attn_map_weight_margin
        self.chn_map_weight_pos     = chn_map_weight_pos
        self.chn_map_weight_neg     = chn_map_weight_neg
        self.chn_map_weight_margin  = chn_map_weight_margin

        # 发作检测 CE（官方此处没有 class weight，直接标准 CE）
        self.classification_loss = nn.CrossEntropyLoss()

        # Map losses
        self.map_loss_pos    = MapLossL2PosSum(scale=True)
        self.map_loss_neg    = MapLossL2Neg(scale=True)
        self.map_loss_margin = MapLossMargin()

    def forward(
        self,
        chn_sz_logits:   torch.Tensor,    # [B, T, 2]  通道发作 logits
        tot_sz_logits:   torch.Tensor,    # [B, T, 2]  全局发作 logits
        attn_onset_map:  torch.Tensor,    # [B, C]     注意力 SOZ 图
        chn_onset_map:   torch.Tensor,    # [B, C]     梯度 SOZ 图
        sz_labels:       torch.Tensor,    # [B, T]     int64  发作标签
        onset_map_gt:    torch.Tensor,    # [B, C]     float  SOZ 真实标签
        sz_label_idx:    Optional[list] = None,  # 用于 CE 的时间步索引子集
        return_components: bool = False,
    ):
        """
        官方 szloc_train.py 中 loss 计算逻辑的等价实现。

        sz_label_idx：官方取 list(range(15)) + list(range(30,45))，
                      即发作前 15 帧和发作后 15 帧。
                      本实现若不传则使用全部时间步。
        """
        # ── 发作检测 CE ──────────────────────────────────────────────
        if sz_label_idx is not None:
            chn_in  = chn_sz_logits[:, sz_label_idx, :]   # [B, idx_len, 2]
            tot_in  = tot_sz_logits[:, sz_label_idx, :]
            lbl_in  = sz_labels[:, sz_label_idx]
        else:
            chn_in  = chn_sz_logits
            tot_in  = tot_sz_logits
            lbl_in  = sz_labels

        # CrossEntropy 期望 logits: [N, C], labels: [N]
        chn_ce = self.chn_sz_weight * self.classification_loss(
            chn_in.reshape(-1, 2),
            lbl_in.reshape(-1).long()
        )
        tot_ce = self.tot_sz_weight * self.classification_loss(
            tot_in.reshape(-1, 2),
            lbl_in.reshape(-1).long()
        )
        total_loss = chn_ce + tot_ce

        # ── attn_onset_map Map 损失 ───────────────────────────────────
        attn_pos    = self.attn_map_weight_pos    * self.map_loss_pos(attn_onset_map,  onset_map_gt)
        attn_neg    = self.attn_map_weight_neg    * self.map_loss_neg(attn_onset_map,  onset_map_gt)
        attn_margin = self.attn_map_weight_margin * self.map_loss_margin(attn_onset_map, onset_map_gt)
        total_loss += attn_pos + attn_neg + attn_margin

        # ── chn_onset_map Map 损失 ────────────────────────────────────
        chn_pos    = self.chn_map_weight_pos    * self.map_loss_pos(chn_onset_map,  onset_map_gt)
        chn_neg    = self.chn_map_weight_neg    * self.map_loss_neg(chn_onset_map,  onset_map_gt)
        chn_margin = self.chn_map_weight_margin * self.map_loss_margin(chn_onset_map, onset_map_gt)
        total_loss += chn_pos + chn_neg + chn_margin

        if return_components:
            return total_loss, {
                'chn_ce':      chn_ce.item(),
                'tot_ce':      tot_ce.item(),
                'attn_pos':    attn_pos.item(),
                'attn_neg':    attn_neg.item(),
                'attn_margin': attn_margin.item(),
                'chn_pos':     chn_pos.item(),
                'chn_neg':     chn_neg.item(),
                'chn_margin':  chn_margin.item(),
            }
        return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, T, C = 4, 45, 19

    # Stage-1
    s1_loss = Stage1DetectionLoss()
    logits_s1 = torch.randn(B, T, 2)
    labels_s1 = torch.randint(0, 2, (B, T))
    print(f'Stage-1 loss: {s1_loss(logits_s1, labels_s1).item():.4f}')

    # Stage-2
    s2_loss = Stage2SOZLoss()
    chn_logits  = torch.randn(B, T, 2)
    tot_logits  = torch.randn(B, T, 2)
    attn_map    = torch.rand(B, C)
    chn_map     = torch.rand(B, C)
    sz_labels   = torch.randint(0, 2, (B, T))
    onset_gt    = torch.zeros(B, C)
    onset_gt[:, [3, 7, 8]] = 1.0

    # 官方 sz_label_idx = list(range(15)) + list(range(30, 45))
    idx = list(range(15)) + list(range(30, 45))
    total, comps = s2_loss(
        chn_logits, tot_logits, attn_map, chn_map,
        sz_labels, onset_gt, sz_label_idx=idx,
        return_components=True
    )
    print(f'Stage-2 total: {total.item():.4f}')
    for k, v in comps.items():
        print(f'  {k}: {v:.4f}')
