#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数模块

参考DeepSOZ的损失函数设计，实现多标签分类训练的损失函数组合。

支持三种分类粒度:
1. onset_zone - 脑区级别多标签分类
2. hemi - 半球级别单标签分类
3. channel - 通道级别多标签分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ==============================================================================
# 基础损失函数
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: 正类权重。可以是：
            - float: 所有类别使用相同权重
            - list/tensor: 每个类别使用不同权重，如 [1.0, 1.0, 1.5, 1.0, 3.0] 
        gamma: 聚焦参数，越大越关注难分类样本。建议范围 [2.0, 5.0]
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, alpha=0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        # 支持 list/tensor 类型的 alpha（每个类别不同权重）
        if isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = alpha  # float 标量
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) 预测logits
            targets: (B, C) 目标标签 (0/1)
        """
        inputs, targets = ensure_same_device(inputs, targets)
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # 处理 alpha
        if isinstance(self.alpha, torch.Tensor):
            # 确保 alpha 在正确设备上，并扩展到 (1, C) 以便广播
            alpha = self.alpha.to(inputs.device)
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(0)  # (1, C)
            F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - 用于处理类别不平衡的分割/多标签任务
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) 预测概率 (经过sigmoid)
            targets: (B, C) 目标标签 (0/1)
        """
        inputs = torch.sigmoid(inputs)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BCEWithLogitsLossWrapper(nn.Module):
    """
    BCEWithLogitsLoss包装器，确保pos_weight在正确的设备上
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) 预测logits
            targets: (B, C) 目标标签 (0/1)
        """
        inputs, targets = ensure_same_device(inputs, targets)
        
        # 确保 pos_weight 在正确的设备上
        pos_weight = self.pos_weight
        if pos_weight is not None and pos_weight.device != inputs.device:
            pos_weight = pos_weight.to(inputs.device)
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight
        )


class LabelSmoothingBCE(nn.Module):
    """
    带标签平滑的BCE损失
    """
    
    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) 预测logits
            targets: (B, C) 目标标签 (0/1)
        """
        inputs, targets = ensure_same_device(inputs, targets)
        # 标签平滑
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # 确保 pos_weight 在正确的设备上
        pos_weight = self.pos_weight
        if pos_weight is not None and pos_weight.device != inputs.device:
            pos_weight = pos_weight.to(inputs.device)
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight
        )


# ==============================================================================
# DeepSOZ风格的Map损失函数
# ==============================================================================

class MapLossL1Pos(nn.Module):
    """
    正类位置损失 - 最大化正类预测值
    
    Loss = 1 - max(pred * target)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测概率
            target: (B, C) 目标标签 (0/1)
        """
        pred, target = ensure_same_device(pred, target)
        if self.normalize:
            # 按batch归一化
            maxes, _ = torch.max(pred, dim=1, keepdim=True)
            pred = pred / (maxes + 1e-6)
        
        pos_loc_max, _ = torch.max(pred * target, dim=1)
        return torch.mean(1 - pos_loc_max)


class MapLossL2PosSum(nn.Module):
    """
    正类位置L2损失 - 最小化正类预测与目标的差距
    """
    
    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = pred.shape
            maxes, _ = torch.max(pred, dim=1, keepdim=True)
            pred = pred / (maxes + 1e-6)
        
        pos_loc_sum = torch.sum((target - pred * target) ** 2, dim=1)
        
        if self.scale:
            factor = torch.sum(target, dim=1) + 1e-6
            pos_loc_sum = pos_loc_sum / factor
        
        return torch.mean(pos_loc_sum)


class MapLossL2Neg(nn.Module):
    """
    负类位置L2损失 - 最小化负类的预测值
    """
    
    def __init__(self, normalize: bool = True, scale: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = pred.shape
            maxes, _ = torch.max(pred, dim=1, keepdim=True)
            pred = pred / (maxes + 1e-6)
        
        neg_loc_sum = torch.sum((pred * (1 - target)) ** 2, dim=1)
        
        if self.scale:
            factor = torch.sum(1 - target, dim=1) + 1e-6
            neg_loc_sum = neg_loc_sum / factor
        
        return torch.mean(neg_loc_sum)


class MapLossMargin(nn.Module):
    """
    Margin损失 - 确保正类预测值高于负类
    """
    
    def __init__(self, normalize: bool = True, margin: float = 0.5):
        super().__init__()
        self.normalize = normalize
        self.margin = margin
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = pred.shape
            maxes, _ = torch.max(pred, dim=1, keepdim=True)
            pred = pred / (maxes + 1e-6)
        
        # 正类最大预测值
        pos_mask = target > 0.5
        neg_mask = target < 0.5
        
        pos_loc_max = torch.where(
            pos_mask.any(dim=1),
            (pred * target).max(dim=1)[0],
            torch.zeros_like(pred[:, 0])
        )
        
        neg_loc_max = torch.where(
            neg_mask.any(dim=1),
            (pred * (1 - target)).max(dim=1)[0],
            torch.zeros_like(pred[:, 0])
        )
        
        # Margin loss: pos_max should be at least `margin` higher than neg_max
        margin_loss = F.relu(self.margin + neg_loc_max - pos_loc_max)
        
        return torch.mean(margin_loss)


class MapLossL2(nn.Module):
    """
    完整的L2 Map损失（正类+负类）
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            B, C = pred.shape
            maxes, _ = torch.max(pred, dim=1, keepdim=True)
            pred = pred / (maxes + 1e-6)
        
        C = pred.shape[1]
        neg_loc_sum = torch.sum((pred * (1 - target)) ** 2, dim=1) / C
        pos_loc_sum = torch.sum((target - pred * target) ** 2, dim=1) / C
        
        return torch.mean(neg_loc_sum + pos_loc_sum)


# ==============================================================================
# 组合损失函数
# ==============================================================================
def ensure_same_device(pred, target):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 确保预测值和目标值在同一设备上
    if pred.device != device:
        pred = pred.to(device)
    if target.device != device:
        target = target.to(device)

    return pred, target
class SOZLocalizationLoss(nn.Module):
    """
    SOZ定位组合损失函数
    
    结合多种损失:
    1. BCE/Focal Loss - 主要分类损失
    2. Map正类损失 - 确保正类预测值高
    3. Map负类损失 - 确保负类预测值低
    4. Margin损失 - 确保正负类有明显差距
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',  # 'bce', 'focal', 'dice'
        pos_weight: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        map_pos_weight: float = 1.0,
        map_neg_weight: float = 0.5,
        map_margin_weight: float = 0.5,
        label_smoothing: float = 0.0,
        n_channels: int = 19
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.map_pos_weight = map_pos_weight
        self.map_neg_weight = map_neg_weight
        self.map_margin_weight = map_margin_weight
        
        # 主分类损失
        if loss_type == 'focal':
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'dice':
            self.cls_loss = DiceLoss()
        else:
            pw = torch.ones(n_channels) * pos_weight
            if label_smoothing > 0:
                self.cls_loss = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pw)
            else:
                # 使用包装器确保pos_weight在正确的设备上
                self.cls_loss = BCEWithLogitsLossWrapper(pos_weight=pw)
        
        # Map损失
        self.map_loss_pos = MapLossL2PosSum(normalize=True, scale=True)
        self.map_loss_neg = MapLossL2Neg(normalize=True, scale=True)
        self.map_loss_margin = MapLossMargin(normalize=True)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测logits
            target: (B, C) 目标标签
            return_components: 是否返回各组件损失
        
        Returns:
            total_loss 或 (total_loss, loss_dict)
        """
        # 主分类损失
        cls_loss = self.cls_loss(pred, target)
        
        # 转换为概率用于map损失
        pred_prob = torch.sigmoid(pred)
        
        # Map损失
        map_loss_pos = self.map_loss_pos(pred_prob, target) * self.map_pos_weight
        map_loss_neg = self.map_loss_neg(pred_prob, target) * self.map_neg_weight
        map_loss_margin = self.map_loss_margin(pred_prob, target) * self.map_margin_weight
        
        # 总损失
        total_loss = cls_loss + map_loss_pos + map_loss_neg + map_loss_margin
        
        if return_components:
            return total_loss, {
                'cls_loss': cls_loss.item(),
                'map_pos': map_loss_pos.item(),
                'map_neg': map_loss_neg.item(),
                'map_margin': map_loss_margin.item(),
            }
        
        return total_loss


class OnsetZoneLoss(nn.Module):
    """
    onset_zone/chain 多标签分类损失函数
    
    支持动态类别数量和每类别不同权重
    
    Args:
        loss_type: 'bce', 'focal', 'dice'
        pos_weight: 正类权重。可以是：
            - float: 所有类别使用相同权重
            - list/tensor: 每个类别不同权重，如 [2.0, 2.0, 3.0, 2.0, 5.0]
        focal_alpha: Focal Loss 的 alpha 参数。可以是 float 或 list/tensor
        focal_gamma: Focal Loss 的 gamma 参数。越大越关注难分类样本
        label_smoothing: 标签平滑系数
        n_classes: 类别数（默认5，可根据label_type调整）
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        pos_weight=2.0,  # float 或 list/tensor
        focal_alpha=0.25,  # float 或 list/tensor
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            # focal_alpha 可以是 float 或 list/tensor（每类别不同）
            self.loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'dice':
            self.loss = DiceLoss()
        else:
            # 处理 pos_weight
            if isinstance(pos_weight, (list, tuple)):
                pw = torch.tensor(pos_weight, dtype=torch.float32)
            elif isinstance(pos_weight, torch.Tensor):
                pw = pos_weight.float()
            else:
                pw = torch.ones(n_classes) * pos_weight
            
            if label_smoothing > 0:
                self.loss = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pw)
            else:
                # 使用包装器确保pos_weight在正确的设备上
                self.loss = BCEWithLogitsLossWrapper(pos_weight=pw)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, n_classes) 预测logits
            target: (B, n_classes) 目标标签
        """
        pred, target = ensure_same_device(pred, target)
        return self.loss(pred, target)


class HemiLoss(nn.Module):
    """
    hemi半球级别损失函数
    
    单标签分类损失（4类: L, R, B, U）
    """
    
    def __init__(
        self,
        loss_type: str = 'ce',  # 'ce' for cross-entropy
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.n_classes = 4
        
        if label_smoothing > 0:
            self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 4) 半球预测logits
            target: (B, 4) one-hot标签 或 (B,) 类别索引
        """
        # 如果target是one-hot，转换为类别索引
        if target.dim() > 1 and target.shape[-1] == self.n_classes:
            target = target.argmax(dim=-1)
        
        return self.loss(pred, target)


class ChannelLoss(SOZLocalizationLoss):
    """
    channel通道级别损失函数
    
    继承SOZLocalizationLoss，针对19通道多标签分类
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        pos_weight: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        map_pos_weight: float = 1.0,
        map_neg_weight: float = 0.5,
        map_margin_weight: float = 0.5,
        label_smoothing: float = 0.1
    ):
        super().__init__(
            loss_type=loss_type,
            pos_weight=pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            map_pos_weight=map_pos_weight,
            map_neg_weight=map_neg_weight,
            map_margin_weight=map_margin_weight,
            label_smoothing=label_smoothing,
            n_channels=19
        )


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    同时优化三种粒度的分类任务
    """
    
    def __init__(
        self,
        onset_zone_weight: float = 1.0,
        hemi_weight: float = 1.0,
        channel_weight: float = 1.0,
        loss_config: dict = None
    ):
        super().__init__()
        
        self.onset_zone_weight = onset_zone_weight
        self.hemi_weight = hemi_weight
        self.channel_weight = channel_weight
        
        config = loss_config or {}
        
        self.onset_zone_loss = OnsetZoneLoss(
            loss_type=config.get('loss_type', 'bce'),
            pos_weight=config.get('pos_weight', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        self.hemi_loss = HemiLoss(
            loss_type='ce',
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        self.channel_loss = ChannelLoss(
            loss_type=config.get('loss_type', 'bce'),
            pos_weight=config.get('pos_weight', 5.0),
            map_pos_weight=config.get('map_pos_weight', 1.0),
            map_neg_weight=config.get('map_neg_weight', 0.5),
            map_margin_weight=config.get('map_margin_weight', 0.5),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    
    def forward(
        self,
        onset_zone_pred: torch.Tensor,
        onset_zone_target: torch.Tensor,
        hemi_pred: torch.Tensor,
        hemi_target: torch.Tensor,
        channel_pred: torch.Tensor,
        channel_target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            onset_zone_pred: (B, 5) 脑区预测
            onset_zone_target: (B, 5) 脑区标签
            hemi_pred: (B, 4) 半球预测
            hemi_target: (B, 4) 半球标签
            channel_pred: (B, 19) 通道预测
            channel_target: (B, 19) 通道标签
        """
        loss_onset_zone = self.onset_zone_loss(onset_zone_pred, onset_zone_target)
        loss_hemi = self.hemi_loss(hemi_pred, hemi_target)
        loss_channel, channel_components = self.channel_loss(
            channel_pred, channel_target, return_components=True
        )
        
        total_loss = (
            self.onset_zone_weight * loss_onset_zone +
            self.hemi_weight * loss_hemi +
            self.channel_weight * loss_channel
        )
        
        if return_components:
            return total_loss, {
                'onset_zone': loss_onset_zone.item(),
                'hemi': loss_hemi.item(),
                'channel': loss_channel.item(),
                **{f'channel_{k}': v for k, v in channel_components.items()}
            }
        
        return total_loss


# ==============================================================================
# 工具函数
# ==============================================================================

def get_loss_function(
    task_type: str,
    config: dict = None
) -> nn.Module:
    """
    获取指定任务的损失函数
    
    Args:
        task_type: 'onset_zone', 'hemi', 'channel', 'multi'
        config: 损失函数配置
    
    Returns:
        损失函数模块
    """
    config = config or {}
    
    if task_type == 'onset_zone':
        return OnsetZoneLoss(
            loss_type=config.get('classification_loss', 'bce'),
            pos_weight=config.get('pos_weight', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    elif task_type == 'hemi':
        return HemiLoss(
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    elif task_type == 'channel':
        return ChannelLoss(
            loss_type=config.get('classification_loss', 'bce'),
            pos_weight=config.get('pos_weight', 5.0),
            map_pos_weight=config.get('map_loss_pos_weight', 1.0),
            map_neg_weight=config.get('map_loss_neg_weight', 0.5),
            map_margin_weight=config.get('map_loss_margin_weight', 0.5),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    elif task_type == 'multi':
        return MultiTaskLoss(
            onset_zone_weight=1.0,
            hemi_weight=1.0,
            channel_weight=1.0,
            loss_config=config
        )
    else:
        raise ValueError(f"未知任务类型: {task_type}")


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    # 测试损失函数
    batch_size = 4
    
    print("测试SOZLocalizationLoss...")
    loss_fn = SOZLocalizationLoss(loss_type='bce', pos_weight=5.0)
    pred = torch.randn(batch_size, 19)
    target = torch.zeros(batch_size, 19)
    target[:, [3, 7, 8]] = 1  # 模拟SOZ通道
    
    loss, components = loss_fn(pred, target, return_components=True)
    print(f"总损失: {loss.item():.4f}")
    print(f"组件: {components}")
    
    print("\n测试OnsetZoneLoss...")
    onset_loss = OnsetZoneLoss()
    pred = torch.randn(batch_size, 5)
    target = torch.zeros(batch_size, 5)
    target[:, [0, 1]] = 1  # frontal, temporal
    loss = onset_loss(pred, target)
    print(f"损失: {loss.item():.4f}")
    
    print("\n测试HemiLoss...")
    hemi_loss = HemiLoss()
    pred = torch.randn(batch_size, 4)
    target = torch.zeros(batch_size, 4)
    target[:, 0] = 1  # L
    loss = hemi_loss(pred, target)
    print(f"损失: {loss.item():.4f}")
    
    print("\n测试ChannelLoss...")
    channel_loss = ChannelLoss()
    pred = torch.randn(batch_size, 19)
    target = torch.zeros(batch_size, 19)
    target[:, [3, 7, 8]] = 1
    loss, components = channel_loss(pred, target, return_components=True)
    print(f"总损失: {loss.item():.4f}")
    print(f"组件: {components}")
    
    print("\n所有损失函数测试通过!")
