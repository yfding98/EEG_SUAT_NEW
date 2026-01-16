"""
损失函数模块
迁移DeepSOZ项目的损失函数设计，包括：
- MapLossL2PosSum: 正样本位置的L2损失
- MapLossL2Neg: 负样本位置的L2损失  
- MapLossMargin: 正负样本边界损失
- 脑区级别分类损失
- 通道级别分类损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MapLossL1Pos(nn.Module):
    """L1正样本损失 - 最大化正样本位置的预测值"""
    
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        """
        Args:
            onset_map_pred: 预测的onset map [B, C]
            onset_map: 真实的onset map [B, C]
        """
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=1, keepdim=True)
            onset_map_pred = onset_map_pred / (maxes + 1e-6)
        
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map, dim=1)
        return torch.mean(1 - pos_loc_max)


class MapLossL1PosSum(nn.Module):
    """L1正样本求和损失"""
    
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=1, keepdim=True)
            onset_map_pred = onset_map_pred / (maxes + 1e-6)
        
        pos_loc_sum = torch.sum(onset_map_pred * onset_map, dim=1)
        
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            factor = torch.clamp(factor, min=1.0)
            pos_loc_sum = pos_loc_sum / factor
        
        return torch.mean(1 - pos_loc_sum)


class MapLossL2PosSum(nn.Module):
    """
    L2正样本求和损失
    最小化正样本位置的预测值与目标值之间的平方差
    """
    
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        """
        Args:
            onset_map_pred: 预测的onset map [B, C]
            onset_map: 真实的onset map [B, C]
        
        Returns:
            L2损失值
        """
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        
        # 计算正样本位置的L2损失: (target - pred*target)^2
        pos_loc_sum = torch.sum((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            factor = torch.clamp(factor, min=1.0)
            pos_loc_sum = pos_loc_sum / factor
        
        return torch.mean(pos_loc_sum)


class MapLossL2PosMax(nn.Module):
    """L2正样本最大损失"""
    
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        
        pos_loc_max, _ = torch.max((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        return torch.mean(pos_loc_max)


class MapLossL2Neg(nn.Module):
    """
    L2负样本损失
    最小化负样本位置的预测值
    """
    
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        """
        Args:
            onset_map_pred: 预测的onset map [B, C]
            onset_map: 真实的onset map [B, C]
        
        Returns:
            L2负样本损失值
        """
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        
        # 计算负样本位置的L2损失: (pred * (1-target))^2
        neg_loc_sum = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1)
        
        if self.scale:
            factor = torch.sum(1 - onset_map, dim=1)
            factor = torch.clamp(factor, min=1.0)
            neg_loc_sum = neg_loc_sum / factor
        
        return torch.mean(neg_loc_sum)


class MapLossMargin(nn.Module):
    """
    边界损失
    确保正样本的最大预测值大于负样本的最大预测值
    """
    
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, onset_map_pred, onset_map):
        """
        Args:
            onset_map_pred: 预测的onset map [B, C]
            onset_map: 真实的onset map [B, C]
        
        Returns:
            边界损失值
        """
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        
        # 正样本位置的最大预测值
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map, dim=1)
        # 负样本位置的最大预测值
        neg_loc_max, _ = torch.max(onset_map_pred * (1 - onset_map), dim=1)
        
        # 希望 pos_loc_max > neg_loc_max
        return torch.mean((1 - pos_loc_max ** 2 + neg_loc_max ** 2) / 2)


class MapLossL2(nn.Module):
    """综合L2损失 (正样本 + 负样本)"""
    
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + 1e-6)
        
        neg_loc_sum = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1) / onset_map.shape[1]
        pos_loc_sum = torch.sum((onset_map - onset_map_pred * onset_map) ** 2, dim=1) / onset_map.shape[1]
        
        return torch.mean(neg_loc_sum + pos_loc_sum)


class RegionClassificationLoss(nn.Module):
    """
    脑区分类损失
    支持单标签和多标签分类
    """
    
    def __init__(self, num_regions=5, multi_label=True, class_weights=None):
        """
        Args:
            num_regions: 脑区数量
            multi_label: 是否为多标签分类
            class_weights: 类别权重
        """
        super().__init__()
        self.num_regions = num_regions
        self.multi_label = multi_label
        
        if multi_label:
            # 多标签分类使用BCEWithLogitsLoss
            self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            # 单标签分类使用CrossEntropyLoss
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, num_regions]
            target: 目标值
                - 多标签: [B, num_regions] 0/1
                - 单标签: [B] 类别索引
        """
        return self.criterion(pred, target)


class ChannelClassificationLoss(nn.Module):
    """
    通道分类损失
    预测每个通道是否为发作起始区域
    """
    
    def __init__(self, num_channels=19, class_weights=None, pos_weight=None):
        """
        Args:
            num_channels: 通道数量
            class_weights: 类别权重
            pos_weight: 正样本权重（用于处理类别不平衡）
        """
        super().__init__()
        self.num_channels = num_channels
        
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight] * num_channels)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, num_channels]
            target: 目标值 [B, num_channels] 0/1
        """
        return self.criterion(pred, target)


class CombinedSOZLoss(nn.Module):
    """
    综合SOZ定位损失
    结合DeepSOZ的损失函数设计
    """
    
    def __init__(self, 
                 chn_sz_weight=1.0,
                 tot_sz_weight=1.0,
                 attn_map_weight_pos=2.0,
                 attn_map_weight_neg=1.0,
                 attn_map_weight_margin=1.0,
                 chn_map_weight_pos=2.0,
                 chn_map_weight_neg=1.0,
                 chn_map_weight_margin=1.0,
                 device='cpu'):
        """
        Args:
            chn_sz_weight: 通道级别癫痫发作分类权重
            tot_sz_weight: 整体癫痫发作分类权重
            attn_map_weight_pos: 注意力图正样本损失权重
            attn_map_weight_neg: 注意力图负样本损失权重
            attn_map_weight_margin: 注意力图边界损失权重
            chn_map_weight_pos: 通道图正样本损失权重
            chn_map_weight_neg: 通道图负样本损失权重
            chn_map_weight_margin: 通道图边界损失权重
            device: 设备
        """
        super().__init__()
        
        self.chn_sz_weight = chn_sz_weight
        self.tot_sz_weight = tot_sz_weight
        self.attn_map_weight_pos = attn_map_weight_pos
        self.attn_map_weight_neg = attn_map_weight_neg
        self.attn_map_weight_margin = attn_map_weight_margin
        self.chn_map_weight_pos = chn_map_weight_pos
        self.chn_map_weight_neg = chn_map_weight_neg
        self.chn_map_weight_margin = chn_map_weight_margin
        
        # 分类损失
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Map损失
        self.map_loss_pos = MapLossL2PosSum(scale=True).to(device)
        self.map_loss_neg = MapLossL2Neg(scale=True).to(device)
        self.map_loss_margin = MapLossMargin().to(device)
    
    def forward(self, outputs, labels, onset_map):
        """
        Args:
            outputs: 模型输出 (chn_sz_pred, sz_pred, attn_onset_map, chn_onset_map)
            labels: 时间序列标签 [B, T] 或 [B*Nsz, T]
            onset_map: 发作起始区域图 [B, C] 或 [B*Nsz, C]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        chn_sz_pred, sz_pred, attn_onset_map, chn_onset_map = outputs
        
        loss_dict = {}
        
        # 通道级别癫痫发作分类损失
        if chn_sz_pred is not None and self.chn_sz_weight > 0:
            # 选择特定时间步的标签
            sz_label_idx = list(range(15)) + list(range(30, 45)) if labels.shape[-1] >= 45 else list(range(labels.shape[-1]))
            chn_sz_loss = self.chn_sz_weight * self.classification_loss(
                chn_sz_pred[:, sz_label_idx, :].transpose(1, 2) if len(chn_sz_pred.shape) == 3 else chn_sz_pred,
                labels[:, sz_label_idx] if len(labels.shape) == 2 else labels
            )
            loss_dict['chn_sz_loss'] = chn_sz_loss.item()
        else:
            chn_sz_loss = 0
        
        # 整体癫痫发作分类损失
        if sz_pred is not None and self.tot_sz_weight > 0:
            sz_label_idx = list(range(15)) + list(range(30, 45)) if labels.shape[-1] >= 45 else list(range(labels.shape[-1]))
            tot_sz_loss = self.tot_sz_weight * self.classification_loss(
                sz_pred[:, sz_label_idx, :].transpose(1, 2) if len(sz_pred.shape) == 3 else sz_pred,
                labels[:, sz_label_idx] if len(labels.shape) == 2 else labels
            )
            loss_dict['tot_sz_loss'] = tot_sz_loss.item()
        else:
            tot_sz_loss = 0
        
        total_loss = chn_sz_loss + tot_sz_loss
        
        # 注意力onset map损失
        if attn_onset_map is not None:
            attn_map_loss_pos = self.attn_map_weight_pos * self.map_loss_pos(attn_onset_map, onset_map)
            attn_map_loss_neg = self.attn_map_weight_neg * self.map_loss_neg(attn_onset_map, onset_map)
            attn_map_loss_margin = self.attn_map_weight_margin * self.map_loss_margin(attn_onset_map, onset_map)
            
            total_loss = total_loss + attn_map_loss_pos + attn_map_loss_neg + attn_map_loss_margin
            
            loss_dict['attn_map_loss_pos'] = attn_map_loss_pos.item()
            loss_dict['attn_map_loss_neg'] = attn_map_loss_neg.item()
            loss_dict['attn_map_loss_margin'] = attn_map_loss_margin.item()
        
        # 通道onset map损失
        if chn_onset_map is not None:
            chn_map_loss_pos = self.chn_map_weight_pos * self.map_loss_pos(chn_onset_map, onset_map)
            chn_map_loss_neg = self.chn_map_weight_neg * self.map_loss_neg(chn_onset_map, onset_map)
            chn_map_loss_margin = self.chn_map_weight_margin * self.map_loss_margin(chn_onset_map, onset_map)
            
            total_loss = total_loss + chn_map_loss_pos + chn_map_loss_neg + chn_map_loss_margin
            
            loss_dict['chn_map_loss_pos'] = chn_map_loss_pos.item()
            loss_dict['chn_map_loss_neg'] = chn_map_loss_neg.item()
            loss_dict['chn_map_loss_margin'] = chn_map_loss_margin.item()
        
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


class SimplifiedSOZLoss(nn.Module):
    """
    简化的SOZ定位损失
    适用于直接预测通道级别SOZ标签的情况
    """
    
    def __init__(self, pos_weight=3.0, use_focal=False, gamma=2.0):
        """
        Args:
            pos_weight: 正样本权重（处理类别不平衡）
            use_focal: 是否使用Focal Loss
            gamma: Focal Loss的gamma参数
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.use_focal = use_focal
        self.gamma = gamma
        
        if not use_focal:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, num_channels]
            target: 目标值 [B, num_channels]
        """
        if self.use_focal:
            # Focal Loss
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            p = torch.sigmoid(pred)
            pt = p * target + (1 - p) * (1 - target)
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * bce
            
            # 应用正样本权重
            weights = torch.ones_like(target)
            weights[target == 1] = self.pos_weight
            loss = loss * weights
            
            return loss.mean()
        else:
            return self.criterion(pred, target)


class RegionSOZLoss(nn.Module):
    """
    脑区级别SOZ定位损失
    将通道级别预测聚合到脑区级别
    """
    
    def __init__(self, num_regions=5, channel_to_region_mapping=None):
        """
        Args:
            num_regions: 脑区数量
            channel_to_region_mapping: 通道到脑区的映射 {channel_idx: region_idx}
        """
        super().__init__()
        self.num_regions = num_regions
        
        # 默认的通道到脑区映射 (基于标准19通道)
        if channel_to_region_mapping is None:
            # frontal: 0, central: 1, temporal: 2, parietal: 3, occipital: 4
            self.channel_to_region = {
                0: 0, 1: 0,  # fp1, fp2 -> frontal
                2: 0, 3: 0, 4: 0, 5: 0, 6: 0,  # f7, f3, fz, f4, f8 -> frontal
                7: 2, 10: 2, 12: 2, 16: 2,  # t3, t4, t5, t6 -> temporal
                8: 1, 9: 1, 11: 1,  # c3, cz, c4 -> central
                13: 3, 14: 3, 15: 3,  # p3, pz, p4 -> parietal
                17: 4, 18: 4,  # o1, o2 -> occipital
            }
        else:
            self.channel_to_region = channel_to_region_mapping
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def aggregate_to_regions(self, channel_pred):
        """
        将通道级别预测聚合到脑区级别
        
        Args:
            channel_pred: 通道级别预测 [B, num_channels]
        
        Returns:
            region_pred: 脑区级别预测 [B, num_regions]
        """
        B, C = channel_pred.shape
        region_pred = torch.zeros(B, self.num_regions, device=channel_pred.device)
        region_counts = torch.zeros(self.num_regions, device=channel_pred.device)
        
        for ch_idx, region_idx in self.channel_to_region.items():
            if ch_idx < C:
                region_pred[:, region_idx] += channel_pred[:, ch_idx]
                region_counts[region_idx] += 1
        
        # 平均
        region_counts = region_counts.clamp(min=1)
        region_pred = region_pred / region_counts.unsqueeze(0)
        
        return region_pred
    
    def forward(self, channel_pred, channel_target):
        """
        Args:
            channel_pred: 通道级别预测 [B, num_channels]
            channel_target: 通道级别目标 [B, num_channels]
        
        Returns:
            loss: 脑区级别损失
        """
        region_pred = self.aggregate_to_regions(channel_pred)
        region_target = self.aggregate_to_regions(channel_target)
        region_target = (region_target > 0).float()  # 二值化
        
        return self.criterion(region_pred, region_target)
