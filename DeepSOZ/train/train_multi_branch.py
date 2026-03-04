#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多分支融合模型训练脚本

训练MultiBranchFusionModel，结合:
1. EEGNet分支 - 原始EEG波形
2. GAT分支 - 连接性矩阵
3. MLP分支 - 图网络指标

关键特性:
- 可配置的融合策略
- 滑动窗口数据增强
- 交叉验证训练
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from dataset_with_connectivity import MultiModalEEGDataset, create_multimodal_dataloader
from preprocess_data import PreprocessedDataset  # 预处理数据加载
from losses import OnsetZoneLoss
from multi_branch_model import MultiBranchFusionModel, create_multi_branch_model
from trainer import create_optimizer, create_scheduler, set_seed, get_device, count_parameters

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# 标签类型配置
# ==============================================================================

LABEL_TYPE_CONFIG = {
    'onset_zone': {
        'name': '脑区分类',
        'classes': ['frontal', 'temporal', 'central', 'parietal', 'occipital'],
        'n_classes': 5,
        'task_type': 'multi_label',  # 多标签分类
    },
    'hemi': {
        'name': '半球分类',
        'classes': ['L', 'R', 'B', 'U'],  # Left, Right, Bilateral, Unknown
        'n_classes': 4,
        'task_type': 'single_label',  # 单标签分类
    },
    'channel': {
        'name': '通道级SOZ',
        'classes': ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
                    't3', 'c3', 'cz', 'c4', 't4',
                    't5', 'p3', 'pz', 'p4', 't6',
                    'o1', 'o2'],
        'n_classes': 19,
        'task_type': 'multi_label',  # 多标签分类
    },
    'chain': {
        'name': '五链分区',
        'classes': ['left_temporal', 'right_temporal', 'left_parasagittal', 
                    'right_parasagittal', 'midline'],
        'n_classes': 5,
        'task_type': 'multi_label',  # 多标签分类
    },
    'region_5': {
        'name': '五脑区(双极导联)',
        'classes': ['left_frontal', 'left_temporal', 'parietal', 
                    'right_frontal', 'right_temporal'],
        'n_classes': 5,
        'task_type': 'multi_label',  # 多标签分类
        # 每个脑区对应的双极导联
        'bipolar_leads': {
            'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
            'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
            'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
            'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
            'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
        },
    },
}


def get_label_config(label_type: str, exclude_classes: list = None):
    """
    获取标签类型配置
    
    Args:
        label_type: 标签类型
        exclude_classes: 要排除的类别列表
    
    Returns:
        config: 配置字典，包含classes, n_classes等
    """
    if label_type not in LABEL_TYPE_CONFIG:
        raise ValueError(f"未知的标签类型: {label_type}")
    
    config = LABEL_TYPE_CONFIG[label_type].copy()
    config['classes'] = config['classes'].copy()
    
    if exclude_classes:
        # 排除指定类别
        exclude_set = {c.lower() for c in exclude_classes}
        config['classes'] = [c for c in config['classes'] if c.lower() not in exclude_set]
        config['n_classes'] = len(config['classes'])
        config['excluded'] = list(exclude_classes)
    
    return config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多分支融合模型训练')
    
    # 数据
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    parser.add_argument('--preprocessed-dir', type=str, default=r"E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ\preprocessed",
                        help='预处理数据目录(使用此选项则从缓存加载，跳过实时处理)')
    
    # 标签类型
    parser.add_argument('--label-type', type=str, default='region_5',
                        choices=['onset_zone', 'hemi', 'channel', 'chain', 'region_5'],
                        help='标签类型: onset_zone(5脑区), hemi(4半球), channel(19通道), chain(5链), region_5(5脑区双极导联)')
    parser.add_argument('--exclude-classes', type=str, nargs='*', default=None,
                        help='排除的类别名称(如 --exclude-classes occipital parietal)')

    
    # 滑动窗口参数
    parser.add_argument('--segment-length', type=float, default=20.0,
                        help='滑动窗口片段长度(秒)')
    parser.add_argument('--segment-overlap', type=float, default=0.5,
                        help='滑动窗口重叠比例')
    parser.add_argument('--window-length', type=float, default=1.0,
                        help='EEGNet内部窗口长度(秒)')
    
    # 模型参数
    parser.add_argument('--fusion-type', type=str, default='attention',
                        choices=['concat', 'attention', 'gated'],
                        help='特征融合策略')
    parser.add_argument('--fusion-feature-dim', type=int, default=64,
                        help='各分支统一输出维度(必须是16的倍数)')
    parser.add_argument('--fusion-dim', type=int, default=128,
                        help='融合后维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')
    
    # EEGNet参数
    parser.add_argument('--F1', type=int, default=8,
                        help='EEGNet时间卷积滤波器数量')
    parser.add_argument('--D', type=int, default=2,
                        help='EEGNet深度乘数')
    parser.add_argument('--F2', type=int, default=16,
                        help='EEGNet分离卷积滤波器数量')
    parser.add_argument('--kernel-length', type=int, default=64,
                        help='EEGNet时间卷积核长度')
    
    # GAT参数
    parser.add_argument('--gat-hidden', type=int, default=32,
                        help='GAT隐藏层维度')
    parser.add_argument('--gat-heads', type=int, default=4,
                        help='GAT注意力头数')
    parser.add_argument('--gat-layers', type=int, default=2,
                        help='GAT层数')
    
    # 连接性参数
    parser.add_argument('--include-directed', action='store_true', default=True,
                        help='包含有向连接性指标 (Granger, TE)')
    parser.add_argument('--no-directed', action='store_false', dest='include_directed',
                        help='不包含有向连接性指标')
    parser.add_argument('--freq-band', type=float, nargs=2, default=[8, 30],
                        help='连接性计算频带 (Hz)')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    
    # 损失函数
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['bce', 'focal', 'dice'],
                        help='损失函数类型')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='正类权重 (统一值，若指定 --class-weights 则忽略)')
    parser.add_argument('--focal-gamma', type=float, default=3.0,
                        help='Focal Loss gamma参数，越大越关注难分类样本 (建议2.0-5.0)')
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                        help='Focal Loss alpha参数 (统一值，若指定 --class-weights 则忽略)')
    parser.add_argument('--class-weights', type=float, nargs='+',default=[0.05, 0.22, 1.5, 0.20, 0.20],
                        help='每类别权重列表，如 --class-weights 1.0 1.0 1.5 1.0 3.0 (顺序对应类别)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='标签平滑')
    
    # 交叉验证
    parser.add_argument('--n-folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--fold', type=int, default=None,
                        help='指定运行某一折')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--dry-run', action='store_true',
                        help='干运行模式（只验证不训练）')
    
    return parser.parse_args()


class MultiBranchTrainer:
    """多分支模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler = None,
        device: str = 'cuda',
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'multi_branch',
        label_config: Dict = None,  # 标签配置
        use_bipolar: bool = False   # 是否使用双极导联数据训练
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.label_config = label_config or {}  # 保存标签配置
        self.use_bipolar = use_bipolar  # 是否使用双极导联数据
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 检查点
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # 训练状态
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            # 获取数据 - 根据配置选择使用原始数据还是双极导联数据
            if self.use_bipolar and 'bipolar_eeg_data' in batch:
                eeg_data = batch['bipolar_eeg_data'].to(self.device)
                connectivity = batch['bipolar_connectivity'].to(self.device)
                graph_metrics = batch['bipolar_graph_metrics'].to(self.device)
            else:
                eeg_data = batch['eeg_data'].to(self.device)
                connectivity = batch['connectivity'].to(self.device)
                graph_metrics = batch['graph_metrics'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(eeg_data, connectivity, graph_metrics)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(eeg_data, connectivity, graph_metrics)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            # 根据配置选择使用原始数据还是双极导联数据
            if self.use_bipolar and 'bipolar_eeg_data' in batch:
                eeg_data = batch['bipolar_eeg_data'].to(self.device)
                connectivity = batch['bipolar_connectivity'].to(self.device)
                graph_metrics = batch['bipolar_graph_metrics'].to(self.device)
            else:
                eeg_data = batch['eeg_data'].to(self.device)
                connectivity = batch['connectivity'].to(self.device)
                graph_metrics = batch['graph_metrics'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(eeg_data, connectivity, graph_metrics)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(eeg_data, connectivity, graph_metrics)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            n_batches += 1
            
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # 第一个batch打印概率范围
            if n_batches == 1:
                logger.info(f"  [调试] Sigmoid输出范围: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        
        avg_loss = total_loss / n_batches
        
        # 计算F1
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # 调试信息：检查预测分布
        pred_positive_rate = all_preds.mean()
        label_positive_rate = all_labels.mean()
        logger.info(f"  [调试] 预测正例比例: {pred_positive_rate:.4f}, 真实正例比例: {label_positive_rate:.4f}")
        
        # 详细分析：每个类别的情况
        from sklearn.metrics import f1_score, accuracy_score
        n_classes = all_labels.shape[1] if len(all_labels.shape) > 1 else 1
        
        if n_classes > 1:
            # 使用label_config中的类别名称
            class_names = self.label_config.get('classes', 
                ['frontal', 'temporal', 'central', 'parietal', 'occipital'])[:n_classes]
            class_f1s = []
            class_info = []
            for c in range(n_classes):
                c_f1 = f1_score(all_labels[:, c], all_preds[:, c], zero_division=0)
                c_pred_pos = all_preds[:, c].sum()  # 预测正样本数
                c_label_pos = all_labels[:, c].sum()  # 真实正样本数
                class_f1s.append(c_f1)
                class_info.append(f"{class_names[c] if c < len(class_names) else f'class{c}'}[{int(c_label_pos)}/{int(c_pred_pos)}]")
            logger.info(f"  [调试] 各类[真实正样本/预测正样本]: {', '.join(class_info)}")
            logger.info(f"  [调试] 各类F1: {[f'{f:.3f}' for f in class_f1s]}")
        
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, f1
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 20,
        save_every: int = 10,
        metric_for_best: str = 'f1'
    ) -> Dict:
        """完整训练流程"""
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_f1 = self.validate()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # 检查是否为最佳
            current_metric = val_f1 if metric_for_best == 'f1' else -val_loss
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                no_improve_count = 0
                
                # 保存最佳模型
                self._save_checkpoint(epoch, is_best=True)
            else:
                no_improve_count += 1
            
            # 日志
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch)
            
            # 早停
            if no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Best F1: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        else:
            path = self.checkpoint_dir / f"{self.experiment_name}_epoch{epoch+1}.pt"
        
        torch.save(checkpoint, path)


def get_patient_ids_from_manifest(manifest_path: str) -> List[str]:
    """从manifest获取患者ID列表"""
    import pandas as pd
    df = pd.read_csv(manifest_path)
    return df['pt_id'].unique().tolist()


def create_cross_validation_splits(
    manifest_path: str,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """创建交叉验证划分"""
    from sklearn.model_selection import KFold
    
    patient_ids = get_patient_ids_from_manifest(manifest_path)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for train_idx, val_idx in kf.split(patient_ids):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        splits.append((train_ids, val_ids))
    
    return splits


def train_fold(
    fold: int,
    train_ids: List[str],
    val_ids: List[str],
    config: Config,
    args,
    checkpoint_dir: str = None  # 新增：统一的检查点目录
) -> Dict:
    """训练单个fold"""
    
    # 获取标签配置
    label_config = get_label_config(args.label_type, args.exclude_classes)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练 Fold {fold} (Multi-Branch Fusion)")
    logger.info(f"训练患者: {len(train_ids)}, 验证患者: {len(val_ids)}")
    logger.info(f"融合策略: {args.fusion_type}")
    logger.info(f"标签类型: {label_config['name']} ({args.label_type})")
    logger.info(f"类别数: {label_config['n_classes']}")
    logger.info(f"类别: {label_config['classes']}")
    if args.exclude_classes:
        logger.info(f"排除类别: {args.exclude_classes}")
    logger.info(f"{'='*60}")
    
    # 确定连接性类型
    if args.include_directed:
        connectivity_types = ['plv', 'wpli', 'aec', 'pearson', 'granger', 'transfer_entropy']
    else:
        connectivity_types = ['plv', 'wpli', 'aec', 'pearson']
    
    # 根据是否有预处理数据选择加载方式
    if args.preprocessed_dir is not None:
        # 从预处理数据加载（快速）
        logger.info(f"从预处理数据加载: {args.preprocessed_dir}")
        
        train_dataset = PreprocessedDataset(
            preprocessed_dir=args.preprocessed_dir,
            patient_ids=train_ids,
            label_type=args.label_type,
            label_config=label_config,
            manifest_path=config.data.manifest_path  # 传入manifest用于准确的chain标签
        )
        
        val_dataset = PreprocessedDataset(
            preprocessed_dir=args.preprocessed_dir,
            patient_ids=val_ids,
            label_type=args.label_type,
            label_config=label_config,
            manifest_path=config.data.manifest_path  # 传入manifest用于准确的chain标签
        )
        
        # 读取并显示预处理配置信息
        preprocess_config = train_dataset.config_info
        if preprocess_config:
            logger.info(f"预处理数据配置:")
            logger.info(f"  use_21_channels: {preprocess_config.get('use_21_channels', False)}")
            logger.info(f"  use_bipolar: {preprocess_config.get('use_bipolar', False)}")
            logger.info(f"  label_type (预处理时): {preprocess_config.get('label_type', 'unknown')}")
            logger.info(f"  segment_length: {preprocess_config.get('segment_length', 'unknown')}s")
            logger.info(f"  window_length: {preprocess_config.get('window_length', 'unknown')}s")
            logger.info(f"  connectivity_types: {preprocess_config.get('connectivity_types', [])}")
            
            # 验证 region_5 标签类型需要双极导联数据
            if args.label_type == 'region_5':
                if not preprocess_config.get('use_bipolar', False):
                    logger.warning("警告: region_5 标签类型需要使用双极导联数据(26通道)")
                    logger.warning("当前预处理数据可能不包含双极导联。请使用 --use-bipolar 重新预处理")
                if not preprocess_config.get('use_21_channels', False):
                    logger.warning("警告: region_5 标签类型需要21电极配置(包含SPHL/SPHR)")
                    logger.warning("当前预处理数据可能缺少SPHL/SPHR电极。请使用 --use-21-channels 重新预处理")
    else:
        # 实时处理数据（慢但灵活）
        logger.info("实时处理数据...")
        
        # 当使用 region_5 时，确保启用 21 电极和双极导联配置
        if args.label_type == 'region_5':
            if not getattr(config.data, 'use_21_channels', False):
                logger.info("region_5 标签需要21电极配置，自动启用 use_21_channels=True")
                config.data.use_21_channels = True
            if not getattr(config.data, 'use_bipolar', False):
                logger.info("region_5 标签需要双极导联，自动启用 use_bipolar=True")
                config.data.use_bipolar = True
        
        logger.info(f"实时处理配置:")
        logger.info(f"  use_21_channels: {getattr(config.data, 'use_21_channels', False)}")
        logger.info(f"  use_bipolar: {getattr(config.data, 'use_bipolar', False)}")
        
        # 创建训练数据集
        train_dataset = MultiModalEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type=args.label_type,  # 使用指定的标签类型
            patient_ids=train_ids,
            config=config.data,
            segment_length=args.segment_length,
            segment_overlap=args.segment_overlap,
            window_length=args.window_length,
            connectivity_types=connectivity_types,
            connectivity_freq_band=tuple(args.freq_band),
            compute_online=True,
            include_directed=args.include_directed
        )
        
        # 创建验证数据集 (无重叠)
        val_dataset = MultiModalEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type=args.label_type,  # 使用指定的标签类型
            patient_ids=val_ids,
            config=config.data,
            segment_length=args.segment_length,
            segment_overlap=0.0,  # 验证集不重叠
            window_length=args.window_length,
            connectivity_types=connectivity_types,
            connectivity_freq_band=tuple(args.freq_band),
            compute_online=True,
            include_directed=args.include_directed
        )
    
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = create_multimodal_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_multimodal_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 计算窗口数
    n_windows = int(args.segment_length / args.window_length)
    n_connectivity_types = len(connectivity_types)
    
    # 动态获取实际数据维度
    # 从数据集中获取第一个样本来确定所有维度
    sample_data = train_dataset[0]
    eeg_shape = sample_data['eeg_data'].shape  # 原始单极数据: (n_windows, n_channels_raw, n_samples)
    conn_shape = sample_data['connectivity'].shape  # 基于原始数据: (n_connectivity_types, n_channels_raw, n_channels_raw)
    graph_shape = sample_data['graph_metrics'].shape  # 基于原始数据: (n_channels_raw, n_graph_features)
    labels_shape = sample_data['labels'].shape  # (n_classes,)
    
    # 检查是否有双极导联相关数据
    has_bipolar = 'bipolar_eeg_data' in sample_data
    has_bipolar_conn = 'bipolar_connectivity' in sample_data
    has_bipolar_graph = 'bipolar_graph_metrics' in sample_data
    
    bipolar_eeg_shape = None
    bipolar_conn_shape = None
    bipolar_graph_shape = None
    
    if has_bipolar:
        bipolar_eeg_shape = sample_data['bipolar_eeg_data'].shape
    if has_bipolar_conn:
        bipolar_conn_shape = sample_data['bipolar_connectivity'].shape
    if has_bipolar_graph:
        bipolar_graph_shape = sample_data['bipolar_graph_metrics'].shape
    
    # 从实际数据中提取维度
    actual_n_windows = eeg_shape[0]
    n_channels_raw = eeg_shape[1]  # 原始单极通道数
    actual_n_samples = eeg_shape[2]
    n_graph_features = graph_shape[1] if len(graph_shape) > 1 else 5
    
    # 原始数据的 connectivity 和 graph_metrics 通道数
    conn_channels_raw = conn_shape[1] if len(conn_shape) >= 3 else n_channels_raw
    graph_channels_raw = graph_shape[0] if len(graph_shape) >= 2 else n_channels_raw
    
    # 确定是否使用双极导联数据训练
    # 条件：有完整的双极导联数据（eeg, connectivity, graph_metrics）
    use_bipolar_for_training = has_bipolar and has_bipolar_conn and has_bipolar_graph
    
    if use_bipolar_for_training:
        n_channels_bipolar = bipolar_eeg_shape[1]
        conn_channels_bipolar = bipolar_conn_shape[1] if len(bipolar_conn_shape) >= 3 else n_channels_bipolar
        graph_channels_bipolar = bipolar_graph_shape[0] if len(bipolar_graph_shape) >= 2 else n_channels_bipolar
        
        # 验证双极导联数据的通道数一致性
        if n_channels_bipolar != conn_channels_bipolar or n_channels_bipolar != graph_channels_bipolar:
            logger.warning(f"双极导联数据通道数不一致: EEG={n_channels_bipolar}, Conn={conn_channels_bipolar}, Graph={graph_channels_bipolar}")
        
        n_channels = n_channels_bipolar  # 模型使用双极导联通道数
    else:
        n_channels = n_channels_raw  # 模型使用原始通道数
    
    # 使用label_config的n_classes（更可靠）
    n_classes = label_config['n_classes']
    
    # 验证原始数据通道维度一致性
    if conn_channels_raw != graph_channels_raw:
        logger.error(f"原始数据通道维度不一致！")
        logger.error(f"  Connectivity通道数: {conn_channels_raw}")
        logger.error(f"  Graph Metrics通道数: {graph_channels_raw}")
        raise ValueError("数据通道维度不一致，请检查预处理数据")
    
    # 确定通道类型描述
    if n_channels == 26:
        channel_desc = "26通道双极导联 (21电极配置)"
    elif n_channels == 18:
        channel_desc = "18通道双极导联 (19电极配置)"
    elif n_channels == 21:
        channel_desc = "21通道单极 (含SPHL/SPHR)"
    elif n_channels == 19:
        channel_desc = "19通道单极 (标准配置)"
    else:
        channel_desc = f"{n_channels}通道 (非标准配置)"
    
    logger.info(f"数据维度检测:")
    logger.info(f"  模型输入通道配置: {channel_desc}")
    logger.info(f"  原始单极数据: {eeg_shape}")
    logger.info(f"  原始Connectivity: {conn_shape}")
    logger.info(f"  原始Graph Metrics: {graph_shape}")
    if has_bipolar:
        logger.info(f"  双极导联数据: {bipolar_eeg_shape}")
    if has_bipolar_conn:
        logger.info(f"  双极导联Connectivity: {bipolar_conn_shape}")
    if has_bipolar_graph:
        logger.info(f"  双极导联Graph Metrics: {bipolar_graph_shape}")
    logger.info(f"  Labels: n_classes={n_classes} (来自label_config)")
    logger.info(f"  训练使用数据: {'双极导联数据' if use_bipolar_for_training else '原始单极数据'}")
    
    # 对于 region_5 标签类型，验证通道配置
    if args.label_type == 'region_5':
        if use_bipolar_for_training and n_channels == 26:
            logger.info(f"  ✓ region_5 使用26通道双极导联数据训练，脑区标签基于双极导联定义")
        else:
            logger.warning(f"  ⚠ region_5 建议使用26通道双极导联数据，当前为 {n_channels} 通道")
            logger.warning(f"    请确保预处理时启用了 use_21_channels=True 和 use_bipolar=True")
    
    # 从预处理数据的config_info中获取连接性类型数量（如果有）
    if args.preprocessed_dir is not None and hasattr(train_dataset, 'config_info'):
        preprocess_config = train_dataset.config_info
        connectivity_types_from_config = preprocess_config.get('connectivity_types', [])
        if connectivity_types_from_config:
            n_connectivity_types = len(connectivity_types_from_config)
            logger.info(f"从预处理配置获取连接性类型数: {n_connectivity_types}")
    else:
        # 从实际数据中获取
        n_connectivity_types = conn_shape[0]
    
    # 创建模型 - 使用实际数据维度
    # 注意：所有三个分支使用相同的 n_channels，确保维度匹配
    model_config = {
        'n_channels': n_channels,  # 从数据获取（26通道双极导联或其他）
        'n_samples': actual_n_samples,  # 从数据获取，而不是计算
        'n_windows': actual_n_windows,  # 从数据获取
        'n_classes': n_classes,  # 从标签配置获取
        'n_connectivity_types': n_connectivity_types,  # 连接性矩阵类型数
        'n_graph_features': n_graph_features,  # 图指标数量
        'fusion_feature_dim': args.fusion_feature_dim,  # 各分支统一输出维度
        'fusion_type': args.fusion_type,
        'fusion_dim': args.fusion_dim,
        'dropout': args.dropout,
        'eegnet': {
            'F1': args.F1,
            'D': args.D,
            'F2': args.F2,
            'kernel_length': args.kernel_length,
            'temporal_aggregation': 'attention'
        },
        'gat': {
            'hidden_dim': args.gat_hidden,
            'n_heads': args.gat_heads,
            'n_layers': args.gat_layers
        }
    }
    
    logger.info(f"模型配置:")
    logger.info(f"  三分支输入通道数: {n_channels} (EEGNet/GAT/MLP共用)")
    logger.info(f"  EEGNet输入: ({actual_n_windows}, {n_channels}, {actual_n_samples})")
    logger.info(f"  GAT输入: ({n_connectivity_types}, {n_channels}, {n_channels})")
    logger.info(f"  MLP输入: ({n_channels}, {n_graph_features})")
    logger.info(f"  各分支输出维度: {args.fusion_feature_dim}")
    logger.info(f"  融合后维度: {args.fusion_dim}")
    logger.info(f"  输出类别数: {n_classes}")
    
    model = create_multi_branch_model(model_config)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    # 如果是dry-run，验证模型后返回
    if args.dry_run:
        logger.info("Dry-run模式: 验证模型前向传播...")
        model = model.to(args.device)
        for batch in train_loader:
            # 根据配置选择使用原始数据还是双极导联数据
            if use_bipolar_for_training and 'bipolar_eeg_data' in batch:
                eeg = batch['bipolar_eeg_data'].to(args.device)
                conn = batch['bipolar_connectivity'].to(args.device)
                graph = batch['bipolar_graph_metrics'].to(args.device)
            else:
                eeg = batch['eeg_data'].to(args.device)
                conn = batch['connectivity'].to(args.device)
                graph = batch['graph_metrics'].to(args.device)
            
            with torch.no_grad():
                out = model(eeg, conn, graph)
            
            logger.info(f"输入: EEG {eeg.shape}, Conn {conn.shape}, Graph {graph.shape}")
            logger.info(f"输出: {out.shape}")
            break
        
        return {'fold': fold, 'status': 'dry-run'}
    
    # 创建损失函数
    # 处理类别权重
    if args.class_weights is not None:
        class_weights = args.class_weights
        # 确保权重数量与类别数匹配
        if len(class_weights) != n_classes:
            logger.warning(f"类别权重数量({len(class_weights)})与类别数({n_classes})不匹配，使用统一权重")
            class_weights = None
    else:
        class_weights = None
    
    # 根据损失类型创建
    if args.loss_type == 'focal':
        # Focal Loss 使用 alpha 作为权重
        focal_alpha = class_weights if class_weights is not None else args.focal_alpha
        criterion = OnsetZoneLoss(
            loss_type='focal',
            focal_alpha=focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            n_classes=n_classes
        )
        logger.info(f"使用 Focal Loss: gamma={args.focal_gamma}, alpha={focal_alpha}")
    else:
        # BCE 使用 pos_weight 作为权重
        pos_weight = class_weights if class_weights is not None else args.pos_weight
        criterion = OnsetZoneLoss(
            loss_type=args.loss_type,
            pos_weight=pos_weight,
            label_smoothing=args.label_smoothing,
            n_classes=n_classes
        )
        logger.info(f"使用 {args.loss_type} Loss: pos_weight={pos_weight}")
    
    # 创建优化器
    optimizer = create_optimizer(
        model,
        optimizer_name='adamw',
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建调度器
    scheduler = create_scheduler(
        optimizer,
        scheduler_name='cosine',
        num_epochs=args.epochs
    )
    
    # 实验名称 - 使用fold编号区分，不再添加时间戳（时间戳在目录层级）
    experiment_name = args.experiment_name or f'multi_branch_{args.fusion_type}'
    experiment_name = f'{experiment_name}_fold{fold}'
    
    # 使用统一的检查点目录（包含时间戳）
    if checkpoint_dir is None:
        # 兼容旧的调用方式：如果没有传入checkpoint_dir，则使用默认逻辑
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = str(Path(config.training.checkpoint_dir) / 'multi_branch' / timestamp)
    
    # 创建训练器
    trainer = MultiBranchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        use_amp=True,
        grad_clip_norm=1.0,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        label_config=label_config,  # 传递标签配置
        use_bipolar=use_bipolar_for_training  # 是否使用双极导联数据训练
    )
    
    # 训练
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_every=10,
        metric_for_best='f1'
    )
    
    return {
        'fold': fold,
        'best_metric': trainer.best_metric,
        'best_epoch': trainer.best_epoch,
        'history': history
    }


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = get_config()
    
    # 覆盖配置
    if args.manifest:
        config.data.manifest_path = args.manifest
    if args.data_roots:
        config.data.edf_data_roots = args.data_roots
    
    # 设置设备
    device = get_device(args.device == 'cuda')
    args.device = device
    logger.info(f"使用设备: {device}")
    
    # 打印配置
    logger.info(f"\n多分支融合模型配置:")
    logger.info(f"  融合策略: {args.fusion_type}")
    logger.info(f"  片段长度: {args.segment_length}s, 重叠: {args.segment_overlap*100:.0f}%")
    logger.info(f"  EEGNet: F1={args.F1}, D={args.D}, F2={args.F2}")
    logger.info(f"  GAT: hidden={args.gat_hidden}, heads={args.gat_heads}, layers={args.gat_layers}")
    logger.info(f"  有向连接性: {args.include_directed}")
    
    # 创建交叉验证划分
    splits = create_cross_validation_splits(
        config.data.manifest_path,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    logger.info(f"数据划分完成: {args.n_folds} 折交叉验证")
    
    # 创建统一的时间戳目录，用于保存本次训练的所有fold模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path(config.training.checkpoint_dir) / 'multi_branch' / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型保存目录: {checkpoint_dir}")
    
    # 训练
    results = []
    
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))
    
    for fold in folds_to_train:
        train_ids, val_ids = splits[fold]
        result = train_fold(fold, train_ids, val_ids, config, args, checkpoint_dir=str(checkpoint_dir))
        results.append(result)
    
    # 汇总结果
    if not args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("训练完成 - 多分支融合模型")
        logger.info("="*60)
        
        best_metrics = [r['best_metric'] for r in results]
        logger.info(f"各Fold最佳F1-Macro: {best_metrics}")
        logger.info(f"平均F1-Macro: {sum(best_metrics)/len(best_metrics):.4f}")
        
        if len(best_metrics) > 1:
            std = (sum((x - sum(best_metrics)/len(best_metrics))**2 for x in best_metrics) / len(best_metrics)) ** 0.5
            logger.info(f"标准差: {std:.4f}")


if __name__ == '__main__':
    main()
