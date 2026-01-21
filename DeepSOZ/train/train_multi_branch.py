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
from losses import OnsetZoneLoss
from multi_branch_model import MultiBranchFusionModel, create_multi_branch_model
from trainer import create_optimizer, create_scheduler, set_seed, get_device, count_parameters

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多分支融合模型训练')
    
    # 数据
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    
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
    parser.add_argument('--fusion-dim', type=int, default=128,
                        help='融合层维度')
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
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    
    # 损失函数
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['bce', 'focal', 'dice'],
                        help='损失函数类型')
    parser.add_argument('--pos-weight', type=float, default=2.0,
                        help='正类权重')
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
        experiment_name: str = 'multi_branch'
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
            # 获取数据
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
            
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / n_batches
        
        # 计算F1
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        from sklearn.metrics import f1_score
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
    args
) -> Dict:
    """训练单个fold"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练 Fold {fold} (Multi-Branch Fusion)")
    logger.info(f"训练患者: {len(train_ids)}, 验证患者: {len(val_ids)}")
    logger.info(f"融合策略: {args.fusion_type}")
    logger.info(f"{'='*60}")
    
    # 确定连接性类型
    if args.include_directed:
        connectivity_types = ['plv', 'wpli', 'aec', 'pearson', 'granger', 'transfer_entropy']
    else:
        connectivity_types = ['plv', 'wpli', 'aec', 'pearson']
    
    # 创建训练数据集
    train_dataset = MultiModalEEGDataset(
        manifest_path=config.data.manifest_path,
        data_roots=config.data.edf_data_roots,
        label_type='onset_zone',
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
        label_type='onset_zone',
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
    
    # 动态获取实际通道数（支持额外通道如sph-l, sph-r）
    # 从数据集中获取第一个样本来确定通道数
    sample_data = train_dataset[0]
    n_channels = sample_data['eeg_data'].shape[1]  # (n_windows, n_channels, n_samples)
    n_classes = sample_data['labels'].shape[0]  # 标签维度
    logger.info(f"检测到通道数: {n_channels}, 分类数: {n_classes}")
    
    # 创建模型
    model_config = {
        'n_channels': n_channels,  # 动态通道数
        'n_samples': int(config.data.target_fs * args.window_length),
        'n_windows': n_windows,
        'n_classes': n_classes,  # 动态分类数
        'n_connectivity_types': n_connectivity_types,
        'n_graph_features': 5,
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
            'output_dim': 64,
            'n_heads': args.gat_heads,
            'n_layers': args.gat_layers
        }
    }
    
    model = create_multi_branch_model(model_config)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    # 如果是dry-run，验证模型后返回
    if args.dry_run:
        logger.info("Dry-run模式: 验证模型前向传播...")
        model = model.to(args.device)
        for batch in train_loader:
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
    criterion = OnsetZoneLoss(
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        label_smoothing=args.label_smoothing
    )
    
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
    
    # 实验名称
    experiment_name = args.experiment_name or f'multi_branch_{args.fusion_type}_fold{fold}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'{experiment_name}_{timestamp}'
    
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
        checkpoint_dir=str(Path(config.training.checkpoint_dir) / 'multi_branch'),
        experiment_name=experiment_name
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
    
    # 训练
    results = []
    
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))
    
    for fold in folds_to_train:
        train_ids, val_ids = splits[fold]
        result = train_fold(fold, train_ids, val_ids, config, args)
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
