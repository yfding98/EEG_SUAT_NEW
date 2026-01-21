#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEGNet onset_zone 脑区级别分类训练脚本

使用经典EEGNet模型进行脑区级别的SOZ定位（5类多标签分类）：
- frontal
- temporal  
- central
- parietal
- occipital
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from dataset import PrivateEEGDataset, create_cross_validation_splits, create_dataloader
from losses import OnsetZoneLoss
from eegnet_model import EEGNetOnsetZone, create_eegnet_model
from trainer import Trainer, create_optimizer, create_scheduler, set_seed, get_device, count_parameters

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EEGNet onset_zone脑区级别分类训练')
    
    # 数据
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    
    # EEGNet模型参数
    parser.add_argument('--F1', type=int, default=8,
                        help='时间卷积滤波器数量')
    parser.add_argument('--D', type=int, default=2,
                        help='深度乘数')
    parser.add_argument('--F2', type=int, default=16,
                        help='分离卷积滤波器数量')
    parser.add_argument('--kernel-length', type=int, default=64,
                        help='时间卷积核长度')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')
    parser.add_argument('--temporal-agg', type=str, default='attention',
                        choices=['mean', 'max', 'attention'],
                        help='时间窗口聚合方式')
    
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
    parser.add_argument('--loss-type', type=str, default='bce',
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
                        help='指定运行某一折（None表示运行所有折）')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--bipolar', action='store_true',
                        help='使用TCP双极导联(通道数从19变为18)')
    
    return parser.parse_args()


def train_fold(
    fold: int,
    train_ids: list,
    val_ids: list,
    config: Config,
    args
) -> dict:
    """
    训练单个fold
    
    Args:
        fold: fold编号
        train_ids: 训练患者ID列表
        val_ids: 验证患者ID列表
        config: 配置
        args: 命令行参数
    
    Returns:
        训练结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练 Fold {fold} (EEGNet)")
    logger.info(f"训练患者: {len(train_ids)}, 验证患者: {len(val_ids)}")
    logger.info(f"{'='*60}")
    
    # 创建数据集
    train_dataset = PrivateEEGDataset(
        manifest_path=config.data.manifest_path,
        data_roots=config.data.edf_data_roots,
        label_type='onset_zone',
        patient_ids=train_ids,
        config=config.data
    )
    
    val_dataset = PrivateEEGDataset(
        manifest_path=config.data.manifest_path,
        data_roots=config.data.edf_data_roots,
        label_type='onset_zone',
        patient_ids=val_ids,
        config=config.data
    )
    
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建EEGNet模型
    model_config = {
        'n_channels': config.model.n_channels,
        'n_samples': config.model.time_steps,
        'n_windows': config.data.n_windows,
        'n_classes': 5,  # onset_zone 5个脑区
        'dropout': args.dropout,
        'F1': args.F1,
        'D': args.D,
        'F2': args.F2,
        'kernel_length': args.kernel_length,
        'temporal_aggregation': args.temporal_agg
    }
    model = create_eegnet_model('onset_zone', model_config)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"EEGNet模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    # 创建损失函数
    criterion = OnsetZoneLoss(
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        label_smoothing=args.label_smoothing
    )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(
        model,
        optimizer_name='adamw',
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name='cosine',
        num_epochs=args.epochs
    )
    
    # 实验名称
    experiment_name = args.experiment_name or f'eegnet_onset_zone_fold{fold}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'{experiment_name}_{timestamp}'
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        task_type='multilabel',
        use_amp=True,
        grad_clip_norm=1.0,
        checkpoint_dir=str(Path(config.training.checkpoint_dir) / 'eegnet_onset_zone'),
        log_dir=str(Path(config.training.log_dir) / 'eegnet_onset_zone'),
        experiment_name=experiment_name
    )
    
    # 训练
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_every=10,
        metric_for_best='f1_macro'
    )
    
    return {
        'fold': fold,
        'best_metric': trainer.best_metric,
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
    
    # 设置双极导联模式
    # 命令行参数可以覆盖config默认设置
    if args.bipolar:
        config.data.use_bipolar = True
    
    # 根据config中的use_bipolar设置正确的通道数
    if config.data.use_bipolar:
        config.model.n_channels = 18
        logger.info("使用TCP双极导联模式，通道数: 18")
    else:
        config.model.n_channels = 19
        logger.info("使用单极导联模式，通道数: 19")
    
    # 设置设备
    device = get_device(args.device == 'cuda')
    args.device = device
    logger.info(f"使用设备: {device}")
    
    # 创建交叉验证划分
    splits = create_cross_validation_splits(
        config.data.manifest_path,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    logger.info(f"数据划分完成: {args.n_folds} 折交叉验证")
    
    # 打印EEGNet模型配置
    logger.info(f"\nEEGNet配置:")
    logger.info(f"  F1={args.F1}, D={args.D}, F2={args.F2}")
    logger.info(f"  kernel_length={args.kernel_length}")
    logger.info(f"  dropout={args.dropout}")
    logger.info(f"  temporal_aggregation={args.temporal_agg}")
    
    # 训练
    results = []
    
    if args.fold is not None:
        # 只训练指定fold
        folds_to_train = [args.fold]
    else:
        # 训练所有fold
        folds_to_train = list(range(args.n_folds))
    
    for fold in folds_to_train:
        train_ids, val_ids = splits[fold]
        result = train_fold(fold, train_ids, val_ids, config, args)
        results.append(result)
    
    # 汇总结果
    logger.info("\n" + "="*60)
    logger.info("训练完成 - EEGNet onset_zone 脑区级别分类")
    logger.info("="*60)
    
    best_metrics = [r['best_metric'] for r in results]
    logger.info(f"各Fold最佳F1-Macro: {best_metrics}")
    logger.info(f"平均F1-Macro: {sum(best_metrics)/len(best_metrics):.4f}")
    logger.info(f"标准差: {(sum((x-sum(best_metrics)/len(best_metrics))**2 for x in best_metrics)/len(best_metrics))**0.5:.4f}")


if __name__ == '__main__':
    main()
