#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel 通道级别分类训练脚本

训练STGNN模型进行通道级别的SOZ定位（19通道多标签分类）
使用DeepSOZ风格的损失函数组合
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
from losses import ChannelLoss, get_loss_function
from model_wrapper import create_model
from trainer import Trainer, create_optimizer, create_scheduler, set_seed, get_device, count_parameters

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='channel通道级别分类训练')
    
    # 数据
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    
    # 模型
    parser.add_argument('--temporal-hidden-dim', type=int, default=32,
                        help='时间特征隐藏维度')
    parser.add_argument('--graph-hidden-dim', type=int, default=32,
                        help='图特征隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout率')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    
    # 损失函数
    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['bce', 'focal', 'dice'],
                        help='损失函数类型')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='正类权重')
    parser.add_argument('--map-pos-weight', type=float, default=1.0,
                        help='Map正类损失权重')
    parser.add_argument('--map-neg-weight', type=float, default=0.5,
                        help='Map负类损失权重')
    parser.add_argument('--map-margin-weight', type=float, default=0.5,
                        help='Map margin损失权重')
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
    parser.add_argument('--visualize', action='store_true',
                        help='训练后生成样本级别可视化报告')
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
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练 Fold {fold}")
    logger.info(f"训练患者: {len(train_ids)}, 验证患者: {len(val_ids)}")
    logger.info(f"{'='*60}")
    
    # 创建数据集
    train_dataset = PrivateEEGDataset(
        manifest_path=config.data.manifest_path,
        data_roots=config.data.edf_data_roots,
        label_type='channel',
        patient_ids=train_ids,
        config=config.data
    )
    
    val_dataset = PrivateEEGDataset(
        manifest_path=config.data.manifest_path,
        data_roots=config.data.edf_data_roots,
        label_type='channel',
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
    
    # 创建模型
    model_config = {
        'n_channels': config.model.n_channels,
        'n_bands': config.model.n_bands,
        'time_steps': config.model.time_steps,
        'n_windows': config.data.n_windows,
        'temporal_hidden_dim': args.temporal_hidden_dim,
        'graph_hidden_dim': args.graph_hidden_dim,
        'dropout': args.dropout
    }
    model = create_model('channel', model_config)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数: {total_params:,} 总计, {trainable_params:,} 可训练")
    
    # 创建损失函数
    criterion = ChannelLoss(
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        map_pos_weight=args.map_pos_weight,
        map_neg_weight=args.map_neg_weight,
        map_margin_weight=args.map_margin_weight,
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
    experiment_name = args.experiment_name or f'channel_fold{fold}'
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
        task_type='multilabel',  # 通道级别是多标签分类
        use_amp=True,
        grad_clip_norm=1.0,
        checkpoint_dir=str(Path(config.training.checkpoint_dir) / 'channel'),
        log_dir=str(Path(config.training.log_dir) / 'channel'),
        experiment_name=experiment_name
    )
    
    # 训练
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_every=10,
        metric_for_best='f1_macro'  # 对于多标签使用F1-Macro
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
    if args.bipolar:
        config.data.use_bipolar = True
        config.model.n_channels = 18  # 双极导联后通道数变为18
        logger.info("启用TCP双极导联模式，通道数: 18")
    
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
    logger.info("\n" + "="*60)
    logger.info("训练完成 - channel 通道级别分类")
    logger.info("="*60)
    
    best_metrics = [r['best_metric'] for r in results]
    logger.info(f"各Fold最佳F1-Macro: {best_metrics}")
    logger.info(f"平均F1-Macro: {sum(best_metrics)/len(best_metrics):.4f}")
    logger.info(f"标准差: {(sum((x-sum(best_metrics)/len(best_metrics))**2 for x in best_metrics)/len(best_metrics))**0.5:.4f}")
    
    # 生成样本级别可视化报告
    if args.visualize:
        logger.info("\n生成样本级别可视化报告...")
        try:
            from evaluate_samples import run_evaluation
            
            for fold in folds_to_train:
                checkpoint_dir = Path(config.training.checkpoint_dir) / 'channel'
                # 查找最佳模型
                best_checkpoints = list(checkpoint_dir.glob(f'*fold{fold}*best*.pth'))
                if best_checkpoints:
                    checkpoint_path = str(best_checkpoints[0])
                    output_dir = str(checkpoint_dir / f'evaluation_fold{fold}')
                    
                    run_evaluation(
                        checkpoint_path=checkpoint_path,
                        task_type='channel',
                        output_dir=output_dir,
                        manifest_path=config.data.manifest_path,
                        data_roots=config.data.edf_data_roots,
                        fold=fold,
                        n_folds=args.n_folds,
                        batch_size=args.batch_size,
                        device=args.device,
                        seed=args.seed
                    )
                    logger.info(f"Fold {fold} 可视化报告已生成: {output_dir}")
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
