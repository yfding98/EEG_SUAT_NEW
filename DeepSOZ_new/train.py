#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ_new 两阶段训练脚本

两阶段训练流程（忠实复现官方 DeepSOZ）：
  Stage-1 (--stage 1 或 --stage both)
    模型  : TransformerLSTM
    任务  : 时序癫痫检测（每 1s 窗口二分类）
    损失  : CrossEntropyLoss(weight=[0.2, 0.8])
    默认优化: Adam, lr=1e-5, 30 epochs
    数据  : Stage1Dataset（npy）或 OnlineStage1Dataset（EDF）

  Stage-2 (--stage 2 或 --stage both)
    模型  : DeepSOZLocator (ctg_11_8)
    任务  : 通道级 SOZ 定位（19 通道多标签）
    损失  : Stage2SOZLoss（CE + MapLoss 组合）
    默认优化: Adam, lr=1e-4, 50 epochs
    数据  : Stage2Dataset（npy）或 OnlineStage2Dataset（EDF）

数据模式（--data-mode）：
  online  : 直接读取 EDF 文件（适合私有数据集，无需预处理）
  offline : 读取预处理好的 .npy 文件（适合 TUSZ 大规模数据集）

示例：
  # 两阶段全流程（在线 EDF 模式，5 折）
  python train.py \\
      --manifest /data/manifest.csv \\
      --data-roots /data/edf \\
      --stage both \\
      --n-folds 5 \\
      --output-dir runs/channel_cv

  # 仅 Stage-2（离线 npy 模式，单折）
  python train.py \\
      --manifest /data/manifest.csv \\
      --data-roots /data/npy \\
      --data-mode offline \\
      --stage 2 \\
      --fold 0 \\
      --stage2-epochs 50 \\
      --output-dir runs/s2_fold0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    OFFICIAL_19_CHANNELS, TCP_BIPOLAR_COLUMNS, TCP_BIPOLAR_NAMES,
    Stage1Dataset, Stage2Dataset,
    OnlineStage1Dataset, OnlineStage2Dataset,
    make_kfold_splits, make_dataloader, read_manifest_csv,
    get_patient_ids,
)
from deepsoz_model import build_stage1_model, build_stage2_model
from losses import Stage2SOZLoss
from trainer import (
    Stage1Trainer, Stage2Trainer,
    count_parameters, set_seed,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

LABEL_NAMES_19 = [c.upper() for c in OFFICIAL_19_CHANNELS]
LABEL_NAMES_BIPOLAR = TCP_BIPOLAR_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='DeepSOZ_new 两阶段训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据
    g = p.add_argument_group('数据')
    g.add_argument('--manifest',    required=True,
                   help='manifest CSV 路径（combined_manifest.csv）')
    g.add_argument('--data-roots',  nargs='+', required=True,
                   help='数据根目录（EDF 或 npy）')
    g.add_argument('--data-mode',   default='online',
                   choices=['online', 'offline'],
                   help='online=实时读 EDF；offline=读预处理 npy')
    g.add_argument('--source',      default=None,
                   choices=['tusz', 'private'],
                   help='数据源过滤：tusz=仅TUSZ，private=仅私有，'
                        '不指定=混合使用全部')
    g.add_argument('--use-bipolar', action='store_true',
                   help='使用 TCP 22 双极导联（默认 19 单极）')
    g.add_argument('--n-windows',   type=int,   default=45,
                   help='Stage-1 时间窗口数（Stage-2 固定为 45）')
    g.add_argument('--target-fs',   type=float, default=200.0)
    g.add_argument('--f-low',       type=float, default=1.6)
    g.add_argument('--f-high',      type=float, default=30.0)
    g.add_argument('--cache-dir',   default='./cache/deepsoz_edf',
                   help='EDF 预处理缓存目录。启用后将预处理结果缓存为 .npz，'
                        '后续 epoch 直接加载，大幅加速训练。'
                        '设为 None 可禁用缓存')

    # 训练阶段
    g = p.add_argument_group('训练阶段')
    g.add_argument('--stage',       default='both',
                   choices=['1', '2', 'both'],
                   help='训练哪个阶段（1=仅检测预训练，2=仅SOZ定位，both=两阶段）')
    g.add_argument('--stage1-ckpt', default=None,
                   help='Stage-1 预训练 checkpoint（仅跑 stage=2 时加载）')

    # 模型
    g = p.add_argument_group('模型')
    g.add_argument('--tf-dropout',  type=float, default=0.15,
                   help='Transformer dropout（官方 0.15）')
    g.add_argument('--cnn-dropout', type=float, default=0.15,
                   help='Stage-2 CNN dropout（官方 0.15）')
    g.add_argument('--gru-dropout', type=float, default=0.0,
                   help='Stage-2 GRU dropout（官方 0.0）')

    # Stage-1 超参
    g = p.add_argument_group('Stage-1 超参')
    g.add_argument('--stage1-lr',       type=float, default=1e-5,
                   help='Stage-1 学习率（官方 1e-5）')
    g.add_argument('--stage1-epochs',   type=int,   default=30,
                   help='Stage-1 最大 epoch（官方 30）')
    g.add_argument('--stage1-patience', type=int,   default=10)
    g.add_argument('--stage1-batch',    type=int,   default=1,
                   help='Stage-1 batch size（官方 1）')

    # Stage-2 超参
    g = p.add_argument_group('Stage-2 超参')
    g.add_argument('--stage2-lr',           type=float, default=1e-4,
                   help='Stage-2 学习率（官方 1e-4）')
    g.add_argument('--stage2-epochs',       type=int,   default=50,
                   help='Stage-2 最大 epoch（官方 50）')
    g.add_argument('--stage2-patience',     type=int,   default=15)
    g.add_argument('--stage2-batch',        type=int,   default=1,
                   help='Stage-2 batch size（官方 1）')
    g.add_argument('--chn-sz-weight',       type=float, default=1.0)
    g.add_argument('--tot-sz-weight',       type=float, default=1.0)
    g.add_argument('--attn-map-w-pos',      type=float, default=2.0)
    g.add_argument('--attn-map-w-neg',      type=float, default=1.0)
    g.add_argument('--attn-map-w-margin',   type=float, default=1.0)
    g.add_argument('--chn-map-w-pos',       type=float, default=2.0)
    g.add_argument('--chn-map-w-neg',       type=float, default=1.0)
    g.add_argument('--chn-map-w-margin',    type=float, default=1.0)

    # 通用训练
    g = p.add_argument_group('通用训练')
    g.add_argument('--grad-clip',    type=float, default=1.0)
    g.add_argument('--no-amp',       action='store_true')
    g.add_argument('--num-workers',  type=int,   default=4)
    g.add_argument('--device',       default='cuda')

    # 交叉验证
    g = p.add_argument_group('交叉验证')
    g.add_argument('--n-folds',      type=int, default=5)
    g.add_argument('--fold',         type=int, default=None,
                   help='只训练指定折（None = 全部折）')
    g.add_argument('--seed',         type=int, default=42)

    # 输出
    g = p.add_argument_group('输出')
    g.add_argument('--output-dir',   default='runs/deepsoz_new')
    g.add_argument('--exp-prefix',   default='deepsoz')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 数据集构建
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    args,
    train_ids: List,
    val_ids: List,
) -> Tuple:
    """
    根据 --data-mode 构建 (s1_train_loader, s1_val_loader,
                            s2_train_loader, s2_val_loader)
    """
    source      = args.source
    use_bipolar = args.use_bipolar

    if args.data_mode == 'offline':
        manifest = read_manifest_csv(args.manifest)

        s1_tr_ds = Stage1Dataset(
            data_root=args.data_roots[0],
            manifest=manifest,
            patient_ids=train_ids,
        )
        s1_va_ds = Stage1Dataset(
            data_root=args.data_roots[0],
            manifest=manifest,
            patient_ids=val_ids,
        )
        s2_tr_ds = Stage2Dataset(
            data_root=args.data_roots[0],
            manifest=manifest,
            patient_ids=train_ids,
            win_len=45,
        )
        s2_va_ds = Stage2Dataset(
            data_root=args.data_roots[0],
            manifest=manifest,
            patient_ids=val_ids,
            win_len=45,
        )
    else:
        # online EDF 模式
        common = dict(
            manifest_path=args.manifest,
            data_roots=args.data_roots,
            source=source,
            use_bipolar=use_bipolar,
            target_fs=args.target_fs,
            f_low=args.f_low,
            f_high=args.f_high,
            cache_dir=args.cache_dir,
        )
        s1_tr_ds = OnlineStage1Dataset(
            patient_ids=train_ids,
            n_windows=args.n_windows,
            **common,
        )
        s1_va_ds = OnlineStage1Dataset(
            patient_ids=val_ids,
            n_windows=args.n_windows,
            **common,
        )
        s2_tr_ds = OnlineStage2Dataset(
            patient_ids=train_ids,
            n_windows=args.n_windows,
            win_len=45,
            **common,
        )
        s2_va_ds = OnlineStage2Dataset(
            patient_ids=val_ids,
            n_windows=args.n_windows,
            win_len=45,
            **common,
        )

    nw = args.num_workers
    if args.cache_dir is not None and nw == 0 and args.data_mode == 'online':
        nw = 4
        logger.info(f'缓存已启用，自动设置 num_workers={nw}')
    s1_tr = make_dataloader(s1_tr_ds, batch_size=args.stage1_batch,
                            shuffle=True,  num_workers=nw)
    s1_va = make_dataloader(s1_va_ds, batch_size=args.stage1_batch,
                            shuffle=False, num_workers=nw)
    s2_tr = make_dataloader(s2_tr_ds, batch_size=args.stage2_batch,
                            shuffle=True,  num_workers=nw)
    s2_va = make_dataloader(s2_va_ds, batch_size=args.stage2_batch,
                            shuffle=False, num_workers=nw)

    return s1_tr, s1_va, s2_tr, s2_va


# ─────────────────────────────────────────────────────────────────────────────
# 单折训练
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(fold: int, train_ids: List, val_ids: List, args) -> dict:
    logger.info(f'\n{"="*60}')
    logger.info(f'Fold {fold}  train_pts={len(train_ids)}  val_pts={len(val_ids)}')
    logger.info(f'{"="*60}')

    set_seed(args.seed + fold)

    ckpt_dir = Path(args.output_dir) / f'fold{fold}'
    use_amp  = not args.no_amp
    n_channels = 22 if args.use_bipolar else 19

    s1_tr, s1_va, s2_tr, s2_va = build_datasets(args, train_ids, val_ids)
    logger.info(f'Stage-1 数据: train={len(s1_tr.dataset)}  '
                f'val={len(s1_va.dataset)}')
    logger.info(f'Stage-2 数据: train={len(s2_tr.dataset)}  '
                f'val={len(s2_va.dataset)}')

    result: dict = {'fold': fold}

    # 标签名称
    label_names = LABEL_NAMES_BIPOLAR if args.use_bipolar else LABEL_NAMES_19

    # ── Stage-1 ──────────────────────────────────────────────────────────
    s1_ckpt_path = None
    if args.stage in ('1', 'both'):
        if len(s1_tr.dataset) == 0:
            logger.warning(f'Fold {fold}: Stage-1 训练集为空，跳过')
        else:
            s1_model = build_stage1_model(
                n_channels=n_channels,
                transformer_dropout=args.tf_dropout,
                device=args.device,
            )
            total, trainable = count_parameters(s1_model)
            logger.info(f'Stage-1 参数: {total:,} 总 / {trainable:,} 可训练')

            s1_trainer = Stage1Trainer(
                model=s1_model,
                train_loader=s1_tr,
                val_loader=s1_va,
                lr=args.stage1_lr,
                n_epochs=args.stage1_epochs,
                patience=args.stage1_patience,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                device=args.device,
                ckpt_dir=str(ckpt_dir),
                exp_name=f'{args.exp_prefix}_fold{fold}_s1',
            )
            s1_trainer.train()
            s1_ckpt_path = str(
                ckpt_dir / f'{args.exp_prefix}_fold{fold}_s1_best.pth'
            )
            result['s1_best_metric'] = s1_trainer.best_metric
            result['s1_best_epoch']  = s1_trainer.best_epoch

    # ── Stage-2 ──────────────────────────────────────────────────────────
    if args.stage in ('2', 'both'):
        if len(s2_tr.dataset) == 0:
            logger.warning(f'Fold {fold}: Stage-2 训练集为空，跳过')
        else:
            s2_model = build_stage2_model(
                n_channels=n_channels,
                cnn_dropout=args.cnn_dropout,
                gru_dropout=args.gru_dropout,
                transformer_dropout=args.tf_dropout,
            )
            total, trainable = count_parameters(s2_model)
            logger.info(f'Stage-2 参数: {total:,} 总 / {trainable:,} 可训练')

            s2_criterion = Stage2SOZLoss(
                chn_sz_weight=args.chn_sz_weight,
                tot_sz_weight=args.tot_sz_weight,
                attn_map_weight_pos=args.attn_map_w_pos,
                attn_map_weight_neg=args.attn_map_w_neg,
                attn_map_weight_margin=args.attn_map_w_margin,
                chn_map_weight_pos=args.chn_map_w_pos,
                chn_map_weight_neg=args.chn_map_w_neg,
                chn_map_weight_margin=args.chn_map_w_margin,
            )

            # stage1_ckpt：优先使用本折 Stage-1 的 best，其次用命令行指定的
            stage1_ckpt = (
                s1_ckpt_path
                if s1_ckpt_path and Path(s1_ckpt_path).exists()
                else args.stage1_ckpt
            )

            s2_trainer = Stage2Trainer(
                model=s2_model,
                criterion=s2_criterion,
                train_loader=s2_tr,
                val_loader=s2_va,
                lr=args.stage2_lr,
                n_epochs=args.stage2_epochs,
                patience=args.stage2_patience,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                device=args.device,
                ckpt_dir=str(ckpt_dir),
                exp_name=f'{args.exp_prefix}_fold{fold}_s2',
                label_names=label_names,
                stage1_ckpt=stage1_ckpt,
            )
            s2_trainer.train()

            # 最终评估
            try:
                s2_trainer.load_best()
                val_metrics = s2_trainer.evaluate(s2_va, mc_samples=20)
                logger.info(f'Fold {fold} Stage-2 最优验证指标:')
                for k, v in val_metrics.items():
                    if not isinstance(v, dict):
                        logger.info(f'  {k}: {v:.4f}')

                eval_path = ckpt_dir / f'{args.exp_prefix}_fold{fold}_s2_eval.json'
                with open(eval_path, 'w', encoding='utf-8') as f:
                    json.dump(val_metrics, f, indent=2, ensure_ascii=False,
                              default=str)
                result['s2_best_metric'] = s2_trainer.best_metric
                result['s2_best_epoch']  = s2_trainer.best_epoch
                result['val_metrics']    = {
                    k: v for k, v in val_metrics.items()
                    if not isinstance(v, dict)
                }
            except Exception as e:
                logger.warning(f'Fold {fold} 评估失败: {e}')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # 支持命令行 --cache-dir None 禁用缓存
    if args.cache_dir and args.cache_dir.lower() == 'none':
        args.cache_dir = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存参数
    with open(out_dir / 'args.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info(f'输出目录  : {out_dir}')
    logger.info(f'训练阶段  : {args.stage}')
    logger.info(f'数据模式  : {args.data_mode}')
    logger.info(f'数据源    : {args.source or "全部(tusz+private)"}')
    logger.info(f'导联模式  : {"TCP双极22ch" if args.use_bipolar else "单极19ch"}')
    logger.info(f'设备      : {args.device}')
    logger.info(f'缓存目录  : {args.cache_dir or "无(未启用缓存)"}')

    # K 折划分（按患者，仅在所选 source 范围内划分）
    patient_ids = get_patient_ids(args.manifest, source=args.source)
    splits      = make_kfold_splits(patient_ids, n_folds=args.n_folds,
                                    seed=args.seed)
    logger.info(f'{args.n_folds} 折划分  总患者数: {len(patient_ids)}')

    folds_to_run = ([args.fold] if args.fold is not None
                    else list(range(args.n_folds)))

    all_results = []
    for fold in folds_to_run:
        train_ids, val_ids = splits[fold]
        result = train_fold(fold, train_ids, val_ids, args)
        all_results.append(result)

    # 汇总 Stage-2 F1
    logger.info('\n' + '='*60)
    logger.info(f'全部折训练完成  stage={args.stage}')
    logger.info('='*60)

    if args.stage in ('2', 'both'):
        f1s = [r.get('val_metrics', {}).get('f1_macro', 0.0)
               for r in all_results]
        aucs = [r.get('val_metrics', {}).get('auc_macro', 0.0)
                for r in all_results]
        corr_szs = [r.get('val_metrics', {}).get('mc_corr_sz', 0.0)
                    for r in all_results]
        acc_pts = [r.get('val_metrics', {}).get('mc_acc_pt', 0.0)
                   for r in all_results]
        import numpy as np
        logger.info(f'各折 F1_macro : {[f"{v:.4f}" for v in f1s]}')
        logger.info(f'均值±std     : {np.mean(f1s):.4f}±{np.std(f1s):.4f}')
        logger.info(f'AUC_macro 均值: {np.mean(aucs):.4f}')
        if any(v > 0 for v in corr_szs):
            logger.info(f'MC corr_sz 均值: '
                        f'{np.mean(corr_szs):.3f}±{np.std(corr_szs):.3f}')
            logger.info(f'MC acc_pt  均值: '
                        f'{np.mean(acc_pts):.3f}±{np.std(acc_pts):.3f}')

        summary = {
            'stage':     args.stage,
            'n_folds':   args.n_folds,
            'folds':     all_results,
            'f1_mean':   float(np.mean(f1s)),
            'f1_std':    float(np.std(f1s)),
            'auc_mean':  float(np.mean(aucs)),
            'mc_corr_sz_mean': float(np.mean(corr_szs)),
            'mc_acc_pt_mean':  float(np.mean(acc_pts)),
        }
        with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f'汇总保存: {out_dir}/summary.json')


if __name__ == '__main__':
    main()
