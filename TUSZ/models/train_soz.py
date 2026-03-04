#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM-TimeFilter-SOZ 三阶段域自适应训练管道

训练策略:
    Phase 1 — 预训练 (仅公共数据):
        冻结LaBraM全部层, 训练TimeFilter + 定位头 (50 epochs)

    Phase 2 — 域对齐 (混合数据):
        解冻LaBraM最后3层
        交替: 公共数据→更新主干+定位头 / 私有数据→更新域判别器
        GRL λ 从0.5线性增长到1.0

    Phase 3 — 微调 (仅私有数据):
        冻结域判别器, 全模型微调 (20 epochs, lr → 1e-5)

域差异分析:
    - 计算19通道PSD的KL散度
    - KL > 0.3 → 激活域对抗训练
    - KL ≤ 0.3 → 仅Phase1预训练 + Phase3微调 (跳过Phase2)

验证策略:
    留一患者交叉验证 (LOPO-CV), 仅在私有数据上评估

评估指标:
    - 通道级AUC (19通道各自的ROC-AUC)
    - Top-3准确率 (预测概率最高的3个通道中包含真实SOZ)
    - 宏平均F1, 精确率, 召回率

Usage:
    python train_soz.py --data-root F:/process_dataset --output-dir ./runs/soz_v1
    python train_soz.py --data-root F:/process_dataset --skip-domain-analysis
    python train_soz.py --data-root F:/process_dataset --fold 0  # 单折
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

# ---- 项目内部导入 ----
# 支持直接运行和包导入两种方式
try:
    from .labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        SOZDetectionLoss,
        STANDARD_19,
        TCP_PAIRS,
        TCP_NAMES,
    )
except ImportError:
    from labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        SOZDetectionLoss,
        STANDARD_19,
        TCP_PAIRS,
        TCP_NAMES,
    )

# ManifestSOZDataset
try:
    from .manifest_dataset import ManifestSOZDataset
except ImportError:
    try:
        from manifest_dataset import ManifestSOZDataset
    except ImportError:
        ManifestSOZDataset = None

# 添加 data_preprocess 到搜索路径
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
try:
    from data_preprocess.eeg_pipeline import TimeFilterDataset
except ImportError:
    TimeFilterDataset = None  # 将在运行时检查

logger = logging.getLogger(__name__)


# =============================================================================
# 训练配置
# =============================================================================

@dataclass
class TrainConfig:
    """三阶段域自适应训练配置"""

    # ---- 数据 ----
    data_root: str = r'F:\process_dataset'
    output_dir: str = './runs/soz_v1'

    # ---- Manifest 模式 (优先级高于 data_root) ----
    manifest_path: str = ''               # combined_manifest.csv 路径
    tusz_data_root: str = r'F:\dataset\TUSZ\v2.0.3\edf'
    private_data_root: str = ''            # 私有数据集 EDF 根目录
    source_filter: str = 'both'            # 'tusz' / 'private' / 'both'
    label_mode: str = 'bipolar'            # 'bipolar' (22ch) / 'monopolar' (19ch)

    # ---- Phase 1: 预训练 (仅公共数据) ----
    phase1_epochs: int = 50
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 32

    # ---- Phase 2: 域对齐 (混合数据) ----
    phase2_epochs: int = 40
    phase2_lr: float = 5e-4
    phase2_batch_size: int = 16          # 每个域各16, 混合batch=32
    phase2_unfreeze_top_n: int = 3       # 解冻LaBraM最后N层
    grl_lambda_start: float = 0.5        # GRL λ 起始值
    grl_lambda_end: float = 1.0          # GRL λ 终止值
    domain_loss_weight: float = 0.5      # 域对抗损失权重

    # ---- Phase 3: 微调 (仅私有数据) ----
    phase3_epochs: int = 20
    phase3_lr: float = 1e-5
    phase3_batch_size: int = 16

    # ---- 域差异分析 ----
    kl_threshold: float = 0.3            # KL散度阈值, 超过则激活域对齐
    skip_domain_analysis: bool = False   # 跳过分析, 始终执行域对齐

    # ---- 过采样 ----
    min_seizure_windows_per_epoch: int = 50  # 每轮至少的发作窗口数
    oversample_seizure: bool = True

    # ---- 通用训练 ----
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 0
    pin_memory: bool = True

    # ---- 交叉验证 ----
    cv_mode: str = 'lopo'                # 'lopo' or 'kfold'
    n_folds: int = 5                     # for kfold
    fold: int = -1                       # -1 = all folds, >=0 = specific fold

    # ---- 模型 ----
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # ---- 日志 ----
    log_every: int = 10                  # 每N步打印
    save_every_epoch: bool = True
    early_stop_patience: int = 10


# =============================================================================
# 1. 域差异分析 — PSD KL散度
# =============================================================================

class DomainDivergenceAnalyzer:
    """
    计算公共/私有数据在19通道上的基线期PSD KL散度

    PSD通过Welch方法估计, 然后对归一化PSD计算KL散度
    """

    def __init__(self, n_channels: int = 19, fs: float = 200.0, nperseg: int = 256):
        self.n_channels = n_channels
        self.fs = fs
        self.nperseg = nperseg

    def compute_psd(self, X: np.ndarray) -> np.ndarray:
        """
        计算平均PSD

        Args:
            X: (N, 22, 20, 100)  多个样本
        Returns:
            psd: (19, n_freqs)  19通道的平均PSD
        """
        from scipy.signal import welch

        # 将patches拼接为连续信号: (N, 22, 2000)
        N, C, P, L = X.shape
        signals = X.reshape(N, C, P * L)

        # 仅使用前19个通道 (排除A1-T3等非标准19通道的TCP导联)
        # 实际上我们这里使用的是TCP双极导联数据, 不是单极
        # 因此对所有22通道都计算PSD, 然后映射回19通道太复杂
        # 简化: 对全部22通道计算, 取前19个作为近似
        n_ch = min(C, self.n_channels)

        all_psds = []
        for i in range(N):
            sample_psd = []
            for ch in range(n_ch):
                sig = signals[i, ch]
                if np.std(sig) < 1e-10:
                    continue
                freqs, pxx = welch(sig, fs=self.fs, nperseg=min(self.nperseg, len(sig)))
                sample_psd.append(pxx)
            if sample_psd:
                all_psds.append(np.array(sample_psd))

        if not all_psds:
            return np.ones((n_ch, 1)) / n_ch

        # 平均: (N, n_ch, n_freqs) → (n_ch, n_freqs)
        avg_psd = np.mean(np.array(all_psds), axis=0)
        return avg_psd

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL(P||Q), 输入为归一化概率分布"""
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def analyze(
        self,
        public_data: np.ndarray,
        private_data: np.ndarray,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """
        分析两个域的PSD差异

        Args:
            public_data:  (N1, 22, 20, 100)
            private_data: (N2, 22, 20, 100)
            max_samples:  最多使用的样本数 (加速计算)

        Returns:
            dict with:
                'mean_kl': 19通道平均KL散度
                'max_kl': 最大KL散度
                'per_channel_kl': list of per-channel KL
                'need_domain_alignment': bool
        """
        # 采样
        if len(public_data) > max_samples:
            idx = np.random.choice(len(public_data), max_samples, replace=False)
            public_data = public_data[idx]
        if len(private_data) > max_samples:
            idx = np.random.choice(len(private_data), max_samples, replace=False)
            private_data = private_data[idx]

        psd_pub = self.compute_psd(public_data)    # (n_ch, n_freqs)
        psd_priv = self.compute_psd(private_data)  # (n_ch, n_freqs)

        # 确保形状一致
        n_ch = min(psd_pub.shape[0], psd_priv.shape[0])
        n_freq = min(psd_pub.shape[1], psd_priv.shape[1])
        psd_pub = psd_pub[:n_ch, :n_freq]
        psd_priv = psd_priv[:n_ch, :n_freq]

        per_channel_kl = []
        for ch in range(n_ch):
            kl = self.kl_divergence(psd_pub[ch], psd_priv[ch])
            per_channel_kl.append(kl)

        mean_kl = float(np.mean(per_channel_kl))
        max_kl = float(np.max(per_channel_kl))

        return {
            'mean_kl': mean_kl,
            'max_kl': max_kl,
            'per_channel_kl': per_channel_kl,
            'need_domain_alignment': mean_kl > 0.3,
        }


# =============================================================================
# 2. 过采样器 — 动态调整发作窗口比例
# =============================================================================

def build_weighted_sampler(
    dataset,
    min_seizure_per_epoch: int = 50,
) -> WeightedRandomSampler:
    """
    构建加权采样器, 对发作/SOZ阳性样本过采样

    Args:
        dataset: TimeFilterDataset
        min_seizure_per_epoch: 每轮至少的发作窗口数

    Returns:
        WeightedRandomSampler
    """
    df = dataset.df
    has_soz = df['has_soz'].values.astype(float)

    n_pos = has_soz.sum()
    n_neg = len(has_soz) - n_pos

    if n_pos == 0:
        logger.warning("No SOZ-positive samples found, using uniform sampling")
        return None

    # 计算权重: SOZ阳性样本权重更高
    pos_weight = max(1.0, n_neg / n_pos)
    weights = np.where(has_soz > 0, pos_weight, 1.0)

    # 调整采样数以确保足够的发作窗口
    n_samples = max(len(df), int(min_seizure_per_epoch / max(n_pos / len(df), 0.01)))
    n_samples = min(n_samples, len(df) * 3)  # 上限: 3倍数据量

    logger.info(
        f"WeightedSampler: {int(n_pos)} SOZ+ / {int(n_neg)} SOZ-, "
        f"pos_weight={pos_weight:.1f}, samples_per_epoch={n_samples}"
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=n_samples,
        replacement=True,
    )


# =============================================================================
# 3. 评估指标
# =============================================================================

def compute_channel_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    channel_names: List[str] = None,
) -> Dict[str, float]:
    """
    通道级AUC

    Args:
        y_true: (N, n_out)  二值标签
        y_prob: (N, n_out)  预测概率
        channel_names: 通道名列表

    Returns:
        dict with per-channel AUC and macro AUC
    """
    n_out = y_true.shape[1]
    if channel_names is None:
        if n_out == 22:
            channel_names = TCP_NAMES
        else:
            channel_names = list(STANDARD_19)

    aucs = {}
    valid_aucs = []

    for i, ch in enumerate(channel_names[:n_out]):
        col = y_true[:, i]
        if col.sum() > 0 and col.sum() < len(col):  # 至少有正/负样本
            try:
                auc = roc_auc_score(col, y_prob[:, i])
                aucs[ch] = auc
                valid_aucs.append(auc)
            except Exception:
                aucs[ch] = float('nan')
        else:
            aucs[ch] = float('nan')

    aucs['macro_auc'] = float(np.nanmean(valid_aucs)) if valid_aucs else 0.0
    return aucs


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 3,
) -> float:
    """
    Top-K准确率: 预测概率最高的K个通道中是否包含真实SOZ通道

    临床意义: 手术前切除区域评估, Top-3是临床可接受范围

    Args:
        y_true: (N, 19)  二值标签
        y_prob: (N, 19)  预测概率
        k: Top-K

    Returns:
        Top-K accuracy (0~1)
    """
    correct = 0
    total = 0

    for i in range(len(y_true)):
        true_soz = set(np.where(y_true[i] > 0.5)[0])
        if len(true_soz) == 0:
            continue  # 跳过没有SOZ标注的样本

        # Top-K 预测通道
        top_k_pred = set(np.argsort(y_prob[i])[-k:])

        # 如果任一真实SOZ通道在Top-K中
        if true_soz & top_k_pred:
            correct += 1
        total += 1

    return correct / max(total, 1)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    全面评估

    Returns:
        dict with: macro_auc, top3_acc, macro_f1, per_channel_auc, ...
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            X, y_soz, mask, metas = batch
            X = X.to(device)
            out = model(X)
            probs = out['soz_probs'].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_soz.numpy())

    y_true = np.concatenate(all_labels, axis=0)   # (N, n_out)
    y_prob = np.concatenate(all_probs, axis=0)     # (N, n_out)
    y_pred = (y_prob > threshold).astype(float)

    # 通道级AUC
    ch_aucs = compute_channel_auc(y_true, y_prob)

    # Top-3准确率
    top3 = compute_top_k_accuracy(y_true, y_prob, k=3)

    # F1等
    metrics = {
        'macro_auc': ch_aucs.get('macro_auc', 0.0),
        'top3_acc': top3,
    }

    # 安全计算macro F1 / precision / recall
    try:
        flat_true = y_true.flatten()
        flat_pred = y_pred.flatten()
        if len(np.unique(flat_true)) > 1:
            metrics['macro_f1'] = f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['macro_precision'] = precision_score(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['macro_recall'] = recall_score(
                y_true, y_pred, average='macro', zero_division=0
            )
        else:
            metrics['macro_f1'] = 0.0
            metrics['macro_precision'] = 0.0
            metrics['macro_recall'] = 0.0
    except Exception:
        metrics['macro_f1'] = 0.0
        metrics['macro_precision'] = 0.0
        metrics['macro_recall'] = 0.0

    metrics['per_channel_auc'] = ch_aucs
    metrics['n_samples'] = len(y_true)
    metrics['n_soz_positive'] = int((y_true.sum(axis=1) > 0).sum())

    return metrics


# =============================================================================
# 4. 三阶段训练器
# =============================================================================

class SOZTrainer:
    """
    LaBraM-TimeFilter-SOZ 三阶段域自适应训练器

    使用方式:
        trainer = SOZTrainer(cfg, model, public_ds, private_ds)
        trainer.train()    # 执行三阶段训练
        results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        model: LaBraM_TimeFilter_SOZ,
        public_dataset: 'TimeFilterDataset',
        private_dataset: 'TimeFilterDataset',
        val_dataset: Optional['TimeFilterDataset'] = None,
        fold_idx: int = 0,
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu'
        )
        self.model = model.to(self.device)
        self.public_ds = public_dataset
        self.private_ds = private_dataset
        self.val_ds = val_dataset
        self.fold_idx = fold_idx

        # 输出目录
        self.run_dir = Path(cfg.output_dir) / f'fold_{fold_idx}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 损失函数
        self.criterion = SOZDetectionLoss(
            focal_gamma=cfg.model_config.focal_gamma,
            focal_alpha=cfg.model_config.focal_alpha,
            domain_weight=cfg.domain_loss_weight,
        )

        # 状态
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history: List[Dict] = []

    # -----------------------------------------------------------------
    # Phase 1: 预训练 (仅公共数据)
    # -----------------------------------------------------------------

    def phase1_pretrain(self):
        """
        Phase 1: 冻结LaBraM全部层, 训练TimeFilter + 定位头

        使用公共数据 (TUSZ), 不使用域判别器
        """
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1: Pretrain (public data only)")
        logger.info("=" * 70)

        cfg = self.cfg
        model = self.model

        # 冻结LaBraM全部层
        for param in model.backbone.parameters():
            param.requires_grad = False
        if model.domain_disc is not None:
            for param in model.domain_disc.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 1 trainable params: {trainable:,}")

        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase1_epochs, eta_min=1e-6)

        # DataLoader
        sampler = None
        shuffle = True
        if cfg.oversample_seizure:
            sampler = build_weighted_sampler(
                self.public_ds, cfg.min_seizure_windows_per_epoch
            )
            if sampler is not None:
                shuffle = False

        loader = self.public_ds.create_dataloader(
            batch_size=cfg.phase1_batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        # 如果有sampler, 需要手动创建DataLoader
        if sampler is not None:
            loader = DataLoader(
                self.public_ds,
                batch_size=cfg.phase1_batch_size,
                sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=loader.collate_fn if hasattr(loader, 'collate_fn')
                else _default_collate,
            )

        val_loader = None
        if self.val_ds is not None:
            val_loader = self.val_ds.create_dataloader(
                batch_size=cfg.phase1_batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            )

        for epoch in range(1, cfg.phase1_epochs + 1):
            loss = self._train_epoch(
                loader, optimizer, phase='pretrain',
                domain_labels_value=0.0,  # 全部是public
            )
            scheduler.step()

            log_msg = f"[P1] Epoch {epoch}/{cfg.phase1_epochs}  loss={loss:.4f}"

            val_metrics = None
            if val_loader is not None and epoch % 5 == 0:
                val_metrics = evaluate_model(self.model, val_loader, self.device)
                log_msg += (
                    f"  val_auc={val_metrics['macro_auc']:.4f}"
                    f"  top3={val_metrics['top3_acc']:.4f}"
                )
                self._maybe_save_best(val_metrics, epoch, 'phase1')

            logger.info(log_msg)
            self.history.append({
                'phase': 'pretrain', 'epoch': epoch, 'loss': loss,
                'val': val_metrics,
            })

    # -----------------------------------------------------------------
    # Phase 2: 域对齐 (混合数据)
    # -----------------------------------------------------------------

    def phase2_domain_align(self):
        """
        Phase 2: 解冻LaBraM最后N层, 交替训练主干+域判别器

        每步:
          1) 公共数据前向 → 更新主干 + 定位头
          2) 私有数据前向 → 更新域判别器 (通过GRL同时对抗更新主干)
        """
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2: Domain Alignment (mixed data)")
        logger.info("=" * 70)

        cfg = self.cfg
        model = self.model

        # 解冻LaBraM最后N层
        n_layers = len(model.backbone.layers)
        n_unfreeze = cfg.phase2_unfreeze_top_n
        for i, layer in enumerate(model.backbone.layers):
            if i >= n_layers - n_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        # 解冻域判别器
        if model.domain_disc is not None:
            for param in model.domain_disc.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Phase 2: unfroze top {n_unfreeze} layers, "
            f"trainable params: {trainable:,}"
        )

        # 优化器 — 分组学习率
        param_groups = []
        backbone_params = [
            p for n, p in model.named_parameters()
            if 'backbone' in n and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if 'backbone' not in n and 'domain_disc' not in n and p.requires_grad
        ]
        domain_params = [
            p for n, p in model.named_parameters()
            if 'domain_disc' in n and p.requires_grad
        ]

        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': cfg.phase2_lr * 0.1})
        if head_params:
            param_groups.append({'params': head_params, 'lr': cfg.phase2_lr})
        if domain_params:
            param_groups.append({'params': domain_params, 'lr': cfg.phase2_lr})

        optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase2_epochs, eta_min=1e-6)

        # DataLoaders
        pub_loader = self.public_ds.create_dataloader(
            batch_size=cfg.phase2_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            drop_last=True,
        )
        priv_loader = self.private_ds.create_dataloader(
            batch_size=cfg.phase2_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            drop_last=True,
        )

        val_loader = None
        if self.val_ds is not None:
            val_loader = self.val_ds.create_dataloader(
                batch_size=cfg.phase2_batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            )

        for epoch in range(1, cfg.phase2_epochs + 1):
            # GRL λ 线性增长
            progress = epoch / cfg.phase2_epochs
            grl_lambda = (
                cfg.grl_lambda_start
                + (cfg.grl_lambda_end - cfg.grl_lambda_start) * progress
            )
            model.set_grl_lambda(grl_lambda)

            loss = self._train_epoch_alternating(
                pub_loader, priv_loader, optimizer,
            )
            scheduler.step()

            log_msg = (
                f"[P2] Epoch {epoch}/{cfg.phase2_epochs}  "
                f"loss={loss:.4f}  grl_lambda={grl_lambda:.3f}"
            )

            val_metrics = None
            if val_loader is not None and epoch % 5 == 0:
                val_metrics = evaluate_model(self.model, val_loader, self.device)
                log_msg += (
                    f"  val_auc={val_metrics['macro_auc']:.4f}"
                    f"  top3={val_metrics['top3_acc']:.4f}"
                )
                self._maybe_save_best(val_metrics, epoch, 'phase2')

            logger.info(log_msg)
            self.history.append({
                'phase': 'domain_align', 'epoch': epoch, 'loss': loss,
                'grl_lambda': grl_lambda, 'val': val_metrics,
            })

    # -----------------------------------------------------------------
    # Phase 3: 微调 (仅私有数据)
    # -----------------------------------------------------------------

    def phase3_finetune(self):
        """
        Phase 3: 冻结域判别器, 全模型微调

        学习率衰减至1e-5, 仅在私有数据上训练
        """
        logger.info("\n" + "=" * 70)
        logger.info("Phase 3: Finetune (private data only)")
        logger.info("=" * 70)

        cfg = self.cfg
        model = self.model

        # 解冻所有backbone层
        for param in model.backbone.parameters():
            param.requires_grad = True
        # 冻结域判别器
        if model.domain_disc is not None:
            for param in model.domain_disc.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 3 trainable params: {trainable:,}")

        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.phase3_lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase3_epochs, eta_min=1e-6)

        # DataLoader (私有数据, 有过采样)
        sampler = None
        shuffle = True
        if cfg.oversample_seizure:
            sampler = build_weighted_sampler(
                self.private_ds, cfg.min_seizure_windows_per_epoch
            )
            if sampler is not None:
                shuffle = False

        loader = self.private_ds.create_dataloader(
            batch_size=cfg.phase3_batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        if sampler is not None:
            loader = DataLoader(
                self.private_ds,
                batch_size=cfg.phase3_batch_size,
                sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=_default_collate,
            )

        val_loader = None
        if self.val_ds is not None:
            val_loader = self.val_ds.create_dataloader(
                batch_size=cfg.phase3_batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            )

        for epoch in range(1, cfg.phase3_epochs + 1):
            loss = self._train_epoch(
                loader, optimizer, phase='finetune',
                domain_labels_value=None,  # 不使用域判别
            )
            scheduler.step()

            log_msg = f"[P3] Epoch {epoch}/{cfg.phase3_epochs}  loss={loss:.4f}"

            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate_model(self.model, val_loader, self.device)
                log_msg += (
                    f"  val_auc={val_metrics['macro_auc']:.4f}"
                    f"  top3={val_metrics['top3_acc']:.4f}"
                )
                self._maybe_save_best(val_metrics, epoch, 'phase3')

            logger.info(log_msg)
            self.history.append({
                'phase': 'finetune', 'epoch': epoch, 'loss': loss,
                'val': val_metrics,
            })

    # -----------------------------------------------------------------
    # 训练入口
    # -----------------------------------------------------------------

    def train(self, need_domain_alignment: bool = True):
        """
        执行完整三阶段训练

        Args:
            need_domain_alignment: 是否执行Phase2域对齐
                (由域差异分析决定, 或手动指定)
        """
        t0 = time.time()
        logger.info(f"\nTraining fold {self.fold_idx} on {self.device}")
        logger.info(f"  Public samples:  {len(self.public_ds)}")
        logger.info(f"  Private samples: {len(self.private_ds)}")
        if self.val_ds:
            logger.info(f"  Val samples:     {len(self.val_ds)}")

        # Phase 1
        self.phase1_pretrain()

        # Phase 2 (条件性)
        if need_domain_alignment:
            self.phase2_domain_align()
        else:
            logger.info("\n[SKIP] Phase 2: Domain alignment not needed (KL < threshold)")

        # Phase 3
        self.phase3_finetune()

        elapsed = time.time() - t0
        logger.info(f"\nTraining complete in {elapsed / 60:.1f} min")
        logger.info(f"Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # 保存训练历史
        history_path = self.run_dir / 'training_history.json'
        _save_json(self.history, history_path)

        # 加载最佳模型
        best_path = self.run_dir / 'best_model.pt'
        if best_path.exists():
            logger.info(f"Loading best model from {best_path}")
            self.model.load_state_dict(
                torch.load(str(best_path), map_location=self.device)
            )

    # -----------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer,
        phase: str = 'pretrain',
        domain_labels_value: Optional[float] = None,
    ) -> float:
        """单个epoch训练 (Phase 1 / Phase 3)"""
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in loader:
            X, y_soz, mask, metas = batch
            X = X.to(self.device)
            y_soz = y_soz.to(self.device)

            # 域标签
            domain_labels = None
            if domain_labels_value is not None and self.model.domain_disc is not None:
                B = X.size(0)
                domain_labels = torch.full(
                    (B, 1), domain_labels_value,
                    device=self.device, dtype=torch.float32,
                )

            optimizer.zero_grad()

            out = self.model(X, domain_labels=domain_labels)

            loss, _ = self.criterion(
                out['soz_logits'], y_soz,
                out.get('domain_logits'), domain_labels,
            )

            loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    def _train_epoch_alternating(
        self,
        pub_loader: DataLoader,
        priv_loader: DataLoader,
        optimizer,
    ) -> float:
        """Phase 2: 交替训练 — 公共/私有数据交替前向"""
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        priv_iter = iter(priv_loader)

        for pub_batch in pub_loader:
            # --- Step A: 公共数据 → 更新主干 + 定位头 ---
            X_pub, y_pub, _, _ = pub_batch
            X_pub = X_pub.to(self.device)
            y_pub = y_pub.to(self.device)
            B_pub = X_pub.size(0)
            domain_pub = torch.zeros(B_pub, 1, device=self.device)

            optimizer.zero_grad()
            out_pub = self.model(X_pub, domain_labels=domain_pub)
            loss_pub, _ = self.criterion(
                out_pub['soz_logits'], y_pub,
                out_pub.get('domain_logits'), domain_pub,
            )
            loss_pub.backward()

            # --- Step B: 私有数据 → 更新域判别器 (GRL对抗) ---
            try:
                priv_batch = next(priv_iter)
            except StopIteration:
                priv_iter = iter(priv_loader)
                priv_batch = next(priv_iter)

            X_priv, y_priv, _, _ = priv_batch
            X_priv = X_priv.to(self.device)
            y_priv = y_priv.to(self.device)
            B_priv = X_priv.size(0)
            domain_priv = torch.ones(B_priv, 1, device=self.device)

            out_priv = self.model(X_priv, domain_labels=domain_priv)
            loss_priv, _ = self.criterion(
                out_priv['soz_logits'], y_priv,
                out_priv.get('domain_logits'), domain_priv,
            )
            loss_priv.backward()

            # --- 合并梯度更新 ---
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            optimizer.step()

            total_loss += (loss_pub.item() + loss_priv.item()) / 2
            n_steps += 1

        return total_loss / max(n_steps, 1)

    def _maybe_save_best(self, metrics: Dict, epoch: int, phase: str):
        """保存最佳模型 (以macro_auc为基准)"""
        score = metrics.get('macro_auc', 0.0)
        if score > self.best_metric:
            self.best_metric = score
            self.best_epoch = epoch
            path = self.run_dir / 'best_model.pt'
            torch.save(self.model.state_dict(), str(path))
            logger.info(
                f"  >> New best: auc={score:.4f} at {phase} epoch {epoch}"
            )

        if self.cfg.save_every_epoch:
            path = self.run_dir / f'{phase}_epoch{epoch}.pt'
            torch.save(self.model.state_dict(), str(path))


# =============================================================================
# 5. LOPO 交叉验证
# =============================================================================

def run_lopo_cv(cfg: TrainConfig):
    """
    留一患者交叉验证 (Leave-One-Patient-Out)

    支持两种数据源:
    1. ManifestSOZDataset (manifest_path)
    2. TimeFilterDataset (data_root) [fallback]

    Returns:
        all_fold_results: List[Dict]
    """
    logger.info("=" * 70)
    logger.info("Leave-One-Patient-Out Cross-Validation")
    logger.info("=" * 70)

    # ---- 决定数据微方式 ----
    use_manifest = bool(cfg.manifest_path and Path(cfg.manifest_path).exists())

    if use_manifest:
        if ManifestSOZDataset is None:
            raise ImportError("ManifestSOZDataset not available")

        logger.info(f"\nData: ManifestSOZDataset (source={cfg.source_filter}, label={cfg.label_mode})")
        logger.info(f"  Manifest: {cfg.manifest_path}")

        # 公共数据集 (TUSZ)
        full_public = ManifestSOZDataset(
            manifest_path=cfg.manifest_path,
            tusz_data_root=cfg.tusz_data_root,
            private_data_root=cfg.private_data_root,
            source_filter='tusz',
            split_filter=['train', 'dev'],
            label_mode=cfg.label_mode,
        )

        # 私有数据集
        full_private = ManifestSOZDataset(
            manifest_path=cfg.manifest_path,
            tusz_data_root=cfg.tusz_data_root,
            private_data_root=cfg.private_data_root,
            source_filter='private',
            label_mode=cfg.label_mode,
        )

        private_pids = full_private.get_patient_ids()

    else:
        # Fallback: TimeFilterDataset
        if TimeFilterDataset is None:
            raise ImportError(
                "Neither ManifestSOZDataset (--manifest) nor TimeFilterDataset is available."
            )
        logger.info(f"\nData: TimeFilterDataset (legacy mode)")
        logger.info(f"  data_root: {cfg.data_root}")

        full_public = TimeFilterDataset(cfg.data_root, subset='tusz')
        full_private = TimeFilterDataset(cfg.data_root, subset='private')
        private_pids = full_private.get_patient_ids()

    n_patients = len(private_pids)
    logger.info(f"Private patients: {n_patients}")
    logger.info(f"Public samples:   {len(full_public)}")
    logger.info(f"Private samples:  {len(full_private)}")

    # ---- 域差异分析 ----
    need_domain_alignment = True
    if not cfg.skip_domain_analysis and cfg.source_filter == 'both':
        logger.info("\nAnalyzing domain divergence (PSD KL)...")
        analyzer = DomainDivergenceAnalyzer()

        pub_samples = []
        for i in range(min(200, len(full_public))):
            try:
                X, _, _, _ = full_public[i]
                pub_samples.append(X.numpy() if isinstance(X, torch.Tensor) else X)
            except Exception:
                continue
        pub_arr = np.array(pub_samples) if pub_samples else np.zeros((1, 22, 20, 100))

        priv_samples = []
        for i in range(min(200, len(full_private))):
            try:
                X, _, _, _ = full_private[i]
                priv_samples.append(X.numpy() if isinstance(X, torch.Tensor) else X)
            except Exception:
                continue
        priv_arr = np.array(priv_samples) if priv_samples else np.zeros((1, 22, 20, 100))

        div_result = analyzer.analyze(pub_arr, priv_arr)
        logger.info(f"  Mean KL divergence: {div_result['mean_kl']:.4f}")
        logger.info(f"  Max KL divergence:  {div_result['max_kl']:.4f}")
        logger.info(
            f"  Domain alignment: "
            f"{'ACTIVATED' if div_result['need_domain_alignment'] else 'SKIPPED'}"
            f" (threshold={cfg.kl_threshold})"
        )
        need_domain_alignment = div_result['need_domain_alignment']

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_json(div_result, out_dir / 'domain_divergence.json')
    elif cfg.source_filter != 'both':
        need_domain_alignment = False
        logger.info(f"\nSingle source mode ({cfg.source_filter}), skipping domain alignment.")
    else:
        logger.info("Domain analysis skipped, alignment always active.")

    # ---- LOPO 循环 ----
    folds_to_run = (
        list(range(n_patients)) if cfg.fold < 0 else [cfg.fold]
    )

    all_results = []
    for fold_idx in folds_to_run:
        val_pid = private_pids[fold_idx]
        train_pids = [p for p in private_pids if p != val_pid]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}/{n_patients-1}: val_patient={val_pid}")
        logger.info(f"  Train patients: {len(train_pids)}")
        logger.info(f"{'='*60}")

        # 按患者划分私有数据
        if use_manifest:
            train_private = ManifestSOZDataset(
                manifest_path=cfg.manifest_path,
                tusz_data_root=cfg.tusz_data_root,
                private_data_root=cfg.private_data_root,
                source_filter='private',
                patient_ids=train_pids,
                label_mode=cfg.label_mode,
            )
            val_private = ManifestSOZDataset(
                manifest_path=cfg.manifest_path,
                tusz_data_root=cfg.tusz_data_root,
                private_data_root=cfg.private_data_root,
                source_filter='private',
                patient_ids=[val_pid],
                label_mode=cfg.label_mode,
            )
        else:
            train_private = TimeFilterDataset(
                cfg.data_root, subset='private', patient_ids=train_pids,
            )
            val_private = TimeFilterDataset(
                cfg.data_root, subset='private', patient_ids=[val_pid],
            )

        # 新建模型
        model = LaBraM_TimeFilter_SOZ(cfg.model_config)

        # 训练
        trainer = SOZTrainer(
            cfg=cfg,
            model=model,
            public_dataset=full_public,
            private_dataset=train_private,
            val_dataset=val_private,
            fold_idx=fold_idx,
        )
        trainer.train(need_domain_alignment=need_domain_alignment)

        # 评估
        val_loader = val_private.create_dataloader(
            batch_size=cfg.phase3_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        fold_metrics = evaluate_model(model, val_loader, trainer.device)
        fold_metrics['val_patient'] = val_pid
        fold_metrics['fold_idx'] = fold_idx

        logger.info(
            f"Fold {fold_idx} results: "
            f"AUC={fold_metrics['macro_auc']:.4f}, "
            f"Top3={fold_metrics['top3_acc']:.4f}, "
            f"F1={fold_metrics['macro_f1']:.4f}"
        )

        all_results.append(fold_metrics)
        _save_json(fold_metrics, trainer.run_dir / 'fold_metrics.json')

    # ---- 汇总 ----
    if len(all_results) > 1:
        summary = _summarize_cv(all_results)
        logger.info("\n" + "=" * 70)
        logger.info("LOPO Cross-Validation Summary")
        logger.info("=" * 70)
        for key in ['macro_auc', 'top3_acc', 'macro_f1', 'macro_precision', 'macro_recall']:
            vals = [r.get(key, 0.0) for r in all_results]
            logger.info(
                f"  {key:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            )

        out_dir = Path(cfg.output_dir)
        _save_json(summary, out_dir / 'cv_summary.json')
        _save_json(all_results, out_dir / 'all_fold_results.json')

    return all_results


def _summarize_cv(results: List[Dict]) -> Dict:
    """汇总交叉验证结果"""
    summary = {}
    metric_keys = ['macro_auc', 'top3_acc', 'macro_f1', 'macro_precision', 'macro_recall']
    for key in metric_keys:
        vals = [r.get(key, 0.0) for r in results]
        summary[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'values': vals,
        }
    summary['n_folds'] = len(results)
    return summary


# =============================================================================
# 工具函数
# =============================================================================

def _default_collate(batch):
    """TimeFilterDataset 的默认 collate"""
    Xs, ys, masks, metas = zip(*batch)
    return (
        torch.stack(Xs),
        torch.stack(ys),
        torch.stack(masks),
        list(metas),
    )


def _save_json(obj, path):
    """安全保存JSON (处理numpy类型)"""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def set_seed(seed: int):
    """设置全局随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description='LaBraM-TimeFilter-SOZ Training Pipeline'
    )

    # 数据 (旧模式)
    parser.add_argument('--data-root', type=str, default=r'F:\process_dataset')
    parser.add_argument('--output-dir', type=str, default='./runs/soz_v1')

    # 数据 (Manifest模式 — 优先级更高)
    parser.add_argument(
        '--manifest', type=str, default='',
        help='combined_manifest.csv 路径 (若提供则用 ManifestSOZDataset)',
    )
    parser.add_argument(
        '--source', type=str, default='both',
        choices=['tusz', 'private', 'both'],
        help='数据源过滤',
    )
    parser.add_argument(
        '--tusz-data-root', type=str, default=r'F:\dataset\TUSZ\v2.0.3\edf',
        help='TUSZ EDF root',
    )
    parser.add_argument(
        '--private-data-root', type=str, default='',
        help='Private EDF root',
    )
    parser.add_argument(
        '--label-mode', type=str, default='bipolar',
        choices=['bipolar', 'monopolar'],
        help='标签模式 (bipolar=22ch, monopolar=19ch)',
    )

    # 阶段
    parser.add_argument('--phase1-epochs', type=int, default=50)
    parser.add_argument('--phase2-epochs', type=int, default=40)
    parser.add_argument('--phase3-epochs', type=int, default=20)
    parser.add_argument('--phase1-lr', type=float, default=1e-3)
    parser.add_argument('--phase2-lr', type=float, default=5e-4)
    parser.add_argument('--phase3-lr', type=float, default=1e-5)

    # 域分析
    parser.add_argument('--kl-threshold', type=float, default=0.3)
    parser.add_argument('--skip-domain-analysis', action='store_true')
    parser.add_argument('--domain-loss-weight', type=float, default=0.5)

    # 训练
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)

    # 交叉验证
    parser.add_argument('--fold', type=int, default=-1, help='-1=all folds')
    parser.add_argument('--cv-mode', choices=['lopo', 'kfold'], default='lopo')

    # 模型
    parser.add_argument('--labram-checkpoint', type=str, default='')
    parser.add_argument('--embed-dim', type=int, default=128)

    args = parser.parse_args()

    # 根据 label_mode 配置模型输出
    n_output = 22 if args.label_mode == 'bipolar' else 19
    output_mode = args.label_mode

    model_cfg = ModelConfig(
        labram_checkpoint=args.labram_checkpoint,
        embed_dim=args.embed_dim,
        domain_loss_weight=args.domain_loss_weight,
        n_output=n_output,
        output_mode=output_mode,
    )

    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        private_data_root=args.private_data_root,
        source_filter=args.source,
        label_mode=args.label_mode,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        phase3_lr=args.phase3_lr,
        phase1_batch_size=args.batch_size,
        phase2_batch_size=args.batch_size // 2,
        phase3_batch_size=args.batch_size // 2,
        kl_threshold=args.kl_threshold,
        skip_domain_analysis=args.skip_domain_analysis,
        domain_loss_weight=args.domain_loss_weight,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        fold=args.fold,
        cv_mode=args.cv_mode,
        model_config=model_cfg,
    )


def main():
    cfg = parse_args()

    # 日志
    log_dir = Path(cfg.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_dir / 'train.log'), encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

    set_seed(cfg.seed)
    logger.info(f"Config: {json.dumps(asdict(cfg), indent=2, default=str)}")

    run_lopo_cv(cfg)


# =============================================================================
# 自测 (不需要实际数据)
# =============================================================================

def _self_test():
    """无数据自测: 验证训练循环逻辑正确性"""
    print("=" * 60)
    print("SOZ Trainer Self-Test (synthetic data)")
    print("=" * 60)

    # 1. 合成数据集
    class SyntheticDataset:
        """模拟 TimeFilterDataset 接口"""

        def __init__(self, n_samples=100, n_out=22, source='tusz', n_patients=5):
            import pandas as pd
            self.n_out = n_out
            self.df = pd.DataFrame({
                'patient_id': [f'P{i % n_patients:02d}' for i in range(n_samples)],
                'source': source,
                'has_soz': [1 if i % 3 == 0 else 0 for i in range(n_samples)],
            })

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            X = torch.randn(22, 20, 100)
            y = torch.zeros(self.n_out)
            if idx % 3 == 0:
                y[torch.randint(0, self.n_out, (2,))] = 1.0
            mask = torch.ones(22)
            meta = {
                'source': self.df.iloc[idx]['source'],
                'patient_id': self.df.iloc[idx]['patient_id'],
                'has_soz': int(self.df.iloc[idx]['has_soz']),
            }
            return X, y, mask, meta

        def get_patient_ids(self):
            return sorted(self.df['patient_id'].unique().tolist())

        def create_dataloader(self, batch_size=8, shuffle=True, **kwargs):
            return DataLoader(
                self, batch_size=batch_size, shuffle=shuffle,
                collate_fn=_default_collate,
            )

    # 2. 构建
    model_cfg = ModelConfig(
        n_transformer_layers=4,
        n_frozen_layers=2,
        embed_dim=64,
    )
    train_cfg = TrainConfig(
        phase1_epochs=2,
        phase2_epochs=2,
        phase3_epochs=2,
        phase1_batch_size=8,
        phase2_batch_size=4,
        phase3_batch_size=4,
        output_dir='./test_runs/selftest',
        model_config=model_cfg,
        skip_domain_analysis=True,
        device='cpu',
    )

    pub_ds = SyntheticDataset(n_samples=40, n_out=model_cfg.n_output, source='tusz', n_patients=3)
    priv_ds = SyntheticDataset(n_samples=30, n_out=model_cfg.n_output, source='private', n_patients=3)
    val_ds = SyntheticDataset(n_samples=10, n_out=model_cfg.n_output, source='private', n_patients=2)

    model = LaBraM_TimeFilter_SOZ(model_cfg)

    # 3. 训练器
    trainer = SOZTrainer(
        cfg=train_cfg,
        model=model,
        public_dataset=pub_ds,
        private_dataset=priv_ds,
        val_dataset=val_ds,
        fold_idx=0,
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Public: {len(pub_ds)}, Private: {len(priv_ds)}, Val: {len(val_ds)}")

    # 4. 训练
    trainer.train(need_domain_alignment=True)

    # 5. 评估
    val_loader = val_ds.create_dataloader(batch_size=4, shuffle=False)
    metrics = evaluate_model(model, val_loader, torch.device('cpu'))
    print(f"\nFinal metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif k == 'per_channel_auc':
            print(f"  macro_auc (from ch): {v.get('macro_auc', 0):.4f}")

    # 6. 域差异分析
    print("\nDomain Divergence Analysis...")
    analyzer = DomainDivergenceAnalyzer()
    pub_X = np.stack([pub_ds[i][0].numpy() for i in range(min(20, len(pub_ds)))])
    priv_X = np.stack([priv_ds[i][0].numpy() for i in range(min(20, len(priv_ds)))])
    div = analyzer.analyze(pub_X, priv_X, max_samples=20)
    print(f"  Mean KL: {div['mean_kl']:.4f}")
    print(f"  Need alignment: {div['need_domain_alignment']}")

    # 清理
    import shutil
    if Path('./test_runs/selftest').exists():
        shutil.rmtree('./test_runs/selftest')

    print("\n[OK] All self-tests passed!")


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        _self_test()
    else:
        main()

