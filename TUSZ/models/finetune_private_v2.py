#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_private_v2.py — 私有数据 Stratified Group K-Fold 微调 (v2)

相比 v1 (finetune_private_loocv.py) 的核心改进:
  1. 5-fold 分层分组交叉验证 (替代 LOOCV 单患者验证)
  2. EWC 在冻结前对完整模型计算 Fisher (修复旧版 bug)
  3. 知识蒸馏: TUSZ 预训练模型作为 teacher
  4. Label Smoothing + 校准后的 focal_alpha
  5. 三阶段渐进解冻 + LR warmup
  6. 课程式 TUSZ 混合比例 (50% → 15%)
  7. Source-aware 数据增强 (仅增强私有数据)
  8. 梯度累积 (有效 batch = 32)
  9. 无数据泄漏的 fold-aware 统计量
  10. SWA 权重平均
  11. 带正则化的 Temperature Scaling

用法:
  python finetune_private_v2.py \
      --checkpoint output/TUSZ/train/best_model.pt \
      --manifest TUSZ/private_manifest_clean.csv \
      --tusz-manifest TUSZ/combined_manifest.csv \
      --private-data-root "E:/DataSet/EEG/EEG dataset_SUAT" \
      --tusz-data-root "F:/dataset/TUSZ/v2.0.3/edf" \
      --output-dir output/private_finetune_v2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    ConcatDataset, DataLoader, Subset, WeightedRandomSampler,
)

# ─── 项目路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ'))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ' / 'models'))

from models.integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
    FocalLoss,
)
from models.manifest_dataset import ManifestSOZDataset

try:
    from models.train_soz_locator_with_brain_networks import (
        SOZBrainNetworkDataset,
        collate_fn,
        analyze_training_labels,
        compute_pos_weight_from_analysis,
        build_private_channel_weight,
        build_generalized_sample_weight,
        compute_localization_ranking_metrics,
        build_selection_key,
        EEGWindowAugmentor,
        apply_lateral_mirror_augmentation,
    )
except ImportError:
    from train_soz_locator_with_brain_networks import (
        SOZBrainNetworkDataset,
        collate_fn,
        analyze_training_labels,
        compute_pos_weight_from_analysis,
        build_private_channel_weight,
        build_generalized_sample_weight,
        compute_localization_ranking_metrics,
        build_selection_key,
        EEGWindowAugmentor,
        apply_lateral_mirror_augmentation,
    )

try:
    from models.evaluate_integration_deepsoz import (
        DEEPSOZ_19,
        reorder_to_deepsoz,
        mc_inference_single,
        final_loc,
        find_best_threshold,
    )
except ImportError:
    from evaluate_integration_deepsoz import (
        DEEPSOZ_19,
        reorder_to_deepsoz,
        mc_inference_single,
        final_loc,
        find_best_threshold,
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FinetuneConfig:
    # Paths
    checkpoint: str = ''
    manifest: str = 'TUSZ/private_manifest_clean.csv'
    tusz_manifest: str = 'TUSZ/combined_manifest.csv'
    private_data_root: str = 'E:/DataSet/EEG/EEG dataset_SUAT'
    tusz_data_root: str = 'F:/dataset/TUSZ/v2.0.3/edf'
    output_dir: str = 'output/private_finetune_v2'

    # Cross-validation
    n_folds: int = 5
    seed: int = 42

    # Phase durations (epochs)
    phase1_epochs: int = 12
    phase2_epochs: int = 20
    phase3_epochs: int = 8
    warmup_epochs: int = 3

    # Learning rates
    phase1_lr: float = 3e-4
    phase2_lr: float = 1e-4
    phase3_lr: float = 2e-5
    weight_decay: float = 5e-3

    # Gradient accumulation
    batch_size: int = 4
    grad_accum_steps: int = 8

    # EWC
    ewc_lambda: float = 500.0
    fisher_samples: int = 500

    # Knowledge distillation
    kd_temperature: float = 4.0
    kd_alpha: float = 0.3

    # TUSZ replay
    tusz_replay_size: int = 400
    tusz_mix_start_ratio: float = 0.5
    tusz_mix_end_ratio: float = 0.15

    # Class imbalance
    focal_gamma: float = 2.0
    focal_alpha: float = 0.85
    label_smoothing: float = 0.05
    pos_weight_clamp: float = 30.0

    # SWA
    swa_start_epoch: int = 30
    swa_lr: float = 5e-5

    # Augmentation
    lr_mirror_prob: float = 0.3

    # Evaluation
    mc_samples: int = 30
    patience: int = 10

    # Multi-task weights (for private data, slightly lower auxiliary weights)
    w_region: float = 0.3
    w_hemisphere: float = 0.3
    w_transition: float = 0.2
    w_pattern: float = 0.1

    # Phase 3 backbone unfreezing
    n_backbone_frozen_layers: int = 8  # Keep bottom 8/12 layers frozen

    # Fold filter
    fold_indices: Optional[List[int]] = None


# ═════════════════════════════════════════════════════════════════════════════
# Smoothed Focal Loss
# ═════════════════════════════════════════════════════════════════════════════

class SmoothedFocalLoss(nn.Module):
    """
    Focal loss with label smoothing for binary multi-label classification.

    Smoothed targets: y_smooth = y * (1 - epsilon) + (1 - y) * epsilon
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.85,
        epsilon: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
        channel_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        if channel_weight is not None:
            self.register_buffer('channel_weight', channel_weight)
        else:
            self.channel_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Label smoothing
        targets_smooth = targets * (1.0 - self.epsilon) + (1.0 - targets) * self.epsilon

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none',
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        # pt uses original (unsmoothed) targets for focal weighting
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce

        if self.channel_weight is not None:
            focal = focal * self.channel_weight.view(1, -1)
        if sample_weight is not None:
            focal = focal * sample_weight.unsqueeze(1)
        return focal.mean()


# ═════════════════════════════════════════════════════════════════════════════
# Knowledge Distillation
# ═════════════════════════════════════════════════════════════════════════════

class SOZKnowledgeDistiller:
    """
    Knowledge distillation from the pretrained TUSZ model.

    L_KD = BCE(sigmoid(student_logits/T), sigmoid(teacher_logits/T)) * T^2
    """

    def __init__(
        self,
        teacher_model: TimeFilter_LaBraM_BrainNetwork_Integration,
        temperature: float = 4.0,
        alpha: float = 0.3,
    ):
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.T = temperature
        self.alpha = alpha

    @torch.no_grad()
    def get_teacher_logits(
        self, x: torch.Tensor, onset: torch.Tensor, start: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.teacher(x, onset, start)
        return outputs['soz_logits'].detach()

    def kd_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        s_prob = torch.sigmoid(student_logits / self.T)
        t_prob = torch.sigmoid(teacher_logits / self.T)
        loss = F.binary_cross_entropy(
            s_prob.clamp(1e-7, 1 - 1e-7),
            t_prob.clamp(1e-7, 1 - 1e-7),
            reduction='mean',
        )
        return loss * (self.T ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# EWC (Fixed)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ewc_reference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 500,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute EWC reference on the FULL unfrozen model BEFORE any freezing.

    Returns: (fisher, star_params) covering ALL model parameters.
    """
    # Save current requires_grad state
    saved_grad_state = {n: p.requires_grad for n, p in model.named_parameters()}

    # Temporarily unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    model.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters()}
    count = 0

    for batch in dataloader:
        if count >= n_samples:
            break
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        label = batch['label'].to(device)

        model.zero_grad()
        outputs = model(x, onset, start)
        log_prob = F.logsigmoid(outputs['soz_logits'])
        loss = -(log_prob * label).sum()
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        count += x.size(0)

    # Normalize
    for n in fisher:
        fisher[n] /= max(count, 1)

    # Save pretrained weights (the reference for EWC)
    star_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    # Restore original requires_grad state
    for n, p in model.named_parameters():
        p.requires_grad = saved_grad_state[n]

    log.info(f'EWC: computed Fisher on {count} samples, '
             f'{len(fisher)} params, '
             f'mean Fisher = {np.mean([f.mean().item() for f in fisher.values()]):.6f}')
    return fisher, star_params


def ewc_penalty(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    star_params: Dict[str, torch.Tensor],
    lambda_ewc: float,
) -> torch.Tensor:
    """EWC penalty: lambda * sum(F_i * (theta_i - theta*_i)^2)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if p.requires_grad and n in fisher and n in star_params:
            loss = loss + (fisher[n] * (p - star_params[n]) ** 2).sum()
    return lambda_ewc * loss


# ═════════════════════════════════════════════════════════════════════════════
# Curriculum TUSZ Mixing Scheduler
# ═════════════════════════════════════════════════════════════════════════════

class CurriculumMixingScheduler:
    """
    Dynamically adjust TUSZ/private mixing ratio.

    Early: more TUSZ (stabilize training)
    Late: more private (domain adaptation)
    """

    def __init__(
        self,
        n_private: int,
        n_tusz: int,
        start_ratio: float = 0.5,
        end_ratio: float = 0.15,
        total_epochs: int = 40,
    ):
        self.n_private = n_private
        self.n_tusz = n_tusz
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.total_epochs = total_epochs

    def get_tusz_ratio(self, epoch: int) -> float:
        t = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        return self.start_ratio * (1 - t) + self.end_ratio * t

    def get_sampler(self, epoch: int) -> WeightedRandomSampler:
        tusz_ratio = self.get_tusz_ratio(epoch)
        # Weight so that sampling probability matches desired ratio
        # P(tusz) / (P(tusz) + P(private)) = tusz_ratio
        # w_tusz * n_tusz / (w_tusz * n_tusz + w_private * n_private) = tusz_ratio
        # Set w_private = 1.0, solve for w_tusz:
        if tusz_ratio <= 0 or self.n_tusz == 0:
            weights = [1.0] * self.n_private + [0.0] * self.n_tusz
        else:
            w_private = 1.0
            w_tusz = (tusz_ratio / (1 - tusz_ratio)) * (self.n_private / max(self.n_tusz, 1))
            weights = ([w_private] * self.n_private) + ([w_tusz] * self.n_tusz)

        total_samples = self.n_private + self.n_tusz
        return WeightedRandomSampler(
            weights, num_samples=total_samples, replacement=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Progressive Unfreezer
# ═════════════════════════════════════════════════════════════════════════════

class ProgressiveUnfreezer:
    """
    Three-phase progressive unfreezing with warmup.

    Phase 1: Heads + fusion only
    Phase 2: + TimeFilter + BrainTimeFilter + net_evolution
    Phase 3: + Top backbone transformer layers
    """

    def __init__(self, model: TimeFilter_LaBraM_BrainNetwork_Integration, cfg: FinetuneConfig):
        self.model = model
        self.cfg = cfg
        self.phase1_end = cfg.phase1_epochs
        self.phase2_end = cfg.phase1_epochs + cfg.phase2_epochs
        self.total_epochs = self.phase2_end + cfg.phase3_epochs
        self._current_phase = 0
        self._optimizer = None
        self._scheduler = None

    def step(self, epoch: int) -> Tuple[torch.optim.Optimizer, object, int]:
        """
        Called at the start of each epoch.
        Returns (optimizer, scheduler, current_phase).
        Creates new optimizer/scheduler on phase transitions.
        """
        if epoch < self.phase1_end:
            phase = 1
            if self._current_phase != 1:
                self._freeze_for_phase1()
                self._optimizer, self._scheduler = self._build_phase1()
                self._current_phase = 1
                log.info(f'Phase 1 (epoch {epoch}): training heads + fusion only')
        elif epoch < self.phase2_end:
            phase = 2
            if self._current_phase != 2:
                self._unfreeze_for_phase2()
                self._optimizer, self._scheduler = self._build_phase2()
                self._current_phase = 2
                log.info(f'Phase 2 (epoch {epoch}): + TimeFilter + BrainTimeFilter + net_evolution')
        else:
            phase = 3
            if self._current_phase != 3:
                self._unfreeze_for_phase3()
                self._optimizer, self._scheduler = self._build_phase3()
                self._current_phase = 3
                log.info(f'Phase 3 (epoch {epoch}): + top {12 - self.cfg.n_backbone_frozen_layers} backbone layers')

        return self._optimizer, self._scheduler, phase

    def _freeze_for_phase1(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for module in [self.model.soz_head, self.model.region_head,
                       self.model.hemisphere_head, self.model.fusion]:
            for p in module.parameters():
                p.requires_grad = True

    def _unfreeze_for_phase2(self):
        for module in [self.model.timefilter, self.model.brain_timefilter,
                       self.model.net_evolution]:
            for p in module.parameters():
                p.requires_grad = True

    def _unfreeze_for_phase3(self):
        # Unfreeze entire backbone first
        for p in self.model.backbone.parameters():
            p.requires_grad = True
        # Re-freeze bottom layers
        n_frozen = self.cfg.n_backbone_frozen_layers
        if hasattr(self.model.backbone, 'blocks'):
            for i, layer in enumerate(self.model.backbone.blocks):
                if i < n_frozen:
                    for p in layer.parameters():
                        p.requires_grad = False
        elif hasattr(self.model.backbone, 'transformer'):
            blocks = getattr(self.model.backbone.transformer, 'layers',
                             getattr(self.model.backbone.transformer, 'blocks', []))
            for i, layer in enumerate(blocks):
                if i < n_frozen:
                    for p in layer.parameters():
                        p.requires_grad = False

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _build_phase1(self):
        params = self._trainable_params()
        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.phase1_lr, weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(self.cfg.phase1_epochs, 1), T_mult=1,
        )
        return optimizer, scheduler

    def _build_phase2(self):
        param_groups = self.model.get_param_groups(self.cfg.phase2_lr)
        # Filter to only trainable params
        param_groups = [
            {**g, 'params': [p for p in g['params'] if p.requires_grad]}
            for g in param_groups
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.cfg.weight_decay,
        )
        scheduler = _WarmupCosineScheduler(
            optimizer, self.cfg.warmup_epochs, self.cfg.phase2_epochs,
        )
        return optimizer, scheduler

    def _build_phase3(self):
        param_groups = self.model.get_param_groups(self.cfg.phase3_lr)
        param_groups = [
            {**g, 'params': [p for p in g['params'] if p.requires_grad]}
            for g in param_groups
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.cfg.weight_decay,
        )
        scheduler = _WarmupCosineScheduler(
            optimizer, self.cfg.warmup_epochs, self.cfg.phase3_epochs,
        )
        return optimizer, scheduler


class _WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup for warmup_epochs, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return max(0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 0.01)
        super().__init__(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════════════════
# Stratified Group K-Fold
# ═════════════════════════════════════════════════════════════════════════════

def _extract_base_patient(patient_id: str) -> str:
    """'刘娟_SZ1' → '刘娟'"""
    return re.sub(r'_SZ\d+.*$', '', str(patient_id))


def build_stratified_group_kfold(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Build k-fold splits with:
    1. Patient-level grouping (same patient's seizures in same fold)
    2. Hemisphere stratification (L/R balanced across folds)
    3. Multi-patient validation sets

    Returns list of dicts with train/val/test indices.
    """
    # Build patient-level grouping
    patient_groups = defaultdict(list)
    for i, row in df.iterrows():
        base = _extract_base_patient(str(row['patient_id']))
        patient_groups[base].append(i)

    patients = sorted(patient_groups.keys())

    # Get hemisphere per patient (majority vote if multiple seizures)
    patient_hemisphere = {}
    for pt in patients:
        indices = patient_groups[pt]
        hemispheres = df.iloc[indices]['hemisphere'].tolist()
        patient_hemisphere[pt] = max(set(hemispheres), key=hemispheres.count)

    # Use sklearn if available, otherwise manual stratified split
    try:
        from sklearn.model_selection import StratifiedKFold
        hemisphere_labels = [patient_hemisphere[pt] for pt in patients]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        patient_fold_assignment = {}
        for fold_idx, (_, fold_patients_idx) in enumerate(skf.split(patients, hemisphere_labels)):
            for pi in fold_patients_idx:
                patient_fold_assignment[patients[pi]] = fold_idx
    except ImportError:
        # Fallback: manual stratified split
        rng = np.random.RandomState(seed)
        l_patients = [p for p in patients if patient_hemisphere[p] == 'L']
        r_patients = [p for p in patients if patient_hemisphere[p] == 'R']
        rng.shuffle(l_patients)
        rng.shuffle(r_patients)
        all_shuffled = []
        for i in range(max(len(l_patients), len(r_patients))):
            if i < len(l_patients):
                all_shuffled.append(l_patients[i])
            if i < len(r_patients):
                all_shuffled.append(r_patients[i])
        patient_fold_assignment = {}
        for i, pt in enumerate(all_shuffled):
            patient_fold_assignment[pt] = i % n_folds

    # Build fold structures (circular: test=fold_i, val=fold_(i+1), train=rest)
    folds = []
    for test_fold_idx in range(n_folds):
        val_fold_idx = (test_fold_idx + 1) % n_folds

        test_patients = [pt for pt, f in patient_fold_assignment.items() if f == test_fold_idx]
        val_patients = [pt for pt, f in patient_fold_assignment.items() if f == val_fold_idx]
        train_patients = [pt for pt, f in patient_fold_assignment.items()
                          if f != test_fold_idx and f != val_fold_idx]

        test_indices = []
        for pt in test_patients:
            test_indices.extend(patient_groups[pt])
        val_indices = []
        for pt in val_patients:
            val_indices.extend(patient_groups[pt])
        train_indices = []
        for pt in train_patients:
            train_indices.extend(patient_groups[pt])

        fold = {
            'fold': test_fold_idx,
            'test_patients': sorted(test_patients),
            'val_patients': sorted(val_patients),
            'train_patients': sorted(train_patients),
            'test_indices': sorted(test_indices),
            'val_indices': sorted(val_indices),
            'train_indices': sorted(train_indices),
        }
        folds.append(fold)

        # Log hemisphere distribution per split
        test_hemi = [patient_hemisphere[p] for p in test_patients]
        val_hemi = [patient_hemisphere[p] for p in val_patients]
        log.info(f'  Fold {test_fold_idx}: test={len(test_patients)}pts '
                 f'(L={test_hemi.count("L")},R={test_hemi.count("R")}), '
                 f'val={len(val_patients)}pts '
                 f'(L={val_hemi.count("L")},R={val_hemi.count("R")}), '
                 f'train={len(train_patients)}pts '
                 f'({len(train_indices)} samples)')

    return folds


# ═════════════════════════════════════════════════════════════════════════════
# Fold-Aware Data Pipeline (No Leakage)
# ═════════════════════════════════════════════════════════════════════════════

def _analyze_subset_labels(
    dataset: SOZBrainNetworkDataset,
    indices: List[int],
) -> Optional[Dict[str, object]]:
    """Analyze labels ONLY from specified indices (no leakage)."""
    sub = Subset(dataset, indices)
    return analyze_training_labels(sub)


def build_fold_data_pipeline(
    private_ds: SOZBrainNetworkDataset,
    fold_info: Dict,
    tusz_replay_ds: Optional[Subset],
    cfg: FinetuneConfig,
    device: torch.device,
) -> Dict:
    """
    Build data pipeline for a single fold.
    All statistics computed ONLY from train_indices.
    """
    train_sub = Subset(private_ds, fold_info['train_indices'])
    val_sub = Subset(private_ds, fold_info['val_indices'])
    test_sub = Subset(private_ds, fold_info['test_indices'])

    # Analyze ONLY training fold labels
    train_analysis = analyze_training_labels(train_sub)

    # Compute pos_weight from training fold only
    pos_weight = None
    if train_analysis is not None:
        try:
            pos_weight = compute_pos_weight_from_analysis(train_analysis, device)
            # Apply clamp
            pos_weight = pos_weight.clamp(max=cfg.pos_weight_clamp)
        except Exception as e:
            log.warning(f'Failed to compute pos_weight: {e}')

    # Compute channel_weight from training fold only
    channel_weight = None
    if train_analysis is not None:
        try:
            channel_weight, _ = build_private_channel_weight(
                train_analysis,
                min_weight=0.5, max_weight=4.0,
                zero_positive_weight=0.1, device=device,
            )
        except Exception as e:
            log.warning(f'Failed to compute channel_weight: {e}')

    # Build val/test loaders
    val_loader = DataLoader(
        val_sub, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_sub, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return {
        'train_sub': train_sub,
        'val_sub': val_sub,
        'test_sub': test_sub,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'pos_weight': pos_weight,
        'channel_weight': channel_weight,
        'train_analysis': train_analysis,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Source-Aware Augmentation
# ═════════════════════════════════════════════════════════════════════════════

def _source_aware_augment(
    x: torch.Tensor,
    source_list: List[str],
    augmentor: EEGWindowAugmentor,
    bipolar_label: torch.Tensor,
) -> torch.Tensor:
    """Apply augmentation ONLY to private samples, not TUSZ replay."""
    private_mask = torch.tensor(
        [str(s).strip().lower() == 'private' for s in source_list],
        device=x.device, dtype=torch.bool,
    )
    if not private_mask.any():
        return x

    x = x.clone()
    private_indices = torch.nonzero(private_mask, as_tuple=False).flatten()
    x_private = x[private_indices]
    bip_private = bipolar_label[private_indices]
    x_aug = augmentor(x_private, bipolar_label=bip_private)
    x[private_indices] = x_aug
    return x


def build_augmentor() -> EEGWindowAugmentor:
    return EEGWindowAugmentor(
        fs=200.0,
        gaussian_prob=0.5,
        gaussian_std_scale=0.015,
        bandstop_prob=0.2,
        bandstop_width_hz=2.0,
        channel_dropout_prob=0.1,
        max_channel_drops=1,
        time_mask_prob=0.3,
        time_mask_max_ratio=0.15,
        amplitude_scale_prob=0.4,
        amplitude_scale_min=0.75,
        amplitude_scale_max=1.25,
        freq_shift_prob=0.15,
        freq_shift_max_hz=1.5,
        time_shift_prob=0.3,
        time_shift_max_samples=40,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Temperature Scaling (Robust)
# ═════════════════════════════════════════════════════════════════════════════

class RobustTemperatureScaler(nn.Module):
    """Temperature scaling with L2 regularization."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_iter: int = 100,
        reg_lambda: float = 0.1,
    ):
        """Fit temperature with L2 regularization on T."""
        self.temperature.data.fill_(1.0)
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.01, max_iter=n_iter,
        )

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled, targets)
            loss = loss + reg_lambda * (self.temperature - 1.0) ** 2
            loss.backward()
            return loss

        optimizer.step(closure)
        log.info(f'Temperature scaling: T={self.temperature.item():.4f}')


# ═════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_pretrained_model(
    checkpoint_path: str, device: torch.device,
) -> Tuple[TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig]:
    """Load pretrained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    cfg = ckpt.get('config', None)
    if cfg is not None and isinstance(cfg, dict):
        cfg = IntegrationConfig(**cfg)
    if cfg is None:
        raise ValueError('Checkpoint missing config')

    cfg.n_frozen_layers = 0
    cfg.labram_checkpoint = ''
    cfg.use_checkpoint = False

    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    own = model.state_dict()
    filtered = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state.items()
        if (k[7:] if k.startswith('module.') else k) in own
        and own[(k[7:] if k.startswith('module.') else k)].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    log.info(f'Loaded pretrained model: {len(filtered)}/{len(own)} params')
    return model, cfg


# ═════════════════════���═══════════════════════════════════════════════════════
# Training Loop (with Gradient Accumulation)
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch_v2(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmentor: Optional[EEGWindowAugmentor],
    distiller: Optional[SOZKnowledgeDistiller],
    fisher: Optional[Dict[str, torch.Tensor]],
    star_params: Optional[Dict[str, torch.Tensor]],
    cfg: FinetuneConfig,
    epoch: int,
) -> Dict[str, float]:
    """Single epoch training with gradient accumulation, KD, and EWC."""
    model.train()
    if distiller is not None:
        distiller.teacher.eval()

    optimizer.zero_grad()
    total_loss = 0.0
    total_soz = 0.0
    total_kd = 0.0
    total_ewc = 0.0
    n_steps = 0
    accum = cfg.grad_accum_steps

    for step, batch in enumerate(loader):
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        label = batch['label'].to(device)
        bip_label = batch['bipolar_label'].to(device)
        mono_label = batch['monopolar_label'].to(device)
        region_label = batch['region_label'].to(device)
        hemi_label = batch['hemisphere_label'].to(device)
        source_list = batch['source']

        # Source-aware augmentation (only private samples)
        if augmentor is not None:
            x = _source_aware_augment(x, source_list, augmentor, bip_label)

        # Lateral mirror augmentation (only private L/R samples)
        if cfg.lr_mirror_prob > 0:
            x, label, bip_label, mono_label, region_label, hemi_label = (
                apply_lateral_mirror_augmentation(
                    x, label, bip_label, mono_label,
                    region_label, hemi_label,
                    mirror_prob=cfg.lr_mirror_prob,
                )
            )

        outputs = model(x, onset, start)

        # Sample weight (downweight diffuse seizures)
        sample_w = build_generalized_sample_weight(label, device, 0.4, 0.1)

        # Task loss
        soz_target = mono_label
        task_loss, losses = model.compute_loss(
            outputs,
            soz_targets=soz_target,
            region_targets=region_label,
            hemisphere_targets=hemi_label,
            sample_weight=sample_w,
        )

        # Knowledge distillation loss
        kd_loss_val = torch.tensor(0.0, device=device)
        if distiller is not None:
            teacher_logits = distiller.get_teacher_logits(x, onset, start)
            kd_loss_val = distiller.kd_loss(outputs['soz_logits'], teacher_logits)

        # Combined loss: (1 - alpha) * task + alpha * KD
        combined = (1 - cfg.kd_alpha) * task_loss + cfg.kd_alpha * kd_loss_val

        # EWC penalty (added on top, not blended with alpha)
        ewc_loss = torch.tensor(0.0, device=device)
        if fisher is not None and star_params is not None and cfg.ewc_lambda > 0:
            ewc_loss = ewc_penalty(model, fisher, star_params, cfg.ewc_lambda)

        loss = combined + ewc_loss

        # Gradient accumulation
        scaled_loss = loss / accum
        scaled_loss.backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_soz += losses['soz'].item()
        total_kd += kd_loss_val.item()
        total_ewc += ewc_loss.item()
        n_steps += 1

    n = max(n_steps, 1)
    return {
        'loss': total_loss / n,
        'loss_soz': total_soz / n,
        'loss_kd': total_kd / n,
        'loss_ewc': total_ewc / n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Validation & Test Evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_val_robust(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on multi-patient validation set."""
    model.eval()
    all_probs, all_targets = [], []

    for batch in val_loader:
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)

        outputs = model(x, onset, start)
        probs = outputs['soz_probs'].cpu().numpy()
        targets = batch['monopolar_label'].numpy()
        all_probs.append(probs)
        all_targets.append(targets)

    if not all_probs:
        return {'composite_score': 0.0, 'recall_at_3': 0.0, 'auc': 0.0}

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    metrics = compute_localization_ranking_metrics(probs, targets, ks=(1, 3, 5))

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        valid_aucs = []
        for c in range(targets.shape[1]):
            if 0 < targets[:, c].sum() < len(targets):
                valid_aucs.append(roc_auc_score(targets[:, c], probs[:, c]))
        metrics['auc'] = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    except Exception:
        metrics['auc'] = 0.0

    # Best F1
    probs_dsz = reorder_to_deepsoz(probs)
    targets_dsz = reorder_to_deepsoz(targets)
    best_th, best_f1 = find_best_threshold(targets_dsz, probs_dsz)
    metrics['best_f1'] = best_f1
    metrics['best_threshold'] = best_th

    # Composite score for model selection
    metrics['composite_score'] = (
        0.4 * metrics.get('auc', 0) +
        0.3 * metrics.get('recall_at_3', 0) +
        0.3 * metrics.get('ndcg_at_3', 0)
    )

    return metrics


@torch.no_grad()
def evaluate_fold_test(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    test_loader: DataLoader,
    temp_scaler: RobustTemperatureScaler,
    device: torch.device,
    mc_samples: int,
    fold_info: Dict,
    fold_dir: Path,
) -> Dict:
    """Full test evaluation: channel-level + MC dropout seizure/patient-level."""
    model.eval()
    all_probs, all_logits, all_targets = [], [], []
    all_pids = []

    for batch in test_loader:
        x = batch['x'].to(device)
        outputs = model(x, batch['onset_sec'].to(device),
                        batch['start_sec'].to(device))

        scaled_logits = temp_scaler(outputs['soz_logits'])
        scaled_probs = torch.sigmoid(scaled_logits)

        all_probs.append(scaled_probs.cpu().numpy())
        all_logits.append(scaled_logits.cpu().numpy())
        all_targets.append(batch['monopolar_label'].numpy())
        all_pids.extend(batch['patient_id'])

    if not all_probs:
        return {'test_patients': fold_info['test_patients'], 'n_test': 0}

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    # Reorder to DeepSOZ order
    probs_dsz = reorder_to_deepsoz(probs)
    targets_dsz = reorder_to_deepsoz(targets)

    # Channel-level ranking metrics
    ranking = compute_localization_ranking_metrics(probs, targets)

    # Optimal threshold
    best_th, best_f1 = find_best_threshold(targets_dsz, probs_dsz)

    # MC dropout seizure-level & patient-level
    mc_results = _mc_evaluate_fold(
        model, test_loader, device, mc_samples, temp_scaler,
    )

    # ECE
    ece = _compute_ece(probs.flatten(), targets.flatten())

    results = {
        'test_patients': fold_info['test_patients'],
        'n_test': len(probs),
        'n_test_patients': len(fold_info['test_patients']),
        'recall_at_1': ranking.get('recall_at_1', 0),
        'recall_at_3': ranking.get('recall_at_3', 0),
        'recall_at_5': ranking.get('recall_at_5', 0),
        'ndcg_at_3': ranking.get('ndcg_at_3', 0),
        'mrr': ranking.get('mrr', 0),
        'best_threshold': best_th,
        'best_f1': best_f1,
        'ece': ece,
        'temperature': float(temp_scaler.temperature.item()),
        **mc_results,
    }

    log.info(
        f'  Fold {fold_info["fold"]} test '
        f'({len(fold_info["test_patients"])} patients, {len(probs)} samples): '
        f'R@3={results["recall_at_3"]:.3f} '
        f'bestF1={best_f1:.3f}(th={best_th:.2f}) '
        f'ECE={ece:.4f} T={results["temperature"]:.3f} '
        f'sz_corr={mc_results.get("sz_correct_rate", 0):.3f} '
        f'pt_corr={mc_results.get("pt_correct", 0)}/{mc_results.get("pt_total", 0)}'
    )
    return results


def _mc_evaluate_fold(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    test_loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    temp_scaler: RobustTemperatureScaler,
) -> Dict:
    """MC dropout evaluation (seizure + patient level)."""
    # Group by patient
    patient_mc_maps = defaultdict(list)
    patient_true_onset = {}

    for batch in test_loader:
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        mono_label = batch['monopolar_label'].numpy()
        pids = batch['patient_id']

        for i in range(x.size(0)):
            xi = x[i:i + 1]
            oi = onset[i:i + 1]
            si = start[i:i + 1]

            mc_maps = mc_inference_single(
                model, xi, oi, si, device, n_samples=mc_samples,
            )
            pid = _extract_base_patient(str(pids[i]))
            patient_mc_maps[pid].append(mc_maps)

            true_dsz = reorder_to_deepsoz(mono_label[i:i + 1])[0]
            patient_true_onset[pid] = true_dsz

    if not patient_mc_maps:
        return {}

    # Seizure-level evaluation
    sz_correct_list = []
    for pid, mc_list in patient_mc_maps.items():
        for mc_map in mc_list:
            _, _, correct = final_loc(mc_map, patient_true_onset[pid])
            sz_correct_list.append(correct)

    # Patient-level evaluation
    pt_correct = 0
    pt_total = 0
    for pid, mc_list in patient_mc_maps.items():
        group_mc = np.concatenate(mc_list, axis=0)
        _, _, correct = final_loc(group_mc, patient_true_onset[pid])
        pt_correct += int(correct)
        pt_total += 1

    return {
        'sz_correct_rate': float(np.mean(sz_correct_list)) if sz_correct_list else 0.0,
        'n_seizures': len(sz_correct_list),
        'pt_correct': pt_correct,
        'pt_total': pt_total,
        'pt_accuracy': pt_correct / max(pt_total, 1),
    }


def _compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = targets[mask].mean()
        ece += mask.sum() / total * abs(avg_conf - avg_acc)
    return float(ece)


# ═════════════════════════════════════════════════════════════════════════════
# Single Fold Training
# ═════════════════════════════════════════════════════════════════════════════

def train_single_fold(
    fold_info: Dict,
    private_ds: SOZBrainNetworkDataset,
    tusz_replay_ds: Optional[Subset],
    cfg: FinetuneConfig,
    output_dir: Path,
    device: torch.device,
) -> Dict:
    """Complete training pipeline for one fold."""
    fold_idx = fold_info['fold']
    fold_dir = output_dir / f'fold_{fold_idx:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'\n{"=" * 60}')
    log.info(f'Fold {fold_idx}: test={fold_info["test_patients"]}, '
             f'val={fold_info["val_patients"]}, '
             f'train={len(fold_info["train_indices"])} samples')
    log.info(f'{"=" * 60}')

    # 1. Load model
    model, model_cfg = load_pretrained_model(cfg.checkpoint, device)
    model_cfg.w_region = cfg.w_region
    model_cfg.w_hemisphere = cfg.w_hemisphere
    model_cfg.w_transition = cfg.w_transition
    model_cfg.w_pattern = cfg.w_pattern
    model.cfg = model_cfg

    # 2. Create teacher (frozen copy for KD)
    teacher_model, _ = load_pretrained_model(cfg.checkpoint, device)
    teacher_model = teacher_model.to(device)
    distiller = SOZKnowledgeDistiller(teacher_model, cfg.kd_temperature, cfg.kd_alpha)

    # 3. Build fold data pipeline (leak-free)
    pipeline = build_fold_data_pipeline(private_ds, fold_info, tusz_replay_ds, cfg, device)

    # 4. Set fold-specific pos_weight and channel_weight
    if pipeline['pos_weight'] is not None:
        model.set_pos_weight(pipeline['pos_weight'])
    if pipeline['channel_weight'] is not None:
        model.set_channel_weight(pipeline['channel_weight'])

    # 5. Replace focal loss with smoothed version
    model.focal_loss = SmoothedFocalLoss(
        gamma=cfg.focal_gamma,
        alpha=cfg.focal_alpha,
        epsilon=cfg.label_smoothing,
        pos_weight=pipeline['pos_weight'],
        channel_weight=pipeline['channel_weight'],
    )

    model = model.to(device)

    # 6. Compute EWC reference BEFORE any freezing
    fisher, star_params = None, None
    if cfg.ewc_lambda > 0 and tusz_replay_ds is not None:
        log.info('Computing EWC reference (Fisher + star_params) on full model...')
        ewc_loader = DataLoader(
            tusz_replay_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        fisher, star_params = compute_ewc_reference(
            model, ewc_loader, device, cfg.fisher_samples,
        )

    # 7. Prepare unfreezer and mixing scheduler
    unfreezer = ProgressiveUnfreezer(model, cfg)

    n_tusz = len(tusz_replay_ds) if tusz_replay_ds is not None else 0
    mixing_scheduler = CurriculumMixingScheduler(
        n_private=len(fold_info['train_indices']),
        n_tusz=n_tusz,
        start_ratio=cfg.tusz_mix_start_ratio,
        end_ratio=cfg.tusz_mix_end_ratio,
        total_epochs=unfreezer.total_epochs,
    )

    augmentor = build_augmentor()

    # 8. SWA setup (will be initialized when needed)
    swa_model = None
    swa_active = False

    # 9. Training loop
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0

    training_history = []

    for epoch in range(unfreezer.total_epochs):
        # Phase transition check
        optimizer, scheduler, phase = unfreezer.step(epoch)

        # Build epoch-specific data loader with curriculum mixing
        if tusz_replay_ds is not None and n_tusz > 0:
            train_dataset = ConcatDataset([pipeline['train_sub'], tusz_replay_ds])
            epoch_sampler = mixing_scheduler.get_sampler(epoch)
        else:
            train_dataset = pipeline['train_sub']
            epoch_sampler = None

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            sampler=epoch_sampler,
            shuffle=(epoch_sampler is None),
            collate_fn=collate_fn, num_workers=0, drop_last=False,
        )

        # Train one epoch
        train_metrics = train_one_epoch_v2(
            model, train_loader, optimizer, device, augmentor,
            distiller, fisher, star_params, cfg, epoch,
        )
        if scheduler is not None:
            scheduler.step()

        # SWA
        if epoch >= cfg.swa_start_epoch:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                log.info(f'SWA started at epoch {epoch}')
            swa_model.update_parameters(model)
            swa_active = True

        # Validate
        val_metrics = evaluate_val_robust(model, pipeline['val_loader'], device)

        tusz_ratio = mixing_scheduler.get_tusz_ratio(epoch)
        log.info(
            f'Fold {fold_idx} Epoch {epoch:02d} (P{phase}) | '
            f'loss={train_metrics["loss"]:.4f} soz={train_metrics["loss_soz"]:.4f} '
            f'kd={train_metrics["loss_kd"]:.4f} ewc={train_metrics["loss_ewc"]:.4f} | '
            f'val R@3={val_metrics.get("recall_at_3", 0):.3f} '
            f'AUC={val_metrics.get("auc", 0):.3f} '
            f'F1={val_metrics.get("best_f1", 0):.3f} '
            f'score={val_metrics["composite_score"]:.3f} | '
            f'tusz_ratio={tusz_ratio:.2f}'
        )

        training_history.append({
            'epoch': epoch, 'phase': phase,
            **train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()},
            'tusz_ratio': tusz_ratio,
        })

        # Model selection
        score = val_metrics['composite_score']
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            model.save_checkpoint(str(fold_dir / 'best_model.pt'), extra={
                'epoch': epoch, 'fold': fold_idx, 'val_metrics': val_metrics,
                'phase': phase,
            })
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info(f'Early stopping at epoch {epoch} (patience={cfg.patience})')
                break

    log.info(f'Best epoch: {best_epoch}, best score: {best_score:.4f}')

    # 10. Finalize: choose between SWA model and best checkpoint
    if swa_active and swa_model is not None:
        # Update BN statistics for SWA model
        swa_train_loader = DataLoader(
            pipeline['train_sub'], batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        torch.optim.swa_utils.update_bn(swa_train_loader, swa_model, device=device)

        # Compare SWA vs best checkpoint on validation
        swa_val_metrics = evaluate_val_robust(swa_model.module, pipeline['val_loader'], device)
        swa_score = swa_val_metrics['composite_score']
        log.info(f'SWA val score: {swa_score:.4f} vs best checkpoint: {best_score:.4f}')

        if swa_score > best_score:
            log.info('Using SWA model (better than best checkpoint)')
            eval_model = swa_model.module
            # Save SWA model
            eval_model.save_checkpoint(str(fold_dir / 'swa_model.pt'), extra={
                'swa': True, 'val_metrics': swa_val_metrics,
            })
        else:
            log.info('Using best checkpoint (better than SWA)')
            eval_model, _ = load_pretrained_model(str(fold_dir / 'best_model.pt'), device)
            eval_model = eval_model.to(device)
    else:
        # Load best checkpoint
        best_ckpt = fold_dir / 'best_model.pt'
        if best_ckpt.exists():
            eval_model, _ = load_pretrained_model(str(best_ckpt), device)
            eval_model = eval_model.to(device)
        else:
            eval_model = model

    # 11. Temperature scaling on validation set
    temp_scaler = RobustTemperatureScaler().to(device)
    eval_model.eval()
    val_logits_list, val_targets_list = [], []
    with torch.no_grad():
        for batch in pipeline['val_loader']:
            x = batch['x'].to(device)
            outputs = eval_model(x, batch['onset_sec'].to(device),
                                 batch['start_sec'].to(device))
            val_logits_list.append(outputs['soz_logits'])
            val_targets_list.append(batch['monopolar_label'].to(device))

    if val_logits_list:
        all_logits = torch.cat(val_logits_list)
        all_targets = torch.cat(val_targets_list)
        temp_scaler.fit(all_logits.detach(), all_targets.float())

    # 12. Test evaluation
    fold_results = evaluate_fold_test(
        eval_model, pipeline['test_loader'], temp_scaler, device,
        cfg.mc_samples, fold_info, fold_dir,
    )
    fold_results['best_epoch'] = best_epoch
    fold_results['best_val_score'] = best_score

    # Save training history and fold results
    with open(fold_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)

    serializable = {k: v for k, v in fold_results.items()
                    if isinstance(v, (int, float, str, list, dict, bool))}
    with open(fold_dir / 'fold_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Cleanup teacher to free GPU memory
    del teacher_model, distiller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_results


# ═════════════════════════════════════════════════════════════════════════════
# Results Aggregation
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_kfold_results(
    all_results: List[Dict], output_dir: Path,
) -> Dict:
    """Aggregate results across all k folds."""
    valid = [r for r in all_results if r.get('n_test', 0) > 0]
    n_folds = len(all_results)

    # Patient-level accuracy
    pt_correct = sum(r.get('pt_correct', 0) for r in valid)
    pt_total = sum(r.get('pt_total', 0) for r in valid)
    pt_acc = pt_correct / max(pt_total, 1)

    # Seizure-level accuracy
    total_sz = sum(r.get('n_seizures', 0) for r in valid)
    weighted_sz_corr = sum(
        r.get('sz_correct_rate', 0) * r.get('n_seizures', 0)
        for r in valid
    )
    sz_corr = weighted_sz_corr / max(total_sz, 1)

    # Per-fold metric averages
    metrics_to_avg = [
        'recall_at_1', 'recall_at_3', 'recall_at_5',
        'ndcg_at_3', 'mrr', 'best_f1', 'ece', 'temperature',
    ]
    summary = {}
    for m in metrics_to_avg:
        vals = [r[m] for r in valid if m in r]
        summary[f'mean_{m}'] = float(np.mean(vals)) if vals else 0.0
        summary[f'std_{m}'] = float(np.std(vals)) if vals else 0.0

    summary.update({
        'n_folds': n_folds,
        'n_valid_folds': len(valid),
        'patient_accuracy': pt_acc,
        'patient_correct': pt_correct,
        'patient_total': pt_total,
        'seizure_correct_rate': sz_corr,
        'total_seizures': total_sz,
    })

    # Print summary
    log.info(f'\n{"=" * 60}')
    log.info(f'K-Fold CV Summary ({n_folds} folds)')
    log.info(f'{"=" * 60}')
    log.info(f'  Patient-level accuracy: {pt_acc:.3f} ({pt_correct}/{pt_total})')
    log.info(f'  Seizure-level correct:  {sz_corr:.3f} ({total_sz} seizures)')
    log.info(f'  Mean Recall@1:          {summary["mean_recall_at_1"]:.3f} +/- {summary["std_recall_at_1"]:.3f}')
    log.info(f'  Mean Recall@3:          {summary["mean_recall_at_3"]:.3f} +/- {summary["std_recall_at_3"]:.3f}')
    log.info(f'  Mean Recall@5:          {summary["mean_recall_at_5"]:.3f} +/- {summary["std_recall_at_5"]:.3f}')
    log.info(f'  Mean Best F1:           {summary["mean_best_f1"]:.3f} +/- {summary["std_best_f1"]:.3f}')
    log.info(f'  Mean NDCG@3:            {summary["mean_ndcg_at_3"]:.3f} +/- {summary["std_ndcg_at_3"]:.3f}')
    log.info(f'  Mean MRR:               {summary["mean_mrr"]:.3f} +/- {summary["std_mrr"]:.3f}')
    log.info(f'  Mean ECE:               {summary["mean_ece"]:.4f} +/- {summary["std_ece"]:.4f}')
    log.info(f'  Mean Temperature:       {summary["mean_temperature"]:.3f} +/- {summary["std_temperature"]:.3f}')

    # Per-fold table
    log.info(f'\n--- Per-Fold Results ---')
    log.info(f'| {"Fold":>4} | {"#Pts":>4} | {"#Sz":>3} | {"R@3":>5} | {"F1":>5} | '
             f'{"SzCorr":>6} | {"PtAcc":>5} | {"Score":>5} |')
    log.info(f'|{"-" * 6}|{"-" * 6}|{"-" * 5}|{"-" * 7}|{"-" * 7}|'
             f'{"-" * 8}|{"-" * 7}|{"-" * 7}|')
    for r in valid:
        log.info(
            f'| {r.get("fold", "?"):>4} | {r.get("n_test_patients", 0):>4} | '
            f'{r.get("n_seizures", 0):>3} | '
            f'{r.get("recall_at_3", 0):>5.3f} | {r.get("best_f1", 0):>5.3f} | '
            f'{r.get("sz_correct_rate", 0):>6.3f} | '
            f'{r.get("pt_accuracy", 0):>5.3f} | '
            f'{r.get("best_val_score", 0):>5.3f} |'
        )

    # Save CSV
    rows = []
    for r in valid:
        row = {k: v for k, v in r.items()
               if isinstance(v, (int, float, str, bool))}
        if 'test_patients' in r:
            row['test_patients'] = ';'.join(str(p) for p in r['test_patients'])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'kfold_summary.csv', index=False, encoding='utf-8-sig')

    # Save JSON
    with open(output_dir / 'kfold_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='私有数据 Stratified Group K-Fold 微调 v2',
    )
    # Paths
    p.add_argument('--checkpoint', required=True, help='TUSZ 预训练 checkpoint')
    p.add_argument('--manifest', default='TUSZ/private_manifest_clean.csv')
    p.add_argument('--tusz-manifest', default='TUSZ/combined_manifest.csv')
    p.add_argument('--private-data-root', default='E:/DataSet/EEG/EEG dataset_SUAT')
    p.add_argument('--tusz-data-root', default='F:/dataset/TUSZ/v2.0.3/edf')
    p.add_argument('--output-dir', default='output/private_finetune_v2')

    # Cross-validation
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    # Phase durations
    p.add_argument('--phase1-epochs', type=int, default=12)
    p.add_argument('--phase2-epochs', type=int, default=20)
    p.add_argument('--phase3-epochs', type=int, default=8)
    p.add_argument('--warmup-epochs', type=int, default=3)

    # Learning rates
    p.add_argument('--phase1-lr', type=float, default=3e-4)
    p.add_argument('--phase2-lr', type=float, default=1e-4)
    p.add_argument('--phase3-lr', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=5e-3)

    # Gradient accumulation
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum-steps', type=int, default=8)

    # EWC
    p.add_argument('--ewc-lambda', type=float, default=500.0)
    p.add_argument('--fisher-samples', type=int, default=500)

    # Knowledge distillation
    p.add_argument('--kd-temperature', type=float, default=4.0)
    p.add_argument('--kd-alpha', type=float, default=0.3)

    # TUSZ replay
    p.add_argument('--tusz-replay-size', type=int, default=400)
    p.add_argument('--tusz-mix-start-ratio', type=float, default=0.5)
    p.add_argument('--tusz-mix-end-ratio', type=float, default=0.15)

    # Class imbalance
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--focal-alpha', type=float, default=0.85)
    p.add_argument('--label-smoothing', type=float, default=0.05)

    # SWA
    p.add_argument('--swa-start-epoch', type=int, default=30)
    p.add_argument('--swa-lr', type=float, default=5e-5)

    # Augmentation
    p.add_argument('--lr-mirror-prob', type=float, default=0.3)

    # Evaluation
    p.add_argument('--mc-samples', type=int, default=30)
    p.add_argument('--patience', type=int, default=10)

    # Debug
    p.add_argument('--fold-indices', type=int, nargs='+', default=None,
                   help='仅跑指定折 (如 --fold-indices 0 1)')

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # Build config from args
    cfg = FinetuneConfig(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        tusz_manifest=args.tusz_manifest,
        private_data_root=args.private_data_root,
        tusz_data_root=args.tusz_data_root,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        seed=args.seed,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        warmup_epochs=args.warmup_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        phase3_lr=args.phase3_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        tusz_replay_size=args.tusz_replay_size,
        tusz_mix_start_ratio=args.tusz_mix_start_ratio,
        tusz_mix_end_ratio=args.tusz_mix_end_ratio,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr,
        lr_mirror_prob=args.lr_mirror_prob,
        mc_samples=args.mc_samples,
        patience=args.patience,
        fold_indices=[int(i) for i in args.fold_indices] if args.fold_indices else None,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        cfg_dict = asdict(cfg)
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    # ── Load private dataset ──
    log.info(f'Loading private dataset: {cfg.manifest}')
    private_manifest_ds = ManifestSOZDataset(
        manifest_path=cfg.manifest,
        tusz_data_root=cfg.tusz_data_root,
        private_data_root=cfg.private_data_root,
        source_filter='private',
        label_mode='monopolar',
        region_label_mode='coarse',
    )
    private_ds = SOZBrainNetworkDataset(private_manifest_ds)
    log.info(f'Private dataset: {len(private_ds)} samples')

    # ── Load TUSZ replay buffer ──
    tusz_replay_ds = None
    if cfg.tusz_replay_size > 0:
        log.info(f'Loading TUSZ replay buffer ({cfg.tusz_replay_size} samples)...')
        try:
            tusz_manifest_ds = ManifestSOZDataset(
                manifest_path=cfg.tusz_manifest,
                tusz_data_root=cfg.tusz_data_root,
                source_filter='tusz',
                split_filter=['train'],
                label_mode='monopolar',
                region_label_mode='coarse',
            )
            tusz_full_ds = SOZBrainNetworkDataset(tusz_manifest_ds)
            n_replay = min(cfg.tusz_replay_size, len(tusz_full_ds))
            rng = np.random.RandomState(cfg.seed)
            replay_indices = rng.choice(len(tusz_full_ds), n_replay, replace=False)
            tusz_replay_ds = Subset(tusz_full_ds, replay_indices.tolist())
            log.info(f'TUSZ replay buffer: {len(tusz_replay_ds)} samples')
        except Exception as e:
            log.warning(f'Failed to load TUSZ replay: {e}')
            tusz_replay_ds = None

    # ── Build stratified group k-fold ──
    log.info(f'Building {cfg.n_folds}-fold stratified group CV...')
    folds = build_stratified_group_kfold(
        private_manifest_ds.df, n_folds=cfg.n_folds, seed=cfg.seed,
    )

    if cfg.fold_indices is not None:
        folds = [f for f in folds if f['fold'] in cfg.fold_indices]
        log.info(f'Running subset: {len(folds)} folds')

    # ── Run all folds ──
    all_results = []
    t0 = time.time()

    for fold in folds:
        fold_result = train_single_fold(
            fold, private_ds, tusz_replay_ds, cfg, output_dir, device,
        )
        all_results.append(fold_result)

    elapsed = time.time() - t0
    log.info(f'\nTotal time: {elapsed / 3600:.1f} hours')

    # ── Aggregate results ──
    if all_results:
        aggregate_kfold_results(all_results, output_dir)


if __name__ == '__main__':
    main()
