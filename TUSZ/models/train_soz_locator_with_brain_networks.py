#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_soz_locator_with_brain_networks.py

端到端 SOZ 定位训练脚本，集成脑网络特征。

流程:
  1. 数据加载 (ManifestSOZDataset)
  2. (可选) 对比学习预训练
  3. 三阶段微调 (冻结骨干 -> 解冻TimeFilter -> 全模型)
  4. 测试评估 + 可解释性报告

支持: DDP / AMP / 断点续训 / TensorBoard
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# ── project path ──
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── imports ──
try:
    from models.integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from models.contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from models.brain_network_extractor import MultiScaleBrainNetworkExtractor
    from models.dynamic_network_evolution import DynamicNetworkEvolutionModel
    from models.manifest_dataset import ManifestSOZDataset
    from tasks.stage_detection import (
        EEGStagePretrainDataset,
        NON_SEIZURE_LABEL,
        SEIZURE_LABEL,
        assign_patch_binary_labels,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )
except ImportError:
    from .integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from .contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from .brain_network_extractor import MultiScaleBrainNetworkExtractor
    from .dynamic_network_evolution import DynamicNetworkEvolutionModel
    from .manifest_dataset import ManifestSOZDataset
    from ..tasks.stage_detection import (
        EEGStagePretrainDataset,
        NON_SEIZURE_LABEL,
        SEIZURE_LABEL,
        assign_patch_binary_labels,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

log = logging.getLogger('train_bn')


# =====================================================================
# Helpers
# =====================================================================

def setup_logging(output_dir: Path, rank: int = 0):
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / 'train.log'))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def setup_ddp():
    """Initialise DDP if launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def _summarize_manifest_subset(manifest_ds: ManifestSOZDataset) -> Dict[str, object]:
    df = manifest_ds.df
    if len(df) == 0:
        return {'rows': 0, 'patients': 0, 'sources': {}, 'hemisphere': {}}
    return {
        'rows': int(len(df)),
        'patients': int(df['patient_id'].nunique()),
        'sources': {str(k): int(v) for k, v in df['source'].value_counts().to_dict().items()},
        'hemisphere': {str(k): int(v) for k, v in df['hemisphere'].value_counts().to_dict().items()},
    }


def _format_subset_summary(name: str, summary: Dict[str, object]) -> str:
    return (
        f"{name}: rows={summary['rows']} patients={summary['patients']} "
        f"sources={summary['sources']} hemisphere={summary['hemisphere']}"
    )


def _resolve_holdout_patient_counts(
    n_patients: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    if n_patients < 3:
        raise ValueError(
            f"private_target split requires at least 3 private patients, got {n_patients}"
        )
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_split and test_split must be >= 0")

    n_val = int(round(n_patients * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n_patients * test_ratio)) if test_ratio > 0 else 0
    if val_ratio > 0:
        n_val = max(1, n_val)
    if test_ratio > 0:
        n_test = max(1, n_test)

    max_holdout = max(n_patients - 1, 0)
    while n_val + n_test > max_holdout:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    n_train = n_patients - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Invalid private_target split: train={n_train}, val={n_val}, test={n_test}"
        )
    return {'train': n_train, 'val': n_val, 'test': n_test}


def _allocate_group_targets(
    total_target: int,
    group_sizes: Dict[str, int],
) -> Dict[str, int]:
    if total_target <= 0 or not group_sizes:
        return {str(k): 0 for k in group_sizes}

    total = max(sum(group_sizes.values()), 1)
    allocated = {
        str(k): min(int(v), int(np.floor(v * total_target / total)))
        for k, v in group_sizes.items()
    }
    remaining = total_target - sum(allocated.values())
    if remaining <= 0:
        return allocated

    fractions = sorted(
        group_sizes.items(),
        key=lambda kv: (
            (kv[1] * total_target / total) - allocated[str(kv[0])],
            kv[1],
            str(kv[0]),
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key, capacity in fractions:
            key = str(key)
            if allocated[key] >= int(capacity):
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated


def _build_private_patient_infos(private_manifest_ds: ManifestSOZDataset) -> List[Dict[str, object]]:
    patient_infos: List[Dict[str, object]] = []
    grouped = private_manifest_ds.df.groupby('patient_id', sort=False)
    for patient_id, patient_df in grouped:
        hemi_values = [
            str(v).strip()
            for v in patient_df['hemisphere'].tolist()
            if str(v).strip()
        ]
        hemisphere = Counter(hemi_values).most_common(1)[0][0] if hemi_values else 'U'
        patient_infos.append(
            {
                'patient_id': str(patient_id),
                'n_rows': int(len(patient_df)),
                'hemisphere': hemisphere,
            }
        )
    return patient_infos


def _split_private_patients(
    patient_infos: List[Dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    target_counts = _resolve_holdout_patient_counts(
        len(patient_infos),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    total_rows = sum(int(info['n_rows']) for info in patient_infos)
    avg_rows = total_rows / max(len(patient_infos), 1)
    hemisphere_sizes = Counter(str(info['hemisphere']) for info in patient_infos)
    target_hemi = {
        split: _allocate_group_targets(count, hemisphere_sizes)
        for split, count in target_counts.items()
    }
    target_rows = {
        split: total_rows * count / max(len(patient_infos), 1)
        for split, count in target_counts.items()
    }

    rng = np.random.default_rng(seed)
    ordered_patients: List[Dict[str, object]] = []
    for info in patient_infos:
        item = dict(info)
        item['rand'] = float(rng.random())
        ordered_patients.append(item)
    ordered_patients.sort(
        key=lambda item: (-int(item['n_rows']), float(item['rand']), str(item['patient_id']))
    )

    split_order = ('val', 'test', 'train')
    split_rank = {name: idx for idx, name in enumerate(split_order)}
    split_stats = {
        name: {
            'patient_ids': [],
            'n_rows': 0,
            'hemisphere': Counter(),
        }
        for name in target_counts
    }

    for item in ordered_patients:
        choices = [
            split
            for split in split_order
            if len(split_stats[split]['patient_ids']) < target_counts[split]
        ]
        if not choices:
            raise RuntimeError("No available split bucket while assigning private patients")

        best_key = None
        best_split = None
        for split in choices:
            patient_delta = abs(
                (len(split_stats[split]['patient_ids']) + 1) - target_counts[split]
            )
            row_delta = abs(
                (split_stats[split]['n_rows'] + int(item['n_rows'])) - target_rows[split]
            ) / max(avg_rows, 1.0)
            hemisphere = str(item['hemisphere'])
            hemi_delta = abs(
                (split_stats[split]['hemisphere'][hemisphere] + 1)
                - target_hemi[split].get(hemisphere, 0)
            )
            score = patient_delta * 6.0 + row_delta * 1.5 + hemi_delta * 2.5
            candidate_key = (score, split_rank[split])
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_split = split

        assert best_split is not None
        split_stats[best_split]['patient_ids'].append(str(item['patient_id']))
        split_stats[best_split]['n_rows'] += int(item['n_rows'])
        split_stats[best_split]['hemisphere'][str(item['hemisphere'])] += 1

    return {
        split: sorted(stats['patient_ids'])
        for split, stats in split_stats.items()
    }


def build_soz_datasets(
    args,
    pipeline_cfg,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[str, object]]:
    split_strategy = args.split_strategy
    private_available = args.source in ('all', 'private')

    if split_strategy == 'auto':
        split_strategy = 'private_target' if private_available else 'random'

    if split_strategy == 'private_target' and args.source not in ('all', 'private'):
        raise ValueError(
            f"split_strategy='private_target' requires --source all/private, got {args.source}"
        )

    dataset_kwargs = dict(
        manifest_path=args.manifest,
        private_data_root=args.private_data_root,
        tusz_data_root=args.tusz_data_root,
        label_mode=args.output_mode,
        pipeline_cfg=pipeline_cfg,
    )

    if split_strategy == 'private_target':
        private_all = ManifestSOZDataset(
            source_filter='private',
            **dataset_kwargs,
        )
        if len(private_all) == 0:
            raise ValueError(
                "split_strategy='private_target' requires private samples, but none were found"
            )

        patient_infos = _build_private_patient_infos(private_all)
        patient_split = _split_private_patients(
            patient_infos,
            val_ratio=args.val_split,
            test_ratio=args.test_split,
            seed=args.seed,
        )

        train_parts: List[torch.utils.data.Dataset] = []
        split_meta: Dict[str, object] = {
            'strategy': 'private_target',
            'private_patient_split': patient_split,
            'log_lines': [],
        }

        if args.source == 'all':
            tusz_train_manifest = ManifestSOZDataset(
                source_filter='tusz',
                **dataset_kwargs,
            )
            train_parts.append(
                SOZBrainNetworkDataset(
                    tusz_train_manifest,
                    precomputed_dir=args.precomputed_dir,
                )
            )
            tusz_summary = _summarize_manifest_subset(tusz_train_manifest)
            split_meta['train_tusz_summary'] = tusz_summary
            split_meta['log_lines'].append(_format_subset_summary('train/tusz_all', tusz_summary))

        private_train_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['train'],
            **dataset_kwargs,
        )
        private_val_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['val'],
            **dataset_kwargs,
        )
        private_test_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['test'],
            **dataset_kwargs,
        )

        split_meta['train_private_summary'] = _summarize_manifest_subset(private_train_manifest)
        split_meta['val_summary'] = _summarize_manifest_subset(private_val_manifest)
        split_meta['test_summary'] = _summarize_manifest_subset(private_test_manifest)

        train_parts.append(
            SOZBrainNetworkDataset(
                private_train_manifest,
                precomputed_dir=args.precomputed_dir,
            )
        )
        val_ds = SOZBrainNetworkDataset(
            private_val_manifest,
            precomputed_dir=args.precomputed_dir,
        )
        test_ds = SOZBrainNetworkDataset(
            private_test_manifest,
            precomputed_dir=args.precomputed_dir,
        )

        split_meta['log_lines'].append(
            _format_subset_summary('train/private', split_meta['train_private_summary'])
        )
        split_meta['log_lines'].append(
            f"train/private patients={patient_split['train']}"
        )
        split_meta['log_lines'].append(
            _format_subset_summary('val/private', split_meta['val_summary'])
        )
        split_meta['log_lines'].append(
            f"val/private patients={patient_split['val']}"
        )
        split_meta['log_lines'].append(
            _format_subset_summary('test/private', split_meta['test_summary'])
        )
        split_meta['log_lines'].append(
            f"test/private patients={patient_split['test']}"
        )

        if len(train_parts) == 1:
            train_ds = train_parts[0]
        else:
            train_ds = ConcatDataset(train_parts)
        return train_ds, val_ds, test_ds, split_meta

    manifest_ds = ManifestSOZDataset(
        source_filter=args.source,
        **dataset_kwargs,
    )
    dataset = SOZBrainNetworkDataset(manifest_ds, precomputed_dir=args.precomputed_dir)
    n = len(dataset)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    split_meta = {
        'strategy': 'random',
        'log_lines': [
            _format_subset_summary('all_sources', _summarize_manifest_subset(manifest_ds)),
            f"random_split train={n_train} val={n_val} test={n_test}",
        ],
    }
    return train_ds, val_ds, test_ds, split_meta


# =====================================================================
# Metrics
# =====================================================================

def compute_localization_ranking_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """Ranking metrics better aligned with multi-channel SOZ localization."""
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    metrics: Dict[str, float] = {'mrr': 0.0, 'valid_localization_samples': 0.0}
    if probs.size == 0 or targets.size == 0 or probs.ndim != 2 or targets.ndim != 2:
        for k in ks:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        return metrics

    recall_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    valid = 0

    for p, t in zip(probs, targets):
        pos_idx = np.flatnonzero(t > 0.5)
        if len(pos_idx) == 0:
            continue
        valid += 1
        order = np.argsort(p)[::-1]
        pos_set = set(pos_idx.tolist())

        first_positive_rank = None
        for rank, idx in enumerate(order, start=1):
            if idx in pos_set:
                first_positive_rank = rank
                break
        if first_positive_rank is not None:
            mrr_sum += 1.0 / first_positive_rank

        for k in ks:
            topk = order[:min(k, len(order))]
            hits = sum(1 for idx in topk if idx in pos_set)
            recall_sums[k] += hits / max(len(pos_idx), 1)
            precision_sums[k] += hits / max(len(topk), 1)

            dcg = 0.0
            for rank, idx in enumerate(topk, start=1):
                if idx in pos_set:
                    dcg += 1.0 / np.log2(rank + 1)
            ideal_hits = min(len(pos_idx), len(topk))
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0

    metrics['valid_localization_samples'] = float(valid)
    if valid == 0:
        for k in ks:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        return metrics

    metrics['mrr'] = mrr_sum / valid
    for k in ks:
        metrics[f'recall_at_{k}'] = recall_sums[k] / valid
        metrics[f'precision_at_{k}'] = precision_sums[k] / valid
        metrics[f'ndcg_at_{k}'] = ndcg_sums[k] / valid
    return metrics


def get_auc_valid_mask(targets: np.ndarray) -> np.ndarray:
    """AUC is only defined when a channel has both positive and negative samples."""
    targets = np.asarray(targets)
    if targets.size == 0 or targets.ndim != 2:
        return np.zeros((0,), dtype=bool)
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    return np.logical_and(pos > 0, neg > 0)


def compute_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    if roc_auc_score is None:
        return 0.0
    valid = get_auc_valid_mask(targets)
    if valid.sum() == 0:
        return 0.0
    auc_values: List[float] = []
    for idx in np.where(valid)[0]:
        try:
            auc_values.append(float(roc_auc_score(targets[:, idx], probs[:, idx])))
        except ValueError:
            continue
    return float(np.mean(auc_values)) if auc_values else 0.0


def build_selection_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    """Prefer calibrated ranking quality, then localization quality and auxiliary heads."""
    return (
        float(metrics.get('auc', 0.0)),
        float(metrics.get('ndcg_at_3', 0.0)),
        float(metrics.get('mrr', 0.0)),
        float(metrics.get('recall_at_3', metrics.get('top3', 0.0))),
        float(metrics.get('region_acc', 0.0)),
        float(metrics.get('hemisphere_acc', 0.0)),
    )


def compute_multilabel_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    if len(probs) == 0:
        return 0.0
    preds = probs >= threshold
    truth = targets >= 0.5
    return float((preds == truth).mean())


def compute_multiclass_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
) -> float:
    if len(logits) == 0:
        return 0.0
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(axis=1)
    return float((preds[mask] == targets[mask]).mean())


def compute_patch_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[float, int]:
    if logits.numel() == 0:
        return 0.0, 0
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    valid = int(mask.sum().item())
    if valid == 0:
        return 0.0, 0
    correct = (preds[mask] == targets[mask]).float().mean().item()
    return float(correct), valid


def count_trainable_parameters(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def compute_binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probs.size == 0 or targets.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0,
            'balanced_acc': 0.0,
            'auc': 0.0,
            'tp': 0.0,
            'fp': 0.0,
            'tn': 0.0,
            'fn': 0.0,
        }

    preds = (probs >= threshold).astype(np.int64)
    tp = float(np.logical_and(preds == 1, targets == 1).sum())
    fp = float(np.logical_and(preds == 1, targets == 0).sum())
    tn = float(np.logical_and(preds == 0, targets == 0).sum())
    fn = float(np.logical_and(preds == 0, targets == 1).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_acc = 0.5 * (recall + specificity)

    auc = 0.0
    if roc_auc_score is not None and np.unique(targets).size > 1:
        try:
            auc = float(roc_auc_score(targets, probs))
        except ValueError:
            auc = 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_acc': float(balanced_acc),
        'auc': float(auc),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def compute_binary_metrics_from_counts(
    tp: float,
    fp: float,
    tn: float,
    fn: float,
) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_acc = 0.5 * (recall + specificity)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_acc': float(balanced_acc),
    }


def estimate_stage_patch_statistics(
    dataset: EEGStagePretrainDataset,
    ignore_index: int = -100,
) -> Dict[str, object]:
    cfg = dataset.pipeline.cfg
    pos = 0
    neg = 0
    ignored = 0

    for sample in dataset.samples:
        labels, _ = assign_patch_binary_labels(
            seizure_start_sec=sample.seizure_start_sec,
            seizure_end_sec=sample.seizure_end_sec,
            window_start_sec=float(sample.center_sec - cfg.pre_onset_sec),
            file_duration_sec=sample.duration_sec,
            n_patches=cfg.n_patches,
            patch_len=cfg.patch_len,
            fs=cfg.target_fs,
            ignore_index=ignore_index,
        )
        pos += int((labels == SEIZURE_LABEL).sum())
        neg += int((labels == NON_SEIZURE_LABEL).sum())
        ignored += int((labels == ignore_index).sum())

    total = pos + neg
    counts = np.array([neg, pos], dtype=np.float64)
    if total > 0:
        class_weight = counts.sum() / np.clip(counts, a_min=1.0, a_max=None)
        class_weight = class_weight / class_weight.mean()
    else:
        class_weight = np.ones(2, dtype=np.float64)

    return {
        'valid_patches': int(total),
        'positive_patches': int(pos),
        'negative_patches': int(neg),
        'ignored_patches': int(ignored),
        'positive_rate': float(pos / total) if total > 0 else 0.0,
        'class_weight': torch.tensor(class_weight, dtype=torch.float32),
    }


def stage_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == 'loss':
        return -float(metrics['loss'])
    if metric_name == 'acc':
        return float(metrics['patch_acc'])
    if metric_name == 'auc':
        return float(metrics['auc'])
    if metric_name == 'recall':
        return float(metrics['recall'])
    return float(metrics['f1'])


def stage_metric_display_value(metric_value: float, metric_name: str) -> float:
    if metric_name == 'loss':
        return -float(metric_value)
    return float(metric_value)


def summarize_status_counts(status_counts: Dict[str, int], top_k: int = 6) -> str:
    if not status_counts:
        return 'none'
    counter = Counter({str(k): int(v) for k, v in status_counts.items()})
    return ', '.join(f'{key}:{value}' for key, value in counter.most_common(top_k))


def compute_pos_weight(loader, device='cpu') -> torch.Tensor:
    """Compute pos_weight = n_neg / n_pos per channel from the full dataset."""
    pos_sum = None
    total = 0
    for batch in loader:
        y = batch['label']
        if pos_sum is None:
            pos_sum = torch.zeros(y.shape[1], dtype=torch.float64)
        pos_sum += y.sum(dim=0).double()
        total += y.shape[0]
    neg_sum = total - pos_sum
    pw = (neg_sum / pos_sum.clamp(min=1.0)).float()
    pw = pw.clamp(max=50.0)
    global_pos_rate = pos_sum.sum() / (total * pos_sum.shape[0])
    per_ch_rate = pos_sum / total
    log.info(f"pos_weight per channel: min={pw.min():.1f}, max={pw.max():.1f}, "
             f"mean={pw.mean():.1f}, pos_rate={global_pos_rate:.4f}")
    log.info(f"  per-channel pos_rate: {[f'{r:.2f}' for r in per_ch_rate.tolist()]}")

    if global_pos_rate > 0.40:
        log.warning(
            f"  *** LABEL ANOMALY: global pos_rate={global_pos_rate:.3f} (>{40}%) ***\n"
            f"  This means {global_pos_rate*100:.1f}% of all channel-labels are positive.\n"
            f"  Typical SOZ labeling should have ~10-20% positive rate.\n"
            f"  Likely cause: onset_channels in manifest are the UNION across all\n"
            f"  seizure events per file, inflating labels. Check generate_manifest.py\n"
            f"  and ensure per-event onset channels are used, not file-level union."
        )

    # count samples with extreme positive counts
    n_ch = pos_sum.shape[0]
    all_pos_count = 0
    high_pos_count = 0
    for batch in loader:
        y = batch['label']
        ch_pos = y.sum(dim=1)  # per-sample positive channel count
        all_pos_count += (ch_pos == n_ch).sum().item()
        high_pos_count += (ch_pos > n_ch * 0.5).sum().item()
    if all_pos_count > 0:
        log.warning(f"  {all_pos_count}/{total} samples have ALL {n_ch} channels = 1")
    if high_pos_count > total * 0.3:
        log.warning(f"  {high_pos_count}/{total} samples have >50% channels positive")

    return pw.to(device)


# =====================================================================
# Dataset wrapper (adds onset / window metadata for patching)
# =====================================================================

class SOZBrainNetworkDataset(torch.utils.data.Dataset):
    """
    Wraps ManifestSOZDataset and provides seizure metadata
    needed by SeizureAlignedAdaptivePatching.
    """

    def __init__(self, manifest_ds: ManifestSOZDataset, precomputed_dir: str = None):
        self.ds = manifest_ds
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None

    def __len__(self):
        return len(self.ds)

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if not self.precomputed_dir:
            return None
        row = self.ds.df.iloc[idx]
        edf_rel = Path(str(row.get('edf_path', '')))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        # Keep directory structure: precomputed_dir / dir_of_edf / filename_start_sec.npz
        # Using .with_suffix('') to remove the .edf extension before appending
        rel_path = edf_rel.parent / f"{edf_rel.stem}_w{start_sec:.1f}.npz"
        
        cache_file = self.precomputed_dir / rel_path
        # Ensure the subdirectories exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        return cache_file

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # x = sample['data']                         # [22, 20, 100]
        # label = sample['label']                    # [22] or [19]

        x, label, mask, meta, y_bipolar, y_monopolar, y_region, y_hemisphere = sample

        # flatten to [22, 2000] for patching module
        C, P, L = x.shape
        x_flat = x.reshape(C, P * L)

        # extract onset / start from manifest row
        row = self.ds.df.iloc[idx]
        onset_sec = float(row.get('onset_sec', 5.0))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        ret = {
            'idx': idx,
            'x': x_flat,
            'label': label,
            'bipolar_label': y_bipolar,
            'monopolar_label': y_monopolar,
            'region_label': y_region,
            'hemisphere_label': y_hemisphere,
            'onset_sec': onset_sec,
            'start_sec': start_sec,
            'source': row.get('source', 'unknown'),
            'patient_id': row.get('patient_id', 'unknown'),
            'edf_path': row.get('edf_path', ''),
        }
        
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                data = np.load(str(cache_path))
                ret['brain_nets'] = torch.from_numpy(data['brain_nets'])
                ret['valid_patch_counts'] = torch.tensor(data['valid_patch_counts'])
                ret['rel_time'] = torch.from_numpy(data['rel_time'])
            except Exception as e:
                pass # Fallback to online computation if loading fails
                
        return ret


def collate_fn(batch):
    ret = {
        'idx': [b['idx'] for b in batch],
        'x': torch.stack([b['x'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'bipolar_label': torch.stack([b['bipolar_label'] for b in batch]),
        'monopolar_label': torch.stack([b['monopolar_label'] for b in batch]),
        'region_label': torch.stack([b['region_label'] for b in batch]),
        'hemisphere_label': torch.stack([b['hemisphere_label'] for b in batch]),
        'onset_sec': torch.tensor([b['onset_sec'] for b in batch]),
        'start_sec': torch.tensor([b['start_sec'] for b in batch]),
        'source': [b['source'] for b in batch],
        'patient_id': [b['patient_id'] for b in batch],
        'edf_path': [b['edf_path'] for b in batch],
    }
    
    if all('brain_nets' in b for b in batch):
        ret['brain_nets'] = torch.stack([b['brain_nets'] for b in batch])
        ret['valid_patch_counts'] = torch.stack([b['valid_patch_counts'] for b in batch])
        ret['rel_time'] = torch.stack([b['rel_time'] for b in batch])
        
    return ret


# =====================================================================
# Training loops
# =====================================================================

def train_one_epoch(
    model, loader, optimizer, scaler, device, epoch, cfg, writer=None,
):
    model.train()
    base = model.module if hasattr(model, 'module') else model
    total_loss, n_batches = 0.0, 0
    all_probs, all_targets = [], []
    all_region_probs, all_region_targets = [], []
    all_hemi_logits, all_hemi_targets = [], []
    loss_sums: Dict[str, float] = {}

    for step, batch in enumerate(loader):
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        region_label = batch['region_label'].to(device)
        hemisphere_label = batch['hemisphere_label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        
        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(
                x, onset, start,
                valid_patch_counts=vp_counts,
                brain_networks=brain_nets,
                rel_time=rel_time,
            )

            # build aux targets
            vm = DynamicNetworkEvolutionModel._build_valid_mask(
                outputs['valid_patch_counts'],
                outputs['transition_probs'].size(1),
            )
            aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
                outputs['seizure_relative_time'], vm,
            )

            # Compute sample_weight for soft suppression of generalized seizures (pos_ratio > 0.5)
            pos_ratio = label.sum(dim=1) / max(label.shape[1], 1)
            sample_weight = torch.where(
                pos_ratio > 0.5, 
                torch.tensor(0.05, device=device, dtype=torch.float32), 
                torch.tensor(1.0, device=device, dtype=torch.float32)
            )

            loss, losses = base.compute_loss(
                outputs, label,
                region_targets=region_label,
                hemisphere_targets=hemisphere_label,
                transition_targets=aux['transition_targets'].to(device),
                pattern_targets=aux['pattern_targets'].to(device),
                sample_weight=sample_weight,
            )

        # NaN check
        if torch.isnan(loss):
            log.warning(f"NaN loss at epoch {epoch} step {step}, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())
        all_probs.append(outputs['soz_probs'].detach().cpu().numpy())
        all_targets.append(label.cpu().numpy())
        all_region_probs.append(outputs['region_probs'].detach().cpu().numpy())
        all_region_targets.append(region_label.cpu().numpy())
        all_hemi_logits.append(outputs['hemisphere_logits'].detach().cpu().numpy())
        all_hemi_targets.append(hemisphere_label.cpu().numpy())

        # logits monitoring (every 50 steps)
        if step % 50 == 0:
            with torch.no_grad():
                soz_l = outputs['soz_logits'].detach()
                soz_p = outputs['soz_probs'].detach()
                gate_w = outputs.get('gate_weights')
                region_loss = losses.get('region')
                hemisphere_loss = losses.get('hemisphere')
                log.info(
                    f"  [E{epoch} S{step}] "
                    f"logits(min={soz_l.min():.3f}, max={soz_l.max():.3f}, "
                    f"mean={soz_l.mean():.3f}, std={soz_l.std():.3f}) "
                    f"probs(min={soz_p.min():.3f}, max={soz_p.max():.3f}, "
                    f"mean={soz_p.mean():.3f}) "
                    f"loss={loss.item():.4f}"
                )
                if region_loss is not None or hemisphere_loss is not None:
                    log.info(
                        f"           aux(region={region_loss.detach().item():.4f}, "
                        f"hemisphere={hemisphere_loss.detach().item():.4f})"
                    )
                if gate_w is not None:
                    log.info(
                        f"           gate(min={gate_w.min():.3f}, "
                        f"max={gate_w.max():.3f}, mean={gate_w.mean():.3f})"
                    )

    avg_loss = total_loss / max(n_batches, 1)
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    region_probs = np.concatenate(all_region_probs, axis=0)
    region_targets = np.concatenate(all_region_targets, axis=0)
    hemi_logits = np.concatenate(all_hemi_logits, axis=0)
    hemi_targets = np.concatenate(all_hemi_targets, axis=0)
    rank_metrics = compute_localization_ranking_metrics(probs, targets, ks=(1, 3, 5))
    recall_at_1 = rank_metrics['recall_at_1']
    recall_at_3 = rank_metrics['recall_at_3']
    recall_at_5 = rank_metrics['recall_at_5']
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0
    auc_valid_channels = int(get_auc_valid_mask(targets).sum())
    region_acc = compute_multilabel_accuracy(region_probs, region_targets)
    hemisphere_acc = compute_multiclass_accuracy(hemi_logits, hemi_targets)
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/recall_at_1', recall_at_1, epoch)
        writer.add_scalar('train/recall_at_3', recall_at_3, epoch)
        writer.add_scalar('train/recall_at_5', recall_at_5, epoch)
        writer.add_scalar('train/precision_at_3', rank_metrics['precision_at_3'], epoch)
        writer.add_scalar('train/ndcg_at_3', rank_metrics['ndcg_at_3'], epoch)
        writer.add_scalar('train/mrr', rank_metrics['mrr'], epoch)
        writer.add_scalar('train/auc', auc, epoch)
        writer.add_scalar('train/region_acc', region_acc, epoch)
        writer.add_scalar('train/hemisphere_acc', hemisphere_acc, epoch)
        for name, value in avg_losses.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        with torch.no_grad():
            for name, param in base.named_parameters():
                if param.grad is not None and param.requires_grad:
                    writer.add_scalar(f'grad_norm/{name}',
                                      param.grad.norm().item(), epoch)

    return {
        'loss': avg_loss,
        'top1': recall_at_1,
        'top3': recall_at_3,
        'top5': recall_at_5,
        'auc': auc,
        'auc_valid_channels': auc_valid_channels,
        'region_acc': region_acc,
        'hemisphere_acc': hemisphere_acc,
        **rank_metrics,
        **avg_losses,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    base = model.module if hasattr(model, 'module') else model
    all_probs, all_targets, all_logits = [], [], []
    all_region_probs, all_region_targets = [], []
    all_hemi_logits, all_hemi_targets = [], []
    loss_sums: Dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        region_label = batch['region_label'].to(device)
        hemisphere_label = batch['hemisphere_label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        
        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        out = model(
            x, onset, start,
            valid_patch_counts=vp_counts,
            brain_networks=brain_nets,
            rel_time=rel_time,
        )

        vm = DynamicNetworkEvolutionModel._build_valid_mask(
            out['valid_patch_counts'],
            out['transition_probs'].size(1),
        )
        aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
            out['seizure_relative_time'], vm,
        )
        pos_ratio = label.sum(dim=1) / max(label.shape[1], 1)
        sample_weight = torch.where(
            pos_ratio > 0.5,
            torch.tensor(0.05, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32),
        )
        _, losses = base.compute_loss(
            out,
            label,
            region_targets=region_label,
            hemisphere_targets=hemisphere_label,
            transition_targets=aux['transition_targets'].to(device),
            pattern_targets=aux['pattern_targets'].to(device),
            sample_weight=sample_weight,
        )

        all_probs.append(out['soz_probs'].cpu().numpy())
        all_targets.append(label.cpu().numpy())
        all_logits.append(out['soz_logits'].cpu().numpy())
        all_region_probs.append(out['region_probs'].cpu().numpy())
        all_region_targets.append(region_label.cpu().numpy())
        all_hemi_logits.append(out['hemisphere_logits'].cpu().numpy())
        all_hemi_targets.append(hemisphere_label.cpu().numpy())
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())
        n_batches += 1

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    region_probs = np.concatenate(all_region_probs, axis=0)
    region_targets = np.concatenate(all_region_targets, axis=0)
    hemi_logits = np.concatenate(all_hemi_logits, axis=0)
    hemi_targets = np.concatenate(all_hemi_targets, axis=0)
    rank_metrics = compute_localization_ranking_metrics(probs, targets, ks=(1, 3, 5))
    recall_at_1 = rank_metrics['recall_at_1']
    recall_at_3 = rank_metrics['recall_at_3']
    recall_at_5 = rank_metrics['recall_at_5']
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0
    auc_valid_channels = int(get_auc_valid_mask(targets).sum())
    region_acc = compute_multilabel_accuracy(region_probs, region_targets)
    hemisphere_acc = compute_multiclass_accuracy(hemi_logits, hemi_targets)
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    log.info(
        f"  [eval] logits(min={logits.min():.3f}, max={logits.max():.3f}, "
        f"mean={logits.mean():.3f}, std={logits.std():.3f}) "
        f"probs(min={probs.min():.3f}, max={probs.max():.3f}, "
        f"mean={probs.mean():.3f}) "
        f"label_pos_rate={targets.mean():.4f} "
        f"r3={recall_at_3:.3f} "
        f"ndcg3={rank_metrics['ndcg_at_3']:.3f} "
        f"mrr={rank_metrics['mrr']:.3f} "
        f"auc_valid_ch={auc_valid_channels}/{targets.shape[1]} "
        f"region_acc={region_acc:.3f} "
        f"hemi_acc={hemisphere_acc:.3f}"
    )

    return {
        'top1': recall_at_1,
        'top3': recall_at_3,
        'top5': recall_at_5,
        'auc': auc,
        'auc_valid_channels': auc_valid_channels,
        'region_acc': region_acc,
        'hemisphere_acc': hemisphere_acc,
        **rank_metrics,
        'probs': probs,
        'targets': targets,
        'region_probs': region_probs,
        'region_targets': region_targets,
        'hemisphere_logits': hemi_logits,
        'hemisphere_targets': hemi_targets,
        **avg_losses,
    }


def train_stage_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    writer=None,
    show_progress: bool = False,
    log_every: int = 20,
):
    model.train()
    base = model.module if hasattr(model, 'module') else model
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0
    total_pos = 0
    n_batches = 0
    loss_sums: Dict[str, float] = {}
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    seen_windows = 0
    effective_windows = 0
    skipped_windows = 0
    status_counts: Counter = Counter()

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=f'stage-train {epoch + 1}',
            leave=False,
            dynamic_ncols=True,
        )

    for step, batch in enumerate(iterator, start=1):
        x = batch['x'].to(device)
        stage_labels = batch['stage_labels'].to(device)
        load_status = [str(s) for s in batch.get('load_status', [])]
        stage_valid_count = batch.get('stage_valid_count', None)
        if stage_valid_count is not None:
            batch_valid_counts = stage_valid_count.cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        else:
            batch_valid_counts = (stage_labels != base.cfg.stage_ignore_index).sum(dim=1).cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        seen_windows += len(load_status) if load_status else int(stage_labels.size(0))
        effective_windows += effective_step_windows
        skipped_windows += skipped_step_windows
        if load_status:
            status_counts.update(load_status)

        valid_patches = int((stage_labels != base.cfg.stage_ignore_index).sum().item())
        if valid_patches == 0:
            continue

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(x)
            loss, losses = base.compute_stage_loss(outputs, stage_labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()

        acc, n_valid = compute_patch_accuracy(
            outputs['stage_logits'].detach(),
            stage_labels,
            ignore_index=base.cfg.stage_ignore_index,
        )
        valid_mask = stage_labels != base.cfg.stage_ignore_index
        valid_probs = torch.softmax(outputs['stage_logits'].detach(), dim=-1)[..., 1][valid_mask]
        valid_targets = stage_labels[valid_mask]
        total_correct += acc * n_valid
        total_valid += n_valid
        total_pos += int((stage_labels == 1).sum().item())
        total_loss += float(loss.detach().item())
        n_batches += 1
        if valid_probs.numel() > 0:
            all_probs.append(valid_probs.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())
            valid_preds = (valid_probs >= 0.5).long()
            tp += float(((valid_preds == 1) & (valid_targets == 1)).sum().item())
            fp += float(((valid_preds == 1) & (valid_targets == 0)).sum().item())
            tn += float(((valid_preds == 0) & (valid_targets == 0)).sum().item())
            fn += float(((valid_preds == 0) & (valid_targets == 1)).sum().item())
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())

        running_loss = total_loss / max(n_batches, 1)
        running_acc = total_correct / max(total_valid, 1)
        running_pos = total_pos / max(total_valid, 1)
        running_binary = compute_binary_metrics_from_counts(tp, fp, tn, fn)
        running_f1 = running_binary['f1']
        running_recall = running_binary['recall']
        running_skip_rate = skipped_windows / max(seen_windows, 1)
        if show_progress:
            iterator.set_postfix(
                loss=f'{running_loss:.4f}',
                acc=f'{running_acc:.3f}',
                rec=f'{running_recall:.3f}',
                f1=f'{running_f1:.3f}',
                skip=f'{running_skip_rate:.2%}',
                pos=f'{running_pos:.3f}',
            )
        if log_every > 0 and (step == 1 or step % log_every == 0):
            log.info(
                "  [stage train] epoch=%d step=%d/%d loss=%.4f acc=%.3f rec=%.3f f1=%.3f "
                "pos=%.3f valid_patches=%d seen=%d effective=%d skipped=%d status=%s",
                epoch + 1,
                step,
                len(loader),
                running_loss,
                running_acc,
                running_recall,
                running_f1,
                running_pos,
                total_valid,
                seen_windows,
                effective_windows,
                skipped_windows,
                summarize_status_counts(status_counts),
            )

    avg_loss = total_loss / max(n_batches, 1)
    patch_acc = total_correct / max(total_valid, 1)
    pos_rate = total_pos / max(total_valid, 1)
    binary_metrics = compute_binary_metrics(
        np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float64),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64),
    )
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/patch_acc', patch_acc, epoch)
        writer.add_scalar('train/positive_rate', pos_rate, epoch)
        writer.add_scalar('train/precision', binary_metrics['precision'], epoch)
        writer.add_scalar('train/recall', binary_metrics['recall'], epoch)
        writer.add_scalar('train/f1', binary_metrics['f1'], epoch)
        writer.add_scalar('train/balanced_acc', binary_metrics['balanced_acc'], epoch)
        writer.add_scalar('train/auc', binary_metrics['auc'], epoch)
        writer.add_scalar('train/valid_patches', total_valid, epoch)
        writer.add_scalar('train/seen_windows', seen_windows, epoch)
        writer.add_scalar('train/effective_windows', effective_windows, epoch)
        writer.add_scalar('train/skipped_windows', skipped_windows, epoch)
        writer.add_scalar('train/skip_rate', skipped_windows / max(seen_windows, 1), epoch)
        writer.add_scalar(
            'train/mean_valid_patches_per_effective_window',
            total_valid / max(effective_windows, 1),
            epoch,
        )
        for name, value in avg_losses.items():
            writer.add_scalar(f'train/{name}', value, epoch)

    return {
        'loss': avg_loss,
        'patch_acc': patch_acc,
        'positive_rate': pos_rate,
        'valid_patches': int(total_valid),
        'seen_windows': int(seen_windows),
        'effective_windows': int(effective_windows),
        'skipped_windows': int(skipped_windows),
        'skip_rate': float(skipped_windows / max(seen_windows, 1)),
        'mean_valid_patches_per_effective_window': float(total_valid / max(effective_windows, 1)),
        'load_status_counts': dict(status_counts),
        **binary_metrics,
        **avg_losses,
    }


@torch.no_grad()
def evaluate_stage(
    model,
    loader,
    device,
    show_progress: bool = False,
    log_every: int = 20,
):
    model.eval()
    base = model.module if hasattr(model, 'module') else model
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0
    total_pos = 0
    n_batches = 0
    loss_sums: Dict[str, float] = {}
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    seen_windows = 0
    effective_windows = 0
    skipped_windows = 0
    status_counts: Counter = Counter()

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc='stage-val',
            leave=False,
            dynamic_ncols=True,
        )

    for step, batch in enumerate(iterator, start=1):
        x = batch['x'].to(device)
        stage_labels = batch['stage_labels'].to(device)
        load_status = [str(s) for s in batch.get('load_status', [])]
        stage_valid_count = batch.get('stage_valid_count', None)
        if stage_valid_count is not None:
            batch_valid_counts = stage_valid_count.cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        else:
            batch_valid_counts = (stage_labels != base.cfg.stage_ignore_index).sum(dim=1).cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        seen_windows += len(load_status) if load_status else int(stage_labels.size(0))
        effective_windows += effective_step_windows
        skipped_windows += skipped_step_windows
        if load_status:
            status_counts.update(load_status)

        valid_patches = int((stage_labels != base.cfg.stage_ignore_index).sum().item())
        if valid_patches == 0:
            continue

        outputs = model(x)
        loss, losses = base.compute_stage_loss(outputs, stage_labels)
        acc, n_valid = compute_patch_accuracy(
            outputs['stage_logits'],
            stage_labels,
            ignore_index=base.cfg.stage_ignore_index,
        )
        valid_mask = stage_labels != base.cfg.stage_ignore_index
        valid_probs = torch.softmax(outputs['stage_logits'], dim=-1)[..., 1][valid_mask]
        valid_targets = stage_labels[valid_mask]
        total_correct += acc * n_valid
        total_valid += n_valid
        total_pos += int((stage_labels == 1).sum().item())
        total_loss += float(loss.detach().item())
        n_batches += 1
        if valid_probs.numel() > 0:
            all_probs.append(valid_probs.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())
            valid_preds = (valid_probs >= 0.5).long()
            tp += float(((valid_preds == 1) & (valid_targets == 1)).sum().item())
            fp += float(((valid_preds == 1) & (valid_targets == 0)).sum().item())
            tn += float(((valid_preds == 0) & (valid_targets == 0)).sum().item())
            fn += float(((valid_preds == 0) & (valid_targets == 1)).sum().item())
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())

        running_binary = compute_binary_metrics_from_counts(tp, fp, tn, fn)
        running_f1 = running_binary['f1']
        running_recall = running_binary['recall']
        if hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                loss=f'{total_loss / max(n_batches, 1):.4f}',
                acc=f'{total_correct / max(total_valid, 1):.3f}',
                rec=f'{running_recall:.3f}',
                f1=f'{running_f1:.3f}',
                skip=f'{skipped_windows / max(seen_windows, 1):.2%}',
                pos=f'{total_pos / max(total_valid, 1):.3f}',
            )
        if log_every > 0 and (step == 1 or step % log_every == 0):
            log.info(
                "  [stage val] step=%d/%d loss=%.4f acc=%.3f rec=%.3f f1=%.3f "
                "pos=%.3f valid_patches=%d seen=%d effective=%d skipped=%d status=%s",
                step,
                len(loader),
                total_loss / max(n_batches, 1),
                total_correct / max(total_valid, 1),
                running_recall,
                running_f1,
                total_pos / max(total_valid, 1),
                total_valid,
                seen_windows,
                effective_windows,
                skipped_windows,
                summarize_status_counts(status_counts),
            )

    avg_loss = total_loss / max(n_batches, 1)
    patch_acc = total_correct / max(total_valid, 1)
    pos_rate = total_pos / max(total_valid, 1)
    binary_metrics = compute_binary_metrics(
        np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float64),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64),
    )
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }
    return {
        'loss': avg_loss,
        'patch_acc': patch_acc,
        'positive_rate': pos_rate,
        'valid_patches': int(total_valid),
        'seen_windows': int(seen_windows),
        'effective_windows': int(effective_windows),
        'skipped_windows': int(skipped_windows),
        'skip_rate': float(skipped_windows / max(seen_windows, 1)),
        'mean_valid_patches_per_effective_window': float(total_valid / max(effective_windows, 1)),
        'load_status_counts': dict(status_counts),
        **binary_metrics,
        **avg_losses,
    }


def run_stage_pretraining(
    args,
    output_dir: Path,
    device,
    rank: int,
    world: int,
    local_rank: int,
    patch_len: int,
):
    log.info("=== Stage pretraining (binary seizure vs non-seizure) ===")
    support = inspect_stage_annotation_support(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        source_filter='tusz',
    )
    log.info(
        "  Stage support: classes=%s valid_events=%s raw=%s",
        support.get('supported_classes'),
        support.get('n_valid_events'),
        support.get('raw_annotation_counts', {}),
    )

    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from ..data_preprocess.eeg_pipeline import PipelineConfig

    stage_pre_sec = float(args.stage_pre_onset_sec)
    stage_post_sec = float(args.stage_post_onset_sec)
    if stage_pre_sec <= 0.0 or stage_post_sec <= 0.0:
        raise ValueError(
            f"stage_pre_onset_sec and stage_post_onset_sec must be > 0, got "
            f"{stage_pre_sec} and {stage_post_sec}"
        )
    stage_roles = tuple(
        str(role).strip().lower()
        for role in args.stage_sample_roles
        if str(role).strip()
    )
    if not stage_roles:
        stage_roles = ('onset',)
    stage_n_pre_patches = int(np.ceil(stage_pre_sec / args.patch_duration))
    stage_n_post_patches = int(np.ceil(stage_post_sec / args.patch_duration))
    log.info(
        "  Stage sampling: roles=%s pre=%.1fs post=%.1fs pre_patches=%d post_patches=%d",
        list(stage_roles),
        stage_pre_sec,
        stage_post_sec,
        stage_n_pre_patches,
        stage_n_post_patches,
    )

    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=stage_pre_sec,
        post_onset_sec=stage_post_sec,
        n_patches=stage_n_pre_patches + stage_n_post_patches,
        patch_len=patch_len,
    )

    train_ds = EEGStagePretrainDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        pipeline_cfg=pipeline_cfg,
        source_filter='tusz',
        split_filter=['train'],
        roles=stage_roles,
    )
    val_splits = ['dev']
    val_ds = EEGStagePretrainDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        pipeline_cfg=pipeline_cfg,
        source_filter='tusz',
        split_filter=val_splits,
        roles=stage_roles,
    )
    if len(val_ds) == 0:
        val_splits = ['eval']
        val_ds = EEGStagePretrainDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            pipeline_cfg=pipeline_cfg,
            source_filter='tusz',
            split_filter=val_splits,
            roles=stage_roles,
        )

    train_meta = summarize_stage_dataset(train_ds)
    val_meta = summarize_stage_dataset(val_ds)
    log.info("  Stage train windows: %s", train_meta)
    log.info("  Stage val windows: %s", val_meta)
    train_patch_stats = estimate_stage_patch_statistics(
        train_ds,
        ignore_index=-100,
    )
    val_patch_stats = estimate_stage_patch_statistics(
        val_ds,
        ignore_index=-100,
    )
    log.info(
        "  Stage patch stats(train): valid=%d pos=%d neg=%d pos_rate=%.3f class_weight=%s",
        train_patch_stats['valid_patches'],
        train_patch_stats['positive_patches'],
        train_patch_stats['negative_patches'],
        train_patch_stats['positive_rate'],
        [round(float(x), 4) for x in train_patch_stats['class_weight'].tolist()],
    )
    log.info(
        "  Stage patch stats(val): valid=%d pos=%d neg=%d pos_rate=%.3f",
        val_patch_stats['valid_patches'],
        val_patch_stats['positive_patches'],
        val_patch_stats['negative_patches'],
        val_patch_stats['positive_rate'],
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        log.warning("  Stage pretraining skipped because train/val windows are empty.")
        return None

    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
    else:
        train_sampler = RandomSampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=stage_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=stage_collate_fn,
        pin_memory=True,
    )
    log.info(
        "  Stage loaders ready: train_batches=%d val_batches=%d batch_size=%d workers=%d",
        len(train_loader),
        len(val_loader),
        args.batch_size,
        args.workers,
    )

    cfg = IntegrationConfig(
        task_mode='stage_pretrain',
        embed_dim=args.embed_dim,
        patch_len=patch_len,
        n_pre_patches=stage_n_pre_patches,
        n_post_patches=stage_n_post_patches,
        fs=args.fs,
        labram_checkpoint=args.labram_ckpt,
        output_mode=args.output_mode,
        w_region=args.w_region,
        w_hemisphere=args.w_hemisphere,
        n_frozen_layers=0,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device)
    base_model = model
    base_model.configure_stage_pretraining(train_backbone=args.stage_train_backbone)
    if args.stage_use_class_weight:
        class_weight = train_patch_stats['class_weight'].to(device)
        base_model.set_stage_class_weight(class_weight)
    trainable_params, total_params = count_trainable_parameters(base_model)
    log.info(
        "  Stage param setup: train_backbone=%s use_class_weight=%s trainable params=%d/%d",
        args.stage_train_backbone,
        args.stage_use_class_weight,
        trainable_params,
        total_params,
    )
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        base_model = model.module

    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.stage_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.stage_epochs, 1),
    )
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    writer = SummaryWriter(str(output_dir / 'tb_stage_pretrain')) if (_HAS_TB and is_main(rank)) else None

    best_metric = float('-inf')
    best_epoch = -1
    patience_counter = 0
    best_path = output_dir / 'best_pretrain_ckpt.pth'

    for epoch in range(args.stage_epochs):
        if is_main(rank):
            log.info("  [stage] starting epoch %03d/%03d", epoch + 1, args.stage_epochs)
        if world > 1:
            train_sampler.set_epoch(epoch)

        train_metrics = train_stage_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            writer,
            show_progress=is_main(rank),
            log_every=args.stage_log_every,
        )
        val_metrics = evaluate_stage(
            model,
            val_loader,
            device,
            show_progress=is_main(rank),
            log_every=args.stage_log_every,
        )
        scheduler.step()

        current_metric = stage_metric_value(val_metrics, args.stage_selection_metric)
        best_metric_for_log = stage_metric_display_value(best_metric, args.stage_selection_metric)
        improved = current_metric > best_metric + 1e-6
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            best_metric_for_log = stage_metric_display_value(best_metric, args.stage_selection_metric)
        else:
            patience_counter += 1

        if is_main(rank):
            train_coverage = train_metrics['valid_patches'] / max(train_patch_stats['valid_patches'], 1)
            val_coverage = val_metrics['valid_patches'] / max(val_patch_stats['valid_patches'], 1)
            log.info(
                "  [stage] epoch %03d/%03d "
                "train_loss=%.4f train_acc=%.3f train_rec=%.3f train_f1=%.3f train_auc=%.3f "
                "val_loss=%.4f val_acc=%.3f val_rec=%.3f val_f1=%.3f val_auc=%.3f val_pos=%.3f",
                epoch + 1,
                args.stage_epochs,
                train_metrics['loss'],
                train_metrics['patch_acc'],
                train_metrics['recall'],
                train_metrics['f1'],
                train_metrics['auc'],
                val_metrics['loss'],
                val_metrics['patch_acc'],
                val_metrics['recall'],
                val_metrics['f1'],
                val_metrics['auc'],
                val_metrics['positive_rate'],
            )
            log.info(
                "  [stage data] train_valid=%d/%d coverage=%.3f effective=%d/%d skip_rate=%.3f "
                "status=%s",
                train_metrics['valid_patches'],
                train_patch_stats['valid_patches'],
                train_coverage,
                train_metrics['effective_windows'],
                train_metrics['seen_windows'],
                train_metrics['skip_rate'],
                summarize_status_counts(train_metrics['load_status_counts']),
            )
            log.info(
                "  [stage data] val_valid=%d/%d coverage=%.3f effective=%d/%d skip_rate=%.3f "
                "status=%s",
                val_metrics['valid_patches'],
                val_patch_stats['valid_patches'],
                val_coverage,
                val_metrics['effective_windows'],
                val_metrics['seen_windows'],
                val_metrics['skip_rate'],
                summarize_status_counts(val_metrics['load_status_counts']),
            )
            if train_coverage < 0.8 or val_coverage < 0.8:
                log.warning(
                    "  [stage data] low valid-patch coverage detected "
                    "(train=%.3f, val=%.3f); many windows may be failing in __getitem__",
                    train_coverage,
                    val_coverage,
                )
            if writer:
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/patch_acc', val_metrics['patch_acc'], epoch)
                writer.add_scalar('val/positive_rate', val_metrics['positive_rate'], epoch)
                writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                writer.add_scalar('val/recall', val_metrics['recall'], epoch)
                writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                writer.add_scalar('val/balanced_acc', val_metrics['balanced_acc'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/valid_patches', val_metrics['valid_patches'], epoch)
                writer.add_scalar('val/seen_windows', val_metrics['seen_windows'], epoch)
                writer.add_scalar('val/effective_windows', val_metrics['effective_windows'], epoch)
                writer.add_scalar('val/skipped_windows', val_metrics['skipped_windows'], epoch)
                writer.add_scalar('val/skip_rate', val_metrics['skip_rate'], epoch)
                writer.add_scalar(
                    'val/mean_valid_patches_per_effective_window',
                    val_metrics['mean_valid_patches_per_effective_window'],
                    epoch,
                )
                writer.add_scalar('val/coverage_vs_static', val_coverage, epoch)
                writer.add_scalar('train/coverage_vs_static', train_coverage, epoch)
                for key in ('loss_stage', 'loss_moe', 'loss_total'):
                    if key in val_metrics:
                        writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

            if improved:
                base_model.save_checkpoint(
                    str(best_path),
                    extra={
                        'epoch': epoch,
                        'best_stage_metric': best_metric,
                        'best_stage_metric_name': args.stage_selection_metric,
                        'stage_metrics': val_metrics,
                    },
                )
                log.info(
                    "  [stage] new best %s=%.4f at epoch %03d -> %s",
                    args.stage_selection_metric,
                    best_metric_for_log,
                    epoch + 1,
                    best_path,
                )
            else:
                log.info(
                    "  [stage] no improvement in %s for %d epoch(s) "
                    "(best=%.4f @ epoch %03d)",
                    args.stage_selection_metric,
                    patience_counter,
                    best_metric_for_log,
                    best_epoch + 1 if best_epoch >= 0 else 0,
                )
        if args.stage_early_stop_patience > 0 and patience_counter >= args.stage_early_stop_patience:
            if is_main(rank):
                log.info(
                    "  [stage] early stopping triggered at epoch %03d "
                    "(patience=%d, best_%s=%.4f @ epoch %03d)",
                    epoch + 1,
                    args.stage_early_stop_patience,
                    args.stage_selection_metric,
                    best_metric_for_log,
                    best_epoch + 1 if best_epoch >= 0 else 0,
                )
            break

    if is_main(rank):
        log.info(
            "  [stage] finished with best_%s=%.4f at epoch %03d",
            args.stage_selection_metric,
            stage_metric_display_value(best_metric, args.stage_selection_metric),
            best_epoch + 1 if best_epoch >= 0 else 0,
        )
    if writer:
        writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_path if best_path.exists() else None


# =====================================================================
# Contrastive pretraining
# =====================================================================

def run_contrastive_pretraining(
    model_pretrain, train_loader, device, args, writer=None,
):
    log.info("=== Contrastive pretraining ===")
    net_ext = MultiScaleBrainNetworkExtractor(
        n_channels=22, patch_len=int(args.patch_duration * args.fs), fs=args.fs,
    ).to(device)
    optimizer = torch.optim.AdamW(model_pretrain.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs,
    )

    for epoch in range(args.pretrain_epochs):
        model_pretrain.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            # compute brain networks from raw patches
            B, C, T = x.shape
            P = T // 100
            patches = x.reshape(B, C, P, 100).permute(0, 2, 1, 3)  # [B,P,22,100]
            with torch.no_grad():
                nets = net_ext(patches)['all']                        # [B,P,22,22,4]

            # for contrastive: use same data with augmentation as pos,
            # circularly shifted as neg
            neg = nets.roll(1, dims=0)
            out = model_pretrain(nets, neg)
            loss = out['contrastive_loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_pretrain.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(train_loader), 1)
        if writer:
            writer.add_scalar('pretrain/loss', avg, epoch)
        if epoch % 10 == 0:
            log.info(f"  Pretrain epoch {epoch}/{args.pretrain_epochs}  loss={avg:.4f}")

    save_path = Path(args.output_dir) / 'pretrained_encoder.pt'
    model_pretrain.save_pretrained_encoder(str(save_path))
    log.info(f"Pretrained encoder saved to {save_path}")
    return save_path


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description='SOZ Locator with Brain Networks')

    # data
    p.add_argument('--manifest', required=True, help='combined_manifest.csv')
    p.add_argument('--private-data-root', default='', help='preprocessed data root')
    p.add_argument('--tusz-data-root', default='', help='TUSZ EDF root')
    p.add_argument('--source', default='all', choices=['tusz', 'private', 'all'])
    p.add_argument(
        '--split-strategy',
        default='auto',
        choices=['auto', 'random', 'private_target'],
        help=(
            "Dataset split strategy for SOZ finetuning: "
            "auto=use private patient-wise split when source is all/private, "
            "otherwise random"
        ),
    )

    # model
    p.add_argument('--labram-ckpt', default='', help='LaBraM pretrained weights')
    p.add_argument('--patch-duration', type=float, default=1.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--output-mode', default='monopolar', choices=['monopolar', 'bipolar'])

    # brain networks
    p.add_argument('--brain-network-features', default='gc,te,aec,wpli')
    p.add_argument('--use-contrastive', action='store_true')
    p.add_argument('--pretrain-epochs', type=int, default=50)
    p.add_argument('--use-pretrain-stage', action='store_true',
                   help='Run binary seizure/non-seizure stage pretraining on TUSZ before SOZ finetuning')
    p.add_argument('--stage-only', action='store_true',
                   help='Run only stage-1 binary seizure/non-seizure pretraining and exit')
    p.add_argument('--stage-pretrain-ckpt', default='',
                   help='Path to a stage-1 checkpoint to load for SOZ-only training')
    p.add_argument('--freeze-labram', action='store_true',
                   help='Freeze LaBraM backbone during SOZ finetuning')
    p.add_argument('--stage-epochs', type=int, default=20,
                   help='Epochs for binary stage pretraining')
    p.add_argument('--stage-lr', type=float, default=1e-4,
                   help='Learning rate for stage pretraining')
    p.add_argument('--stage-log-every', type=int, default=20,
                   help='Log every N steps during stage pretraining')
    p.add_argument('--stage-early-stop-patience', type=int, default=6,
                   help='Early-stop patience for stage pretraining (0 disables)')
    p.add_argument('--stage-selection-metric', default='f1',
                   choices=['f1', 'recall', 'auc', 'acc', 'loss'],
                   help='Validation metric used to save best stage checkpoint and early stop')
    p.add_argument('--stage-train-backbone', dest='stage_train_backbone',
                   action='store_true',
                   help='Train the full LaBraM backbone during stage pretraining')
    p.add_argument('--no-stage-train-backbone', dest='stage_train_backbone',
                   action='store_false',
                   help='Freeze LaBraM backbone and train only TimeFilter + stage head')
    p.set_defaults(stage_train_backbone=True)
    p.add_argument('--stage-use-class-weight', dest='stage_use_class_weight',
                   action='store_true',
                   help='Use inverse-frequency class weights for stage CrossEntropy')
    p.add_argument('--no-stage-use-class-weight', dest='stage_use_class_weight',
                   action='store_false',
                   help='Disable class weighting for stage CrossEntropy')
    p.set_defaults(stage_use_class_weight=True)
    p.add_argument('--stage-pre-onset-sec', type=float, default=8.0,
                   help='Seconds before seizure onset used only for stage-1 sampling')
    p.add_argument('--stage-post-onset-sec', type=float, default=4.0,
                   help='Seconds after seizure onset used only for stage-1 sampling')
    p.add_argument('--stage-sample-roles', nargs='+',
                   default=['onset'],
                   choices=['onset', 'mid', 'offset'],
                   help='Stage-1 sampling centers to include')

    # Sequence length configurations
    p.add_argument('--pre-onset-sec', type=float, default=5.0, help='Seconds before onset to extract')
    p.add_argument('--post-onset-sec', type=float, default=5.0, help='Seconds after onset to extract')

    # training
    p.add_argument('--finetune-epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--w-region', type=float, default=0.5,
                   help='Loss weight for coarse region classifier')
    p.add_argument('--w-hemisphere', type=float, default=0.5,
                   help='Loss weight for hemisphere classifier')
    p.add_argument('--amp', action='store_true', help='mixed precision')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)

    # output
    p.add_argument('--output-dir', default='output/train_bn')
    p.add_argument('--precomputed-dir', default=None, help='Directory with precomputed brain networks')
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    p.add_argument('--save-every', type=int, default=10)

    # validation
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)

    return p.parse_args()


def main():
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Save config ──
    if is_main(rank):
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        log.info(f"Config saved to {output_dir / 'config.json'}")

    # ── 1. Data loading ──
    log.info("=== Step 1: Loading data ===")
    
    # Calculate patch configurations
    patch_len = int(args.patch_duration * args.fs)
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    n_patches = n_pre_patches + n_post_patches

    if args.stage_only:
        stage_pretrain_ckpt = run_stage_pretraining(
            args=args,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world=world,
            local_rank=local_rank,
            patch_len=patch_len,
        )
        if is_main(rank):
            log.info("Stage-1 pretraining finished: %s", stage_pretrain_ckpt)
        if world > 1:
            dist.destroy_process_group()
        log.info("Done.")
        return 0
    
    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from ..data_preprocess.eeg_pipeline import PipelineConfig
        
    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_patches,
        patch_len=patch_len
    )
    
    if args.source in ('all', 'private') and not args.private_data_root:
        log.warning(
            "  --private-data-root is empty while private samples are enabled; "
            "private EDF relative paths may fail to resolve."
        )

    train_ds, val_ds, test_ds, split_meta = build_soz_datasets(
        args=args,
        pipeline_cfg=pipeline_cfg,
    )
    log.info("  SOZ split strategy: %s", split_meta['strategy'])
    for line in split_meta.get('log_lines', []):
        log.info("  %s", line)

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    log.info("  Final dataset sizes: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # check sample size
    if n_train < 50:
        log.warning(f"  Small dataset ({n_train} samples) — strong augmentation recommended")

    # loaders
    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
    else:
        train_sampler = RandomSampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    # ── 2. ModelInit ──
    stage_pretrain_ckpt = None
    if args.stage_pretrain_ckpt:
        candidate = Path(args.stage_pretrain_ckpt)
        if not candidate.exists():
            raise FileNotFoundError(f"Stage pretrain checkpoint not found: {candidate}")
        stage_pretrain_ckpt = candidate
        log.info("Using provided stage-1 checkpoint for SOZ training: %s", stage_pretrain_ckpt)
    elif args.use_pretrain_stage:
        stage_pretrain_ckpt = run_stage_pretraining(
            args=args,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world=world,
            local_rank=local_rank,
            patch_len=patch_len,
        )

    log.info("=== Step 2: Initializing model ===")
    cfg = IntegrationConfig(
        task_mode='soz',
        embed_dim=args.embed_dim,
        patch_len=patch_len,
        n_pre_patches=n_pre_patches,
        n_post_patches=n_post_patches,
        fs=args.fs,
        labram_checkpoint=args.labram_ckpt,
        output_mode=args.output_mode,
        w_region=args.w_region,
        w_hemisphere=args.w_hemisphere,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device)
    log.info(model.summary())
    if stage_pretrain_ckpt is not None:
        load_info = model.load_a_branch_weights(str(stage_pretrain_ckpt), map_location=device)
        log.info(
            "  Loaded stage-pretrained A-branch from %s (loaded=%d, missing=%d, unexpected=%d)",
            stage_pretrain_ckpt,
            len(load_info['loaded_keys']),
            len(load_info['missing_keys']),
            len(load_info['unexpected_keys']),
        )

    # ── 2b. Compute class balance and set pos_weight ──
    log.info("  Computing pos_weight from training labels...")
    pw = compute_pos_weight(train_loader, device=device)
    model.set_pos_weight(pw)
    log.info(f"  pos_weight set (shape={pw.shape})")

    # ── 3. Contrastive pretraining (optional) ──
    pretrain_encoder_path = None
    if args.use_contrastive:
        log.info("=== Step 3: Contrastive pretraining ===")
        pt_cfg = PretrainConfig(embed_dim=cfg.embed_dim)
        pretrain_model = BrainNetworkContrastivePretrainer(pt_cfg).to(device)
        writer_pt = SummaryWriter(str(output_dir / 'tb_contrastive_pretrain')) if (_HAS_TB and is_main(rank)) else None
        pretrain_encoder_path = run_contrastive_pretraining(
            pretrain_model, train_loader, device, args, writer_pt,
        )
        if writer_pt:
            writer_pt.close()

    # ── 4. Fine-tuning ──
    log.info("=== Step 4: Fine-tuning ===")
    writer = SummaryWriter(str(output_dir / 'tb_finetune')) if (_HAS_TB and is_main(rank)) else None
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # DDP
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # resume
    start_epoch = 0
    best_top3 = 0.0
    best_selection_key = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_top3 = ckpt.get('best_top3', 0.0)
        raw_best_key = ckpt.get('best_selection_key', None)
        if raw_best_key is not None:
            best_selection_key = tuple(float(x) for x in raw_best_key)
            if len(best_selection_key) < 6:
                best_selection_key = best_selection_key + (-1.0,) * (6 - len(best_selection_key))
        else:
            best_selection_key = (0.0, 0.0, 0.0, float(best_top3), 0.0, 0.0)
        log.info(
            "  Resumed from epoch %d, best_recall@3=%.4f, best_key=%s",
            start_epoch,
            best_top3,
            best_selection_key,
        )

    total_epochs = args.finetune_epochs
    phase1_end = total_epochs // 5       # 20% frozen backbone
    phase2_end = total_epochs * 3 // 5   # next 40% unfreeze timefilter

    has_stage_init = stage_pretrain_ckpt is not None

    if args.freeze_labram:
        base_model.freeze_backbone()
        base_model.unfreeze_timefilter()
        log.info("  Finetune setup: LaBraM backbone frozen, TimeFilter + downstream heads trainable")
    elif has_stage_init:
        base_model.unfreeze_all()
        log.info("  Finetune setup: loaded stage-pretrained A-branch, all parameters trainable")

    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.lr), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2,
    )

    for epoch in range(start_epoch, total_epochs):
        # phase transitions
        if not has_stage_init and not args.freeze_labram:
            if epoch == 0:
                base_model.freeze_backbone()
                log.info("  Phase 1: backbone frozen")
            elif epoch == phase1_end:
                base_model.unfreeze_timefilter()
                optimizer = torch.optim.AdamW(
                    base_model.get_param_groups(args.lr * 0.5), weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2,
                )
                log.info("  Phase 2: TimeFilter + network unfrozen")
            elif epoch == phase2_end:
                base_model.unfreeze_all()
                optimizer = torch.optim.AdamW(
                    base_model.get_param_groups(args.lr * 0.1), weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2,
                )
                log.info("  Phase 3: full model unfrozen")

        if world > 1:
            train_sampler.set_epoch(epoch)

        # train
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg, writer,
        )
        scheduler.step()
        dt = time.time() - t0

        # validate
        val_metrics = evaluate(model, val_loader, device)

        if is_main(rank):
            val_summary = {
                key: value for key, value in val_metrics.items()
                if not isinstance(value, np.ndarray)
            }
            log.info(
                f"Epoch {epoch:3d}/{total_epochs} "
                f"loss={train_metrics['loss']:.4f} "
                f"soz={train_metrics.get('loss_soz', 0.0):.4f} "
                f"region={train_metrics.get('loss_region', 0.0):.4f} "
                f"hemi={train_metrics.get('loss_hemisphere', 0.0):.4f} "
                f"train_r3={train_metrics['recall_at_3']:.3f} "
                f"train_ndcg3={train_metrics['ndcg_at_3']:.3f} "
                f"train_mrr={train_metrics['mrr']:.3f} "
                f"train_region_acc={train_metrics['region_acc']:.3f} "
                f"train_hemi_acc={train_metrics['hemisphere_acc']:.3f} "
                f"val_r3={val_metrics['recall_at_3']:.3f} "
                f"val_ndcg3={val_metrics['ndcg_at_3']:.3f} "
                f"val_mrr={val_metrics['mrr']:.3f} "
                f"val_auc={val_metrics['auc']:.3f} "
                f"val_region_acc={val_metrics['region_acc']:.3f} "
                f"val_hemi_acc={val_metrics['hemisphere_acc']:.3f} "
                f"({dt:.1f}s)"
            )
            if writer:
                writer.add_scalar('val/recall_at_1', val_metrics['recall_at_1'], epoch)
                writer.add_scalar('val/recall_at_3', val_metrics['recall_at_3'], epoch)
                writer.add_scalar('val/recall_at_5', val_metrics['recall_at_5'], epoch)
                writer.add_scalar('val/precision_at_3', val_metrics['precision_at_3'], epoch)
                writer.add_scalar('val/ndcg_at_3', val_metrics['ndcg_at_3'], epoch)
                writer.add_scalar('val/mrr', val_metrics['mrr'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/region_acc', val_metrics['region_acc'], epoch)
                writer.add_scalar('val/hemisphere_acc', val_metrics['hemisphere_acc'], epoch)
                for key in ('loss_total', 'loss_soz', 'loss_region', 'loss_hemisphere'):
                    if key in val_metrics:
                        writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

            # save best
            val_selection_key = build_selection_key(val_metrics)
            if val_selection_key > best_selection_key:
                best_selection_key = val_selection_key
                best_top3 = val_metrics['recall_at_3']
                base_model.save_checkpoint(
                    str(output_dir / 'best_model.pt'),
                    extra={
                        'epoch': epoch,
                        'best_top3': best_top3,
                        'best_recall_at_3': best_top3,
                        'best_selection_key': list(best_selection_key),
                        'val_metrics': val_summary,
                    },
                )
                log.info(
                    "  ** New best: auc=%.4f ndcg3=%.4f mrr=%.4f r3=%.4f region_acc=%.4f hemi_acc=%.4f",
                    val_metrics['auc'],
                    val_metrics['ndcg_at_3'],
                    val_metrics['mrr'],
                    val_metrics['recall_at_3'],
                    val_metrics['region_acc'],
                    val_metrics['hemisphere_acc'],
                )

            # periodic save
            if (epoch + 1) % args.save_every == 0:
                base_model.save_checkpoint(
                    str(output_dir / f'ckpt_epoch{epoch:03d}.pt'),
                    extra={
                        'epoch': epoch,
                        'best_top3': best_top3,
                        'best_recall_at_3': best_top3,
                        'best_selection_key': list(best_selection_key),
                    },
                )

    # ── 5. Test evaluation ──
    log.info("=== Step 5: Test evaluation ===")
    # load best
    best_path = output_dir / 'best_model.pt'
    if best_path.exists():
        ckpt_best = torch.load(str(best_path), map_location=device)
        base_model.load_state_dict(ckpt_best['model_state'])

    test_metrics = evaluate(model, test_loader, device)

    if is_main(rank):
        log.info(
            f"\nTest results:\n"
            f"  Recall@1: {test_metrics['recall_at_1']:.4f}\n"
            f"  Recall@3: {test_metrics['recall_at_3']:.4f}\n"
            f"  Recall@5: {test_metrics['recall_at_5']:.4f}\n"
            f"  Precision@3: {test_metrics['precision_at_3']:.4f}\n"
            f"  nDCG@3: {test_metrics['ndcg_at_3']:.4f}\n"
            f"  MRR:   {test_metrics['mrr']:.4f}\n"
            f"  AUC:   {test_metrics['auc']:.4f}\n"
            f"  Region acc: {test_metrics['region_acc']:.4f}\n"
            f"  Hemisphere acc: {test_metrics['hemisphere_acc']:.4f}"
        )

        # save test report (markdown)
        report = (
            f"# SOZ Localization Report\n\n"
            f"## Configuration\n"
            f"- Manifest: `{args.manifest}`\n"
            f"- LaBraM checkpoint: `{args.labram_ckpt}`\n"
            f"- Contrastive pretraining: {args.use_contrastive}\n"
            f"- Stage pretraining: {args.use_pretrain_stage}\n"
            f"- Stage only mode: {args.stage_only}\n"
            f"- Stage init ckpt: `{args.stage_pretrain_ckpt or stage_pretrain_ckpt or ''}`\n"
            f"- Freeze LaBraM backbone: {args.freeze_labram}\n"
            f"- Finetune epochs: {total_epochs}\n"
            f"- Output mode: {args.output_mode}\n\n"
            f"## Test Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Recall@1 | {test_metrics['recall_at_1']:.4f} |\n"
            f"| Recall@3 | {test_metrics['recall_at_3']:.4f} |\n"
            f"| Recall@5 | {test_metrics['recall_at_5']:.4f} |\n"
            f"| Precision@3 | {test_metrics['precision_at_3']:.4f} |\n"
            f"| nDCG@3 | {test_metrics['ndcg_at_3']:.4f} |\n"
            f"| MRR | {test_metrics['mrr']:.4f} |\n"
            f"| AUC | {test_metrics['auc']:.4f} |\n"
            f"| Region acc | {test_metrics['region_acc']:.4f} |\n"
            f"| Hemisphere acc | {test_metrics['hemisphere_acc']:.4f} |\n\n"
            f"## Best validation key: auc={best_selection_key[0]:.4f}, "
            f"ndcg3={best_selection_key[1]:.4f}, "
            f"mrr={best_selection_key[2]:.4f}, "
            f"r3={best_selection_key[3]:.4f}, "
            f"region_acc={best_selection_key[4]:.4f}, "
            f"hemi_acc={best_selection_key[5]:.4f}\n"
        )
        (output_dir / 'report.md').write_text(report, encoding='utf-8')
        log.info(f"Report saved to {output_dir / 'report.md'}")

        # save test predictions
        np.savez(
            str(output_dir / 'test_predictions.npz'),
            probs=test_metrics['probs'],
            targets=test_metrics['targets'],
            region_probs=test_metrics['region_probs'],
            region_targets=test_metrics['region_targets'],
            hemisphere_logits=test_metrics['hemisphere_logits'],
            hemisphere_targets=test_metrics['hemisphere_targets'],
        )

    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()

    log.info("Done.")
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    sys.exit(main())
