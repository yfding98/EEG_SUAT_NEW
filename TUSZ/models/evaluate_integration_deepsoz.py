#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Model 评估脚本 — 使用 DeepSOZ 官方评估方法

使用 train_soz_locator_with_brain_networks.py 训练的模型
(TimeFilter_LaBraM_BrainNetwork_Integration),
在 combined_manifest.csv 上跑推理, 按 DeepSOZ 官方评估逻辑计算:

1. Channel-level (19 单极通道, DeepSOZ 官方顺序)
   - 逐通道混淆矩阵 (TP/FP/TN/FN/Precision/Recall/Specificity/F1)
   - 多阈值扫描 + 最优阈值

2. Region-level (6 脑区)
   - 逐区混淆矩阵

3. Seizure-level 定位正确率 (corr_sz)
   - MC dropout N 次采样 → 归一化 → 均值 → argmax → 判断命中 SOZ
   - MC 不确定性 (szunc)

4. Patient-level 定位正确率 (corr_pt)
   - 聚合同一患者所有发作的 MC 采样 → argmax
   - MC 不确定性 (ptunc)

5. 邻居放宽判断 (chn_neighbours)
   - 当 SOZ 通道数 ≤ threshold 时, 预测通道落在空间邻居也算正确

���══════════════════════════════════════════════════════════════════════════════
通道映射说明:
═══════════════════════════════════════════════════════════════════════════════

模型输出 soz_probs [B, 19] (output_mode='monopolar'), 顺序为 STANDARD_19:
  [FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ]

DeepSOZ 官方评估需要 19 通道顺序 (DEEPSOZ_19):
  [FP1, FP2, F7, F3, FZ, F4, F8, T3, C3, CZ, C4, T4, T5, P3, PZ, P4, T6, O1, O2]

脚本内置了 reorder_to_deepsoz() 做重排。

用法:
  python evaluate_integration_deepsoz.py \\
      --checkpoint output/TUSZ/train/best_model.pt \\
      --manifest TUSZ/combined_manifest.csv \\
      --tusz-data-root F:/dataset/TUSZ/v2.0.3/edf \\
      --source tusz \\
      --mc-samples 20
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─── 项目路径设置 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # EEG_SUAT_NEW
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ'))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ' / 'models'))

from models.integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
)
from models.manifest_dataset import (
    ManifestSOZDataset,
    TCP_BIPOLAR_NAMES,
    TCP_COL_NAMES,
    COARSE_REGION_NAMES,
    HEMISPHERE_NAMES,
    _build_bipolar_to_monopolar_matrix,
    get_region_names,
)
from models.bipolar_to_monopolar import DEFAULT_MONOPOLAR_19

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 通道定义 & 重排序映射
# ═════════════════════════════════════════════════════════════════════════════

# Integration model 输出 19 通道顺序 (STANDARD_19 / BipolarToMonopolarMapper)
INTEGRATION_19 = list(DEFAULT_MONOPOLAR_19)
# ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
#  'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']

# DeepSOZ 官方 19 通道顺序
DEEPSOZ_19 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3',  'C3',  'CZ', 'C4', 'T4',
    'T5',  'P3',  'PZ', 'P4', 'T6',
    'O1',  'O2',
]

# 构建 Integration → DeepSOZ 重排索引
_int_idx = {ch: i for i, ch in enumerate(INTEGRATION_19)}
INTEGRATION_TO_DEEPSOZ = [_int_idx[ch] for ch in DEEPSOZ_19]

# 反向映射
_dsz_idx = {ch: i for i, ch in enumerate(DEEPSOZ_19)}
DEEPSOZ_TO_INTEGRATION = [_dsz_idx[ch] for ch in INTEGRATION_19]


def reorder_to_deepsoz(arr: np.ndarray) -> np.ndarray:
    """将 STANDARD_19 顺序的 [*, 19] 数组重排为 DeepSOZ 官方顺序。"""
    return arr[..., INTEGRATION_TO_DEEPSOZ]


# ═════════════════════════════════════════════════════════════════════════════
# 双极标签 → 单极标签 (DeepSOZ 顺序)
# ═════════════════════════════════════════════════════════════════════════════

def bipolar22_to_monopolar19_binary(bipolar_labels: np.ndarray) -> np.ndarray:
    """
    22 双极 SOZ 二值标签 → 19 单极 SOZ 二值标签 (DeepSOZ 顺序)

    bipolar_labels: [*, 22] binary
    返回: [*, 19] binary (DeepSOZ 通道顺序)
    """
    b2m = _build_bipolar_to_monopolar_matrix()  # [19, 22] STANDARD_19 顺序
    orig_shape = bipolar_labels.shape
    flat = bipolar_labels.reshape(-1, 22)
    mono = (flat @ b2m.T > 0).astype(np.float32)  # [N, 19] STANDARD_19 顺序
    mono = mono.reshape(*orig_shape[:-1], 19)
    return reorder_to_deepsoz(mono)


# ═════════════════════════════════════════════════════════════════════════════
# DeepSOZ 官方评估工具函数
# ═════════════════════════════════════════════════════════════════════════════

# 官方 chn_neighbours (索引基于 DEEPSOZ_19 顺序)
CHN_NEIGHBOURS_19 = {
    0:  [1, 2, 3, 4],                 # FP1
    1:  [0, 4, 5, 6],                 # FP2
    2:  [0, 3, 4, 7, 8],              # F7
    3:  [0, 2, 4, 8, 9],              # F3
    4:  [0, 1, 3, 5, 9],              # FZ
    5:  [1, 4, 6, 9, 10],             # F4
    6:  [1, 4, 5, 10, 11],            # F8
    7:  [2, 8, 12, 13, 17],           # T3
    8:  [2, 3, 4, 7, 9, 12, 13, 14],  # C3
    9:  [3, 4, 5, 8, 10, 13, 14, 15], # CZ
    10: [4, 5, 6, 9, 11, 14, 15, 16], # C4
    11: [6, 10, 15, 16, 18],          # T4
    12: [7, 8, 13, 17],               # T5
    13: [7, 8, 9, 12, 14, 17],        # P3
    14: [8, 9, 10, 13, 15, 17, 18],   # PZ
    15: [9, 10, 11, 14, 16, 18],      # P4
    16: [10, 11, 15, 18],             # T6
    17: [7, 12, 13, 14, 18],          # O1
    18: [11, 14, 15, 16, 17],         # O2
}


def check_neighborhood(max_chn: int, onset_map: np.ndarray) -> bool:
    for i in range(len(onset_map)):
        if onset_map[i] == 1 and max_chn in CHN_NEIGHBOURS_19.get(i, []):
            return True
    return False


def final_loc(
    psoz: np.ndarray,
    true_onset: np.ndarray,
    neighbour_threshold: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    官方 final_loc (final_eval_all.ipynb):
    psoz: [N, 19], true_onset: [19] (均为 DeepSOZ 顺序)
    返回 (ysoz [19], uncertainty [19], correct 0/1)
    """
    m = psoz.max(axis=1, keepdims=True)
    m = np.where(m > 1e-12, m, 1.0)  # 避免除零
    psoz_norm = psoz / m
    ysoz = psoz_norm.mean(axis=0)

    max_chn = int(np.argmax(ysoz))
    correct = 1 if true_onset[max_chn] == 1 else 0

    if (correct == 0
            and int(true_onset.sum()) <= neighbour_threshold
            and check_neighborhood(max_chn, true_onset)):
        correct = 1

    uncertainty = psoz_norm.var(axis=0)
    # 修复 NaN: 当方差计算结果为 NaN (样本数不足或全零), 替换为 0
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)
    return ysoz, uncertainty, correct


# ═════════════════════════════════════════════════════════════════════════════
# 混淆矩阵
# ═════════════════════════════════════════════════════════════════════════════

def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, support=int(y_true.sum()),
                precision=prec, recall=rec, specificity=spec, f1=f1)


def print_confusion_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    title: str = 'Channel-level',
) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    print(f'\n## {title} Confusion Matrix (threshold={threshold:.3f})\n')
    header = (f'| {"Label":<10} | {"TP":>4} | {"FP":>4} | {"TN":>4} | '
              f'{"FN":>4} | {"Supp":>5} | {"Prec":>6} | {"Rec":>6} | '
              f'{"Spec":>6} | {"F1":>6} |')
    sep = '|' + '-' * 12 + '|' + ('--------|' * 8)
    print(header)
    print(sep)

    per_label = {}
    total_tp = total_fp = total_tn = total_fn = 0
    for i, name in enumerate(label_names):
        cm = binary_confusion(y_true[:, i], y_pred[:, i])
        per_label[name] = cm
        total_tp += cm['tp']
        total_fp += cm['fp']
        total_tn += cm['tn']
        total_fn += cm['fn']
        print(f'| {name:<10} | {cm["tp"]:>4} | {cm["fp"]:>4} | '
              f'{cm["tn"]:>4} | {cm["fn"]:>4} | {cm["support"]:>5} | '
              f'{cm["precision"]:>6.3f} | {cm["recall"]:>6.3f} | '
              f'{cm["specificity"]:>6.3f} | {cm["f1"]:>6.3f} |')

    macro_prec = np.mean([cm['precision'] for cm in per_label.values()])
    macro_rec  = np.mean([cm['recall']    for cm in per_label.values()])
    macro_spec = np.mean([cm['specificity'] for cm in per_label.values()])
    macro_f1   = np.mean([cm['f1']        for cm in per_label.values()])
    print(f'| {"MACRO":<10} |      |      |      |      |       | '
          f'{macro_prec:>6.3f} | {macro_rec:>6.3f} | {macro_spec:>6.3f} | {macro_f1:>6.3f} |')

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    print(f'| {"MICRO":<10} | {total_tp:>4} | {total_fp:>4} | '
          f'{total_tn:>4} | {total_fn:>4} | {int(y_true.sum()):>5} | '
          f'{micro_prec:>6.3f} | {micro_rec:>6.3f} | {micro_spec:>6.3f} | {micro_f1:>6.3f} |')

    per_label['_macro'] = dict(precision=macro_prec, recall=macro_rec,
                               specificity=macro_spec, f1=macro_f1)
    per_label['_micro'] = dict(tp=total_tp, fp=total_fp, tn=total_tn, fn=total_fn,
                               precision=micro_prec, recall=micro_rec,
                               specificity=micro_spec, f1=micro_f1)
    return per_label


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        thresholds: np.ndarray = None) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
    best_th, best_f1 = 0.5, 0.0
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        f1s = []
        for c in range(y_true.shape[1]):
            cm = binary_confusion(y_true[:, c], y_pred[:, c])
            f1s.append(cm['f1'])
        macro = np.mean(f1s)
        if macro > best_f1:
            best_f1, best_th = macro, th
    return float(best_th), float(best_f1)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset wrapper (与 train_soz_locator_with_brain_networks.py 一致)
# ═════════════════════════════════════════════════════════════════════════════

class SOZBrainNetworkDataset(torch.utils.data.Dataset):
    """
    包装 ManifestSOZDataset, 提供 seizure_onset_sec / window_start_sec
    给 SeizureAlignedAdaptivePatching 使用。

    输出 x: [22, T] (展平后的原始波形), 与训练时一致。
    """

    def __init__(self, manifest_ds: ManifestSOZDataset):
        self.ds = manifest_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        x, label, mask, meta, y_bipolar, y_monopolar, y_region, y_hemisphere = sample

        # 展平 [22, P, L] → [22, P*L] 与训练时一致
        C, P, L = x.shape
        x_flat = x.reshape(C, P * L)

        row = self.ds.df.iloc[idx]
        onset_sec = float(row.get('onset_sec', 5.0))
        start_sec = float(row.get('window_start_sec', 0.0))

        return {
            'idx': idx,
            'x': x_flat,
            'label': label,
            'bipolar_label': y_bipolar,
            'monopolar_label': y_monopolar,
            'region_label': y_region,
            'hemisphere_label': y_hemisphere,
            'onset_sec': onset_sec,
            'start_sec': start_sec,
            'patient_id': str(meta.get('patient_id', '')),
            'edf_path': str(meta.get('edf_path', '')),
        }


def eval_collate_fn(batch):
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
        'patient_id': [b['patient_id'] for b in batch],
        'edf_path': [b['edf_path'] for b in batch],
    }
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═════════════════════════════════════════════════════════════════════════════

def build_model(args) -> TimeFilter_LaBraM_BrainNetwork_Integration:
    """
    从 checkpoint 加载 TimeFilter_LaBraM_BrainNetwork_Integration。

    checkpoint 格式 (由 model.save_checkpoint 保存):
      {'model_state': state_dict, 'config': IntegrationConfig, ...}
    """
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # 尝试从 checkpoint 恢复 config
    if 'config' in ckpt:
        cfg = ckpt['config']
        if isinstance(cfg, dict):
            cfg = IntegrationConfig(**cfg)
        logger.info(f'Using config from checkpoint')
    else:
        # 回退: 手动构建 config
        logger.warning('Checkpoint 中无 config, 使用命令行参数构建')
        cfg = IntegrationConfig(
            n_channels=22,
            embed_dim=args.embed_dim,
            patch_len=args.patch_len,
            n_pre_patches=args.n_pre_patches,
            n_post_patches=args.n_post_patches,
            fs=args.fs,
            labram_checkpoint='',
            n_frozen_layers=0,
            output_mode=args.output_mode,
            n_regions=args.n_regions,
            task_mode='soz',
            use_checkpoint=False,
        )

    # 推理时不冻结, 不需 LaBraM pretrain
    cfg.n_frozen_layers = 0
    cfg.labram_checkpoint = ''
    cfg.use_checkpoint = False

    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg)

    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    if not isinstance(state, dict):
        raise KeyError("Checkpoint does not contain a valid model state dict")

    # 兼容 DDP 的 module. 前缀
    own_state = model.state_dict()
    filtered_state = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            filtered_state[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    loaded = len(filtered_state)
    logger.info(f'Loaded checkpoint: {args.checkpoint}')
    logger.info(f'  loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}')
    logger.info(f'  output_mode={cfg.output_mode}, embed_dim={cfg.embed_dim}, '
                f'patches={cfg.n_pre_patches}+{cfg.n_post_patches}, patch_len={cfg.patch_len}')
    return model, cfg


# ═════════════════════════════════════════════════════════════════════════════
# 推理函数
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_standard_inference(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    标准推理 (model.eval(), 无 MC dropout)。

    模型输入: x [B, 22, T], seizure_onset_sec [B], window_start_sec [B]
    模型输出: soz_probs [B, 19] (monopolar, STANDARD_19 顺序)
    """
    model.eval()
    use_amp = device.type == 'cuda'

    all_soz_probs, all_soz_logits, all_bip_labels, all_mono_labels = [], [], [], []
    all_region_probs, all_region_labels = [], []
    all_hemi_probs, all_hemi_labels = [], []
    all_bip_logits = []
    all_pids, all_edfs = [], []

    for batch in loader:
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(x, onset, start)

        soz_probs = outputs['soz_probs'].cpu().numpy()            # [B, 19] STANDARD_19
        soz_logits = outputs['soz_logits'].cpu().numpy()          # [B, 19] STANDARD_19
        bipolar_logits = outputs['bipolar_logits'].cpu().numpy()   # [B, 22]
        region_probs = outputs['region_probs'].cpu().numpy()       # [B, n_regions]
        hemi_probs = outputs['hemisphere_probs'].cpu().numpy()     # [B, 3]

        all_soz_probs.append(soz_probs)
        all_soz_logits.append(soz_logits)
        all_bip_logits.append(bipolar_logits)
        all_bip_labels.append(batch['bipolar_label'].numpy())
        all_mono_labels.append(batch['monopolar_label'].numpy())
        all_region_probs.append(region_probs)
        all_region_labels.append(batch['region_label'].numpy())
        all_hemi_probs.append(hemi_probs)
        all_hemi_labels.append(batch['hemisphere_label'].numpy())
        all_pids.extend(batch['patient_id'])
        all_edfs.extend(batch['edf_path'])

    soz_probs_int19 = np.concatenate(all_soz_probs, axis=0)       # [N, 19] STANDARD_19
    soz_logits_int19 = np.concatenate(all_soz_logits, axis=0)     # [N, 19] STANDARD_19
    mono_labels_int19 = np.concatenate(all_mono_labels, axis=0)    # [N, 19] STANDARD_19

    # 重排到 DeepSOZ 顺序
    soz_probs_dsz19 = reorder_to_deepsoz(soz_probs_int19)
    soz_logits_dsz19 = reorder_to_deepsoz(soz_logits_int19)
    mono_labels_dsz19 = reorder_to_deepsoz(mono_labels_int19)

    return {
        'soz_probs_int19':    soz_probs_int19,
        'soz_probs_dsz19':    soz_probs_dsz19,
        'soz_logits_int19':   soz_logits_int19,
        'soz_logits_dsz19':   soz_logits_dsz19,
        'labels_mono_int19':  mono_labels_int19,
        'labels_mono_dsz19':  mono_labels_dsz19,
        'bipolar_logits':     np.concatenate(all_bip_logits, axis=0),
        'labels_bip22':       np.concatenate(all_bip_labels, axis=0),
        'region_probs':       np.concatenate(all_region_probs, axis=0),
        'region_labels':      np.concatenate(all_region_labels, axis=0),
        'hemisphere_probs':   np.concatenate(all_hemi_probs, axis=0),
        'hemisphere_labels':  np.concatenate(all_hemi_labels, axis=0),
        'patient_ids':        all_pids,
        'edf_paths':          all_edfs,
    }


def mc_inference_single(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    x: torch.Tensor,
    onset: torch.Tensor,
    start: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
) -> np.ndarray:
    """
    对单个 batch 做 MC dropout 采样。

    model.train() 保持 dropout 开启, 前向 n_samples 次。

    x: [B, 22, T], onset: [B], start: [B]
    返回: [n_samples * B, 19] SOZ 概率 (DeepSOZ 通道顺序)
    """
    model.train()
    use_amp = device.type == 'cuda'
    results = []

    for _ in range(n_samples):
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(x.to(device), onset.to(device), start.to(device))
        probs = outputs['soz_probs'].cpu().numpy()          # [B, 19] STANDARD_19
        probs_dsz = reorder_to_deepsoz(probs)                # [B, 19] DeepSOZ
        results.append(probs_dsz)

    model.eval()
    return np.concatenate(results, axis=0)  # [n_samples * B, 19]


# ═════════════════════════════════════════════════════════════════════════════
# 官方 Seizure/Patient-level 评估
# ═════════════════════════════════════════════════════════════════════════════

def official_sz_pt_evaluation(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    dataset: SOZBrainNetworkDataset,
    device: torch.device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
) -> Dict:
    """
    官方 Seizure-level + Patient-level 评估 (MC dropout)。

    按患者分组, 对每个发作做 MC 采样, 然后用 final_loc 判断定位正确性。
    """
    # 按 patient_id 分组
    pid_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        pid_to_indices[sample['patient_id']].append(i)

    corr_pt = 0
    total_pt = 0
    pt_uncs = []
    pt_details = []
    all_corr_sz = []
    all_sz_unc = []
    sz_details = []

    for pt_id, indices in pid_to_indices.items():
        pt_psoz_all = []
        true_onset_dsz = None

        for ds_idx in indices:
            sample = dataset[ds_idx]
            x = sample['x'].unsqueeze(0)               # [1, 22, T]
            onset = torch.tensor([sample['onset_sec']])  # [1]
            start = torch.tensor([sample['start_sec']])  # [1]

            # 真实标签: monopolar [19] (STANDARD_19) → DeepSOZ 顺序
            true_mono_int = sample['monopolar_label'].numpy()
            true_onset_dsz = reorder_to_deepsoz(true_mono_int)

            # MC dropout 采样
            mc_maps = mc_inference_single(
                model, x, onset, start, device,
                n_samples=mc_samples,
            )  # [mc_samples, 19] DeepSOZ 顺序

            # Seizure-level
            ysoz, unc, correct = final_loc(
                mc_maps, true_onset_dsz,
                neighbour_threshold=neighbour_threshold,
            )
            all_corr_sz.append(correct)
            all_sz_unc.append(unc)
            sz_details.append({
                'pt_id': pt_id,
                'edf_path': sample['edf_path'],
                'correct': correct,
                'max_chn': int(np.argmax(ysoz)),
                'max_chn_name': DEEPSOZ_19[int(np.argmax(ysoz))],
                'unc_max': float(unc.max()),
            })

            pt_psoz_all.append(mc_maps)

        if true_onset_dsz is None:
            continue

        # Patient-level
        total_pt += 1
        pt_psoz = np.concatenate(pt_psoz_all, axis=0)
        ysoz_pt, unc_pt, correct_pt = final_loc(
            pt_psoz, true_onset_dsz,
            neighbour_threshold=neighbour_threshold,
        )
        corr_pt += correct_pt
        pt_uncs.append(unc_pt)
        pt_details.append({
            'pt_id':        pt_id,
            'correct':      correct_pt,
            'n_seizures':   len(pt_psoz_all),
            'max_chn':      int(np.argmax(ysoz_pt)),
            'max_chn_name': DEEPSOZ_19[int(np.argmax(ysoz_pt))],
            'true_soz':     [DEEPSOZ_19[i] for i in range(19) if true_onset_dsz[i] == 1],
            'unc_max':      float(unc_pt.max()),
        })

    acc_pt = corr_pt / total_pt if total_pt > 0 else 0.0
    corr_sz_mean = float(np.mean(all_corr_sz)) if all_corr_sz else 0.0
    ptunc_mean = float(np.mean([u.max() for u in pt_uncs])) if pt_uncs else 0.0
    szunc_mean = float(np.mean([u.max() for u in all_sz_unc])) if all_sz_unc else 0.0

    return {
        'patient_level': {
            'corr_pt': corr_pt,
            'total_pt': total_pt,
            'acc_pt': acc_pt,
            'ptunc_mean': ptunc_mean,
            'per_patient': pt_details,
        },
        'seizure_level': {
            'corr_sz_mean': corr_sz_mean,
            'total_sz': len(all_corr_sz),
            'szunc_mean': szunc_mean,
            'per_seizure': sz_details,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 通道映射验证
# ══════════════════════════════════════════════��══════════════════════════════

def verify_channel_mapping():
    print('\n' + '=' * 72)
    print('Channel Mapping Verification')
    print('=' * 72)

    print(f'\n1. Model output (STANDARD_19, via BipolarToMonopolarMapper):')
    for i, ch in enumerate(INTEGRATION_19):
        print(f'   [{i:>2}] {ch}')

    print(f'\n2. DeepSOZ official evaluation order:')
    for i, ch in enumerate(DEEPSOZ_19):
        print(f'   [{i:>2}] {ch}')

    print(f'\n3. Reorder mapping (STANDARD_19 → DeepSOZ):')
    for i, src_idx in enumerate(INTEGRATION_TO_DEEPSOZ):
        assert INTEGRATION_19[src_idx] == DEEPSOZ_19[i]
        print(f'   DeepSOZ[{i:>2}] {DEEPSOZ_19[i]:<5} ← STANDARD_19[{src_idx:>2}] {INTEGRATION_19[src_idx]}')

    # 标签映射验证
    print(f'\n4. Label mapping test:')
    test_bip = np.zeros(22, dtype=np.float32)
    test_bip[15] = 1.0  # P4-O2
    test_mono = bipolar22_to_monopolar19_binary(test_bip.reshape(1, -1)).reshape(19)
    active = [DEEPSOZ_19[i] for i in range(19) if test_mono[i] == 1]
    print(f'   bipolar P4-O2=1 → monopolar SOZ: {active}')
    assert 'P4' in active and 'O2' in active

    print('\n[OK] All channel mappings verified.')
    print('=' * 72 + '\n')


# ═════════════════════════════════════════════════════════════════════════════
# 参数
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Integration Model → DeepSOZ 官方评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument('--checkpoint', required=True,
                   help='TimeFilter_LaBraM_BrainNetwork_Integration .pt checkpoint')
    p.add_argument('--manifest', required=True,
                   help='combined_manifest.csv')
    p.add_argument('--tusz-data-root', default='F:/dataset/TUSZ/v2.0.3/edf')
    p.add_argument('--private-data-root', default='')
    p.add_argument('--source', default=None,
                   choices=['tusz', 'private', 'both'])
    p.add_argument('--split', nargs='+', default=None,
                   help='split 过滤 (train dev eval)')
    p.add_argument('--patient-ids', nargs='+', default=None)

    # 模型架构 (仅在 checkpoint 不含 config 时使用)
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--patch-len', type=int, default=200)
    p.add_argument('--n-pre-patches', type=int, default=5)
    p.add_argument('--n-post-patches', type=int, default=5)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--output-mode', default='monopolar',
                   choices=['monopolar', 'bipolar'])
    p.add_argument('--n-regions', type=int, default=6)

    # 评估
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--mc-samples', type=int, default=20)
    p.add_argument('--neighbour-threshold', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda')

    # 输出
    p.add_argument('--output-dir', default=None)
    p.add_argument('--save-preds', action='store_true')
    p.add_argument('--skip-mc', action='store_true',
                   help='跳过 MC dropout, 仅做标准混淆矩阵')
    p.add_argument('--verify-mapping', action='store_true')
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.verify_mapping:
        verify_channel_mapping()

    device = (torch.device('cuda')
              if args.device == 'cuda' and torch.cuda.is_available()
              else torch.device('cpu'))
    logger.info(f'Device: {device}')

    # ── 构建模型 ──────────────────────────────────────────────────────────
    model, cfg = build_model(args)
    model.to(device)

    # ── 构建数据集 ────────────────────────────────────────────────────────
    source_filter = args.source or 'both'
    manifest_ds = ManifestSOZDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        private_data_root=args.private_data_root or None,
        source_filter=source_filter,
        split_filter=args.split,
        patient_ids=args.patient_ids,
        label_mode='monopolar',  # 需要 monopolar 标签用于 DeepSOZ 评估
    )

    ds = SOZBrainNetworkDataset(manifest_ds)
    logger.info(f'Dataset: {len(ds)} samples, '
                f'{len(manifest_ds.get_patient_ids())} patients, '
                f'source={source_filter}')

    if len(ds) == 0:
        logger.error('数据集为空')
        return

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=eval_collate_fn,
        pin_memory=device.type == 'cuda',
    )

    # ══════════════════════════════════════════════════════════════════════
    # 1. 标准推理 → 混淆矩阵
    # ══════════════════════════════════════════════════════════════════════
    logger.info('Running standard inference...')
    results = run_standard_inference(model, loader, device)
    n_samples = results['soz_probs_dsz19'].shape[0]
    logger.info(f'Inference done: {n_samples} samples')

    # ── 诊断: 打印 raw logits/probs 分布 ─────────────────────────────
    soz_logits = results['soz_logits_dsz19']   # [N, 19]
    soz_probs  = results['soz_probs_dsz19']    # [N, 19]
    bip_logits = results['bipolar_logits']      # [N, 22]
    bip_probs  = 1.0 / (1.0 + np.exp(-bip_logits))  # sigmoid

    print('\n' + '=' * 72)
    print('DIAGNOSTIC: Raw Output Distribution')
    print('=' * 72)

    print('\n-- soz_logits (19ch, after b2m mapping) --')
    print(f'   overall: min={soz_logits.min():.4f}  max={soz_logits.max():.4f}  '
          f'mean={soz_logits.mean():.4f}  std={soz_logits.std():.4f}')
    print(f'   per-channel mean:')
    for i, ch in enumerate(DEEPSOZ_19):
        print(f'     [{i:>2}] {ch:<5}  mean={soz_logits[:, i].mean():>8.4f}  '
              f'std={soz_logits[:, i].std():>7.4f}  '
              f'min={soz_logits[:, i].min():>8.4f}  max={soz_logits[:, i].max():>8.4f}')

    print('\n-- soz_probs (19ch, sigmoid of soz_logits) --')
    print(f'   overall: min={soz_probs.min():.4f}  max={soz_probs.max():.4f}  '
          f'mean={soz_probs.mean():.4f}')
    print(f'   fraction >= 0.5: {(soz_probs >= 0.5).mean():.4f}')
    print(f'   fraction >= 0.3: {(soz_probs >= 0.3).mean():.4f}')
    print(f'   fraction >= 0.1: {(soz_probs >= 0.1).mean():.4f}')
    # argmax 分布
    argmax_19 = soz_probs.argmax(axis=1)
    unique, counts = np.unique(argmax_19, return_counts=True)
    print(f'   argmax distribution (DeepSOZ order):')
    for u, c in zip(unique, counts):
        print(f'     {DEEPSOZ_19[u]:<5} : {c:>5} ({c/n_samples:.1%})')

    print('\n-- bipolar_logits (22ch, before b2m mapping) --')
    print(f'   overall: min={bip_logits.min():.4f}  max={bip_logits.max():.4f}  '
          f'mean={bip_logits.mean():.4f}  std={bip_logits.std():.4f}')
    print(f'   per-channel mean (top 5):')
    bip_means = bip_logits.mean(axis=0)
    bip_names = TCP_BIPOLAR_NAMES
    sorted_idx = np.argsort(bip_means)[::-1]
    for rank, j in enumerate(sorted_idx[:5]):
        print(f'     #{rank+1}  [{j:>2}] {bip_names[j]:<10}  mean={bip_means[j]:>8.4f}')

    print(f'\n-- bipolar_probs (sigmoid of bipolar_logits) --')
    print(f'   overall: min={bip_probs.min():.4f}  max={bip_probs.max():.4f}  '
          f'mean={bip_probs.mean():.4f}')
    print(f'   fraction >= 0.5: {(bip_probs >= 0.5).mean():.4f}')
    print(f'   fraction >= 0.3: {(bip_probs >= 0.3).mean():.4f}')
    print(f'   fraction >= 0.1: {(bip_probs >= 0.1).mean():.4f}')

    print('=' * 72 + '\n')

    # ── 22 双极通道混淆矩阵 (模型直接输出, 训练时的优化目标) ──────────
    bip_report_05 = print_confusion_report(
        results['labels_bip22'], bip_probs,
        TCP_BIPOLAR_NAMES, threshold=0.5,
        title='Bipolar 22ch (sigmoid of bipolar_logits, th=0.5)',
    )
    best_th_bip, best_f1_bip = find_best_threshold(
        results['labels_bip22'], bip_probs,
    )
    if abs(best_th_bip - 0.5) > 0.01:
        print_confusion_report(
            results['labels_bip22'], bip_probs,
            TCP_BIPOLAR_NAMES, threshold=best_th_bip,
            title=f'Bipolar 22ch (sigmoid of bipolar_logits, th={best_th_bip:.2f})',
        )
    logger.info(f'Bipolar best threshold: {best_th_bip:.2f} (macro F1={best_f1_bip:.3f})')

    # ── 19 单极通道混淆矩阵 (DeepSOZ 顺序) ──────────────────────────
    best_th, best_f1 = find_best_threshold(
        results['labels_mono_dsz19'], results['soz_probs_dsz19'],
    )
    logger.info(f'Monopolar best threshold: {best_th:.2f} (macro F1={best_f1:.3f})')

    ch_report_05 = print_confusion_report(
        results['labels_mono_dsz19'], results['soz_probs_dsz19'],
        DEEPSOZ_19, threshold=0.5,
        title='Channel-level 19ch DeepSOZ order (th=0.5)',
    )

    if abs(best_th - 0.5) > 0.01:
        ch_report_best = print_confusion_report(
            results['labels_mono_dsz19'], results['soz_probs_dsz19'],
            DEEPSOZ_19, threshold=best_th,
            title=f'Channel-level 19ch DeepSOZ order (th={best_th:.2f})',
        )
    else:
        ch_report_best = ch_report_05

    # Region-level
    region_report = print_confusion_report(
        results['region_labels'], results['region_probs'],
        list(get_region_names('coarse')), threshold=0.5,
        title='Region-level (6 coarse regions)',
    )

    # Hemisphere
    hemi_true = results['hemisphere_labels']
    valid_hemi = hemi_true >= 0
    hemi_acc = 0.0
    if valid_hemi.sum() > 0:
        hemi_pred = results['hemisphere_probs'].argmax(axis=1)
        hemi_acc = float((hemi_pred[valid_hemi] == hemi_true[valid_hemi]).mean())
        print(f'\n## Hemisphere Classification')
        print(f'  Accuracy: {hemi_acc:.3f} '
              f'({int((hemi_pred[valid_hemi] == hemi_true[valid_hemi]).sum())} '
              f'/ {int(valid_hemi.sum())})')
    else:
        print('\n## Hemisphere: no valid samples')

    # ══════════════════════════════════════════════════════════════════════
    # 2. MC Dropout → Seizure/Patient-level
    # ══════════════════════════════════════════════════════════════════════
    official_results = None
    if not args.skip_mc:
        logger.info(f'\nMC dropout evaluation (samples={args.mc_samples})...')
        official_results = official_sz_pt_evaluation(
            model=model,
            dataset=ds,
            device=device,
            mc_samples=args.mc_samples,
            neighbour_threshold=args.neighbour_threshold,
        )

        pt = official_results['patient_level']
        sz = official_results['seizure_level']

        print(f'\n## Patient-level SOZ Localization (DeepSOZ method)')
        print(f'  Correct: {pt["corr_pt"]} / {pt["total_pt"]}  '
              f'Accuracy: {pt["acc_pt"]:.3f}')
        print(f'  Uncertainty (mean max-var): {pt["ptunc_mean"]:.4f}')

        print(f'\n## Seizure-level SOZ Localization (DeepSOZ method)')
        print(f'  Total seizures: {sz["total_sz"]}')
        print(f'  Correct rate (corr_sz): {sz["corr_sz_mean"]:.3f}')
        print(f'  Uncertainty (mean max-var): {sz["szunc_mean"]:.4f}')

        print(f'\n## Per-patient Details\n')
        print(f'| {"Patient":<20} | {"#Sz":>4} | {"Corr":>5} | '
              f'{"Pred":>10} | {"True SOZ":<30} | {"Unc":>6} |')
        print(f'|{"-"*22}|{"-"*6}|{"-"*7}|{"-"*12}|{"-"*32}|{"-"*8}|')
        for d in pt['per_patient']:
            mark = 'Y' if d['correct'] else 'N'
            true_str = ','.join(d['true_soz'])[:30]
            print(f'| {d["pt_id"]:<20} | {d["n_seizures"]:>4} | '
                  f'{mark:>5} | {d["max_chn_name"]:>10} | '
                  f'{true_str:<30} | {d["unc_max"]:>6.4f} |')

    # ══════════════════════════════════════════════════════════════════════
    # 3. 保存结果
    # ══════════════════════════════════════════════════════════════════════
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    summary = {
        'checkpoint': args.checkpoint,
        'n_samples': n_samples,
        'n_patients': len(manifest_ds.get_patient_ids()),
        'source': source_filter,
        'channel_level': {
            'threshold_05': {
                'macro_f1': ch_report_05.get('_macro', {}).get('f1', 0.0),
                'macro_prec': ch_report_05.get('_macro', {}).get('precision', 0.0),
                'macro_rec': ch_report_05.get('_macro', {}).get('recall', 0.0),
            },
            'best_threshold': best_th,
            'best_macro_f1': best_f1,
            'per_channel': {
                name: cm for name, cm in ch_report_best.items()
                if not name.startswith('_')
            },
        },
        'region_level': {
            name: cm for name, cm in region_report.items()
            if not name.startswith('_')
        },
        'hemisphere_acc': float(hemi_acc),
    }

    if official_results:
        summary['patient_level'] = {
            k: v for k, v in official_results['patient_level'].items()
            if k != 'per_patient'
        }
        summary['patient_details'] = official_results['patient_level']['per_patient']
        summary['seizure_level'] = {
            k: v for k, v in official_results['seizure_level'].items()
            if k != 'per_seizure'
        }

    result_path = out_dir / f'{stem}_deepsoz_eval.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'Results saved: {result_path}')

    if args.save_preds:
        rows = []
        for i in range(n_samples):
            row = {
                'patient_id': results['patient_ids'][i],
                'edf_path': results['edf_paths'][i],
            }
            for j, ch_name in enumerate(DEEPSOZ_19):
                row[f'prob_{ch_name}'] = float(results['soz_probs_dsz19'][i, j])
                row[f'label_{ch_name}'] = int(results['labels_mono_dsz19'][i, j])
            rows.append(row)
        pred_path = out_dir / f'{stem}_deepsoz_preds.csv'
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        logger.info(f'Predictions saved: {pred_path}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
