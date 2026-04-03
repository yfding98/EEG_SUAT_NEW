#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Model 评估脚本 — 使用 DeepSOZ 官方评估方法

使用 TUSZ/models/integration_model.py 训练的模型，
在 combined_manifest.csv 上跑推理，然后按 DeepSOZ 官方评估逻辑计算：

1. Channel-level (19通道)
   - 逐通道混淆矩阵 (TP/FP/TN/FN/Precision/Recall/Specificity/F1)
   - 多阈值扫描 + 最优阈值

2. Region-level (6脑区)
   - 逐区混淆矩阵

3. Seizure-level 定位正确率 (corr_sz)
   - MC dropout N 次采样 → 归一化 → 均值 → argmax → 判断命中 SOZ
   - MC 不确定性 (szunc)

4. Patient-level 定位正确率 (corr_pt)
   - 聚合同一患者所有发作的 MC 采样 → argmax
   - MC 不确定性 (ptunc)

5. 邻居放宽判断 (chn_neighbours)
   - 当 SOZ 通道数 ≤ threshold 时，预测通道落在空间邻居也算正确

关键注意：
   Integration model 输出 19 通道顺序 (STANDARD_19 / BipolarToMonopolarMapper):
     [FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ]

   DeepSOZ 官方 19 通道顺序 (OFFICIAL_19_CHANNELS):
     [FP1, FP2, F7, F3, FZ, F4, F8, T3, C3, CZ, C4, T4, T5, P3, PZ, P4, T6, O1, O2]

   脚本内置了通道重排序映射来对齐两边。

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
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# ─── 项目路径设置 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # EEG_SUAT_NEW
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ' / 'models'))

from manifest_dataset import (
    ManifestSOZDataset, TCP_BIPOLAR_NAMES, TCP_COL_NAMES,
    COARSE_REGION_NAMES, HEMISPHERE_NAMES,
)
from integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
)
from bipolar_to_monopolar import DEFAULT_MONOPOLAR_19

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 通道定义 & 重排序映射
# ═════════════════════════════════════════════════════════════════════════════

# Integration model 输出 19 通道顺序 (BipolarToMonopolarMapper / STANDARD_19)
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
# INTEGRATION_TO_DEEPSOZ[i] = integration_model 输出中第几个通道对应 DeepSOZ 第 i 个通道
_int_idx = {ch: i for i, ch in enumerate(INTEGRATION_19)}
INTEGRATION_TO_DEEPSOZ = [_int_idx[ch] for ch in DEEPSOZ_19]

# 构建 DeepSOZ → Integration 重排索引 (用于将 DeepSOZ 顺序标签映射到 Integration 顺序)
_dsz_idx = {ch: i for i, ch in enumerate(DEEPSOZ_19)}
DEEPSOZ_TO_INTEGRATION = [_dsz_idx[ch] for ch in INTEGRATION_19]


def reorder_to_deepsoz(arr: np.ndarray) -> np.ndarray:
    """将 Integration 模型的 19 通道输出重排为 DeepSOZ 官方顺序。
    arr: [..., 19] — 最后一维是通道
    """
    return arr[..., INTEGRATION_TO_DEEPSOZ]


# ═════════════════════════════════════════════════════════════════════════════
# DeepSOZ 官方评估工具函数 (来自 deeksha-ms/DeepSOZ 源码)
# ═════════════════════════════════════════════════════════════════════════════

# 官方 chn_neighbours (19 通道空间邻接, 索引基于 DEEPSOZ_19 顺序)
# 来源: deeksha-ms/DeepSOZ code/test/final_eval_all.ipynb
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
    """官方 check_neighborhood: max_chn 是否落在任一真实 SOZ 的空间邻居内。"""
    for i in range(len(onset_map)):
        if onset_map[i] == 1:
            if max_chn in CHN_NEIGHBOURS_19.get(i, []):
                return True
    return False


def final_loc(
    psoz: np.ndarray,
    true_onset: np.ndarray,
    neighbour_threshold: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    官方 final_loc (来自 final_eval_all.ipynb):

    psoz        : [N, 19]  N次 MC 采样的通道预测值 (DeepSOZ 通道顺序)
    true_onset  : [19]     0/1 SOZ 标签 (DeepSOZ 通道顺序)

    流程:
      1. 每行除以行最大值 (归一化)
      2. 列均值 → [19] 最终通道概率
      3. argmax 取预测通道
      4. 精确命中 or 邻居放宽

    返回 (ysoz, uncertainty, correct)
    """
    m = psoz.max(axis=1, keepdims=True)
    m = np.where(m > 0, m, 1.0)
    psoz_norm = psoz / m
    ysoz = psoz_norm.mean(axis=0)       # [19]

    max_chn = int(np.argmax(ysoz))
    correct = 1 if true_onset[max_chn] == 1 else 0

    if (correct == 0
            and int(true_onset.sum()) <= neighbour_threshold
            and check_neighborhood(max_chn, true_onset)):
        correct = 1

    uncertainty = psoz_norm.var(axis=0)  # [19]
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
    """打印逐通道/逐区混淆矩阵并返回详情 dict。"""
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

    # MACRO 平均
    macro_prec = np.mean([cm['precision'] for cm in per_label.values()])
    macro_rec  = np.mean([cm['recall']    for cm in per_label.values()])
    macro_spec = np.mean([cm['specificity'] for cm in per_label.values()])
    macro_f1   = np.mean([cm['f1']        for cm in per_label.values()])
    print(f'| {"MACRO":<10} |      |      |      |      |       | '
          f'{macro_prec:>6.3f} | {macro_rec:>6.3f} | {macro_spec:>6.3f} | {macro_f1:>6.3f} |')

    # MICRO 平均
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
    """扫描阈值，找 macro F1 最优值。"""
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
# 模型加载
# ═════════════════════════════════════════════════════════════════════════════

def build_model(args) -> TimeFilter_LaBraM_BrainNetwork_Integration:
    """根据命令行参数构建 Integration model 并加载权重。"""
    cfg = IntegrationConfig(
        n_channels=22,
        embed_dim=args.embed_dim,
        n_transformer_layers=args.n_transformer_layers,
        n_frozen_layers=0,  # 推理时不冻结
        labram_checkpoint='',  # 不重新加载 LaBraM pretrain
        n_pre_patches=args.n_pre_patches,
        n_post_patches=args.n_post_patches,
        patch_len=args.patch_len,
        n_timefilter_blocks=args.n_timefilter_blocks,
        brain_tf_n_blocks=args.brain_tf_n_blocks,
        brain_tf_hidden=args.brain_tf_hidden,
        gru_hidden=args.gru_hidden,
        gcn_hidden=args.gcn_hidden,
        output_mode='monopolar',      # 输出 19 通道
        task_mode='soz',
        n_regions=args.n_regions,
        use_checkpoint=False,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg)

    ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt  # 直接是 state_dict

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f'Missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        logger.warning(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
    logger.info(f'Loaded checkpoint: {ckpt_path}')
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 推理函数
# ═════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """与 ManifestSOZDataset 配套的 collate 函数。"""
    Xs, ys, masks, metas, y_bips, y_monos, y_regs, y_hemis = zip(*batch)
    return (
        torch.stack(Xs),
        torch.stack(ys),
        torch.stack(masks),
        list(metas),
        torch.stack(y_bips),
        torch.stack(y_monos),
        torch.stack(y_regs),
        torch.stack(y_hemis),
    )


@torch.no_grad()
def run_standard_inference(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    标准推理 (model.eval(), 无 MC dropout)。

    返回:
        {
            'soz_probs_19':     [N, 19] (Integration 顺序)
            'soz_probs_dsz19':  [N, 19] (DeepSOZ 顺序)
            'labels_mono_19':   [N, 19] (Integration 顺序)
            'labels_dsz19':     [N, 19] (DeepSOZ 顺序, 通过重排标签得到)
            'labels_bip_22':    [N, 22]
            'region_probs':     [N, 6]
            'region_labels':    [N, 6]
            'hemisphere_probs': [N, 3]
            'hemisphere_labels':[N]
            'patient_ids':      [N] list
            'edf_paths':        [N] list
        }
    """
    model.eval()
    use_amp = device.type == 'cuda'

    all_soz_probs, all_mono_labels, all_bip_labels = [], [], []
    all_region_probs, all_region_labels = [], []
    all_hemi_probs, all_hemi_labels = [], []
    all_pids, all_edfs = [], []

    for batch in loader:
        X, y_soz, mask, metas, y_bip, y_mono, y_reg, y_hemi = batch
        X = X.to(device)

        with autocast(enabled=use_amp):
            outputs = model(X)

        soz_probs = outputs['soz_probs'].cpu().numpy()          # [B, 19]
        region_probs = outputs['region_probs'].cpu().numpy()     # [B, 6]
        hemi_probs = outputs['hemisphere_probs'].cpu().numpy()   # [B, 3]

        all_soz_probs.append(soz_probs)
        all_mono_labels.append(y_mono.numpy())
        all_bip_labels.append(y_bip.numpy())
        all_region_probs.append(region_probs)
        all_region_labels.append(y_reg.numpy())
        all_hemi_probs.append(hemi_probs)
        all_hemi_labels.append(y_hemi.numpy())

        for m in metas:
            all_pids.append(m['patient_id'])
            all_edfs.append(m['edf_path'])

    soz_probs_19 = np.concatenate(all_soz_probs, axis=0)        # [N, 19] Integration 顺序

    # 标签也是 Integration 顺序 (b2m_matrix 生成的 monopolar_label 用的是 STANDARD_19)
    labels_mono_19 = np.concatenate(all_mono_labels, axis=0)     # [N, 19]

    # 重排到 DeepSOZ 顺序
    soz_probs_dsz19 = reorder_to_deepsoz(soz_probs_19)
    labels_dsz19 = reorder_to_deepsoz(labels_mono_19)

    return {
        'soz_probs_19':     soz_probs_19,
        'soz_probs_dsz19':  soz_probs_dsz19,
        'labels_mono_19':   labels_mono_19,
        'labels_dsz19':     labels_dsz19,
        'labels_bip_22':    np.concatenate(all_bip_labels, axis=0),
        'region_probs':     np.concatenate(all_region_probs, axis=0),
        'region_labels':    np.concatenate(all_region_labels, axis=0),
        'hemisphere_probs': np.concatenate(all_hemi_probs, axis=0),
        'hemisphere_labels': np.concatenate(all_hemi_labels, axis=0),
        'patient_ids':      all_pids,
        'edf_paths':        all_edfs,
    }


def mc_inference_single(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    X: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
) -> np.ndarray:
    """
    对单个 batch 做 MC dropout 采样。

    model.train() 保持 dropout 开启，前向 n_samples 次。

    X: [B, 22, n_patches, patch_len]
    返回: [n_samples * B, 19] 通道 SOZ 概率 (DeepSOZ 顺序)
    """
    model.train()
    use_amp = device.type == 'cuda'
    results = []

    for _ in range(n_samples):
        with torch.no_grad():
            with autocast(enabled=use_amp):
                outputs = model(X.to(device))
        probs = outputs['soz_probs'].cpu().numpy()       # [B, 19] Integration 顺序
        probs_dsz = reorder_to_deepsoz(probs)             # [B, 19] DeepSOZ 顺序
        results.append(probs_dsz)

    model.eval()
    return np.concatenate(results, axis=0)  # [n_samples * B, 19]


# ════════════════════════════��════════════════════════════════════════════════
# 官方 Seizure/Patient-level 评估
# ═════════════════════════════════════════════════════════════════════════════

def official_sz_pt_evaluation(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    dataset: ManifestSOZDataset,
    device: torch.device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
) -> Dict:
    """
    官方 Seizure-level + Patient-level 评估 (MC dropout)。

    按患者分组，对每个发作做 MC 采样，然后用 final_loc 判断定位正确性。
    """
    # 按患者分组
    patient_ids = dataset.get_patient_ids()
    df = dataset.df

    corr_pt = 0
    total_pt = 0
    pt_uncs = []
    pt_details = []
    all_corr_sz = []
    all_sz_unc = []
    sz_details = []

    for pt_id in patient_ids:
        # 获取该患者所有样本索引
        pt_mask = df['patient_id'] == pt_id
        indices = df.index[pt_mask].tolist()
        if len(indices) == 0:
            continue

        pt_psoz_all = []
        true_onset_dsz = None

        for idx_in_df in indices:
            # 获取在 dataset 中的位置索引
            ds_idx = dataset.df.index.get_loc(idx_in_df)

            sample = dataset[ds_idx]
            X, y_soz, mask, meta, y_bip, y_mono, y_reg, y_hemi = sample

            # 真实 onset_map: monopolar [19] (Integration 顺序) → DeepSOZ 顺序
            true_onset_int = y_mono.numpy()                       # [19]
            true_onset_dsz = reorder_to_deepsoz(true_onset_int)   # [19]

            # MC dropout 采样
            X_batch = X.unsqueeze(0)                              # [1, 22, P, L]
            mc_maps = mc_inference_single(
                model, X_batch, device,
                n_samples=mc_samples,
            )  # [mc_samples, 19] DeepSOZ 顺序

            # Seizure-level 评估
            ysoz, unc, correct = final_loc(
                mc_maps, true_onset_dsz,
                neighbour_threshold=neighbour_threshold,
            )
            all_corr_sz.append(correct)
            all_sz_unc.append(unc)
            sz_details.append({
                'pt_id': pt_id,
                'edf_path': meta['edf_path'],
                'correct': correct,
                'max_chn': int(np.argmax(ysoz)),
                'max_chn_name': DEEPSOZ_19[int(np.argmax(ysoz))],
                'unc_max': float(unc.max()),
            })

            pt_psoz_all.append(mc_maps)

        if true_onset_dsz is None:
            continue

        # Patient-level 评估
        total_pt += 1
        pt_psoz = np.concatenate(pt_psoz_all, axis=0)  # [N_total, 19]
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
# 通道映射验证 (调试用)
# ═════════════════════════════════════════════════════════════════════════════

def verify_channel_mapping():
    """打印通道映射详情以供人工检查。"""
    print('\n' + '=' * 70)
    print('Channel Mapping Verification')
    print('=' * 70)

    print('\nIntegration model 输出顺序 (STANDARD_19 / BipolarToMonopolarMapper):')
    for i, ch in enumerate(INTEGRATION_19):
        print(f'  [{i:>2}] {ch}')

    print('\nDeepSOZ 官方评估顺序 (OFFICIAL_19_CHANNELS):')
    for i, ch in enumerate(DEEPSOZ_19):
        print(f'  [{i:>2}] {ch}')

    print('\nIntegration → DeepSOZ 重排索引:')
    print(f'  INTEGRATION_TO_DEEPSOZ = {INTEGRATION_TO_DEEPSOZ}')
    print('\n  即: DeepSOZ[i] = Integration[INTEGRATION_TO_DEEPSOZ[i]]')
    for i, src_idx in enumerate(INTEGRATION_TO_DEEPSOZ):
        assert INTEGRATION_19[src_idx] == DEEPSOZ_19[i], \
            f'Mapping error: Integration[{src_idx}]={INTEGRATION_19[src_idx]} != DeepSOZ[{i}]={DEEPSOZ_19[i]}'
        print(f'  DeepSOZ[{i:>2}] {DEEPSOZ_19[i]:<5} ← Integration[{src_idx:>2}] {INTEGRATION_19[src_idx]}')

    print('\n[OK] Channel mapping verified — all 19 channels match correctly.')
    print('=' * 70 + '\n')


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
                   help='Integration model .pt checkpoint')
    p.add_argument('--manifest', required=True,
                   help='combined_manifest.csv 路径')
    p.add_argument('--tusz-data-root', default='F:/dataset/TUSZ/v2.0.3/edf',
                   help='TUSZ EDF 根目录')
    p.add_argument('--private-data-root', default='',
                   help='私有数据 EDF 根目录')
    p.add_argument('--source', default=None,
                   choices=['tusz', 'private', 'both'],
                   help='数据源过滤')
    p.add_argument('--split', nargs='+', default=None,
                   help='split 过滤 (如 train dev eval)')
    p.add_argument('--patient-ids', nargs='+', default=None,
                   help='仅评估这些患者')

    # 模型架构 (需与训练时一致)
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--n-transformer-layers', type=int, default=12)
    p.add_argument('--n-pre-patches', type=int, default=5)
    p.add_argument('--n-post-patches', type=int, default=5)
    p.add_argument('--patch-len', type=int, default=200)
    p.add_argument('--n-timefilter-blocks', type=int, default=2)
    p.add_argument('--brain-tf-n-blocks', type=int, default=1)
    p.add_argument('--brain-tf-hidden', type=int, default=64)
    p.add_argument('--gru-hidden', type=int, default=128)
    p.add_argument('--gcn-hidden', type=int, default=64)
    p.add_argument('--n-regions', type=int, default=6)

    # 评估
    p.add_argument('--threshold', type=float, default=0.5,
                   help='混淆矩阵二值化阈值')
    p.add_argument('--mc-samples', type=int, default=20,
                   help='MC dropout 采样次数')
    p.add_argument('--neighbour-threshold', type=int, default=4,
                   help='邻居放宽 SOZ 通道数上限')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda')

    # 输出
    p.add_argument('--output-dir', default=None,
                   help='结果保存目录 (默认 checkpoint 同级)')
    p.add_argument('--save-preds', action='store_true',
                   help='保存逐样本预测 CSV')
    p.add_argument('--skip-mc', action='store_true',
                   help='跳过 MC dropout 评估 (仅做标准推理)')
    p.add_argument('--verify-mapping', action='store_true',
                   help='打印通道映射验证信息')
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 通道映射验证
    if args.verify_mapping:
        verify_channel_mapping()

    device = (torch.device('cuda')
              if args.device == 'cuda' and torch.cuda.is_available()
              else torch.device('cpu'))
    logger.info(f'Device: {device}')

    # ── 构建模型 ──────────────────────────────────────────────────────────
    model = build_model(args)
    model.to(device)

    # ── 构建数据集 ────────────────────────────────────────────────────────
    source_filter = args.source or 'both'
    ds = ManifestSOZDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        private_data_root=args.private_data_root or None,
        source_filter=source_filter,
        split_filter=args.split,
        patient_ids=args.patient_ids,
        label_mode='monopolar',  # 输出 19 通道标签
    )
    logger.info(f'Dataset: {len(ds)} samples, '
                f'{len(ds.get_patient_ids())} patients, '
                f'source={source_filter}')

    if len(ds) == 0:
        logger.error('数据集为空，请检查 manifest 和数据路径')
        return

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == 'cuda',
    )

    # ══════════════════════════════════════════════════════════════════════
    # 1. 标准推理 → Channel/Region/Hemisphere 混淆矩阵
    # ══════════════════════════════════════════════════════════════════════
    logger.info('Running standard inference...')
    results = run_standard_inference(model, loader, device)

    n_samples = results['soz_probs_dsz19'].shape[0]
    logger.info(f'Inference done: {n_samples} samples')

    # ── 1a. Channel-level 混淆矩阵 (DeepSOZ 通道顺序) ────────────────
    # 先找最优阈值
    best_th, best_f1 = find_best_threshold(
        results['labels_dsz19'], results['soz_probs_dsz19'],
    )
    logger.info(f'Best channel threshold: {best_th:.2f} (macro F1={best_f1:.3f})')

    # 固定阈值 0.5 和最优阈值都打印
    ch_report_05 = print_confusion_report(
        results['labels_dsz19'], results['soz_probs_dsz19'],
        DEEPSOZ_19, threshold=0.5,
        title='Channel-level (DeepSOZ order, th=0.5)',
    )

    if abs(best_th - 0.5) > 0.01:
        ch_report_best = print_confusion_report(
            results['labels_dsz19'], results['soz_probs_dsz19'],
            DEEPSOZ_19, threshold=best_th,
            title=f'Channel-level (DeepSOZ order, th={best_th:.2f})',
        )
    else:
        ch_report_best = ch_report_05

    # ── 1b. Region-level 混淆矩阵 ────────────────────────────────────
    region_report = print_confusion_report(
        results['region_labels'], results['region_probs'],
        COARSE_REGION_NAMES, threshold=0.5,
        title='Region-level (6 coarse regions)',
    )

    # ── 1c. Hemisphere 准确率 ─────────────────────────────────────────
    hemi_pred = results['hemisphere_probs'].argmax(axis=1)
    hemi_true = results['hemisphere_labels']
    valid_hemi = hemi_true >= 0  # 排除 IGNORE_INDEX = -100
    if valid_hemi.sum() > 0:
        hemi_acc = (hemi_pred[valid_hemi] == hemi_true[valid_hemi]).mean()
        print(f'\n## Hemisphere Classification')
        print(f'  Accuracy: {hemi_acc:.3f} ({int((hemi_pred[valid_hemi] == hemi_true[valid_hemi]).sum())} / {int(valid_hemi.sum())})')
        print(f'  Classes: {HEMISPHERE_NAMES}')
    else:
        hemi_acc = 0.0
        print(f'\n## Hemisphere Classification: no valid samples')

    # ══════════════════════════════════════════════════════════════════════
    # 2. MC Dropout → Seizure/Patient-level 评估
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

        # 打印每个患者详情
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
        'n_patients': len(ds.get_patient_ids()),
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

    # ── 逐样本预测 CSV ──────────────────────────────────────────────
    if args.save_preds:
        rows = []
        for i in range(n_samples):
            row = {
                'patient_id': results['patient_ids'][i],
                'edf_path': results['edf_paths'][i],
            }
            # DeepSOZ 顺序通道概率 & 标签
            for j, ch_name in enumerate(DEEPSOZ_19):
                row[f'prob_{ch_name}'] = float(results['soz_probs_dsz19'][i, j])
                row[f'label_{ch_name}'] = int(results['labels_dsz19'][i, j])
            # Region
            for j, reg_name in enumerate(COARSE_REGION_NAMES):
                row[f'reg_prob_{reg_name}'] = float(results['region_probs'][i, j])
                row[f'reg_label_{reg_name}'] = int(results['region_labels'][i, j])
            rows.append(row)
        pred_path = out_dir / f'{stem}_deepsoz_preds.csv'
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        logger.info(f'Predictions saved: {pred_path}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
