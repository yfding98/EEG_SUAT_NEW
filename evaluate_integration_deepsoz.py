#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Model ↔ DeepSOZ 官方评估对比脚本
==============================================

使用 DeepSOZ 官方评估方法（final_eval_all.ipynb + szloc_all.ipynb）来评估
TimeFilter_LaBraM_BrainNetwork_Integration 模型在 SOZ 定位任务上的表现，
方便与 DeepSOZ 基线做公平对比实验。

评估层级：
  1. Channel-level — 逐通道混淆矩阵（TP/FP/TN/FN/Precision/Recall/F1）
  2. Seizure-level — 每次发作的定位正确率 + MC 不确定性
  3. Patient-level — 聚合同患者所有发作 → 定位正确率 + 不确定性

关键适配逻辑：
  - DeepSOZ 数据集: [B, Nsz, T, 19, 200] (19 单极, 1s 帧)
  - Integration 模型: [B, 22, 2000] (22 双极, 10s 窗口)
  → 本脚本直接从 EDF 文件出发，分别为两套管道各自加载所需格式

用法:
  python evaluate_integration_deepsoz.py \
      --checkpoint output/TUSZ/train/best_model.pt \
      --manifest TUSZ/combined_manifest.csv \
      --data-roots F:/dataset/TUSZ/v2.0.3/edf E:/DataSet/EEG/EEG_dataset_SUAT \
      --source private \
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
from torch.utils.data import DataLoader, Dataset

# ── 项目路径设置 ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'TUSZ'))
sys.path.insert(0, str(ROOT / 'TUSZ' / 'models'))
sys.path.insert(0, str(ROOT / 'DeepSOZ_new'))

from TUSZ.models.integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
)
from TUSZ.data_preprocess.eeg_pipeline import (
    EEGPipeline,
    PipelineConfig,
    SeizureEvent,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# =====================================================================
# 官方 DeepSOZ 评估逻辑（完整复现）
# =====================================================================

# 19 单极通道名
OFFICIAL_19_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2',
]

# 22 双极通道名
TCP_BIPOLAR_COLUMNS = [
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
    'A1_T3', 'T3_C3', 'C3_CZ', 'CZ_C4', 'C4_T4', 'T4_A2',
]

# 官方 chn_neighbours（19 通道 10-20 系统拓扑邻接关系）
CHN_NEIGHBOURS_19 = {
    0:  [1, 2, 3, 4],
    1:  [0, 4, 5, 6],
    2:  [0, 3, 4, 7, 8],
    3:  [0, 2, 4, 8, 9],
    4:  [0, 1, 3, 5, 9],
    5:  [1, 4, 6, 9, 10],
    6:  [1, 4, 5, 10, 11],
    7:  [2, 8, 12, 13, 17],
    8:  [2, 3, 4, 7, 9, 12, 13, 14],
    9:  [3, 4, 5, 8, 10, 13, 14, 15],
    10: [4, 5, 6, 9, 11, 14, 15, 16],
    11: [6, 10, 15, 16, 18],
    12: [7, 8, 13, 17],
    13: [7, 8, 9, 12, 14, 17],
    14: [8, 9, 10, 13, 15, 17, 18],
    15: [9, 10, 11, 14, 16, 18],
    16: [10, 11, 15, 18],
    17: [7, 12, 13, 14, 18],
    18: [11, 14, 15, 16, 17],
}


def check_neighborhood(max_chn: int, onset_map: np.ndarray,
                        neighbours: Dict = None) -> bool:
    """官方 check_neighborhood：预测通道落在真实 SOZ 的邻居内则正确。"""
    if neighbours is None:
        neighbours = CHN_NEIGHBOURS_19
    for i in range(len(onset_map)):
        if onset_map[i] == 1:
            if max_chn in neighbours.get(i, []):
                return True
    return False


def final_loc(psoz: np.ndarray, true_onset: np.ndarray,
              neighbour_threshold: int = 4,
              neighbours: Dict = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    官方 final_loc（final_eval_all.ipynb Cell 14）。

    psoz       : [N, C]  N 次采样的通道预测值
    true_onset : [C]     0/1 SOZ 标签

    流程：
      1. 每行除以该行最大值（归一化到 [0,1]）
      2. 列均值 → [C] 最终通道概率
      3. argmax 取预测通道
      4. 判断正确性：精确命中 or 邻居放宽（SOZ ≤ threshold 时）

    返回 (ysoz [C], uncertainty [C], correct 0/1)
    """
    m = psoz.max(axis=1, keepdims=True)
    m = np.where(m > 0, m, 1.0)
    psoz_norm = psoz / m

    ysoz = psoz_norm.mean(axis=0)
    max_chn_loc = int(np.argmax(ysoz))
    max_chn_correct = 1 if true_onset[max_chn_loc] == 1 else 0

    # 邻居放宽
    if (max_chn_correct == 0
            and int(true_onset.sum()) <= neighbour_threshold
            and check_neighborhood(max_chn_loc, true_onset, neighbours)):
        max_chn_correct = 1

    uncertainty = psoz_norm.var(axis=0)
    return ysoz, uncertainty, max_chn_correct


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, support=int(y_true.sum()),
                precision=prec, recall=rec, specificity=spec, f1=f1)


def print_confusion_report(
    y_true: np.ndarray, y_prob: np.ndarray,
    label_names: List[str], threshold: float = 0.5,
) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    print(f'\n## Channel-level Confusion Matrix (threshold={threshold:.3f})\n')
    header = (f'| {"Channel":<10} | {"TP":>4} | {"FP":>4} | {"TN":>4} | '
              f'{"FN":>4} | {"Support":>7} | {"Prec":>6} | {"Rec":>6} | '
              f'{"Spec":>6} | {"F1":>6} |')
    sep = '|' + '-' * 12 + '|' + ('---------|' * 8)
    print(header)
    print(sep)

    per_label = {}
    for i, name in enumerate(label_names):
        cm = binary_confusion(y_true[:, i], y_pred[:, i])
        per_label[name] = cm
        print(f'| {name:<10} | {cm["tp"]:>4} | {cm["fp"]:>4} | '
              f'{cm["tn"]:>4} | {cm["fn"]:>4} | {cm["support"]:>7} | '
              f'{cm["precision"]:>6.3f} | {cm["recall"]:>6.3f} | '
              f'{cm["specificity"]:>6.3f} | {cm["f1"]:>6.3f} |')

    # 汇总
    all_cm = binary_confusion(y_true.flatten(), y_pred.flatten())
    per_label['MACRO_AVG'] = {
        'precision': np.mean([v['precision'] for v in per_label.values()]),
        'recall': np.mean([v['recall'] for v in per_label.values()]),
        'f1': np.mean([v['f1'] for v in per_label.values()]),
        'specificity': np.mean([v['specificity'] for v in per_label.values()]),
    }
    per_label['MICRO_AVG'] = all_cm
    print(f'| {"MACRO":<10} |      |      |      |      |         | '
          f'{per_label["MACRO_AVG"]["precision"]:>6.3f} | '
          f'{per_label["MACRO_AVG"]["recall"]:>6.3f} | '
          f'{per_label["MACRO_AVG"]["specificity"]:>6.3f} | '
          f'{per_label["MACRO_AVG"]["f1"]:>6.3f} |')
    print(f'| {"MICRO":<10} | {all_cm["tp"]:>4} | {all_cm["fp"]:>4} | '
          f'{all_cm["tn"]:>4} | {all_cm["fn"]:>4} | {all_cm["support"]:>7} | '
          f'{all_cm["precision"]:>6.3f} | {all_cm["recall"]:>6.3f} | '
          f'{all_cm["specificity"]:>6.3f} | {all_cm["f1"]:>6.3f} |')
    return per_label


# =====================================================================
# 数据适配器: 从 manifest 读取 EDF → 生成 Integration 模型所需的输入
# =====================================================================

class IntegrationEvalDataset(Dataset):
    """
    为 Integration 模型评估定制的数据集。

    从 combined_manifest.csv 读取发作事件，使用 EEGPipeline 生成
    [22, n_patches, patch_len] 的双极 patch 数据，同时生成 19 通道
    单极 SOZ 标签用于与 DeepSOZ 基线对比。

    __getitem__ 返回 dict:
        x              : [22, window_samples]  原始 EEG（拼接 patches）
        onset_sec      : float  发作起始时间
        start_sec      : float  窗口起始时间
        onset_map_19   : [19]   19 通道单极 SOZ 标签
        onset_map_22   : [22]   22 通道双极 SOZ 标签
        pt_id          : str    患者 ID
        fn             : str    文件名
        source         : str    数据源
    """

    # 通道名归一化映射
    CHANNEL_NAME_MAP = {
        'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
        'A1': 'A1', 'A2': 'A2',
    }

    def __init__(
        self,
        manifest_path: str,
        data_roots: List[str],
        patient_ids: Optional[List[str]] = None,
        source: Optional[str] = None,
        pipeline_cfg: Optional[PipelineConfig] = None,
    ):
        self.data_roots = data_roots
        cfg = pipeline_cfg or PipelineConfig()
        self.pipeline = EEGPipeline(cfg)
        self.patch_len = cfg.patch_len
        self.n_patches = cfg.n_patches

        # 读取 manifest
        df = pd.read_csv(manifest_path)
        if source is not None and 'source' in df.columns:
            df = df[df['source'] == source]
        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)]
        df = df.dropna(subset=['sz_start', 'sz_end'])
        df = df[df['sz_end'] > df['sz_start']]
        # 排除 03_tcp_ar_a montage
        edf_paths = df['edf_path'].astype(str)
        df = df[~edf_paths.apply(lambda p: '03_tcp_ar_a' in p)]
        self.df = df.reset_index(drop=True)
        logger.info(
            f'IntegrationEvalDataset: {len(self.df)} samples, '
            f'{len(self.df["patient_id"].unique())} patients, '
            f'source={source}'
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_edf(self, edf_rel: str, source: str) -> Optional[str]:
        """查找 EDF 文件的绝对路径。"""
        for root in self.data_roots:
            p = Path(root) / edf_rel
            if p.exists():
                return str(p)
        # fallback: 只用文件名搜索
        fname = Path(edf_rel).name
        for root in self.data_roots:
            for f in Path(root).rglob(fname):
                return str(f)
        return None

    @staticmethod
    def _parse_19ch_soz(row: pd.Series) -> np.ndarray:
        """从 manifest 行解析 19 通道单极 SOZ 标签。"""
        soz = np.zeros(19, dtype=np.float32)
        ch_19_lower = [c.lower() for c in OFFICIAL_19_CHANNELS]
        for i, chn in enumerate(ch_19_lower):
            val = row.get(chn, '')
            if val != '' and not pd.isna(val):
                try:
                    soz[i] = float(val)
                except (ValueError, TypeError):
                    pass
        if soz.sum() > 0:
            return soz
        # fallback: 从 onset_channels 文本解析
        onset_str = str(row.get('onset_channels', ''))
        ch_name_map = {
            'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
        }
        ch_19_upper = [c.upper() for c in OFFICIAL_19_CHANNELS]
        if onset_str and onset_str != 'nan':
            for token in onset_str.replace(',', ';').split(';'):
                name = token.strip().upper()
                name = ch_name_map.get(name, name)
                if name in ch_19_upper:
                    soz[ch_19_upper.index(name)] = 1.0
        return soz

    @staticmethod
    def _parse_22ch_soz(row: pd.Series) -> np.ndarray:
        """从 manifest 行解析 22 通道双极 SOZ 标签。"""
        soz = np.zeros(22, dtype=np.float32)
        for i, col in enumerate(TCP_BIPOLAR_COLUMNS):
            val = row.get(col, 0)
            if val != '' and not pd.isna(val):
                try:
                    soz[i] = float(val)
                except (ValueError, TypeError):
                    pass
        return soz

    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df.iloc[idx]
        source = str(row.get('source', 'tusz'))
        edf_rel = str(row.get('edf_path', ''))
        edf_path = self._resolve_edf(edf_rel, source)
        if edf_path is None:
            return None

        sz_start = float(row['sz_start'])
        sz_end = float(row['sz_end'])
        pt_id = str(row.get('patient_id', ''))

        # SOZ 标签
        onset_map_19 = self._parse_19ch_soz(row)
        onset_map_22 = self._parse_22ch_soz(row)

        try:
            event = SeizureEvent(
                edf_path=edf_path,
                onset=sz_start,
                end=sz_end,
                soz_channels=[],
                source=source,
                patient_id=pt_id,
            )
            result = self.pipeline.process_event(event)
            if result is None:
                return None

            X = result['X']  # [22, n_patches, patch_len]
            # 拼接 patches → [22, n_patches * patch_len]
            n_ch, n_p, p_len = X.shape
            x_flat = X.reshape(n_ch, n_p * p_len)  # [22, 2000]

            # 计算 onset_sec 和 window_start_sec
            cfg = self.pipeline.cfg
            onset_sec = sz_start
            window_start_sec = sz_start - cfg.pre_onset_sec

            return {
                'x': torch.from_numpy(x_flat).float(),
                'onset_sec': torch.tensor(onset_sec, dtype=torch.float32),
                'start_sec': torch.tensor(window_start_sec, dtype=torch.float32),
                'onset_map_19': torch.from_numpy(onset_map_19).float(),
                'onset_map_22': torch.from_numpy(onset_map_22).float(),
                'pt_id': pt_id,
                'fn': edf_rel,
                'source': source,
            }
        except Exception as e:
            logger.warning(f'[{idx}] 加载失败 {edf_path}: {e}')
            return None

    def get_patient_ids(self) -> List[str]:
        return sorted(self.df['patient_id'].unique().tolist())


def collate_skip_none(batch):
    """过滤掉 None 样本后 collate。"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'x': torch.stack([b['x'] for b in batch]),
        'onset_sec': torch.stack([b['onset_sec'] for b in batch]),
        'start_sec': torch.stack([b['start_sec'] for b in batch]),
        'onset_map_19': torch.stack([b['onset_map_19'] for b in batch]),
        'onset_map_22': torch.stack([b['onset_map_22'] for b in batch]),
        'pt_id': [b['pt_id'] for b in batch],
        'fn': [b['fn'] for b in batch],
        'source': [b['source'] for b in batch],
    }


# =====================================================================
# MC Dropout 推理（适配 Integration 模型）
# =====================================================================

def mc_inference_integration(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    x: torch.Tensor,
    onset_sec: torch.Tensor,
    start_sec: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
    use_amp: bool = True,
) -> np.ndarray:
    """
    对 Integration 模型做 MC dropout 采样。

    保持 model.train() 使 dropout 生效，前向 n_samples 次。

    x         : [B, 22, T]
    onset_sec : [B]
    start_sec : [B]

    返回 : [n_samples * B, 19]  每次采样的 SOZ 概率
    """
    model.train()
    results = []
    for _ in range(n_samples):
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                outputs = model(
                    x.to(device),
                    seizure_onset_sec=onset_sec.to(device),
                    window_start_sec=start_sec.to(device),
                )
        prob = outputs['soz_probs'].cpu().numpy()  # [B, 19]
        results.append(prob)
    model.eval()
    return np.concatenate(results, axis=0)  # [n_samples * B, 19]


# =====================================================================
# Channel-level 推理（不带 MC，用于混淆矩阵）
# =====================================================================

@torch.no_grad()
def run_channel_inference(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    标准推理：对每个样本获取 SOZ 概率 [B, 19]。
    返回 (all_probs [N, 19], all_labels [N, 19], all_meta)
    """
    model.eval()
    all_probs, all_labels, all_meta = [], [], []

    for batch in loader:
        if batch is None:
            continue
        x = batch['x'].to(device)
        onset_sec = batch['onset_sec'].to(device)
        start_sec = batch['start_sec'].to(device)

        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            outputs = model(
                x,
                seizure_onset_sec=onset_sec,
                window_start_sec=start_sec,
            )
        probs = outputs['soz_probs'].cpu().numpy()  # [B, 19]
        all_probs.append(probs)
        all_labels.append(batch['onset_map_19'].numpy())
        for i in range(x.size(0)):
            all_meta.append({
                'pt_id': batch['pt_id'][i],
                'fn': batch['fn'][i],
            })

    if not all_probs:
        return np.zeros((0, 19)), np.zeros((0, 19)), []
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_meta,
    )


# =====================================================================
# 官方三级评估 (Patient-level + Seizure-level + Channel-level)
# =====================================================================

def official_evaluation(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    dataset: IntegrationEvalDataset,
    patient_ids: List[str],
    device: torch.device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
    use_amp: bool = True,
) -> Dict:
    """
    官方两级评估（Patient-level + Seizure-level），含 MC 不确定性。
    """
    label_names = OFFICIAL_19_CHANNELS

    corr_pt = 0
    total_pt = 0
    pt_uncs = []
    pt_details = []

    all_corr_sz = []
    all_sz_unc = []
    sz_details = []

    for pt_id in patient_ids:
        # 为该患者筛选样本
        pt_mask = dataset.df['patient_id'] == pt_id
        pt_indices = dataset.df.index[pt_mask].tolist()
        if not pt_indices:
            continue

        pt_psoz_all = []
        true_onset = None

        for idx in pt_indices:
            sample = dataset[idx]
            if sample is None:
                continue

            true_onset = sample['onset_map_19'].numpy()  # [19]

            # MC dropout 采样
            mc_maps = mc_inference_integration(
                model,
                sample['x'].unsqueeze(0),         # [1, 22, T]
                sample['onset_sec'].unsqueeze(0),  # [1]
                sample['start_sec'].unsqueeze(0),  # [1]
                device,
                n_samples=mc_samples,
                use_amp=use_amp,
            )  # [mc_samples, 19]

            # --- Seizure-level 评估 ---
            ysoz, unc, correct = final_loc(
                mc_maps, true_onset,
                neighbour_threshold=neighbour_threshold,
            )
            all_corr_sz.append(correct)
            all_sz_unc.append(unc)
            sz_details.append({
                'pt_id': pt_id,
                'correct': correct,
                'max_chn': int(np.argmax(ysoz)),
                'max_chn_name': label_names[int(np.argmax(ysoz))],
                'unc_max': float(unc.max()),
            })

            pt_psoz_all.append(mc_maps)

        if true_onset is None or not pt_psoz_all:
            continue

        # --- Patient-level 评估 ---
        total_pt += 1
        pt_psoz = np.concatenate(pt_psoz_all, axis=0)  # [N_total, 19]
        ysoz_pt, unc_pt, correct_pt = final_loc(
            pt_psoz, true_onset,
            neighbour_threshold=neighbour_threshold,
        )
        corr_pt += correct_pt
        pt_uncs.append(unc_pt)
        pt_details.append({
            'pt_id': pt_id,
            'correct': correct_pt,
            'n_seizures': len(pt_psoz_all),
            'max_chn': int(np.argmax(ysoz_pt)),
            'max_chn_name': label_names[int(np.argmax(ysoz_pt))],
            'true_soz': [label_names[i] for i in range(len(true_onset))
                         if true_onset[i] == 1],
            'unc_max': float(unc_pt.max()),
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


# =====================================================================
# 综合评估入口
# =====================================================================

def full_evaluation(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    dataset: IntegrationEvalDataset,
    patient_ids: List[str],
    args,
) -> Dict:
    """
    对 Integration 模型执行完整三级评估（与 DeepSOZ 基线可直接对比）：
      1. Channel-level 混淆矩阵
      2. Seizure-level 定位正确率 + MC 不确定性
      3. Patient-level 定位正确率 + MC 不确定性
    """
    label_names = OFFICIAL_19_CHANNELS
    device = (
        torch.device('cuda')
        if args.device == 'cuda' and torch.cuda.is_available()
        else torch.device('cpu')
    )
    use_amp = device.type == 'cuda'
    model.to(device)

    # ── 1) Channel-level 混淆矩阵 ──────────────────────────────────
    logger.info('Channel-level 推理...')
    # 用筛选后的 patient_ids 构建子集
    pt_mask = dataset.df['patient_id'].isin(patient_ids)
    eval_indices = dataset.df.index[pt_mask].tolist()

    class SubsetDataset(Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.parent[self.indices[idx]]

    subset = SubsetDataset(dataset, eval_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )

    all_probs, all_labels, all_meta = run_channel_inference(
        model, loader, device, use_amp=use_amp,
    )

    if len(all_probs) > 0:
        channel_report = print_confusion_report(
            all_labels, all_probs, label_names, threshold=args.threshold,
        )
    else:
        logger.warning('无有效评估样本')
        channel_report = {}

    # ── 2) + 3) Seizure-level / Patient-level（MC dropout）──────────
    logger.info(f'\n开始 MC dropout 评估 (samples={args.mc_samples})...')
    official_results = official_evaluation(
        model=model,
        dataset=dataset,
        patient_ids=patient_ids,
        device=device,
        mc_samples=args.mc_samples,
        neighbour_threshold=args.neighbour_threshold,
        use_amp=use_amp,
    )

    pt = official_results['patient_level']
    sz = official_results['seizure_level']

    print(f'\n## Patient-level SOZ Localization')
    print(f'  Correct: {pt["corr_pt"]} / {pt["total_pt"]}  '
          f'Accuracy: {pt["acc_pt"]:.3f}')
    print(f'  Uncertainty (mean max-var): {pt["ptunc_mean"]:.4f}')

    print(f'\n## Seizure-level SOZ Localization')
    print(f'  Total seizures: {sz["total_sz"]}')
    print(f'  Correct rate (corr_sz): {sz["corr_sz_mean"]:.3f}')
    print(f'  Uncertainty (mean max-var): {sz["szunc_mean"]:.4f}')

    # 打印每个患者详情
    print(f'\n## Per-patient Details\n')
    print(f'| {"Patient":<20} | {"#Sz":>4} | {"Correct":>7} | '
          f'{"Pred":>10} | {"True SOZ":<30} | {"Unc":>6} |')
    print(f'|{"-"*22}|{"-"*6}|{"-"*9}|{"-"*12}|{"-"*32}|{"-"*8}|')
    for d in pt['per_patient']:
        correct_mark = 'Y' if d['correct'] else 'N'
        true_soz_str = ','.join(d['true_soz'])[:30]
        print(f'| {d["pt_id"]:<20} | {d["n_seizures"]:>4} | '
              f'{correct_mark:>7} | {d["max_chn_name"]:>10} | '
              f'{true_soz_str:<30} | {d["unc_max"]:>6.4f} |')

    # ── 保存结果 ────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    result = {
        'model': 'TimeFilter_LaBraM_BrainNetwork_Integration',
        'checkpoint': args.checkpoint,
        'eval_method': 'DeepSOZ_official (final_eval_all + szloc_all)',
        'channel_level': channel_report,
        'patient_level': {k: v for k, v in pt.items() if k != 'per_patient'},
        'patient_details': pt['per_patient'],
        'seizure_level': {k: v for k, v in sz.items() if k != 'per_seizure'},
        'seizure_details': sz['per_seizure'],
        'config': {
            'mc_samples': args.mc_samples,
            'neighbour_threshold': args.neighbour_threshold,
            'threshold': args.threshold,
            'source': args.source,
        },
    }
    result_path = out_dir / f'{stem}_deepsoz_eval.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'评估结果保存: {result_path}')

    return result


# =====================================================================
# 参数
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Integration Model — DeepSOZ 官方评估对比',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Integration 模型 .pt checkpoint 路径')
    p.add_argument('--manifest', required=True,
                   help='combined_manifest.csv 路径')
    p.add_argument('--data-roots', nargs='+', required=True,
                   help='EDF 数据根目录列表')
    p.add_argument('--source', default=None,
                   choices=['tusz', 'private'],
                   help='数据源过滤')
    p.add_argument('--patient-ids', nargs='*', default=None,
                   help='指定评估的患者 ID 列表 (默认: 全部)')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Channel-level 二值化阈值')
    p.add_argument('--mc-samples', type=int, default=20,
                   help='MC dropout 采样次数（官方 20）')
    p.add_argument('--neighbour-threshold', type=int, default=4,
                   help='邻居放宽的 SOZ 通道数上限（官方 2~4）')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', default=None,
                   help='结果保存目录 (默认: checkpoint 所在目录)')

    # EEG Pipeline 配置
    p.add_argument('--pre-onset-sec', type=float, default=5.0,
                   help='发作前秒数')
    p.add_argument('--post-onset-sec', type=float, default=5.0,
                   help='发作后秒数')
    p.add_argument('--target-fs', type=float, default=200.0,
                   help='目标采样率')
    p.add_argument('--filter-low', type=float, default=3.0,
                   help='带通滤波下界 (Hz)')
    p.add_argument('--filter-high', type=float, default=45.0,
                   help='带通滤波上界 (Hz)')
    p.add_argument('--patch-len', type=int, default=200,
                   help='patch 长度 (采样点)')
    p.add_argument('--n-patches', type=int, default=10,
                   help='patch 数量')

    # K-Fold 评估
    p.add_argument('--n-folds', type=int, default=0,
                   help='K 折评估 (0=不折叠, 使用全部患者)')
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# =====================================================================
# 主函数
# =====================================================================

def main():
    args = parse_args()

    # ── 加载模型 ────────────────────────────────────────────────
    device = (
        torch.device('cuda')
        if args.device == 'cuda' and torch.cuda.is_available()
        else torch.device('cpu')
    )
    logger.info(f'加载 Integration 模型: {args.checkpoint}')
    model, ckpt = TimeFilter_LaBraM_BrainNetwork_Integration.load_checkpoint(
        args.checkpoint, map_location=device,
    )
    model.to(device)
    model.eval()
    logger.info(f'模型配置: {model.cfg}')

    # ── 构建 Pipeline 配置 ──────────────────────────────────────
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.pre_onset_sec = args.pre_onset_sec
    pipeline_cfg.post_onset_sec = args.post_onset_sec
    pipeline_cfg.target_fs = args.target_fs
    pipeline_cfg.filter_low = args.filter_low
    pipeline_cfg.filter_high = args.filter_high
    pipeline_cfg.patch_len = args.patch_len
    pipeline_cfg.n_patches = args.n_patches
    # 使用命令行提供的数据根目录
    if args.data_roots:
        pipeline_cfg.tusz_data_root = args.data_roots[0]
        if len(args.data_roots) > 1:
            pipeline_cfg.private_data_roots = args.data_roots[1:]

    # ── 构建数据集 ──────────────────────────────────────────────
    dataset = IntegrationEvalDataset(
        manifest_path=args.manifest,
        data_roots=args.data_roots,
        source=args.source,
        pipeline_cfg=pipeline_cfg,
    )

    # ── 获取患者列表 ────────────────────────────────────────────
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = dataset.get_patient_ids()
    logger.info(f'评估患者数: {len(patient_ids)}  source={args.source}')

    # ── 执行评估 ────────────────────────────────────────────────
    if args.n_folds > 1:
        # K-Fold 交叉验证评估
        from DeepSOZ_new.dataset import make_kfold_splits
        splits = make_kfold_splits(patient_ids, n_folds=args.n_folds, seed=args.seed)

        fold_results = []
        for fold, (train_ids, val_ids) in enumerate(splits):
            logger.info(f'\n{"="*60}\nFold {fold}: {len(val_ids)} patients\n{"="*60}')
            result = full_evaluation(model, dataset, val_ids, args)
            if result:
                fold_results.append(result)

        if fold_results:
            pt_accs = [r['patient_level']['acc_pt'] for r in fold_results]
            sz_accs = [r['seizure_level']['corr_sz_mean'] for r in fold_results]
            print(f'\n{"="*60}')
            print(f'K-Fold Summary ({len(fold_results)} folds)')
            print(f'{"="*60}')
            print(f'Patient-level accuracy: '
                  f'{np.mean(pt_accs):.3f} +/- {np.std(pt_accs):.3f}')
            print(f'Seizure-level corr_sz: '
                  f'{np.mean(sz_accs):.3f} +/- {np.std(sz_accs):.3f}')

            out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
            summary_path = out_dir / 'kfold_deepsoz_eval_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': 'TimeFilter_LaBraM_BrainNetwork_Integration',
                    'n_folds': len(fold_results),
                    'pt_acc_mean': float(np.mean(pt_accs)),
                    'pt_acc_std': float(np.std(pt_accs)),
                    'sz_corr_mean': float(np.mean(sz_accs)),
                    'sz_corr_std': float(np.std(sz_accs)),
                }, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f'K-Fold 汇总保存: {summary_path}')
    else:
        # 全量评估
        full_evaluation(model, dataset, patient_ids, args)


if __name__ == '__main__':
    main()
