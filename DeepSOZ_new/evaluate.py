#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ_new 评估脚本

忠实复现官方 DeepSOZ 评估逻辑（final_eval_all.ipynb + szloc_all.ipynb），包括：

1. Channel-level 逐通道混淆矩阵（TP/FP/TN/FN/F1 等）
2. Seizure-level 定位正确率（corr_sz）
   - 20 次 MC dropout 采样 → 归一化 → 均值 → argmax → 判断是否命中 SOZ
3. Patient-level 定位正确率（corr_pt）
   - 聚合同一患者所有发作的 onset_map → 均值 → argmax
4. 邻居放宽判断（chn_neighbours）
   - 当 SOZ 通道数 ≤ 4 时，预测通道落在真实 SOZ 的空间邻居内也算正确
5. MC 不确定性（ptunc / szunc）
   - 通道预测值的方差

支持：
  --source tusz/private  数据源过滤
  --use-bipolar          TCP 22 双极导联
  --mc-samples N         MC 采样次数（默认 20）

用法：
  python evaluate.py \\
      --checkpoint runs/fold0/deepsoz_fold0_s2_best.pth \\
      --manifest combined_manifest.csv \\
      --data-roots /data/edf \\
      --source private \\
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

sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    OFFICIAL_19_CHANNELS, TCP_BIPOLAR_NAMES, TCP_BIPOLAR_COLUMNS,
    OnlineStage2Dataset, Stage2Dataset,
    make_dataloader, make_kfold_splits, get_patient_ids, read_manifest_csv,
)
from deepsoz_model import build_stage2_model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

LABEL_NAMES_19 = [c.upper() for c in OFFICIAL_19_CHANNELS]
LABEL_NAMES_BIPOLAR = TCP_BIPOLAR_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 官方 chn_neighbours（19 单极通道空间邻接关系，10-20 系统拓扑）
# 来源: deeksha-ms/DeepSOZ code/test/final_eval_all.ipynb Cell 14
# 索引: 0=FP1 1=FP2 2=F7 3=F3 4=FZ 5=F4 6=F8
#       7=T3  8=C3  9=CZ 10=C4 11=T4
#       12=T5 13=P3 14=PZ 15=P4 16=T6
#       17=O1 18=O2
# ─────────────────────────────────────────────────────────────────────────────

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


def check_neighborhood(max_chn: int, onset_map: np.ndarray,
                        neighbours: Dict = None) -> bool:
    """
    官方 check_neighborhood：
    如果 max_chn 落在任一真实 SOZ 通道的空间邻居内，返回 True。
    """
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
    官方 final_loc（来自 final_eval_all.ipynb Cell 14）:

    psoz         : [N, C]  N 次采样/发作的通道预测值
    true_onset   : [C]     0/1 SOZ 标签

    流程：
      1. 每行除以该行最大值（归一化到 [0,1]）
      2. 列均值 → [C] 作为最终通道概率
      3. argmax 取预测通道
      4. 判断正确性：精确命中 or 邻居放宽（当 SOZ 通道数 ≤ threshold）

    返回 (ysoz [C], uncertainty [C], correct 0/1)
    """
    n = psoz.shape[0]
    # 归一化：每行除以行最大值
    m = psoz.max(axis=1, keepdims=True)
    m = np.where(m > 0, m, 1.0)  # 避免除零
    psoz_norm = psoz / m
    # 列均值
    ysoz = psoz_norm.mean(axis=0)   # [C]

    max_chn_loc = int(np.argmax(ysoz))
    max_chn_correct = 1 if true_onset[max_chn_loc] == 1 else 0

    # 邻居放宽
    if (max_chn_correct == 0
            and int(true_onset.sum()) <= neighbour_threshold
            and check_neighborhood(max_chn_loc, true_onset, neighbours)):
        max_chn_correct = 1

    # 不确定性：各通道方差
    uncertainty = psoz_norm.var(axis=0)  # [C]

    return ysoz, uncertainty, max_chn_correct


# ─────────────────────────────────────────────────────────────────────────────
# Channel-level 混淆矩阵工具
# ─────────────────────────────────────────────────────────────────────────────

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
    return per_label


# ─────────────────────────────────────────────────────────────────────────────
# MC dropout 采样推理
# ─────────────────────────────────────────────────────────────────────────────

def mc_inference_single(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
    use_amp: bool = True,
) -> np.ndarray:
    """
    对单个样本做 MC dropout 采样。

    官方做法：保持 model.train()（dropout 生效），前向 n_samples 次。

    data : [1, Nsz, T, C, L]
    返回 : [n_samples, C]  每次采样的通道 onset_map（取 chn_onset_map）
    """
    model.train()  # 保持 dropout 开启
    results = []
    for _ in range(n_samples):
        with torch.no_grad():
            with autocast(enabled=use_amp and device.type == 'cuda'):
                _, _, attn_map, chn_map = model(data.to(device))
        # 官方用 psoz（即 chn_onset_map）作为定位依据
        # chn_map: [B, C]
        prob = F.sigmoid(chn_map).cpu().numpy()  # [B, C]
        results.append(prob)
    model.eval()
    return np.concatenate(results, axis=0)   # [n_samples * B, C]


# ─────────────────────────────────────────────────────────────────────────────
# 官方评估：Seizure-level + Patient-level
# ─────────────────────────────────────────────────────────────────────────────

def official_evaluation(
    model: torch.nn.Module,
    manifest_path: str,
    data_roots: List[str],
    patient_ids: List[str],
    device: torch.device,
    n_channels: int = 19,
    use_bipolar: bool = False,
    source: Optional[str] = None,
    n_windows: int = 45,
    target_fs: float = 200.0,
    f_low: float = 1.6,
    f_high: float = 30.0,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
    use_amp: bool = True,
) -> Dict:
    """
    官方两级评估（Patient-level + Seizure-level），含 MC 不确定性。

    返回:
      {
        'patient_level': {'corr_pt': int, 'total_pt': int, 'acc_pt': float,
                          'ptunc_mean': float,
                          'per_patient': [...] },
        'seizure_level': {'corr_sz_mean': float, 'szunc_mean': float,
                          'per_patient': [...] },
      }
    """
    label_names = LABEL_NAMES_BIPOLAR if use_bipolar else LABEL_NAMES_19

    # Patient-level 评估
    corr_pt = 0
    total_pt = 0
    pt_uncs = []
    pt_details = []

    # Seizure-level 评估
    all_corr_sz = []
    all_sz_unc = []
    sz_details = []

    for pt_id in patient_ids:
        # 为每个患者单独构建 DataLoader
        ds = OnlineStage2Dataset(
            manifest_path=manifest_path,
            data_roots=data_roots,
            patient_ids=[pt_id],
            source=source,
            use_bipolar=use_bipolar,
            n_windows=n_windows,
            win_len=45,
            target_fs=target_fs,
            f_low=f_low,
            f_high=f_high,
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        # 收集该患者所有发作的 onset_map（MC 采样）
        pt_psoz_all = []   # 用于 patient-level
        true_onset = None

        for data in loader:
            true_onset = data['onset_map'].numpy().reshape(-1)  # [C]

            # MC dropout 采样
            mc_maps = mc_inference_single(
                model, data['buffers'], device,
                n_samples=mc_samples, use_amp=use_amp
            )   # [mc_samples, C]

            # --- Seizure-level 评估 ---
            for szn in range(data['buffers'].shape[1]):
                # 每次发作对应 mc_samples 行
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

            # 存入 patient-level 聚合池
            pt_psoz_all.append(mc_maps)   # [mc_samples, C]

        if true_onset is None:
            continue

        # --- Patient-level 评估 ---
        total_pt += 1
        # 聚合所有发作的所有 MC 采样 → [N_total, C]
        pt_psoz = np.concatenate(pt_psoz_all, axis=0)
        ysoz_pt, unc_pt, correct_pt = final_loc(
            pt_psoz, true_onset,
            neighbour_threshold=neighbour_threshold,
        )
        corr_pt += correct_pt
        pt_uncs.append(unc_pt)
        pt_details.append({
            'pt_id':         pt_id,
            'correct':       correct_pt,
            'n_seizures':    len(pt_psoz_all),
            'max_chn':       int(np.argmax(ysoz_pt)),
            'max_chn_name':  label_names[int(np.argmax(ysoz_pt))],
            'true_soz':      [label_names[i] for i in range(len(true_onset))
                              if true_onset[i] == 1],
            'unc_max':       float(unc_pt.max()),
        })

    # 汇总
    acc_pt = corr_pt / total_pt if total_pt > 0 else 0.0
    corr_sz_mean = float(np.mean(all_corr_sz)) if all_corr_sz else 0.0
    ptunc_mean = float(np.mean([u.max() for u in pt_uncs])) if pt_uncs else 0.0
    szunc_mean = float(np.mean([u.max() for u in all_sz_unc])) if all_sz_unc else 0.0

    return {
        'patient_level': {
            'corr_pt':    corr_pt,
            'total_pt':   total_pt,
            'acc_pt':     acc_pt,
            'ptunc_mean': ptunc_mean,
            'per_patient': pt_details,
        },
        'seizure_level': {
            'corr_sz_mean': corr_sz_mean,
            'total_sz':     len(all_corr_sz),
            'szunc_mean':   szunc_mean,
            'per_seizure':  sz_details,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Channel-level 标准推理（不带 MC，用于混淆矩阵）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    model.eval()
    all_probs, all_labels, all_meta = [], [], []
    for batch in loader:
        data = batch['buffers'].to(device)
        with autocast(enabled=use_amp and device.type == 'cuda'):
            _, _, attn_map, _ = model(data)
        probs = F.sigmoid(attn_map).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch['onset_map'].numpy())
        for i in range(data.shape[0]):
            all_meta.append({
                'pt_id': batch.get('pt_id', batch['fn'])[i],
                'fn':    batch['fn'][i],
            })
    return (np.concatenate(all_probs, axis=0),
            np.concatenate(all_labels, axis=0),
            all_meta)


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='DeepSOZ_new 官方评估脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint',     default=None,
                   help='单个 .pth checkpoint 路径')
    p.add_argument('--checkpoint-dir', default=None,
                   help='checkpoint 目录（批量评估所有折 best）')
    p.add_argument('--manifest',       required=True)
    p.add_argument('--data-roots',     nargs='+', required=True)
    p.add_argument('--data-mode',      default='online',
                   choices=['online', 'offline'])
    p.add_argument('--source',         default=None,
                   choices=['tusz', 'private'])
    p.add_argument('--use-bipolar',    action='store_true')
    p.add_argument('--n-windows',      type=int,   default=45)
    p.add_argument('--target-fs',      type=float, default=200.0)
    p.add_argument('--f-low',          type=float, default=1.6)
    p.add_argument('--f-high',         type=float, default=30.0)
    p.add_argument('--threshold',      type=float, default=0.5)
    p.add_argument('--mc-samples',     type=int,   default=20,
                   help='MC dropout 采样次数（官方 20）')
    p.add_argument('--neighbour-threshold', type=int, default=4,
                   help='邻居放宽的 SOZ 通道数上限（官方 2~4）')
    p.add_argument('--batch-size',     type=int,   default=1)
    p.add_argument('--num-workers',    type=int,   default=0)
    p.add_argument('--device',         default='cuda')
    p.add_argument('--tf-dropout',     type=float, default=0.15,
                   help='Transformer dropout（MC 时需 >0）')
    p.add_argument('--cnn-dropout',    type=float, default=0.15)
    p.add_argument('--gru-dropout',    type=float, default=0.0)
    p.add_argument('--n-folds',        type=int,   default=5)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--exp-prefix',     default='deepsoz')
    p.add_argument('--output-dir',     default=None)
    p.add_argument('--save-preds',     action='store_true')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 综合评估
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(ckpt_path: str, val_ids: List[str], args) -> Dict:
    """
    对一个 checkpoint 执行完整三级评估：
      1. Channel-level 混淆矩阵（阈值二值化）
      2. Seizure-level 定位正确率 + MC 不确定性
      3. Patient-level 定位正确率 + MC 不确定性
    """
    n_channels  = 22 if args.use_bipolar else 19
    label_names = LABEL_NAMES_BIPOLAR if args.use_bipolar else LABEL_NAMES_19

    device = (torch.device('cuda')
              if args.device == 'cuda' and torch.cuda.is_available()
              else torch.device('cpu'))

    model = build_stage2_model(
        n_channels=n_channels,
        cnn_dropout=args.cnn_dropout,
        gru_dropout=args.gru_dropout,
        transformer_dropout=args.tf_dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    logger.info(f'加载 checkpoint: {ckpt_path}')

    # ── 1) Channel-level 混淆矩阵 ──────────────────────────────────────
    loader = build_val_loader(args, val_ids)
    if len(loader.dataset) == 0:
        logger.warning('验证集为空')
        return {}

    logger.info(f'评估样本数: {len(loader.dataset)}')
    use_amp = device.type == 'cuda'
    all_probs, all_labels, all_meta = run_inference(
        model, loader, device, use_amp=use_amp
    )
    channel_report = print_confusion_report(
        all_labels, all_probs, label_names, threshold=args.threshold
    )

    # ── 2) + 3) Seizure-level / Patient-level（MC dropout）────────────
    logger.info(f'\n开始 MC dropout 评估 (samples={args.mc_samples})...')
    official_results = official_evaluation(
        model=model,
        manifest_path=args.manifest,
        data_roots=args.data_roots,
        patient_ids=val_ids,
        device=device,
        n_channels=n_channels,
        use_bipolar=args.use_bipolar,
        source=args.source,
        n_windows=args.n_windows,
        target_fs=args.target_fs,
        f_low=args.f_low,
        f_high=args.f_high,
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

    # ── 保存结果 ─────────────────────────────────────────────────────
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(ckpt_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).stem

    result = {
        'channel_level': channel_report,
        'patient_level': {k: v for k, v in pt.items() if k != 'per_patient'},
        'patient_details': pt['per_patient'],
        'seizure_level': {k: v for k, v in sz.items() if k != 'per_seizure'},
    }
    result_path = out_dir / f'{stem}_eval_full.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'完整评估结果保存: {result_path}')

    if args.save_preds:
        rows = []
        for i, meta in enumerate(all_meta):
            row = dict(meta)
            for j, name in enumerate(label_names):
                row[f'prob_{name}']  = float(all_probs[i, j])
                row[f'label_{name}'] = int(all_labels[i, j])
            rows.append(row)
        pred_path = out_dir / f'{stem}_preds.csv'
        pd.DataFrame(rows).to_csv(pred_path, index=False, encoding='utf-8')
        logger.info(f'预测结果保存: {pred_path}')

    return result


def build_val_loader(args, val_ids: Optional[List[str]]):
    kw = dict(
        n_windows=args.n_windows,
        target_fs=args.target_fs,
        f_low=args.f_low,
        f_high=args.f_high,
    )
    if args.data_mode == 'offline':
        manifest = read_manifest_csv(args.manifest)
        ds = Stage2Dataset(
            data_root=args.data_roots[0],
            manifest=manifest,
            patient_ids=val_ids,
            win_len=45,
        )
    else:
        ds = OnlineStage2Dataset(
            manifest_path=args.manifest,
            data_roots=args.data_roots,
            patient_ids=val_ids,
            source=args.source,
            use_bipolar=args.use_bipolar,
            win_len=45,
            **kw,
        )
    return make_dataloader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.checkpoint is not None:
        # 单 checkpoint：用全部指定 source 的患者
        val_ids = get_patient_ids(args.manifest, source=args.source)
        logger.info(f'评估患者数: {len(val_ids)}  source={args.source}')
        full_evaluation(args.checkpoint, val_ids, args)

    elif args.checkpoint_dir is not None:
        # K 折批量评估
        patient_ids = get_patient_ids(args.manifest, source=args.source)
        splits = make_kfold_splits(patient_ids, n_folds=args.n_folds,
                                   seed=args.seed)
        ckpt_dir = Path(args.checkpoint_dir)

        fold_results = []
        for fold, (train_ids, val_ids) in enumerate(splits):
            patterns = [
                f'fold{fold}/{args.exp_prefix}_fold{fold}_s2_best.pth',
                f'fold{fold}/*s2*best*.pth',
            ]
            ckpt_path = None
            for pat in patterns:
                matches = list(ckpt_dir.glob(pat))
                if matches:
                    ckpt_path = str(matches[0])
                    break
            if ckpt_path is None:
                logger.warning(f'Fold {fold}: 找不到 checkpoint，跳过')
                continue

            logger.info(f'\n{"="*60}\nFold {fold}: {ckpt_path}\n{"="*60}')
            result = full_evaluation(ckpt_path, val_ids, args)
            fold_results.append(result)

        # K 折汇总
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

            summary_path = (Path(args.output_dir) if args.output_dir
                            else ckpt_dir) / 'kfold_eval_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'n_folds':     len(fold_results),
                    'pt_acc_mean': float(np.mean(pt_accs)),
                    'pt_acc_std':  float(np.std(pt_accs)),
                    'sz_corr_mean': float(np.mean(sz_accs)),
                    'sz_corr_std':  float(np.std(sz_accs)),
                    'per_fold': [
                        {'patient_level': r['patient_level'],
                         'seizure_level': r['seizure_level']}
                        for r in fold_results
                    ],
                }, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f'K-Fold 汇总保存: {summary_path}')
    else:
        logger.error('请指定 --checkpoint 或 --checkpoint-dir')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
