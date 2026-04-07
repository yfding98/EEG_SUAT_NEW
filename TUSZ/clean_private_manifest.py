#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 private_manifest.csv 进行后��理:

1. sz_end 缺失: 通过 --keep-no-end 决定保留(默认60s时长)还是丢弃
2. onset_channels 为空: 直接丢弃
3. SPH-L → T3;F7, SPH-R → T4;F8 替换, 去重后更新 soz_bipolar 和 22ch 列
4. 输出清洗后的 CSV

用法:
  python clean_private_manifest.py [--input INPUT] [--output OUTPUT]
      [--keep-no-end] [--default-sz-duration 60]
"""

import argparse
from typing import Dict, List, Tuple

import pandas as pd

# ─── TCP bipolar 22ch 定义 ───────────────────────────────────────────────────

TCP_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
    ('C4', 'T4'),  ('T4', 'A2'),
]

BIPOLAR_COL_NAMES = [f'{a}_{b}' for a, b in TCP_PAIRS]

# SPH 替换映射
SPH_REPLACEMENT = {
    'SPH-L': ['T3', 'F7'],
    'SPH-R': ['T4', 'F8'],
}


def channels_to_bipolar(channels: List[str]) -> Tuple[str, Dict[str, int]]:
    """单极通道列表 → (soz_bipolar 字符串, 22ch 0/1 dict)。"""
    ch_set = set(channels)
    bipolar_labels = {}
    active_pairs = []
    for a, b in TCP_PAIRS:
        col = f'{a}_{b}'
        if a in ch_set and b in ch_set:
            bipolar_labels[col] = 1
            active_pairs.append(f'{a}-{b}')
        else:
            bipolar_labels[col] = 0
    soz_bipolar = ','.join(active_pairs)
    return soz_bipolar, bipolar_labels


def replace_sph(onset_str: str) -> str:
    """
    替换 SPH-L → T3;F7, SPH-R → T4;F8, 合并去重, 排序返回。
    """
    channels = [c.strip() for c in onset_str.split(';') if c.strip()]
    expanded = []
    for ch in channels:
        if ch in SPH_REPLACEMENT:
            expanded.extend(SPH_REPLACEMENT[ch])
        else:
            expanded.append(ch)
    # 去重并排序
    return ';'.join(sorted(set(expanded)))


def main():
    parser = argparse.ArgumentParser(description='清洗 private_manifest.csv')
    parser.add_argument('--input', '-i', default='TUSZ/private_manifest.csv')
    parser.add_argument('--output', '-o', default='TUSZ/private_manifest_clean.csv')
    parser.add_argument('--keep-no-end', action='store_true',
                        help='保留无 sz_end 的行 (用默认发作时长填充)')
    parser.add_argument('--default-sz-duration', type=float, default=60.0,
                        help='无 sz_end 时使用的默认发作时长 (秒)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding='utf-8-sig')
    n_orig = len(df)
    print(f'读入 {n_orig} 行')

    # ── 步骤 1: 处理 sz_end 缺失 ──
    no_end_mask = df['sz_end'].isna() | (df['sz_end'].astype(str).str.strip() == '')
    n_no_end = no_end_mask.sum()

    if args.keep_no_end:
        # 填充默认时长
        df.loc[no_end_mask, 'sz_end'] = (
            df.loc[no_end_mask, 'sz_start'] + args.default_sz_duration
        )
        df.loc[no_end_mask, 'sz_duration'] = args.default_sz_duration
        print(f'步骤1: 保留无 sz_end 的 {n_no_end} 行, '
              f'填充默认时长 {args.default_sz_duration}s')
    else:
        df = df[~no_end_mask].copy()
        print(f'步骤1: 丢弃无 sz_end 的 {n_no_end} 行, 剩余 {len(df)} 行')

    # ── 步骤 2: 丢弃 onset_channels 为空的行 ──
    empty_ch_mask = df['onset_channels'].isna() | (df['onset_channels'].astype(str).str.strip() == '')
    n_empty_ch = empty_ch_mask.sum()
    df = df[~empty_ch_mask].copy()
    print(f'步骤2: 丢弃 onset_channels 为空的 {n_empty_ch} 行, 剩余 {len(df)} 行')

    # ── 步骤 3: SPH 替换 + 更新 bipolar ──
    n_sph = 0
    for idx in df.index:
        onset_str = str(df.at[idx, 'onset_channels'])
        if 'SPH-L' not in onset_str and 'SPH-R' not in onset_str:
            continue

        n_sph += 1
        new_onset = replace_sph(onset_str)
        df.at[idx, 'onset_channels'] = new_onset

        # 重新计算 bipolar
        channels = [c.strip() for c in new_onset.split(';')]
        soz_bipolar, bipolar_labels = channels_to_bipolar(channels)
        df.at[idx, 'soz_bipolar'] = soz_bipolar
        for col, val in bipolar_labels.items():
            df.at[idx, col] = val

    print(f'步骤3: 替换了 {n_sph} 行中的 SPH-L/SPH-R')

    # ── 步骤 4: 更新 n_seizure_events 并输出 ──
    edf_counts = df.groupby('edf_path').size().to_dict()
    df['n_seizure_events'] = df['edf_path'].map(edf_counts)

    df = df.sort_values(['patient_id', 'sz_start']).reset_index(drop=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f'\n完成! 已保存到: {args.output}')
    print(f'  原始行数: {n_orig}')
    print(f'  最终行数: {len(df)}')
    print(f'  唯一患者: {df["patient_id"].str.rsplit("_", n=1).str[0].nunique()}')
    print(f'  唯一 EDF: {df["edf_path"].nunique()}')


if __name__ == '__main__':
    main()