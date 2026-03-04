#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_manifests.py
────────────────────────────────────────────────────────────────────
合并 TUSZ 与私有数据集的 manifest 文件，生成统一训练用 CSV。

输入:
  - tusz_manifest.csv          (TUSZ数据集, generate_manifest.py生成)
  - bipolar_manifest_pure_v1.csv  (私有数据集)

过滤:
  - TUSZ: 剔除 n_seizure_events == 0 的行（无发作文件）

输出字段（unified_manifest.csv）:
  ─ 通用元信息 ─
  source          : 'tusz' / 'private'
  patient_id      : 患者ID
  edf_path        : EDF文件路径（相对/绝对）
  split           : train/dev/eval  （private 均设为 'private'）
  duration        : 文件时长（秒）
  sz_start        : 发作开始时间（秒）
  sz_end          : 发作结束时间（秒）
  sz_duration     : 本次发作时长（秒）
  n_seizure_events: 该文件的发作事件总数
  hemisphere      : L/R/B/M/U

  ─ SOZ 标注 ─
  onset_channels  : 逗号分隔的双极导联名，如 "FP1-F7,F8-T4"
  soz_bipolar     : 等同 onset_channels（保留原始形式，分隔符统一为逗号）

  ─ 22个 TCP 双极导联 0/1 列 (与 eeg_pipeline.py TCP_PAIRS 顺序一致) ─
  FP1_F7, F7_T3, T3_T5, T5_O1,
  FP2_F8, F8_T4, T4_T6, T6_O2,
  FP1_F3, F3_C3, C3_P3, P3_O1,
  FP2_F4, F4_C4, C4_P4, P4_O2,
  A1_T3,  T3_C3, C3_CZ, CZ_C4, C4_T4, T4_A2
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from pathlib import Path

# ── TCP 22通道双极导联（与 eeg_pipeline.py 完全一致）─────────────────────────
TCP_BIPOLAR = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'A1-T3',  'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
]
# CSV列名（减号→下划线）
TCP_COL_NAMES = [ch.replace('-', '_') for ch in TCP_BIPOLAR]
TCP_SET = {ch: col for ch, col in zip(TCP_BIPOLAR, TCP_COL_NAMES)}


def bipolar_str_to_01(bipolar_str: str) -> dict:
    """
    将双极导联字符串（逗号或分号分隔）转换为22个TCP导联的0/1字典。
    bipolar_str 中的导联名大小写不敏感。
    """
    row = {col: 0 for col in TCP_COL_NAMES}
    if not bipolar_str or str(bipolar_str).strip() in ('', 'nan'):
        return row
    # 支持逗号或分号分隔
    sep = ';' if ';' in str(bipolar_str) else ','
    channels = [c.strip().upper() for c in str(bipolar_str).split(sep) if c.strip()]
    for ch in channels:
        if ch in TCP_SET:
            row[TCP_SET[ch]] = 1
    return row


# ─────────────────────────────────────────────────────────────────────────────
# 处理 TUSZ
# ─────────────────────────────────────────────────────────────────────────────
def process_tusz(tusz_path: str) -> pd.DataFrame:
    df = pd.read_csv(tusz_path, dtype={'n_seizure_events': int})

    # 过滤：去掉无发作事件的行
    before = len(df)
    df = df[df['n_seizure_events'] > 0].copy()
    print(f"[TUSZ] 共 {before} 行，过滤后保留 {len(df)} 行（n_seizure_events > 0）")

    rows = []
    for _, r in df.iterrows():
        # TUSZ 的 sz_starts/sz_ends 为分号分隔的多个时间点（一个文件多次发作）
        # 每次发作单独拆成一行，方便训练时按发作片段索引
        starts = [s.strip() for s in str(r['sz_starts']).split(';') if s.strip()]
        ends   = [e.strip() for e in str(r['sz_ends']).split(';') if e.strip()]

        # onset_channels 字段是逗号分隔的双极导联名（全大写）
        onset_ch_str = str(r['onset_channels']) if pd.notna(r['onset_channels']) else ''
        bipolar_01 = bipolar_str_to_01(onset_ch_str)

        # 规范化 onset_channels（去除空格）
        onset_norm = ','.join(
            c.strip() for c in onset_ch_str.split(',') if c.strip()
        )

        n_events = int(r['n_seizure_events'])
        n_pairs = min(len(starts), len(ends))

        for i in range(max(n_pairs, 1)):
            sz_start = float(starts[i]) if i < len(starts) else float('nan')
            sz_end   = float(ends[i])   if i < len(ends)   else float('nan')
            sz_dur   = (sz_end - sz_start) if (not pd.isna(sz_start) and not pd.isna(sz_end)) else float('nan')

            row = {
                'source':           'tusz',
                'patient_id':       r['patient_id'],
                'edf_path':         r['edf_path'],
                'split':            r['split'],
                'duration':         r['duration'],
                'sz_start':         sz_start,
                'sz_end':           sz_end,
                'sz_duration':      sz_dur,
                'n_seizure_events': n_events,
                'hemisphere':       r.get('hemisphere', 'U'),
                'onset_channels':   onset_norm,
                'soz_bipolar':      onset_norm,
            }
            row.update(bipolar_01)
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 处理私有数据集
# ─────────────────────────────────────────────────────────────────────────────

# 私有数据集的 hemi 字段值映射
HEMI_MAP = {
    'L': 'L', 'R': 'R', 'B': 'B', 'M': 'M',
    'left': 'L', 'right': 'R', 'bilateral': 'B', 'midline': 'M',
}

def process_private(private_path: str) -> pd.DataFrame:
    df = pd.read_csv(private_path, dtype=str)
    print(f"[Private] 共 {len(df)} 行，全部保留")

    rows = []
    for _, r in df.iterrows():
        # soz_bipolar 字段（分号分隔的双极导联名），转为01列 + 逗号分隔的字符串
        soz_bp = str(r.get('soz_bipolar', ''))
        bipolar_01 = bipolar_str_to_01(soz_bp)

        # 统一为逗号分隔
        if ';' in soz_bp:
            onset_norm = ','.join(c.strip() for c in soz_bp.split(';') if c.strip())
        else:
            onset_norm = soz_bp.strip()

        # 时间字段
        try:
            sz_start = float(r.get('sz_start', 'nan'))
        except (ValueError, TypeError):
            sz_start = float('nan')
        try:
            sz_end = float(r.get('sz_end', 'nan'))
        except (ValueError, TypeError):
            sz_end = float('nan')
        try:
            sz_dur = float(r.get('sz_duration', 'nan'))
        except (ValueError, TypeError):
            sz_dur = (sz_end - sz_start) if not (pd.isna(sz_start) or pd.isna(sz_end)) else float('nan')
        try:
            nsz = int(float(str(r.get('nsz', '0'))))
        except (ValueError, TypeError):
            nsz = 0

        # 半球
        hemi_raw = str(r.get('hemi', '')).strip()
        hemisphere = HEMI_MAP.get(hemi_raw, HEMI_MAP.get(hemi_raw.capitalize(), 'U'))

        # edf 路径（fn 字段）
        fn = str(r.get('fn', '')) or str(r.get('loc', ''))

        row = {
            'source':           'private',
            'patient_id':       str(r.get('pt_id', '')),
            'edf_path':         fn,
            'split':            'private',
            'duration':         r.get('duration', float('nan')),
            'sz_start':         sz_start,
            'sz_end':           sz_end,
            'sz_duration':      sz_dur,
            'n_seizure_events': nsz,
            'hemisphere':       hemisphere,
            'onset_channels':   onset_norm,
            'soz_bipolar':      onset_norm,
        }
        row.update(bipolar_01)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base = Path(__file__).parent

    tusz_path    = base / 'tusz_manifest.csv'
    private_path = base.parent / 'DeepSOZ' / 'train' / 'visualizations' / 'bipolar_manifest_pure_v1.csv'
    output_path  = base / 'combined_manifest.csv'

    df_tusz    = process_tusz(str(tusz_path))
    df_private = process_private(str(private_path))

    # 合并（TUSZ在前，私有数据集在后）
    df_combined = pd.concat([df_tusz, df_private], ignore_index=True)

    # 列顺序：通用元信息 + 双极导联01列
    meta_cols = [
        'source', 'patient_id', 'edf_path', 'split', 'duration',
        'sz_start', 'sz_end', 'sz_duration', 'n_seizure_events',
        'hemisphere', 'onset_channels', 'soz_bipolar',
    ]
    final_cols = meta_cols + TCP_COL_NAMES
    df_combined = df_combined[final_cols]

    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')

    # ── 统计报告 ─────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('合并完成！')
    print(f'输出路径: {output_path}')
    print(f'总行数: {len(df_combined)}')
    print()
    print('各来源行数:')
    print(df_combined['source'].value_counts().to_string())
    print()
    print('各split分布:')
    print(df_combined['split'].value_counts().to_string())
    print()
    print('SOZ阳性行数（至少1个双极导联为1）:')
    soz_any = (df_combined[TCP_COL_NAMES].sum(axis=1) > 0).sum()
    print(f'  {soz_any} / {len(df_combined)} ({soz_any/len(df_combined)*100:.1f}%)')
    print()
    print('各来源 SOZ 阳性率:')
    for src in df_combined['source'].unique():
        sub = df_combined[df_combined['source'] == src]
        pos = (sub[TCP_COL_NAMES].sum(axis=1) > 0).sum()
        print(f'  {src}: {pos}/{len(sub)} ({pos/len(sub)*100:.1f}%)')
    print()
    print('22个TCP导联的SOZ频次:')
    for ch, col in zip(TCP_BIPOLAR, TCP_COL_NAMES):
        cnt = int(df_combined[col].sum())
        print(f'  {ch:10s}: {cnt:4d}')
    print('=' * 60)


if __name__ == '__main__':
    main()
