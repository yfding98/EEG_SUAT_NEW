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
  patient_id      : 患者ID (TUSZ: patient_id字段; 私有: fn字段)
  edf_path        : EDF文件路径 (TUSZ: edf_path字段; 私有: loc字段)
  split           : train/dev/eval  （private 均设为 'private'）
  duration        : 文件时长（秒）
  sz_start        : 发作开始时间（秒）
  sz_end          : 发作结束时间（秒）
  sz_duration     : 本次发作时长（秒）
  n_seizure_events: 该文件的发作事件总数
  seizure_type    : 发作类型 (fnsz/gnsz/cpsz/absz/tnsz/mysz/seiz 等)
  hemisphere      : L/R/B/M/U

  ─ SOZ 标注 ─
  onset_channels  : 单极电极名, 分号分隔
                    TUSZ: 从双极导联拆分去重, 如 "T4-T6,T6-O2" → "T4;T6;O2"
                    私有: 直接使用 soz_unipolar 列
  soz_bipolar     : 双极导联名（逗号分隔），如 "FP1-F7,F8-T4"

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
MAX_TUSZ_POSITIVE_CHANNEL_RATIO = 0.5


def infer_hemisphere_from_electrodes(electrodes) -> str:
    hemispheres = set()
    for elec in electrodes:
        e = str(elec).strip().upper()
        if not e:
            continue
        if e.endswith('Z'):
            hemispheres.add('M')
        elif e.endswith(('1', '3', '5', '7', '9')):
            hemispheres.add('L')
        elif e.endswith(('2', '4', '6', '8', '0')):
            hemispheres.add('R')
    if not hemispheres:
        return 'U'
    if hemispheres == {'L'}:
        return 'L'
    if hemispheres == {'R'}:
        return 'R'
    if 'L' in hemispheres and 'R' in hemispheres:
        return 'B'
    if hemispheres == {'M'}:
        return 'M'
    if 'L' in hemispheres:
        return 'L'
    if 'R' in hemispheres:
        return 'R'
    return 'M'


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
    skipped_union_multi = 0
    skipped_missing_onset = 0
    skipped_dense_events = 0
    for _, r in df.iterrows():
        starts = [s.strip() for s in str(r['sz_starts']).split(';') if s.strip()]
        ends   = [e.strip() for e in str(r['sz_ends']).split(';') if e.strip()]

        # onset_channels: pipe-separated per-event format from generate_manifest
        onset_ch_raw = str(r['onset_channels']) if pd.notna(r['onset_channels']) else ''
        per_event_onset = [g.strip() for g in onset_ch_raw.split('|') if g.strip()]

        # backward compatibility: if no pipe separator, treat as single group
        if not per_event_onset and onset_ch_raw.strip():
            per_event_onset = [onset_ch_raw.strip()]

        event_hemi_raw = str(r.get('event_hemispheres', '')) if pd.notna(r.get('event_hemispheres', '')) else ''
        per_event_hemi = [h.strip().upper() for h in event_hemi_raw.split('|') if h.strip()]

        # seizure_types: pipe-separated per-event format (e.g. "fnsz|gnsz|fnsz")
        # backward compat: old format was comma-separated union (e.g. "fnsz,gnsz")
        sz_types_raw = str(r.get('seizure_types', '')) if pd.notna(r.get('seizure_types', '')) else ''
        if '|' in sz_types_raw:
            per_event_types = [t.strip() for t in sz_types_raw.split('|')]
        elif ',' in sz_types_raw:
            # old format: comma-separated union, apply to all events
            union_type = sz_types_raw.strip()
            per_event_types = [union_type]
        elif sz_types_raw.strip():
            per_event_types = [sz_types_raw.strip()]
        else:
            per_event_types = []

        n_events = int(r['n_seizure_events'])
        n_pairs = min(len(starts), len(ends))
        n_rows = max(n_events, n_pairs, len(per_event_onset), len(per_event_types), len(per_event_hemi), 1)

        if n_events > 1 and len(per_event_onset) <= 1:
            skipped_union_multi += 1
            continue

        for i in range(n_rows):
            sz_start = float(starts[i]) if i < len(starts) else float('nan')
            sz_end   = float(ends[i])   if i < len(ends)   else float('nan')
            sz_dur   = (sz_end - sz_start) if (not pd.isna(sz_start) and not pd.isna(sz_end)) else float('nan')

            # per-event onset channels
            if i < len(per_event_onset):
                event_onset_str = per_event_onset[i]
            elif per_event_onset:
                event_onset_str = per_event_onset[-1]
            else:
                event_onset_str = ''

            if not event_onset_str:
                skipped_missing_onset += 1
                continue

            # per-event seizure type
            if i < len(per_event_types):
                event_sz_type = per_event_types[i]
            elif per_event_types:
                event_sz_type = per_event_types[-1]
            else:
                event_sz_type = 'seiz'

            bipolar_01 = bipolar_str_to_01(event_onset_str)

            soz_bipolar_norm = ','.join(
                c.strip() for c in event_onset_str.split(',') if c.strip()
            )
            pos_count = int(sum(bipolar_01[col] for col in TCP_COL_NAMES))
            if pos_count > len(TCP_COL_NAMES) * MAX_TUSZ_POSITIVE_CHANNEL_RATIO:
                skipped_dense_events += 1
                continue

            electrodes = []
            for bp_ch in event_onset_str.split(','):
                bp_ch = bp_ch.strip().upper()
                if bp_ch and '-' in bp_ch:
                    for elec in bp_ch.split('-'):
                        e = elec.strip()
                        if e and e not in electrodes:
                            electrodes.append(e)
                elif bp_ch:
                    if bp_ch not in electrodes:
                        electrodes.append(bp_ch)
            onset_channels_unipolar = ';'.join(electrodes)
            event_hemisphere = infer_hemisphere_from_electrodes(electrodes)
            if event_hemisphere == 'U' and i < len(per_event_hemi):
                event_hemisphere = per_event_hemi[i]
            if event_hemisphere == 'U':
                event_hemisphere = str(r.get('hemisphere', 'U')).strip().upper() or 'U'

            row = {
                'source':           'tusz',
                'patient_id':       r['patient_id'],
                'edf_path':         str(r['edf_path']).replace('\\', '/'),
                'split':            r['split'],
                'duration':         r['duration'],
                'sz_start':         sz_start,
                'sz_end':           sz_end,
                'sz_duration':      sz_dur,
                'n_seizure_events': n_events,
                'seizure_type':     event_sz_type,
                'hemisphere':       event_hemisphere,
                'onset_channels':   onset_channels_unipolar,
                'soz_bipolar':      soz_bipolar_norm,
            }
            row.update(bipolar_01)
            rows.append(row)

    if skipped_union_multi:
        print(f"[TUSZ] skipped {skipped_union_multi} multi-event files with ambiguous file-level onset labels")
    if skipped_missing_onset:
        print(f"[TUSZ] skipped {skipped_missing_onset} seizure events with empty onset labels")
    if skipped_dense_events:
        print(
            f"[TUSZ] skipped {skipped_dense_events} dense seizure events "
            f"(>{MAX_TUSZ_POSITIVE_CHANNEL_RATIO:.0%} positive channels)"
        )

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

        # soz_bipolar: 统一为逗号分隔
        if ';' in soz_bp:
            soz_bipolar_norm = ','.join(c.strip() for c in soz_bp.split(';') if c.strip())
        else:
            soz_bipolar_norm = soz_bp.strip()

        # onset_channels: 使用 soz_unipolar 列（分号分隔的单极通道名）
        soz_uni = str(r.get('soz_unipolar', ''))
        if soz_uni and soz_uni.strip() and soz_uni.strip().lower() != 'nan':
            onset_channels = ';'.join(
                c.strip().upper() for c in soz_uni.split(';') if c.strip()
            )
        else:
            onset_channels = ''

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

        # edf_path ← loc 字段（数据文件相对路径）
        edf_path = str(r.get('loc', '')).replace('\\', '/')
        # patient_id ← fn 字段（患者文件标识）
        patient_id = str(r.get('fn', ''))

        row = {
            'source':           'private',
            'patient_id':       patient_id,
            'edf_path':         edf_path,
            'split':            'private',
            'duration':         r.get('duration', float('nan')),
            'sz_start':         sz_start,
            'sz_end':           sz_end,
            'sz_duration':      sz_dur,
            'n_seizure_events': nsz,
            'seizure_type':     'fnsz',
            'hemisphere':       hemisphere,
            'onset_channels':   onset_channels,
            'soz_bipolar':      soz_bipolar_norm,
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
        'seizure_type', 'hemisphere', 'onset_channels', 'soz_bipolar',
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
    print('发作类型分布:')

    n_ch = len(TCP_COL_NAMES)
    sz_type_counts = df_combined['seizure_type'].value_counts()
    for sz_type, cnt in sz_type_counts.items():
        sub = df_combined[df_combined['seizure_type'] == sz_type]
        sub_pos = sub[TCP_COL_NAMES].sum(axis=1)
        print(f'  {sz_type:8s}: {cnt:4d} events, '
              f'avg_pos_ch={sub_pos.mean():.1f}/{n_ch}, '
              f'>50%: {(sub_pos > n_ch * 0.5).sum()}, '
              f'全通道=1: {(sub_pos == n_ch).sum()}')
    print()
    print('22个TCP导联的SOZ频次:')
    for ch, col in zip(TCP_BIPOLAR, TCP_COL_NAMES):
        cnt = int(df_combined[col].sum())
        print(f'  {ch:10s}: {cnt:4d}')
    print()

    # per-event label quality stats
    ch_pos_per_row = df_combined[TCP_COL_NAMES].sum(axis=1)
    n_total = len(df_combined)
    global_pos_rate = df_combined[TCP_COL_NAMES].values.sum() / (n_total * n_ch)
    print(f'Label质量统计:')
    print(f'  全局正样本率 (channel-level): {global_pos_rate:.4f} ({global_pos_rate*100:.1f}%)')
    print(f'  每行平均阳性通道数: {ch_pos_per_row.mean():.2f} / {n_ch}')
    print(f'  全通道=1 的行: {(ch_pos_per_row == n_ch).sum()} / {n_total}')
    print(f'  >50%通道=1 的行: {(ch_pos_per_row > n_ch * 0.5).sum()} / {n_total}')
    print(f'  0通道=1 的行: {(ch_pos_per_row == 0).sum()} / {n_total}')
    print()

    # per-source label stats
    for src in df_combined['source'].unique():
        sub = df_combined[df_combined['source'] == src]
        sub_pos = sub[TCP_COL_NAMES].values.sum()
        sub_total = len(sub) * n_ch
        sub_rate = sub_pos / sub_total if sub_total > 0 else 0
        sub_per_row = sub[TCP_COL_NAMES].sum(axis=1)
        print(f'  [{src}] pos_rate={sub_rate:.4f}, '
              f'avg_pos_ch={sub_per_row.mean():.2f}/{n_ch}, '
              f'全通道=1: {(sub_per_row == n_ch).sum()}, '
              f'>50%: {(sub_per_row > n_ch * 0.5).sum()}')

    print('=' * 60)


if __name__ == '__main__':
    main()
