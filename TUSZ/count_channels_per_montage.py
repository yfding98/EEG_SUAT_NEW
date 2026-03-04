#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ 通道频次统计脚本

扫描 TUSZ 数据集所有 EDF 文件，按 montage 类型统计每个通道出现的次数和占比。
输出结果到 CSV 和控制台。

用法:
    python count_channels_per_montage.py
    python count_channels_per_montage.py --data-root F:/dataset/TUSZ/v2.0.3/edf
    python count_channels_per_montage.py --normalize  # 标准化通道名后再统计
"""

import os
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# EDF 读取（优先使用 pyedflib，速度比 MNE 快得多，且只需读头部信息）
try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# ==============================================================================
# 通道名标准化（可选）
# ==============================================================================

CHANNEL_NAME_MAP = {
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ', 'Oz': 'OZ',
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
}


def normalize_channel_name(name: str) -> str:
    """标准化通道名：去除 EEG 前缀和 -REF/-LE 等后缀"""
    name = name.strip()
    upper = name.upper()

    # 去除 EEG 前缀
    for prefix in ['EEG ', 'EEG-']:
        if upper.startswith(prefix):
            name = name[len(prefix):]
            upper = name.upper()

    # 去除参考后缀
    for suffix in ['-REF', '-LE', '-AR', '-AVG']:
        if upper.endswith(suffix):
            name = name[:-len(suffix)]
            upper = name.upper()

    # 常见别名映射
    if upper in CHANNEL_NAME_MAP:
        return CHANNEL_NAME_MAP[upper]
    for key, val in CHANNEL_NAME_MAP.items():
        if key.upper() == upper:
            return val

    return upper


# ==============================================================================
# Montage 检测
# ==============================================================================

def detect_montage_from_path(path: Path) -> str:
    """从路径中检测 montage 类型"""
    for part in path.parts:
        part_lower = part.lower()
        if '01_tcp_ar' in part_lower and 'tcp_ar_a' not in part_lower:
            return 'tcp_ar'
        elif '02_tcp_le' in part_lower and 'tcp_le_a' not in part_lower:
            return 'tcp_le'
        elif '03_tcp_ar_a' in part_lower:
            return 'tcp_ar_a'
        elif '04_tcp_le_a' in part_lower:
            return 'tcp_le_a'
    return 'unknown'


# ==============================================================================
# 快速读取 EDF 通道名（只读头部，不读信号数据）
# ==============================================================================

def read_channel_names_pyedflib(filepath: str) -> List[str]:
    """使用 pyedflib 只读取通道名（不加载信号）"""
    f = pyedflib.EdfReader(filepath)
    try:
        ch_names = f.getSignalLabels()
        return list(ch_names)
    finally:
        f._close()


def read_channel_names_mne(filepath: str) -> List[str]:
    """使用 MNE 只读取通道名"""
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose='ERROR')
    return list(raw.ch_names)


def read_channel_names(filepath: str) -> List[str]:
    """读取 EDF 文件的通道名列表"""
    errors = []

    if HAS_PYEDFLIB:
        try:
            return read_channel_names_pyedflib(filepath)
        except Exception as e:
            errors.append(f"pyedflib: {e}")

    if HAS_MNE:
        try:
            return read_channel_names_mne(filepath)
        except Exception as e:
            errors.append(f"mne: {e}")

    raise RuntimeError(f"无法读取 {filepath}: {errors}")


# ==============================================================================
# 主逻辑
# ==============================================================================

def scan_and_count(
    data_root: str,
    do_normalize: bool = False,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    扫描所有 EDF 文件，按 montage 统计通道出现频次。

    Returns:
        montage_channel_counts: {montage: {channel: count}}
        montage_file_counts: {montage: total_files}
    """
    data_root = Path(data_root)
    edf_files = list(data_root.rglob('*.edf'))
    print(f"[INFO] 共找到 {len(edf_files)} 个 EDF 文件")

    montage_channel_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    montage_file_counts: Dict[str, int] = defaultdict(int)
    error_count = 0

    for edf_path in tqdm(edf_files, desc="扫描 EDF 文件"):
        montage = detect_montage_from_path(edf_path)
        try:
            ch_names = read_channel_names(str(edf_path))
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"[WARN] 读取失败: {edf_path} -> {e}")
            continue

        montage_file_counts[montage] += 1

        for ch in ch_names:
            if do_normalize:
                ch = normalize_channel_name(ch)
            montage_channel_counts[montage][ch] += 1

    if error_count > 0:
        print(f"[WARN] 共 {error_count} 个文件读取失败")

    return dict(montage_channel_counts), dict(montage_file_counts)


def print_results(
    montage_channel_counts: Dict[str, Dict[str, int]],
    montage_file_counts: Dict[str, int],
):
    """打印统计结果到控制台"""
    for montage in sorted(montage_channel_counts.keys()):
        file_count = montage_file_counts[montage]
        ch_counts = montage_channel_counts[montage]

        print(f"\n{'=' * 70}")
        print(f"  Montage: {montage}  |  文件数: {file_count}")
        print(f"{'=' * 70}")
        print(f"  {'通道名':<30s} {'出现次数':>10s} {'占比(%)':>10s}")
        print(f"  {'-' * 52}")

        # 按出现次数降序排列
        for ch, count in sorted(ch_counts.items(), key=lambda x: -x[1]):
            pct = (count / file_count * 100) if file_count > 0 else 0
            print(f"  {ch:<30s} {count:>10d} {pct:>10.2f}")

        print(f"  {'-' * 52}")
        print(f"  共 {len(ch_counts)} 种不同通道")


def save_csv(
    montage_channel_counts: Dict[str, Dict[str, int]],
    montage_file_counts: Dict[str, int],
    output_path: str,
):
    """保存统计结果到 CSV"""
    rows = []
    for montage in sorted(montage_channel_counts.keys()):
        file_count = montage_file_counts[montage]
        ch_counts = montage_channel_counts[montage]
        for ch, count in sorted(ch_counts.items(), key=lambda x: -x[1]):
            pct = (count / file_count * 100) if file_count > 0 else 0
            rows.append({
                'montage': montage,
                'channel': ch,
                'count': count,
                'total_files': file_count,
                'percentage': round(pct, 4),
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['montage', 'channel', 'count', 'total_files', 'percentage'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] CSV 已保存到: {output_path}")


def save_json(
    montage_channel_counts: Dict[str, Dict[str, int]],
    montage_file_counts: Dict[str, int],
    output_path: str,
):
    """保存统计结果到 JSON"""
    result = {}
    for montage in sorted(montage_channel_counts.keys()):
        file_count = montage_file_counts[montage]
        ch_counts = montage_channel_counts[montage]
        channels = []
        for ch, count in sorted(ch_counts.items(), key=lambda x: -x[1]):
            pct = (count / file_count * 100) if file_count > 0 else 0
            channels.append({
                'channel': ch,
                'count': count,
                'percentage': round(pct, 4),
            })
        result[montage] = {
            'total_files': file_count,
            'channels': channels,
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[INFO] JSON 已保存到: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='统计 TUSZ 数据集每种 montage 下所有通道的出现频次和占比',
    )
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default='F:/dataset/TUSZ/v2.0.3/edf',
        help='TUSZ 数据集 edf 根目录',
    )
    parser.add_argument(
        '--output-csv', '-o',
        type=str,
        default=None,
        help='输出 CSV 路径 (默认: 脚本同目录下 channel_counts_per_montage.csv)',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='输出 JSON 路径 (默认: 脚本同目录下 channel_counts_per_montage.json)',
    )
    parser.add_argument(
        '--normalize', '-n',
        action='store_true',
        help='是否标准化通道名（去除 EEG 前缀，-REF/-LE 后缀等）',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.output_csv is None:
        args.output_csv = str(script_dir / 'channel_counts_per_montage.csv')
    if args.output_json is None:
        args.output_json = str(script_dir / 'channel_counts_per_montage.json')

    # 执行统计
    montage_channel_counts, montage_file_counts = scan_and_count(
        data_root=args.data_root,
        do_normalize=args.normalize,
    )

    # 打印结果
    print_results(montage_channel_counts, montage_file_counts)

    # 保存
    save_csv(montage_channel_counts, montage_file_counts, args.output_csv)
    save_json(montage_channel_counts, montage_file_counts, args.output_json)

    print("\n[DONE] 统计完成！")


if __name__ == '__main__':
    main()
