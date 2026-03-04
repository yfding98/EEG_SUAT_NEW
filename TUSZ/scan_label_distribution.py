#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ 标签分布扫描脚本

遍历 TUSZ 数据集下所有 EDF 文件对应的标注，按不同 montage 分别统计 label 分布，
便于根据标签分布制定训练策略。

输出内容（按 montage 分别输出）：
- 各 label 出现的文件数、标注段数（segment 数）
- 有/无发作文件数
- 发作类型分布

用法:
    python -m TUSZ.scan_label_distribution
    python -m TUSZ.scan_label_distribution --data-root F:/dataset/TUSZ/v2.0.3/edf
    python -m TUSZ.scan_label_distribution --data-root F:/dataset/TUSZ/v2.0.3/edf --out-dir doc
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# 保证可导入 TUSZ 包内模块
if __name__ == '__main__' and __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = 'TUSZ'

from TUSZ.config import TUSZ_CONFIG, is_seizure_label, BACKGROUND_LABEL
from TUSZ.parse_annotations import parse_csv_annotation
from TUSZ.data_loader import detect_montage_type


# ==============================================================================
# 目录扫描（与 generate_manifest 一致）
# ==============================================================================

def scan_edf_files(data_root: Path, split: str = None) -> List[Dict]:
    """扫描 TUSZ 目录下所有 EDF 文件，返回路径与 montage、split 信息。"""
    if split:
        splits = [split]
    else:
        splits = ['train', 'dev', 'eval']

    files = []
    for split_name in splits:
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        for patient_dir in split_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            for session_dir in patient_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                for montage_dir in session_dir.iterdir():
                    if not montage_dir.is_dir():
                        continue
                    montage = detect_montage_type(str(montage_dir))
                    for edf_file in montage_dir.glob('*.edf'):
                        rel_path = edf_file.relative_to(data_root)
                        files.append({
                            'edf_path': edf_file,
                            'rel_path': str(rel_path),
                            'split': split_name,
                            'montage': montage,
                        })
    return files


# ==============================================================================
# 按 montage 统计 label
# ==============================================================================

def collect_label_stats_per_file(edf_path: Path) -> Tuple[bool, Dict[str, int], bool]:
    """
    解析单个 EDF 对应的 .csv 标注，统计该文件内各 label 的 segment 数量。

    Returns:
        (has_csv, label_to_segment_count, has_seizure)
    """
    csv_path = edf_path.with_suffix('.csv')
    if not csv_path.exists():
        return False, {}, False

    try:
        ann = parse_csv_annotation(str(csv_path))
    except Exception:
        return False, {}, False

    label_counts = defaultdict(int)
    for a in ann.annotations:
        label_counts[a.label.strip().lower()] += 1

    return True, dict(label_counts), ann.has_seizure


def aggregate_by_montage(
    data_root: Path,
    split_filter: str = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    遍历所有 EDF，按 montage 聚合 label 统计。

    Returns:
        montage_label_segments: montage -> label -> 该 label 的 segment 总个数
        montage_label_files:    montage -> label -> 包含该 label 的文件数
        montage_file_stats:     montage -> { 'total_files', 'with_csv', 'with_seizure', 'split_counts' }
    """
    files = scan_edf_files(data_root, split=split_filter)
    montage_label_segments: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    montage_label_files: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    montage_file_stats: Dict[str, Dict] = defaultdict(lambda: {
        'total_files': 0,
        'with_csv': 0,
        'with_seizure': 0,
        'split_counts': defaultdict(int),
    })

    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc='扫描标注')
    except ImportError:
        iterator = files

    for finfo in iterator:
        montage = finfo['montage']
        edf_path = finfo['edf_path']
        split = finfo['split']

        montage_file_stats[montage]['total_files'] += 1
        montage_file_stats[montage]['split_counts'][split] += 1

        has_csv, label_counts, has_seizure = collect_label_stats_per_file(edf_path)
        if not has_csv:
            continue

        montage_file_stats[montage]['with_csv'] += 1
        if has_seizure:
            montage_file_stats[montage]['with_seizure'] += 1

        for label, count in label_counts.items():
            montage_label_segments[montage][label] += count
            montage_label_files[montage][label] += 1

    # 转成普通 dict 便于序列化
    def to_plain(d):
        if isinstance(d, defaultdict):
            return {k: to_plain(v) for k, v in d.items()}
        return d

    montage_file_stats = {m: {**s, 'split_counts': dict(s['split_counts'])} for m, s in montage_file_stats.items()}
    return (
        {m: dict(seg) for m, seg in montage_label_segments.items()},
        {m: dict(fl) for m, fl in montage_label_files.items()},
        montage_file_stats,
    )


# ==============================================================================
# 输出
# ==============================================================================

def write_csv_per_montage(
    montage_label_segments: Dict[str, Dict[str, int]],
    montage_label_files: Dict[str, Dict[str, int]],
    montage_file_stats: Dict,
    output_path: Path,
):
    """按 montage 分表写入 CSV：每个 montage 一个 sheet 或写成一个长表，这里用长表。"""
    rows = []
    for montage in sorted(montage_label_segments.keys()):
        stats = montage_file_stats.get(montage, {})
        total_files = stats.get('total_files', 0)
        with_csv = stats.get('with_csv', 0)
        for label in sorted(montage_label_segments[montage].keys()):
            seg_count = montage_label_segments[montage][label]
            file_count = montage_label_files[montage].get(label, 0)
            rows.append({
                'montage': montage,
                'label': label,
                'segment_count': seg_count,
                'file_count': file_count,
                'total_edf_files': total_files,
                'files_with_csv': with_csv,
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['montage', 'label', 'segment_count', 'file_count', 'total_edf_files', 'files_with_csv'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV 已写入: {output_path}")


def write_summary_md(
    montage_label_segments: Dict[str, Dict[str, int]],
    montage_label_files: Dict[str, Dict[str, int]],
    montage_file_stats: Dict,
    output_path: Path,
):
    """按 montage 分别输出可读的 Markdown 摘要。"""
    lines = [
        '# TUSZ 标签分布（按 Montage）',
        '',
        '用于根据 label 分布制定训练策略。',
        '',
    ]

    for montage in sorted(montage_label_segments.keys()):
        stats = montage_file_stats.get(montage, {})
        total = stats.get('total_files', 0)
        with_csv = stats.get('with_csv', 0)
        with_sz = stats.get('with_seizure', 0)
        split_counts = stats.get('split_counts', {})

        lines.extend([
            f'## Montage: `{montage}`',
            '',
            f'- **EDF 文件总数**: {total}',
            f'- **有 .csv 标注的文件数**: {with_csv}',
            f'- **含发作的文件数**: {with_sz}',
            f'- **按划分**: ' + ', '.join(f'{k}={v}' for k, v in sorted(split_counts.items())),
            '',
            '### Label 分布（该 montage 下）',
            '',
            '| Label | 标注段数(segment) | 出现该 label 的文件数 |',
            '|-------|-------------------|------------------------|',
        ])

        seg = montage_label_segments[montage]
        fl = montage_label_files[montage]
        for label in sorted(seg.keys()):
            lines.append(f'| {label} | {seg[label]} | {fl.get(label, 0)} |')

        lines.append('')
        lines.append('---')
        lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] 摘要已写入: {output_path}")


def print_console_summary(
    montage_label_segments: Dict[str, Dict[str, int]],
    montage_label_files: Dict[str, Dict[str, int]],
    montage_file_stats: Dict,
):
    """在控制台按 montage 打印简要分布。"""
    print('\n' + '=' * 70)
    print('  TUSZ 标签分布（按 Montage）')
    print('=' * 70)

    for montage in sorted(montage_label_segments.keys()):
        stats = montage_file_stats.get(montage, {})
        total = stats.get('total_files', 0)
        with_csv = stats.get('with_csv', 0)
        with_sz = stats.get('with_seizure', 0)

        print(f'\n--- Montage: {montage} ---')
        print(f'  EDF 总数: {total}  有 CSV: {with_csv}  含发作: {with_sz}')
        print(f'  Label 分布:')
        seg = montage_label_segments[montage]
        fl = montage_label_files[montage]
        for label in sorted(seg.keys()):
            print(f'    {label:12s}  segment数: {seg[label]:8d}  文件数: {fl.get(label, 0):8d}')
    print('\n[DONE] 统计完成。')


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='遍历 TUSZ 所有 EDF 的标注，按 montage 统计 label 分布',
    )
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default=None,
        help=f'TUSZ EDF 根目录 (默认: config 中的 {TUSZ_CONFIG.get("data_root", "")})',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='输出目录：写入 CSV 与 summary.md (默认: 项目 doc 目录)',
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev', 'eval'],
        default=None,
        help='只统计指定划分',
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='不写入 CSV',
    )
    parser.add_argument(
        '--no-md',
        action='store_true',
        help='不写入 Markdown 摘要',
    )
    args = parser.parse_args()

    data_root = Path(args.data_root or TUSZ_CONFIG.get('data_root', '.'))
    if not data_root.exists():
        print(f"[ERROR] 数据根目录不存在: {data_root}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / 'doc'
    out_dir.mkdir(parents=True, exist_ok=True)

    montage_label_segments, montage_label_files, montage_file_stats = aggregate_by_montage(
        data_root, split_filter=args.split
    )

    print_console_summary(montage_label_segments, montage_label_files, montage_file_stats)

    if not args.no_csv:
        write_csv_per_montage(
            montage_label_segments,
            montage_label_files,
            montage_file_stats,
            out_dir / 'tusz_label_distribution_by_montage.csv',
        )
    if not args.no_md:
        write_summary_md(
            montage_label_segments,
            montage_label_files,
            montage_file_stats,
            out_dir / 'tusz_label_distribution_by_montage.md',
        )

    return 0


if __name__ == '__main__':
    exit(main() or 0)
