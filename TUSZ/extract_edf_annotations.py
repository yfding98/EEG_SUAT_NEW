#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从私有数据集的所有 EDF 文件中提取 annotation 信息，输出为 CSV。

用法:
  python extract_edf_annotations.py [--data-root DATA_ROOT] [--output OUTPUT_CSV]

默认:
  --data-root  "E:/DataSet/EEG/EEG dataset_SUAT"
  --output     edf_annotations.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import mne


def find_edf_files(root: Path):
    """递归查找所有 .edf / .EDF 文件。"""
    return sorted(root.rglob("*.[eE][dD][fF]"))


def fix_encoding(text: str) -> str:
    """修复 latin-1 读取的 UTF-8 中文文本。"""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def extract_annotations(edf_path: Path, data_root: Path):
    """从单个 EDF 文件中提取所有 annotations。"""
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False,
                                   verbose=False, encoding="latin-1")
    except Exception as e:
        print(f"  [WARN] 无法读取 {edf_path}: {e}", file=sys.stderr)
        return []

    rel_path = str(edf_path.relative_to(data_root))
    # 提取患者名 (父目录名)
    patient = edf_path.parent.name
    # 提取子集名 (祖父目录名)
    subset = edf_path.parent.parent.name if edf_path.parent.parent != data_root else ""
    duration = round(raw.times[-1], 3)
    n_channels = len(raw.ch_names)
    sfreq = raw.info["sfreq"]

    rows = []
    if len(raw.annotations) == 0:
        # 无 annotation 也输出一行，标记为空
        rows.append({
            "edf_path": rel_path,
            "subset": subset,
            "patient": patient,
            "file_name": edf_path.name,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "duration_sec": duration,
            "n_annotations": 0,
            "ann_idx": "",
            "onset_sec": "",
            "duration_ann_sec": "",
            "description": "(no annotations)",
        })
    else:
        for i, (onset, dur, desc) in enumerate(zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )):
            rows.append({
                "edf_path": rel_path,
                "subset": subset,
                "patient": patient,
                "file_name": edf_path.name,
                "sfreq": sfreq,
                "n_channels": n_channels,
                "duration_sec": duration,
                "n_annotations": len(raw.annotations),
                "ann_idx": i,
                "onset_sec": round(onset, 3),
                "duration_ann_sec": round(dur, 3),
                "description": fix_encoding(desc),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="从私有 EDF 数据集中提取所有 annotation 信息"
    )
    parser.add_argument(
        "--data-root",
        default="E:/DataSet/EEG/EEG dataset_SUAT",
        help="EDF 数据根目录",
    )
    parser.add_argument(
        "--output", "-o",
        default="edf_annotations.csv",
        help="输出 CSV 文件路径",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"错误: 数据目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)

    edf_files = find_edf_files(data_root)
    print(f"找到 {len(edf_files)} 个 EDF 文件，开始提取 annotations ...")

    fieldnames = [
        "edf_path", "subset", "patient", "file_name",
        "sfreq", "n_channels", "duration_sec", "n_annotations",
        "ann_idx", "onset_sec", "duration_ann_sec", "description",
    ]

    all_rows = []
    n_with_ann = 0
    n_without_ann = 0
    total_ann = 0

    for i, edf in enumerate(edf_files, 1):
        print(f"  [{i}/{len(edf_files)}] {edf.relative_to(data_root)}")
        rows = extract_annotations(edf, data_root)
        all_rows.extend(rows)
        if rows and rows[0]["n_annotations"] > 0:
            n_with_ann += 1
            total_ann += rows[0]["n_annotations"]
        else:
            n_without_ann += 1

    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n完成! 结果已保存到: {output_path}")
    print(f"  EDF 文件总数: {len(edf_files)}")
    print(f"  有 annotations: {n_with_ann}")
    print(f"  无 annotations: {n_without_ann}")
    print(f"  Annotation 总条数: {total_ann}")


if __name__ == "__main__":
    main()
