#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ Manifest Generator

扫描TUSZ数据集目录，生成训练manifest文件。
Manifest包含所有EDF文件的元信息和派生的SOZ标签。

用法:
    python -m TUSZ.generate_manifest --data-root F:/dataset/TUSZ/v2.0.3/edf --output manifest.csv
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    TUSZ_CONFIG,
    BIPOLAR_CHANNELS,
    BIPOLAR_CHANNELS_18,
    BRAIN_REGIONS,
)
from parse_annotations import (
    parse_csv_annotation,
    parse_csv_bi_annotation,
    extract_seizure_events,
    get_soz_channel_labels,
    get_soz_region_labels,
    FileAnnotation,
    SeizureEvent,
)
from data_loader import detect_montage_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Manifest条目数据结构
# ==============================================================================

@dataclass
class ManifestEntry:
    """Manifest条目"""
    # 基本信息
    patient_id: str             # 患者ID, e.g., 'aaaaaaac'
    session_id: str             # Session ID, e.g., 's001_2002'
    file_id: str                # 文件ID, e.g., 'aaaaaaac_s001_t000'
    edf_path: str               # EDF文件相对路径
    split: str                  # 数据集划分: train/dev/eval
    montage: str                # Montage类型: tcp_ar/tcp_le/etc.
    duration: float             # 文件时长（秒）
    
    # 发作信息
    has_seizure: bool           # 是否包含发作
    n_seizure_events: int       # 发作事件数量
    seizure_types: str          # 逗号分隔的发作类型
    sz_starts: str              # 分号分隔的发作开始时间
    sz_ends: str                # 分号分隔的发作结束时间
    total_seizure_duration: float  # 总发作时长（秒）
    
    # SOZ信息（仅用于有发作的文件）
    onset_channels: str         # 逗号分隔的onset通道
    onset_regions: str          # 逗号分隔的onset脑区
    hemisphere: str             # 半球: L/R/B/M/U
    
    # 通道级SOZ标签（22个通道的01值）
    # 使用独立列存储每个通道的标签
    
    def to_dict(self, channel_list: List[str] = None) -> Dict:
        """转换为字典，包含通道级标签列。
        
        onset_channels 可能是 pipe-separated per-event 格式:
          "T4-T6,T6-O2|FP1-F7"   (2个event)
        文件级 0/1 列使用所有 event 的并集。
        """
        d = asdict(self)
        
        if channel_list is None:
            channel_list = BIPOLAR_CHANNELS

        # flatten pipe-separated per-event groups to get file-level union
        onset_set = set()
        for event_group in self.onset_channels.split('|'):
            for ch in event_group.split(','):
                ch = ch.strip().upper()
                if ch:
                    onset_set.add(ch)

        for ch in channel_list:
            col_name = ch.replace('-', '_')
            d[col_name] = 1 if ch.upper() in onset_set else 0
        
        return d


# ==============================================================================
# 目录扫描
# ==============================================================================

def scan_edf_files(data_root: str, split: str = None) -> List[Dict]:
    """
    扫描TUSZ数据集目录，找出所有EDF文件
    
    TUSZ目录结构:
    edf/
    ├── train/
    │   └── {patient_id}/
    │       └── {session_id}/
    │           └── {montage}/
    │               ├── {file_id}.edf
    │               ├── {file_id}.csv
    │               └── {file_id}.csv_bi
    ├── dev/
    └── eval/
    
    Args:
        data_root: 数据集根目录 (edf目录)
        split: 指定扫描的划分，None则扫描全部
        
    Returns:
        文件信息列表
    """
    data_root = Path(data_root)
    
    if split:
        splits = [split]
    else:
        splits = ['train', 'dev', 'eval']
    
    files = []
    
    for split_name in splits:
        split_dir = data_root / split_name
        if not split_dir.exists():
            logger.warning(f"划分目录不存在: {split_dir}")
            continue
        
        # 遍历患者目录
        for patient_dir in split_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            
            # 遍历session目录
            for session_dir in patient_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                
                # 遍历montage目录
                for montage_dir in session_dir.iterdir():
                    if not montage_dir.is_dir():
                        continue
                    
                    montage = detect_montage_type(str(montage_dir))
                    
                    # 找出EDF文件
                    for edf_file in montage_dir.glob('*.edf'):
                        file_id = edf_file.stem
                        
                        # 计算相对路径
                        rel_path = edf_file.relative_to(data_root)
                        
                        files.append({
                            'patient_id': patient_id,
                            'session_id': session_id,
                            'file_id': file_id,
                            'edf_path': str(rel_path),
                            'edf_full_path': str(edf_file),
                            'split': split_name,
                            'montage': montage,
                        })
    
    logger.info(f"扫描到 {len(files)} 个EDF文件")
    return files


# ==============================================================================
# 标注处理
# ==============================================================================

def process_edf_file(file_info: Dict, data_root: str) -> Optional[ManifestEntry]:
    """
    处理单个EDF文件，提取标注信息并创建Manifest条目
    
    Args:
        file_info: 文件信息字典
        data_root: 数据集根目录
        
    Returns:
        ManifestEntry对象，如果处理失败则返回None
    """
    edf_path = Path(data_root) / file_info['edf_path']
    csv_path = edf_path.with_suffix('.csv')
    csv_bi_path = edf_path.with_name(edf_path.stem + '.csv_bi')
    
    # 默认值
    duration = 0.0
    has_seizure = False
    n_seizure_events = 0
    seizure_types = ''
    sz_starts = ''
    sz_ends = ''
    total_seizure_duration = 0.0
    onset_channels = ''
    onset_regions = ''
    hemisphere = 'U'
    
    try:
        # 解析.csv标注（每通道多类别）
        if csv_path.exists():
            csv_ann = parse_csv_annotation(str(csv_path))
            duration = csv_ann.duration
            has_seizure = csv_ann.has_seizure
            
            if has_seizure:
                # 提取发作事件
                events = extract_seizure_events(csv_ann)
                n_seizure_events = len(events)
                
                if events:
                    all_types = set()
                    per_event_types = []
                    all_starts = []
                    all_ends = []
                    per_event_onset_channels = []
                    all_onset_regions = set()
                    hemispheres = set()
                    
                    for event in events:
                        all_types.add(event.seizure_type)
                        per_event_types.append(event.seizure_type or 'seiz')
                        all_starts.append(f"{event.start_time:.4f}")
                        all_ends.append(f"{event.stop_time:.4f}")
                        per_event_onset_channels.append(
                            ','.join(sorted(event.onset_channels))
                        )
                        all_onset_regions.update(event.onset_regions)
                        hemispheres.add(event.hemisphere)
                        total_seizure_duration += event.duration
                    
                    # pipe-separated per-event seizure types
                    seizure_types = '|'.join(per_event_types)
                    sz_starts = ';'.join(all_starts)
                    sz_ends = ';'.join(all_ends)
                    # pipe-separated per-event onset channels
                    onset_channels = '|'.join(per_event_onset_channels)
                    onset_regions = ','.join(sorted(all_onset_regions))
                    
                    # 确定整体半球
                    if len(hemispheres) == 1:
                        hemisphere = list(hemispheres)[0]
                    elif 'B' in hemispheres or ('L' in hemispheres and 'R' in hemispheres):
                        hemisphere = 'B'
                    elif 'L' in hemispheres:
                        hemisphere = 'L'
                    elif 'R' in hemispheres:
                        hemisphere = 'R'
                    else:
                        hemisphere = 'M'
        
        # 如果没有.csv，尝试从.csv_bi获取基本信息
        elif csv_bi_path.exists():
            csv_bi_ann = parse_csv_bi_annotation(str(csv_bi_path))
            duration = csv_bi_ann.duration
            has_seizure = csv_bi_ann.has_seizure
            
            if has_seizure:
                seizure_events = csv_bi_ann.get_seizure_events()
                n_seizure_events = len(seizure_events)
                all_starts = []
                all_ends = []
                for event in seizure_events:
                    all_starts.append(f"{event.start_time:.4f}")
                    all_ends.append(f"{event.stop_time:.4f}")
                    total_seizure_duration += event.duration
                sz_starts = ';'.join(all_starts)
                sz_ends = ';'.join(all_ends)
                seizure_types = 'seiz'  # 二分类只有seiz
                # csv_bi没有通道级信息，无法确定onset
        
        else:
            logger.warning(f"未找到标注文件: {csv_path} 或 {csv_bi_path}")
            return None
        
    except Exception as e:
        logger.error(f"处理文件失败 {edf_path}: {e}")
        return None
    
    return ManifestEntry(
        patient_id=file_info['patient_id'],
        session_id=file_info['session_id'],
        file_id=file_info['file_id'],
        edf_path=file_info['edf_path'],
        split=file_info['split'],
        montage=file_info['montage'],
        duration=duration,
        has_seizure=has_seizure,
        n_seizure_events=n_seizure_events,
        seizure_types=seizure_types,
        sz_starts=sz_starts,
        sz_ends=sz_ends,
        total_seizure_duration=total_seizure_duration,
        onset_channels=onset_channels,
        onset_regions=onset_regions,
        hemisphere=hemisphere,
    )


# ==============================================================================
# Manifest生成
# ==============================================================================

def generate_manifest(
    data_root: str,
    output_path: str,
    split: str = None,
    limit: int = None,
    n_workers: int = 8,
    channel_list: List[str] = None
) -> List[ManifestEntry]:
    """
    生成TUSZ数据集的训练manifest
    
    Args:
        data_root: 数据集根目录
        output_path: 输出CSV文件路径
        split: 指定处理的划分，None则处理全部
        limit: 限制处理的文件数量（用于测试）
        n_workers: 并行工作线程数
        channel_list: 通道列表，默认使用18通道配置
        
    Returns:
        ManifestEntry列表
    """
    if channel_list is None:
        channel_list = BIPOLAR_CHANNELS  # 默认使用22通道TCP导联（与eeg_pipeline.py对齐）
    
    # 扫描文件
    logger.info(f"扫描数据目录: {data_root}")
    files = scan_edf_files(data_root, split=split)
    
    if limit:
        files = files[:limit]
        logger.info(f"限制处理 {limit} 个文件")
    
    # 处理文件
    entries = []
    
    logger.info(f"处理 {len(files)} 个文件...")
    
    if n_workers > 1:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_edf_file, f, data_root): f 
                for f in files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理文件"):
                entry = future.result()
                if entry:
                    entries.append(entry)
    else:
        # 单线程处理
        for f in tqdm(files, desc="处理文件"):
            entry = process_edf_file(f, data_root)
            if entry:
                entries.append(entry)
    
    logger.info(f"成功处理 {len(entries)} 个文件")
    
    # 统计信息
    seizure_files = sum(1 for e in entries if e.has_seizure)
    total_duration = sum(e.duration for e in entries)
    total_seizure_duration = sum(e.total_seizure_duration for e in entries)
    
    logger.info(f"统计信息:")
    logger.info(f"  总文件数: {len(entries)}")
    logger.info(f"  含发作文件: {seizure_files}")
    logger.info(f"  总时长: {total_duration/3600:.2f} 小时")
    logger.info(f"  发作总时长: {total_seizure_duration/3600:.2f} 小时")
    
    # 写入CSV
    if output_path:
        write_manifest(entries, output_path, channel_list)
        logger.info(f"Manifest已保存到: {output_path}")
    
    return entries


def write_manifest(
    entries: List[ManifestEntry],
    output_path: str,
    channel_list: List[str] = None
):
    """
    将Manifest条目写入CSV文件
    
    Args:
        entries: ManifestEntry列表
        output_path: 输出文件路径
        channel_list: 通道列表
    """
    if not entries:
        logger.warning("没有条目可写入")
        return
    
    if channel_list is None:
        channel_list = BIPOLAR_CHANNELS  # 默认使用22通道TCP导联（与eeg_pipeline.py对齐）
    
    # 获取字段名
    sample_dict = entries[0].to_dict(channel_list)
    fieldnames = list(sample_dict.keys())
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in entries:
            writer.writerow(entry.to_dict(channel_list))


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    加载Manifest CSV文件
    
    Args:
        manifest_path: Manifest文件路径
        
    Returns:
        字典列表
    """
    entries = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换类型
            row['duration'] = float(row['duration'])
            row['has_seizure'] = row['has_seizure'].lower() == 'true'
            row['n_seizure_events'] = int(row['n_seizure_events'])
            row['total_seizure_duration'] = float(row['total_seizure_duration'])
            entries.append(row)
    
    return entries


# ==============================================================================
# 过滤函数
# ==============================================================================

def filter_manifest_by_seizure(
    entries: List[ManifestEntry],
    seizure_only: bool = True
) -> List[ManifestEntry]:
    """
    按是否包含发作过滤Manifest
    
    Args:
        entries: ManifestEntry列表
        seizure_only: True只保留含发作的，False只保留无发作的
        
    Returns:
        过滤后的列表
    """
    if seizure_only:
        return [e for e in entries if e.has_seizure]
    else:
        return [e for e in entries if not e.has_seizure]


def filter_manifest_by_region(
    entries: List[ManifestEntry],
    regions: List[str]
) -> List[ManifestEntry]:
    """
    按onset脑区过滤Manifest
    
    Args:
        entries: ManifestEntry列表
        regions: 要包含的脑区列表
        
    Returns:
        过滤后的列表
    """
    regions_set = set(r.lower() for r in regions)
    
    filtered = []
    for entry in entries:
        if not entry.has_seizure:
            continue
        entry_regions = set(r.lower().strip() for r in entry.onset_regions.split(',') if r.strip())
        if entry_regions & regions_set:
            filtered.append(entry)
    
    return filtered


def filter_manifest_by_duration(
    entries: List[ManifestEntry],
    min_duration: float = 0,
    max_duration: float = float('inf')
) -> List[ManifestEntry]:
    """
    按文件时长过滤Manifest
    
    Args:
        entries: ManifestEntry列表
        min_duration: 最小时长（秒）
        max_duration: 最大时长（秒）
        
    Returns:
        过滤后的列表
    """
    return [e for e in entries if min_duration <= e.duration <= max_duration]


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TUSZ Manifest Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 生成完整manifest
    python -m TUSZ.generate_manifest --data-root F:/dataset/TUSZ/v2.0.3/edf --output manifest.csv
    
    # 只处理train集，限制100个文件
    python -m TUSZ.generate_manifest --data-root F:/dataset/TUSZ/v2.0.3/edf --split train --limit 100 --output train_manifest.csv
        """
    )
    
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default=TUSZ_CONFIG['data_root'],
        help='TUSZ数据集根目录 (edf目录)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='tusz_manifest.csv',
        help='输出manifest文件路径'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        choices=['train', 'dev', 'eval'],
        default=None,
        help='只处理指定的数据集划分'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='限制处理的文件数量（用于测试）'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='并行工作线程数'
    )
    parser.add_argument(
        '--channels',
        type=int,
        choices=[18, 22],
        default=22,
        help='使用的通道配置 (18或22通道)'
    )
    
    args = parser.parse_args()
    
    channel_list = BIPOLAR_CHANNELS if args.channels == 22 else BIPOLAR_CHANNELS_18
    
    generate_manifest(
        data_root=args.data_root,
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        n_workers=args.workers,
        channel_list=channel_list
    )


if __name__ == '__main__':
    main()
