#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ Annotation Parser

解析TUSZ数据集的标注文件：
- .csv 文件：每通道多类别癫痫发作标注
- .csv_bi 文件：二分类标注（seizure/background）

从标注中提取SOZ（发作起始区）信息。
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging

from config import (
    BIPOLAR_CHANNELS, 
    BIPOLAR_TO_REGION, 
    BIPOLAR_TO_HEMISPHERE,
    BRAIN_REGIONS,
    is_seizure_label,
    BACKGROUND_LABEL,
    TUSZ_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class ChannelAnnotation:
    """单通道的标注事件"""
    channel: str
    start_time: float
    stop_time: float
    label: str
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.stop_time - self.start_time
    
    @property
    def is_seizure(self) -> bool:
        return is_seizure_label(self.label)


@dataclass
class FileAnnotation:
    """文件级别的标注信息"""
    file_path: str
    duration: float
    version: str = "csv_v1.0.0"
    montage_file: str = ""
    annotations: List[ChannelAnnotation] = field(default_factory=list)
    
    @property
    def has_seizure(self) -> bool:
        return any(ann.is_seizure for ann in self.annotations)
    
    def get_seizure_events(self) -> List[ChannelAnnotation]:
        """获取所有癫痫发作事件"""
        return [ann for ann in self.annotations if ann.is_seizure]
    
    def get_background_events(self) -> List[ChannelAnnotation]:
        """获取所有背景事件"""
        return [ann for ann in self.annotations if ann.label == BACKGROUND_LABEL]


@dataclass
class SeizureEvent:
    """癫痫发作事件（文件级别）"""
    start_time: float
    stop_time: float
    onset_channels: List[str]           # 最早出现发作的通道
    all_channels: List[str]             # 所有涉及的通道
    seizure_type: str                   # 发作类型
    onset_regions: List[str]            # SOZ脑区
    hemisphere: str                     # 半球 (L/R/B/M)
    
    @property
    def duration(self) -> float:
        return self.stop_time - self.start_time


# ==============================================================================
# CSV标注解析
# ==============================================================================

def parse_csv_header(lines: List[str]) -> Dict[str, str]:
    """解析CSV文件头部元信息"""
    metadata = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # 解析格式: # key = value
            match = re.match(r'#\s*(\w+)\s*=\s*(.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                metadata[key] = value
    return metadata


def parse_csv_annotation(csv_path: str) -> FileAnnotation:
    """
    解析TUSZ的.csv标注文件（每通道多类别标注）
    
    文件格式示例:
    # version = csv_v1.0.0
    # bname = aaaaaaac_s001_t000
    # duration = 301.00 secs
    # montage_file = $NEDC_NFC/lib/nedc_eas_default_montage.txt
    #
    channel,start_time,stop_time,label,confidence
    FP1-F7,0.0000,36.8868,bckg,1.0000
    FP1-F7,36.8868,183.3055,cpsz,1.0000
    ...
    
    Args:
        csv_path: .csv文件路径
        
    Returns:
        FileAnnotation对象
    """
    csv_path = Path(csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析头部元信息
    header_lines = [l for l in lines if l.strip().startswith('#')]
    metadata = parse_csv_header(header_lines)
    
    # 提取duration
    duration_str = metadata.get('duration', '0')
    duration_match = re.search(r'([\d.]+)', duration_str)
    duration = float(duration_match.group(1)) if duration_match else 0.0
    
    # 解析标注行
    annotations = []
    data_started = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 跳过表头
        if line.lower().startswith('channel,'):
            data_started = True
            continue
        
        if not data_started:
            # 如果没遇到表头，也尝试解析（兼容性）
            if ',' in line:
                data_started = True
            else:
                continue
        
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                channel = parts[0].strip().upper()
                start_time = float(parts[1])
                stop_time = float(parts[2])
                label = parts[3].strip().lower()
                confidence = float(parts[4]) if len(parts) > 4 else 1.0
                
                annotations.append(ChannelAnnotation(
                    channel=channel,
                    start_time=start_time,
                    stop_time=stop_time,
                    label=label,
                    confidence=confidence
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"解析标注行失败: {line}, 错误: {e}")
    
    return FileAnnotation(
        file_path=str(csv_path),
        duration=duration,
        version=metadata.get('version', 'csv_v1.0.0'),
        montage_file=metadata.get('montage_file', ''),
        annotations=annotations
    )


def parse_csv_bi_annotation(csv_bi_path: str) -> FileAnnotation:
    """
    解析TUSZ的.csv_bi标注文件（二分类标注）
    
    文件格式示例:
    # version = csv_v1.0.0
    # bname = aaaaaaac_s001_t000
    # duration = 301.0000 secs
    # montage_file = nedc_eas_default_montage.txt
    #
    channel,start_time,stop_time,label,confidence
    TERM,36.8868,237.2101,seiz,1.0000
    
    Args:
        csv_bi_path: .csv_bi文件路径
        
    Returns:
        FileAnnotation对象（只包含文件级别的seizure/background标注）
    """
    csv_bi_path = Path(csv_bi_path)
    
    with open(csv_bi_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析头部元信息
    header_lines = [l for l in lines if l.strip().startswith('#')]
    metadata = parse_csv_header(header_lines)
    
    # 提取duration
    duration_str = metadata.get('duration', '0')
    duration_match = re.search(r'([\d.]+)', duration_str)
    duration = float(duration_match.group(1)) if duration_match else 0.0
    
    # 解析标注行
    annotations = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('channel,'):
            continue
        
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                channel = parts[0].strip().upper()  # 通常是 "TERM"
                start_time = float(parts[1])
                stop_time = float(parts[2])
                label = parts[3].strip().lower()
                confidence = float(parts[4]) if len(parts) > 4 else 1.0
                
                annotations.append(ChannelAnnotation(
                    channel=channel,
                    start_time=start_time,
                    stop_time=stop_time,
                    label=label,
                    confidence=confidence
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"解析标注行失败: {line}, 错误: {e}")
    
    return FileAnnotation(
        file_path=str(csv_bi_path),
        duration=duration,
        version=metadata.get('version', 'csv_v1.0.0'),
        montage_file=metadata.get('montage_file', ''),
        annotations=annotations
    )


# ==============================================================================
# SOZ提取逻辑
# ==============================================================================

def extract_seizure_onset_channels(
    annotations: List[ChannelAnnotation],
    tolerance: float = None
) -> Tuple[List[str], float]:
    """
    从每通道标注中提取最早显示癫痫发作的通道
    
    逻辑：
    1. 找出所有seizure事件的最早开始时间
    2. 在tolerance秒内开始的通道都被认为是onset通道
    
    Args:
        annotations: 通道标注列表
        tolerance: 时间容差（秒），默认使用配置值
        
    Returns:
        (onset_channels, earliest_onset_time)
    """
    if tolerance is None:
        tolerance = TUSZ_CONFIG.get('onset_tolerance', 1.0)
    
    # 筛选癫痫发作事件
    seizure_events = [ann for ann in annotations if ann.is_seizure]
    
    if not seizure_events:
        return [], -1.0
    
    # 找到最早的开始时间
    earliest_time = min(ann.start_time for ann in seizure_events)
    
    # 找出在tolerance内开始的通道
    onset_channels = set()
    for ann in seizure_events:
        if ann.start_time <= earliest_time + tolerance:
            onset_channels.add(ann.channel)
    
    return list(onset_channels), earliest_time


def extract_seizure_events(file_annotation: FileAnnotation) -> List[SeizureEvent]:
    """
    从文件标注中提取癫痫发作事件
    
    将每通道的标注聚合成文件级别的发作事件，
    并确定每个事件的onset通道、脑区和半球。
    
    Args:
        file_annotation: 文件标注对象
        
    Returns:
        SeizureEvent列表
    """
    seizure_anns = file_annotation.get_seizure_events()
    
    if not seizure_anns:
        return []
    
    # 按起始时间分组，找出不重叠的发作事件
    # 首先按开始时间排序
    seizure_anns.sort(key=lambda x: x.start_time)
    
    # 使用区间合并找出独立的发作事件
    events = []
    current_start = seizure_anns[0].start_time
    current_end = seizure_anns[0].stop_time
    current_annotations = [seizure_anns[0]]
    
    for ann in seizure_anns[1:]:
        # 如果当前标注与之前的事件重叠
        if ann.start_time <= current_end:
            current_end = max(current_end, ann.stop_time)
            current_annotations.append(ann)
        else:
            # 保存之前的事件
            events.append(_create_seizure_event(
                current_start, current_end, current_annotations
            ))
            # 开始新事件
            current_start = ann.start_time
            current_end = ann.stop_time
            current_annotations = [ann]
    
    # 保存最后一个事件
    if current_annotations:
        events.append(_create_seizure_event(
            current_start, current_end, current_annotations
        ))
    
    return events


def _create_seizure_event(
    start_time: float,
    stop_time: float,
    annotations: List[ChannelAnnotation]
) -> SeizureEvent:
    """从标注创建SeizureEvent对象"""
    tolerance = TUSZ_CONFIG.get('onset_tolerance', 1.0)
    
    # 找最早开始时间
    earliest_time = min(ann.start_time for ann in annotations)
    
    # Onset通道（在tolerance内开始的）
    onset_channels = [
        ann.channel for ann in annotations 
        if ann.start_time <= earliest_time + tolerance
    ]
    onset_channels = list(set(onset_channels))
    
    # 所有涉及的通道
    all_channels = list(set(ann.channel for ann in annotations))
    
    # 主要的发作类型（取出现最多的）
    from collections import Counter
    types = [ann.label for ann in annotations]
    seizure_type = Counter(types).most_common(1)[0][0]
    
    # 确定onset脑区
    onset_regions = []
    for ch in onset_channels:
        region = BIPOLAR_TO_REGION.get(ch, None)
        if region and region not in onset_regions:
            onset_regions.append(region)
    
    # 确定半球
    hemispheres = set()
    for ch in onset_channels:
        hemi = BIPOLAR_TO_HEMISPHERE.get(ch, None)
        if hemi:
            hemispheres.add(hemi)
    
    if len(hemispheres) == 0:
        hemisphere = 'U'  # Unknown
    elif hemispheres == {'L'}:
        hemisphere = 'L'
    elif hemispheres == {'R'}:
        hemisphere = 'R'
    elif 'L' in hemispheres and 'R' in hemispheres:
        hemisphere = 'B'  # Bilateral
    elif hemispheres == {'M'}:
        hemisphere = 'M'
    else:
        # 混合情况
        if 'L' in hemispheres:
            hemisphere = 'L'
        elif 'R' in hemispheres:
            hemisphere = 'R'
        else:
            hemisphere = 'M'
    
    return SeizureEvent(
        start_time=start_time,
        stop_time=stop_time,
        onset_channels=onset_channels,
        all_channels=all_channels,
        seizure_type=seizure_type,
        onset_regions=onset_regions,
        hemisphere=hemisphere
    )


def get_soz_channel_labels(
    onset_channels: List[str],
    channel_list: List[str] = None
) -> List[int]:
    """
    生成SOZ通道标签向量
    
    Args:
        onset_channels: SOZ通道列表
        channel_list: 目标通道列表，默认使用BIPOLAR_CHANNELS
        
    Returns:
        二值标签列表，1表示SOZ通道
    """
    if channel_list is None:
        channel_list = BIPOLAR_CHANNELS
    
    onset_set = set(ch.upper() for ch in onset_channels)
    
    labels = []
    for ch in channel_list:
        if ch.upper() in onset_set:
            labels.append(1)
        else:
            labels.append(0)
    
    return labels


def get_soz_region_labels(onset_regions: List[str]) -> List[int]:
    """
    生成SOZ脑区标签向量
    
    Args:
        onset_regions: SOZ脑区列表
        
    Returns:
        5维二值标签 [frontal, temporal, central, parietal, occipital]
    """
    onset_set = set(r.lower() for r in onset_regions)
    
    labels = []
    for region in BRAIN_REGIONS:
        if region in onset_set:
            labels.append(1)
        else:
            labels.append(0)
    
    return labels


# ==============================================================================
# 便捷函数
# ==============================================================================

def parse_annotation_pair(edf_path: str) -> Tuple[FileAnnotation, FileAnnotation]:
    """
    解析EDF文件对应的两个标注文件
    
    Args:
        edf_path: EDF文件路径
        
    Returns:
        (csv_annotation, csv_bi_annotation)
    """
    edf_path = Path(edf_path)
    csv_path = edf_path.with_suffix('.csv')
    csv_bi_path = edf_path.with_name(edf_path.stem + '.csv_bi')
    
    csv_ann = None
    csv_bi_ann = None
    
    if csv_path.exists():
        csv_ann = parse_csv_annotation(str(csv_path))
    else:
        logger.warning(f"未找到CSV标注文件: {csv_path}")
    
    if csv_bi_path.exists():
        csv_bi_ann = parse_csv_bi_annotation(str(csv_bi_path))
    else:
        logger.warning(f"未找到CSV_BI标注文件: {csv_bi_path}")
    
    return csv_ann, csv_bi_ann


def analyze_file_soz(edf_path: str) -> Dict:
    """
    分析单个EDF文件的SOZ信息
    
    Args:
        edf_path: EDF文件路径
        
    Returns:
        包含SOZ分析结果的字典
    """
    csv_ann, csv_bi_ann = parse_annotation_pair(edf_path)
    
    result = {
        'edf_path': edf_path,
        'duration': 0.0,
        'has_seizure': False,
        'seizure_events': [],
        'onset_channels': [],
        'onset_regions': [],
        'hemisphere': 'U',
        'soz_channel_labels': [0] * len(BIPOLAR_CHANNELS),
        'soz_region_labels': [0] * len(BRAIN_REGIONS),
    }
    
    if csv_ann is None:
        return result
    
    result['duration'] = csv_ann.duration
    result['has_seizure'] = csv_ann.has_seizure
    
    if not csv_ann.has_seizure:
        return result
    
    # 提取发作事件
    events = extract_seizure_events(csv_ann)
    result['seizure_events'] = events
    
    if events:
        # 合并所有事件的onset信息
        all_onset_channels = set()
        all_onset_regions = set()
        hemispheres = set()
        
        for event in events:
            all_onset_channels.update(event.onset_channels)
            all_onset_regions.update(event.onset_regions)
            hemispheres.add(event.hemisphere)
        
        result['onset_channels'] = list(all_onset_channels)
        result['onset_regions'] = list(all_onset_regions)
        
        # 确定整体半球
        if len(hemispheres) == 1:
            result['hemisphere'] = list(hemispheres)[0]
        elif 'B' in hemispheres or ('L' in hemispheres and 'R' in hemispheres):
            result['hemisphere'] = 'B'
        else:
            result['hemisphere'] = list(hemispheres)[0]
        
        # 生成标签向量
        result['soz_channel_labels'] = get_soz_channel_labels(result['onset_channels'])
        result['soz_region_labels'] = get_soz_region_labels(result['onset_regions'])
    
    return result


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 测试模式：解析示例文件
        test_csv = "F:/dataset/TUSZ/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.csv"
        test_csv_bi = "F:/dataset/TUSZ/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.csv_bi"
        test_edf = "F:/dataset/TUSZ/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf"
        
        print("=" * 60)
        print("测试CSV标注解析")
        print("=" * 60)
        
        if Path(test_csv).exists():
            ann = parse_csv_annotation(test_csv)
            print(f"文件: {ann.file_path}")
            print(f"时长: {ann.duration}s")
            print(f"标注数量: {len(ann.annotations)}")
            print(f"包含发作: {ann.has_seizure}")
            
            seizure_events = ann.get_seizure_events()
            print(f"发作事件数: {len(seizure_events)}")
            
            if seizure_events:
                onset_channels, earliest = extract_seizure_onset_channels(ann.annotations)
                print(f"Onset时间: {earliest:.2f}s")
                print(f"Onset通道: {onset_channels}")
        else:
            print(f"测试文件不存在: {test_csv}")
        
        print("\n" + "=" * 60)
        print("测试CSV_BI标注解析")
        print("=" * 60)
        
        if Path(test_csv_bi).exists():
            ann_bi = parse_csv_bi_annotation(test_csv_bi)
            print(f"文件: {ann_bi.file_path}")
            print(f"时长: {ann_bi.duration}s")
            print(f"标注数量: {len(ann_bi.annotations)}")
            for a in ann_bi.annotations:
                print(f"  {a.start_time:.2f}-{a.stop_time:.2f}: {a.label}")
        else:
            print(f"测试文件不存在: {test_csv_bi}")
        
        print("\n" + "=" * 60)
        print("测试SOZ分析")
        print("=" * 60)
        
        if Path(test_edf).exists():
            result = analyze_file_soz(test_edf)
            print(f"EDF: {result['edf_path']}")
            print(f"时长: {result['duration']}s")
            print(f"包含发作: {result['has_seizure']}")
            print(f"Onset通道: {result['onset_channels']}")
            print(f"Onset脑区: {result['onset_regions']}")
            print(f"半球: {result['hemisphere']}")
            print(f"通道标签: {sum(result['soz_channel_labels'])}/{len(result['soz_channel_labels'])} 个SOZ通道")
            print(f"脑区标签: {result['soz_region_labels']}")
        else:
            print(f"测试文件不存在: {test_edf}")
    
    else:
        print("用法: python -m TUSZ.parse_annotations --test")
