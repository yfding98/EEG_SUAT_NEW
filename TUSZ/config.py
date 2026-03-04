#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ Dataset Configuration

配置文件，定义TUSZ数据处理的各种参数和映射关系。
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 数据集路径配置
# ==============================================================================

TUSZ_CONFIG = {
    # 数据集根目录
    'data_root': 'F:/dataset/TUSZ/v2.0.3/edf',

    # 预处理参数
    'target_fs': 200,           # 目标采样率
    'filter_low': 3,            # 高通截止频率 Hz
    'filter_high': 45,          # 低通截止频率 Hz
    'clip_std': 2,              # 幅值裁剪标准差倍数

    # 窗口参数
    'window_len': 20.0,         # 窗口长度（秒）
    'window_overlap': 0.5,      # 窗口重叠比例

    # 癫痫事件参数
    'min_seizure_duration': 5.0,    # 最小发作持续时间（秒）
    'onset_tolerance': 1.0,         # 判定onset通道的时间容差（秒）
    'pre_seizure_buffer': 5.0,      # 发作前缓冲时间（秒）
    'post_seizure_buffer': 5.0,     # 发作后缓冲时间（秒）

    # 输出配置
    'output_dir': 'F:/dataset/TUSZ/dyf_processed',
}


# ==============================================================================
# TCP双极导联定义 (22通道)
# 顺序与 data_preprocess/eeg_pipeline.py 的 TCP_PAIRS 完全一致
# ==============================================================================

# 标准TCP双极导联对（与 eeg_pipeline.py TCP_PAIRS 完全对齐）
BIPOLAR_PAIRS = [
    # 左颞链 (4对) - channels 0-3
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    # 右颞链 (4对) - channels 4-7
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    # 左副矢状链 (4对) - channels 8-11
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    # 右副矢状链 (4对) - channels 12-15
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    # 中央链 (6对) - channels 16-21
    ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'), ('T4', 'A2'),
]

# 生成双极通道名称列表（22通道，顺序与 eeg_pipeline.py 一致）
BIPOLAR_CHANNELS = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS]

# 18通道配置（去掉含A1/A2的中央链两端，保留16条通用链 + FZ-CZ/CZ-PZ）
BIPOLAR_PAIRS_18 = [
    # 左颞链 (4对)
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    # 右颞链 (4对)
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    # 左副矢状链 (4对)
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    # 右副矢状链 (4对)
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    # 中线链 (2对)
    ('FZ', 'CZ'), ('CZ', 'PZ'),
]

BIPOLAR_CHANNELS_18 = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS_18]


# ==============================================================================
# 脑区映射
# ==============================================================================

# 双极通道到脑区的映射
BIPOLAR_TO_REGION: Dict[str, str] = {
    # 左颞链
    'FP1-F7': 'frontal',
    'F7-T3':  'temporal',
    'T3-T5':  'temporal',
    'T5-O1':  'occipital',
    # 右颞链
    'FP2-F8': 'frontal',
    'F8-T4':  'temporal',
    'T4-T6':  'temporal',
    'T6-O2':  'occipital',
    # 左副矢状链
    'FP1-F3': 'frontal',
    'F3-C3':  'central',
    'C3-P3':  'parietal',
    'P3-O1':  'occipital',
    # 右副矢状链
    'FP2-F4': 'frontal',
    'F4-C4':  'central',
    'C4-P4':  'parietal',
    'P4-O2':  'occipital',
    # 中央链
    'A1-T3':  'temporal',
    'T3-C3':  'central',
    'C3-CZ':  'central',
    'CZ-C4':  'central',
    'C4-T4':  'central',
    'T4-A2':  'temporal',
    # 中线链（18通道配置）
    'FZ-CZ':  'central',
    'CZ-PZ':  'parietal',
}

# 双极通道到半球的映射
BIPOLAR_TO_HEMISPHERE: Dict[str, str] = {
    # 左侧 (L)
    'FP1-F7': 'L', 'F7-T3': 'L', 'T3-T5': 'L', 'T5-O1': 'L',
    'FP1-F3': 'L', 'F3-C3': 'L', 'C3-P3': 'L', 'P3-O1': 'L',
    'A1-T3':  'L', 'T3-C3': 'L',
    # 右侧 (R)
    'FP2-F8': 'R', 'F8-T4': 'R', 'T4-T6': 'R', 'T6-O2': 'R',
    'FP2-F4': 'R', 'F4-C4': 'R', 'C4-P4': 'R', 'P4-O2': 'R',
    'C4-T4':  'R', 'T4-A2': 'R',
    # 中线 (M)
    'C3-CZ':  'M', 'CZ-C4': 'M',
    'FZ-CZ':  'M', 'CZ-PZ': 'M',
}

# 脑区名称列表
BRAIN_REGIONS = ['frontal', 'temporal', 'central', 'parietal', 'occipital']
REGION_TO_IDX = {r: i for i, r in enumerate(BRAIN_REGIONS)}

# 半球名称列表
HEMISPHERES = ['L', 'R', 'B', 'M']  # Left, Right, Bilateral, Midline
HEMI_TO_IDX = {h: i for i, h in enumerate(HEMISPHERES)}


# ==============================================================================
# 标准19通道单极导联（用于双极转换前）
# ==============================================================================

STANDARD_19_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2'
]

# 通道名映射（处理各种变体）
CHANNEL_NAME_MAP = {
    # 大小写变体
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ', 'Oz': 'OZ',
    # 10-10到10-20系统映射
    'T7': 'T3', 'T8': 'T4',
    'P7': 'T5', 'P8': 'T6',
    # EDF中常见的命名格式
    'EEG FP1-REF': 'FP1', 'EEG FP2-REF': 'FP2',
    'EEG F7-REF': 'F7',   'EEG F3-REF': 'F3',   'EEG FZ-REF': 'FZ',
    'EEG F4-REF': 'F4',   'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3',   'EEG C3-REF': 'C3',   'EEG CZ-REF': 'CZ',
    'EEG C4-REF': 'C4',   'EEG T4-REF': 'T4',
    'EEG T5-REF': 'T5',   'EEG P3-REF': 'P3',   'EEG PZ-REF': 'PZ',
    'EEG P4-REF': 'P4',   'EEG T6-REF': 'T6',
    'EEG O1-REF': 'O1',   'EEG O2-REF': 'O2',
    'EEG A1-REF': 'A1',   'EEG A2-REF': 'A2',
    # LE参考
    'EEG FP1-LE': 'FP1', 'EEG FP2-LE': 'FP2',
    'EEG F7-LE': 'F7',   'EEG F3-LE': 'F3',   'EEG FZ-LE': 'FZ',
    'EEG F4-LE': 'F4',   'EEG F8-LE': 'F8',
    'EEG T3-LE': 'T3',   'EEG C3-LE': 'C3',   'EEG CZ-LE': 'CZ',
    'EEG C4-LE': 'C4',   'EEG T4-LE': 'T4',
    'EEG T5-LE': 'T5',   'EEG P3-LE': 'P3',   'EEG PZ-LE': 'PZ',
    'EEG P4-LE': 'P4',   'EEG T6-LE': 'T6',
    'EEG O1-LE': 'O1',   'EEG O2-LE': 'O2',
    'EEG A1-LE': 'A1',   'EEG A2-LE': 'A2',
}


# ==============================================================================
# 癫痫发作类型
# ==============================================================================

SEIZURE_TYPES = {
    'seiz': 'Generic Seizure',
    'fnsz': 'Focal Non-specific Seizure',
    'gnsz': 'Generalized Non-specific Seizure',
    'spsz': 'Simple Partial Seizure',
    'cpsz': 'Complex Partial Seizure',
    'absz': 'Absence Seizure',
    'tnsz': 'Tonic Seizure',
    'cnsz': 'Clonic Seizure',
    'tcsz': 'Tonic-Clonic Seizure',
    'atsz': 'Atonic Seizure',
    'mysz': 'Myoclonic Seizure',
    'nesz': 'Non-Epileptic Seizure',
}

ALL_SEIZURE_LABELS = set(SEIZURE_TYPES.keys()) | {'sz'}
BACKGROUND_LABEL = 'bckg'


# ==============================================================================
# 辅助函数
# ==============================================================================

def normalize_channel_name(name: str) -> str:
    """标准化通道名称"""
    name = name.strip().upper()

    if name in STANDARD_19_CHANNELS or name in ['A1', 'A2']:
        return name

    if name in CHANNEL_NAME_MAP:
        return CHANNEL_NAME_MAP[name]

    name_orig = name
    for key, val in CHANNEL_NAME_MAP.items():
        if key.upper() == name_orig:
            return val

    for prefix in ['EEG ', 'EEG-']:
        if name.startswith(prefix):
            stripped = name[len(prefix):]
            for suffix in ['-REF', '-LE', '-AR', '-AVG']:
                if stripped.endswith(suffix):
                    stripped = stripped[:-len(suffix)]
            if stripped in STANDARD_19_CHANNELS or stripped in ['A1', 'A2']:
                return stripped

    return name


def get_bipolar_index(channel_name: str) -> int:
    """获取双极通道的索引"""
    name = channel_name.upper().strip()
    try:
        return BIPOLAR_CHANNELS.index(name)
    except ValueError:
        return -1


def get_region_from_bipolar(channel_name: str) -> str:
    """从双极通道获取脑区"""
    return BIPOLAR_TO_REGION.get(channel_name.upper().strip(), 'unknown')


def get_hemisphere_from_bipolar(channel_name: str) -> str:
    """从双极通道获取半球"""
    return BIPOLAR_TO_HEMISPHERE.get(channel_name.upper().strip(), 'U')


def is_seizure_label(label: str) -> bool:
    """判断是否为发作标签"""
    label = label.lower().strip()
    return label in ALL_SEIZURE_LABELS or label.endswith('sz')


if __name__ == '__main__':
    print("TUSZ Configuration:")
    print(f"  Data root: {TUSZ_CONFIG['data_root']}")
    print(f"  Target sampling rate: {TUSZ_CONFIG['target_fs']} Hz")
    print(f"  Filter band: {TUSZ_CONFIG['filter_low']}-{TUSZ_CONFIG['filter_high']} Hz")
    print(f"\nBipolar channels ({len(BIPOLAR_CHANNELS)}, eeg_pipeline.py顺序):")
    for i, ch in enumerate(BIPOLAR_CHANNELS):
        print(f"  {i:2d}: {ch} -> {BIPOLAR_TO_REGION.get(ch, 'unknown')} ({BIPOLAR_TO_HEMISPHERE.get(ch, 'U')})")
