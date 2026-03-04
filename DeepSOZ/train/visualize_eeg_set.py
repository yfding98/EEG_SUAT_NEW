#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG脑区可视化脚本 - 支持.set格式 (EEGLAB格式)

功能：
1. 加载与train_onset_zone.py相同的manifest文件
2. 支持.edf和.set两种数据格式
3. 对于.set格式：
   - 数据根目录添加'processed'后缀
   - 文件名添加'_filtered_3_45_postICA_eye'后缀
   - 滤波范围为3-45Hz
4. 按脑区可视化通道数据
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.signal import resample

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, DataConfig
from dataset import (
    extract_standard_channels, bandpass_filter, 
    clip_amplitude, resample_signal, parse_seizure_times,
    parse_onset_zone_label, STANDARD_19_CHANNELS, STANDARD_21_CHANNELS, BRAIN_REGION_MAP,
    convert_to_bipolar, BIPOLAR_PAIRS_18, BIPOLAR_PAIRS_26, map_channel_name,
    parse_mask_segments, extract_segment_with_mask_removal  # mask_segments支持
)
from connectivity import compute_all_connectivity
from connectivity_viz import visualize_all_connectivity
from connectivity_viz_bipolar import visualize_bipolar_connectivity

# 尝试导入MNE用于读取.set文件
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 脑区到标准19通道索引的映射
REGION_TO_CHANNEL_IDX = {
    'frontal': [0, 1, 2, 3, 4, 5, 6],      # FP1, FP2, F7, F3, FZ, F4, F8
    'temporal': [7, 11, 12, 16],            # T3, T4, T5, T6
    'central': [8, 9, 10],                  # C3, CZ, C4
    'parietal': [13, 14, 15],               # P3, PZ, P4
    'occipital': [17, 18]                   # O1, O2
}

# 脑区对应的通道名称
REGION_CHANNEL_NAMES = {
    'frontal': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'central': ['C3', 'CZ', 'C4'],
    'parietal': ['P3', 'PZ', 'P4'],
    'occipital': ['O1', 'O2']
}

# ==============================================================================
# 双极导联五链区域定义（标准18通道 - 19电极）
# ==============================================================================

BIPOLAR_CHAIN_TO_CHANNEL_IDX = {
    'left_temporal': [0, 1, 2, 3],
    'right_temporal': [4, 5, 6, 7],
    'left_parasagittal': [8, 9, 10, 11],
    'right_parasagittal': [12, 13, 14, 15],
    'midline': [16, 17]
}

BIPOLAR_CHAIN_CHANNEL_NAMES = {
    'left_temporal': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1'],
    'right_temporal': ['FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
    'left_parasagittal': ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
    'right_parasagittal': ['FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'midline': ['FZ-CZ', 'CZ-PZ']
}

BIPOLAR_CHAIN_DISPLAY_NAMES = {
    'left_temporal': '左颞链 (L-Temp)',
    'right_temporal': '右颞链 (R-Temp)',
    'left_parasagittal': '左副矢状链 (L-Para)',
    'right_parasagittal': '右副矢状链 (R-Para)',
    'midline': '中线链 (Midline)'
}

# ==============================================================================
# 双极导联五脑区定义（26通道 - 21电极含蝶骨电极）
# ==============================================================================

BIPOLAR_CHAIN_TO_CHANNEL_IDX_26 = {
    'left_frontal': [0, 1, 2, 3],           # 4通道
    'left_temporal': [4, 5, 6, 7, 8, 9],     # 6通道
    'parietal': [10, 11, 12, 13, 14, 15],    # 6通道
    'right_frontal': [16, 17, 18, 19],       # 4通道
    'right_temporal': [20, 21, 22, 23, 24, 25]  # 6通道
}

BIPOLAR_CHAIN_CHANNEL_NAMES_26 = {
    'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
    'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
    'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
    'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
    'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2']
}

BIPOLAR_CHAIN_DISPLAY_NAMES_26 = {
    'left_frontal': '左额 (L-Frontal)',
    'left_temporal': '左颞 (L-Temporal)',
    'parietal': '顶叶 (Parietal)',
    'right_frontal': '右额 (R-Frontal)',
    'right_temporal': '右颞 (R-Temporal)'
}

# 单极电极到双极通道的映射（用于SOZ标记）- 21电极5脑区版本
# 根据用户定义的脑区划分:
# 左额: FP1-F7, FP1-F3, F7-F3, F3-FZ
# 左颞: F7-SPHL, SPHL-T3, T3-T5, T5-O1, T3-C3, T5-P3
# 顶叶: FZ-CZ, C3-CZ, P3-PZ, CZ-PZ, CZ-C4, PZ-P4
# 右额: FP2-F4, FP2-F8, F4-F8, FZ-F4
# 右颞: F8-SPHR, SPHR-T4, C4-T4, T4-T6, P4-T6, T6-O2

UNIPOLAR_TO_BIPOLAR_MAP = {
    # 左额相关电极
    'FP1': [('FP1-F7', 'left_frontal'), ('FP1-F3', 'left_frontal')],
    'F7': [('FP1-F7', 'left_frontal'), ('F7-F3', 'left_frontal'), ('F7-SPHL', 'left_temporal')],
    'F3': [('FP1-F3', 'left_frontal'), ('F7-F3', 'left_frontal'), ('F3-FZ', 'left_frontal')],
    
    # 左颞相关电极
    'SPHL': [('F7-SPHL', 'left_temporal'), ('SPHL-T3', 'left_temporal')],
    'T3': [('SPHL-T3', 'left_temporal'), ('T3-T5', 'left_temporal'), ('T3-C3', 'left_temporal')],
    'T5': [('T3-T5', 'left_temporal'), ('T5-O1', 'left_temporal'), ('T5-P3', 'left_temporal')],
    'O1': [('T5-O1', 'left_temporal')],
    
    # 顶叶相关电极
    'FZ': [('F3-FZ', 'left_frontal'), ('FZ-CZ', 'parietal'), ('FZ-F4', 'right_frontal')],
    'CZ': [('FZ-CZ', 'parietal'), ('C3-CZ', 'parietal'), ('CZ-PZ', 'parietal'), ('CZ-C4', 'parietal')],
    'C3': [('T3-C3', 'left_temporal'), ('C3-CZ', 'parietal')],
    'C4': [('CZ-C4', 'parietal'), ('C4-T4', 'right_temporal')],
    'P3': [('T5-P3', 'left_temporal'), ('P3-PZ', 'parietal')],
    'PZ': [('P3-PZ', 'parietal'), ('CZ-PZ', 'parietal'), ('PZ-P4', 'parietal')],
    'P4': [('PZ-P4', 'parietal'), ('P4-T6', 'right_temporal')],
    
    # 右额相关电极
    'FP2': [('FP2-F4', 'right_frontal'), ('FP2-F8', 'right_frontal')],
    'F4': [('FP2-F4', 'right_frontal'), ('F4-F8', 'right_frontal'), ('FZ-F4', 'right_frontal')],
    'F8': [('FP2-F8', 'right_frontal'), ('F4-F8', 'right_frontal'), ('F8-SPHR', 'right_temporal')],
    
    # 右颞相关电极
    'SPHR': [('F8-SPHR', 'right_temporal'), ('SPHR-T4', 'right_temporal')],
    'T4': [('SPHR-T4', 'right_temporal'), ('C4-T4', 'right_temporal'), ('T4-T6', 'right_temporal')],
    'T6': [('T4-T6', 'right_temporal'), ('P4-T6', 'right_temporal'), ('T6-O2', 'right_temporal')],
    'O2': [('T6-O2', 'right_temporal')],
}




def get_soz_bipolar_info(soz_channels: set) -> Tuple[set, set]:
    """
    根据SOZ单极电极获取相关的双极通道和链
    """
    soz_chains = set()
    soz_bipolar_channels = set()
    
    for ch in soz_channels:
        ch_upper = ch.upper()
        if ch_upper in UNIPOLAR_TO_BIPOLAR_MAP:
            for bipolar_ch, chain_name in UNIPOLAR_TO_BIPOLAR_MAP[ch_upper]:
                soz_chains.add(chain_name)
                soz_bipolar_channels.add(bipolar_ch)
    
    return soz_chains, soz_bipolar_channels


def read_set_file(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    读取EEGLAB .set格式文件
    
    Args:
        filepath: .set文件路径
    
    Returns:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        ch_names: 通道名称列表
    """
    if not HAS_MNE:
        raise ImportError("需要安装MNE库来读取.set文件: pip install mne")
    
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='ERROR')
        data = raw.get_data()
        fs = raw.info['sfreq']
        ch_names = raw.ch_names
        return data, fs, ch_names
    except Exception as e:
        raise RuntimeError(f"无法读取.set文件 {filepath}: {e}")


def normalize_data(data: np.ndarray, global_min: float = None, global_max: float = None) -> np.ndarray:
    """
    对数据进行归一化到 [0, 1]
    
    Args:
        data: 输入数据
        global_min: 全局最小值（如果为None，则使用data自身的最小值）
        global_max: 全局最大值（如果为None，则使用data自身的最大值）
    
    Returns:
        归一化后的数据
    """
    min_val = global_min if global_min is not None else np.min(data)
    max_val = global_max if global_max is not None else np.max(data)
    if max_val - min_val > 1e-10:
        return (data - min_val) / (max_val - min_val)
    return np.zeros_like(data)


def create_heatmap_image(data: np.ndarray, cmap: str = 'RdBu_r') -> np.ndarray:
    """
    将一维EEG数据转换为RGB热力图图像
    
    Args:
        data: 已归一化的1D数据 (n_samples,)
        cmap: colormap名称
    
    Returns:
        RGB图像 (height, n_samples, 3)，height固定为50像素
    """
    height = 50
    colormap = cm.get_cmap(cmap)
    
    # 创建热力图
    rgba = colormap(data)[:, :3]  # (n_samples, 3)
    
    # 扩展为2D图像
    heatmap = np.tile(rgba, (height, 1, 1))  # (height, n_samples, 3)
    
    return heatmap


def plot_waveform(
    region_data: np.ndarray,
    channel_names: List[str],
    fs: float,
    title: str,
    save_path: str,
    global_min: float = None,
    global_max: float = None,
    soz_channels: set = None
):
    """
    绘制脑区通道的原始波形图
    
    Args:
        region_data: (n_channels, n_samples) 该脑区的EEG数据
        channel_names: 通道名称列表
        fs: 采样率
        title: 图标题
        save_path: 保存路径
        global_min: 全局最小值（用于所有通道统一归一化）
        global_max: 全局最大值（用于所有通道统一归一化）
        soz_channels: SOZ通道名称集合（这些通道的Y轴标签显示为红色）
    """
    n_channels, n_samples = region_data.shape
    time = np.arange(n_samples) / fs
    
    if soz_channels is None:
        soz_channels = set()
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        # 使用全局归一化
        normalized_data = normalize_data(region_data[i], global_min, global_max)
        ax.plot(time, normalized_data, 'b-', linewidth=0.5)
        
        # 判断是否为SOZ通道（红色标签）
        is_soz = ch_name.upper() in {ch.upper() for ch in soz_channels}
        label_color = 'red' if is_soz else 'black'
        ax.set_ylabel(ch_name, fontsize=10, rotation=0, ha='right', va='center',
                     color=label_color, fontweight='bold' if is_soz else 'normal')
        
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 0.5, 1.0])
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"保存波形图: {save_path}")


def plot_heatmap(
    region_data: np.ndarray,
    channel_names: List[str],
    fs: float,
    title: str,
    save_path: str,
    global_min: float = None,
    global_max: float = None,
    soz_channels: set = None
):
    """
    绘制脑区通道的RGB热力图
    
    Args:
        region_data: (n_channels, n_samples) 该脑区的EEG数据
        channel_names: 通道名称列表
        fs: 采样率
        title: 图标题
        save_path: 保存路径
        global_min: 全局最小值（用于所有通道统一归一化）
        global_max: 全局最大值（用于所有通道统一归一化）
        soz_channels: SOZ通道名称集合（这些通道的Y轴标签显示为红色）
    """
    n_channels, n_samples = region_data.shape
    
    if soz_channels is None:
        soz_channels = set()
    
    # 为每个通道创建热力图（使用全局归一化）
    heatmaps = []
    for i in range(n_channels):
        normalized_data = normalize_data(region_data[i], global_min, global_max)
        heatmap = create_heatmap_image(normalized_data)
        heatmaps.append(heatmap)
    
    # 计算图像大小
    single_height = 50
    gap = 10
    total_height = n_channels * single_height + (n_channels - 1) * gap
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 0.8 * n_channels + 1), 
                              gridspec_kw={'hspace': 0.3})
    if n_channels == 1:
        axes = [axes]
    
    time_extent = [0, n_samples / fs]
    
    for i, (ax, ch_name, heatmap) in enumerate(zip(axes, channel_names, heatmaps)):
        ax.imshow(heatmap, aspect='auto', extent=[time_extent[0], time_extent[1], 0, 1])
        
        # 判断是否为SOZ通道（红色标签）
        is_soz = ch_name.upper() in {ch.upper() for ch in soz_channels}
        label_color = 'red' if is_soz else 'black'
        ax.set_ylabel(ch_name, fontsize=10, rotation=0, ha='right', va='center',
                     color=label_color, fontweight='bold' if is_soz else 'normal')
        
        ax.set_yticks([])
        ax.set_xlim(time_extent)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"保存热力图: {save_path}")


def find_data_file(
    row: pd.Series, 
    data_roots: List[str], 
    file_format: str = 'edf'
) -> Optional[str]:
    """
    查找数据文件路径
    
    Args:
        row: CSV行数据
        data_roots: 数据根目录列表
        file_format: 文件格式 'edf' 或 'set'
    
    Returns:
        找到的文件路径，未找到返回None
    """
    import re
    loc = row.get('loc', '')
    
    if pd.notna(loc) and loc:
        for root in data_roots:
            # 根据格式调整路径
            if file_format == 'set':
                # 添加 'processed' 后缀到数据根目录
                root_processed = str(root) + '_processed'
                
                # 获取原始文件路径（不含扩展名）
                loc_path = Path(loc)
                base_name = loc_path.stem  # 不含扩展名的文件名
                parent_dir = loc_path.parent
                
                # 添加后缀
                new_filename = f"{base_name}_filtered_3_45_postICA_eye.set"
                new_loc = parent_dir / new_filename
                
                full_path = Path(root_processed) / new_loc
                if full_path.exists():
                    return str(full_path)
                
                # 尝试直接在目录中查找
                if '\\' in loc or '/' in loc:
                    fname = f"{base_name}_filtered_3_45_postICA_eye.set"
                    for set_file in Path(root_processed).rglob(fname):
                        return str(set_file)
            else:
                # EDF格式，使用原始路径
                full_path = Path(root) / loc
                if full_path.exists():
                    return str(full_path)
                
                if '\\' in loc or '/' in loc:
                    fname = Path(loc).name
                    for edf_file in Path(root).rglob(fname):
                        return str(edf_file)
    
    # 根据pt_id和fn查找
    pt_id = row.get('pt_id', '')
    fn = row.get('fn', '')
    
    for root in data_roots:
        if file_format == 'set':
            root_path = Path(str(root) + 'processed')
        else:
            root_path = Path(root)
        
        if not root_path.exists():
            continue
            
        for patient_dir in root_path.rglob(f"*{pt_id}*"):
            if patient_dir.is_dir():
                match = re.search(r'SZ(\d+)', fn, re.IGNORECASE)
                if match:
                    sz_num = match.group(1)
                    ext = 'set' if file_format == 'set' else 'edf'
                    suffix = '_filtered_3_45_postICA_eye' if file_format == 'set' else ''
                    
                    patterns = [
                        f"*SZ{sz_num}*{suffix}.{ext}",
                        f"*sz{sz_num}*{suffix}.{ext}"
                    ]
                    
                    for pattern in patterns:
                        for data_file in patient_dir.glob(pattern):
                            return str(data_file)
    
    return None


def process_and_visualize(
    row: pd.Series,
    config: DataConfig,
    output_dir: Path,
    file_format: str = 'edf',
    use_bipolar: bool = False
):
    """
    处理单条记录并生成可视化
    
    Args:
        row: CSV中的一行数据
        config: 数据配置
        output_dir: 输出目录
        file_format: 文件格式 'edf' 或 'set'
        use_bipolar: 是否使用双极导联
    """
    # 查找数据文件
    data_path = find_data_file(row, config.edf_data_roots, file_format)
    if data_path is None:
        logger.warning(f"未找到{file_format.upper()}文件: {row.get('fn', 'unknown')}")
        return
    
    logger.info(f"处理: {row.get('fn', 'unknown')} -> {data_path}")
    
    # 解析发作时间
    seizure_times = parse_seizure_times(row.get('sz_starts'), row.get('sz_ends'))
    if not seizure_times:
        logger.warning(f"无发作时间: {row.get('fn')}")
        return
    
    # 解析SOZ脑区（用于单极导联模式）
    onset_zone_str = row.get('onset_zone', '')
    soz_regions = set()
    if pd.notna(onset_zone_str) and onset_zone_str:
        for part in str(onset_zone_str).lower().split(','):
            part = part.strip()
            if part in REGION_TO_CHANNEL_IDX:
                soz_regions.add(part)
    
    # 解析通道级别SOZ（用于双极导联模式）
    soz_unipolar_channels = set()
    channel_names_in_csv = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
                             't3', 'c3', 'cz', 'c4', 't4',
                             't5', 'p3', 'pz', 'p4', 't6',
                             'o1', 'o2']
    for ch in channel_names_in_csv:
        ch_value = row.get(ch, 0)
        if ch_value is None:
            ch_value = row.get(ch.upper(), 0)
        try:
            if pd.notna(ch_value) and int(ch_value) == 1:
                soz_unipolar_channels.add(ch.upper())
        except (ValueError, TypeError):
            pass
    
    # 获取双极导联的SOZ信息
    if soz_unipolar_channels:
        soz_chains, soz_bipolar_channels = get_soz_bipolar_info(soz_unipolar_channels)
        logger.info(f"SOZ单极电极: {soz_unipolar_channels}")
        logger.info(f"SOZ双极通道: {soz_bipolar_channels}")
        logger.info(f"SOZ链区域: {soz_chains}")
    else:
        soz_chains = set()
        soz_bipolar_channels = set()
    
    try:
        # 1. 读取数据文件
        if file_format == 'set':
            raw_data, fs, ch_names = read_set_file(data_path)
        else:
            from dataset import read_edf
            raw_data, fs, ch_names = read_edf(data_path)
        
        # 2. 提取标准通道
        # 首先尝试提取21通道（含蝶骨电极），如果失败则使用19通道
        data, found_channels = extract_standard_channels(raw_data, ch_names, target_channels=STANDARD_21_CHANNELS)
        use_21_channels = len(found_channels) == 21
        
        if not use_21_channels:
            # 尝试使用标准19通道
            data, found_channels = extract_standard_channels(raw_data, ch_names, target_channels=STANDARD_19_CHANNELS)
            logger.info(f"使用19电极模式: {len(found_channels)} 通道")
        else:
            logger.info(f"使用21电极模式（含蝶骨电极）: {len(found_channels)} 通道")
        
        # 3. 可选：转换为双极导联
        if use_bipolar:
            if use_21_channels:
                # 使用26对双极导联（21电极，5脑区）
                data, found_channels = convert_to_bipolar(data, found_channels, BIPOLAR_PAIRS_26)
                logger.info(f"使用26对双极导联（5脑区模式）: {len(found_channels)} 通道")
            else:
                # 使用18对双极导联（19电极，5链）
                data, found_channels = convert_to_bipolar(data, found_channels, BIPOLAR_PAIRS_18)
                logger.info(f"使用18对双极导联（5链模式）: {len(found_channels)} 通道")
        
        # 3.5. 解析mask_segments（后续在提取片段时使用）
        mask_segments = parse_mask_segments(row.get('mask_segments'))
        
        # 4. 预处理每个通道
        # 根据文件格式使用不同的滤波范围
        if file_format == 'set':
            filter_low = 3.0   # .set格式使用3-45Hz
            filter_high = 45.0
        else:
            filter_low = config.filter_low   # .edf使用配置中的值
            filter_high = config.filter_high
        
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered = bandpass_filter(
                data[i], fs,
                f_low=filter_low,
                f_high=filter_high
            )
            clipped = clip_amplitude(filtered, config.clip_std)
            processed_data[i] = clipped
        
        # 5. 重采样
        if fs != config.target_fs:
            n_samples_new = int(processed_data.shape[1] * config.target_fs / fs)
            resampled_data = np.zeros((processed_data.shape[0], n_samples_new))
            for i in range(processed_data.shape[0]):
                resampled_data[i] = resample_signal(
                    processed_data[i], fs, config.target_fs
                )
            processed_data = resampled_data
            fs = config.target_fs
        
        # 为每个发作创建可视化
        pt_id = row.get('pt_id', 'unknown')
        fn = row.get('fn', 'unknown')
        
        for sz_idx, (sz_start, sz_end) in enumerate(seizure_times):
            # 提取发作片段 + 移除范围内的坏段（正确处理时间偏移）
            seizure_segment, clean_duration = extract_segment_with_mask_removal(
                processed_data, fs,
                sz_start, sz_end,
                mask_segments
            )
            
            if seizure_segment.shape[1] == 0:
                logger.warning(f"发作片段 {sz_start}-{sz_end} 清洗后没有数据")
                continue
            
            logger.info(f"发作片段提取: [{sz_start:.1f}s, {sz_end:.1f}s] -> 清洗后{clean_duration:.1f}s")
            
            # 计算全局归一化参数（所有通道所有脑区共用）
            global_min = np.min(seizure_segment)
            global_max = np.max(seizure_segment)
            logger.info(f"全局归一化范围: [{global_min:.6f}, {global_max:.6f}]")
            
            # 创建患者/发作目录
            format_suffix = f"_{file_format}"
            patient_dir = output_dir / f"{pt_id}_{fn}_{sz_start}-{sz_end}{format_suffix}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # 按脑区可视化
            if use_bipolar:
                # 双极导联按脑区可视化
                title_base = f"{pt_id} - {fn} - SZ{sz_idx+1} ({sz_start:.1f}s - {sz_end:.1f}s)"
                
                # 根据通道数选择脑区定义
                n_bipolar_channels = seizure_segment.shape[0]
                if n_bipolar_channels == 26:
                    # 使用26通道/5脑区模式（21电极含蝶骨电极）
                    chain_idx_map = BIPOLAR_CHAIN_TO_CHANNEL_IDX_26
                    chain_names_map = BIPOLAR_CHAIN_CHANNEL_NAMES_26
                    chain_display_map = BIPOLAR_CHAIN_DISPLAY_NAMES_26
                    logger.info("使用26通道/5脑区模式可视化")
                else:
                    # 使用18通道/5链模式（标准19电极）
                    chain_idx_map = BIPOLAR_CHAIN_TO_CHANNEL_IDX
                    chain_names_map = BIPOLAR_CHAIN_CHANNEL_NAMES
                    chain_display_map = BIPOLAR_CHAIN_DISPLAY_NAMES
                    logger.info("使用18通道/5链模式可视化")
                
                # 遍历脑区分别可视化
                for chain_name, channel_indices in chain_idx_map.items():
                    # 检查索引是否超出范围
                    if max(channel_indices) >= n_bipolar_channels:
                        logger.warning(f"跳过 {chain_name}: 索引超出范围")
                        continue
                    
                    # 获取该脑区的数据
                    chain_data = seizure_segment[channel_indices, :]
                    chain_ch_names = chain_names_map[chain_name]
                    chain_display_name = chain_display_map[chain_name]
                    
                    # 检查该脑区是否包含SOZ
                    is_soz_chain = chain_name in soz_chains
                    soz_marker = " [SOZ]" if is_soz_chain else ""
                    
                    # 获取该脑区中的SOZ通道（用于红色标签）
                    chain_soz_channels = set()
                    for ch in chain_ch_names:
                        if ch.upper() in {c.upper() for c in soz_bipolar_channels}:
                            chain_soz_channels.add(ch)
                    
                    title_chain = f"{title_base} - {chain_display_name}{soz_marker}"
                    
                    # 波形图
                    waveform_suffix = "_soz" if is_soz_chain else ""
                    waveform_path = patient_dir / f"{chain_name}{waveform_suffix}_waveform.png"
                    plot_waveform(
                        chain_data,
                        chain_ch_names,
                        fs,
                        f"{title_chain} - Waveform",
                        str(waveform_path),
                        global_min=global_min,
                        global_max=global_max,
                        soz_channels=chain_soz_channels
                    )
                    
                    # 热力图
                    heatmap_path = patient_dir / f"{chain_name}{waveform_suffix}_heatmap.png"
                    plot_heatmap(
                        chain_data,
                        chain_ch_names,
                        fs,
                        f"{title_chain} - Heatmap",
                        str(heatmap_path),
                        global_min=global_min,
                        global_max=global_max,
                        soz_channels=chain_soz_channels
                    )
                
                # ===============================================================
                # 双极导联脑网络连接性特征计算与可视化
                # ===============================================================
                # 支持18通道（标准19电极）或26通道（21电极含蝶骨电极）
                if seizure_segment.shape[0] in [18, 26]:
                    logger.info(f"计算双极导联脑网络连接性特征 ({seizure_segment.shape[0]}通道)...")
                    
                    try:
                        connectivity_dict = compute_all_connectivity(
                            seizure_segment, 
                            fs=fs,
                            freq_band=(8, 13),
                            include_directed=True
                        )
                        
                        connectivity_dir = patient_dir / "connectivity_bipolar"
                        
                        visualize_bipolar_connectivity(
                            connectivity_dict,
                            output_dir=str(connectivity_dir),
                            prefix=f"SZ{sz_idx+1}",
                            percentile=70.0,
                            zscore_threshold=3.0,
                            max_lines=60
                        )
                        
                        logger.info(f"双极导联连接性可视化完成: {connectivity_dir}")
                        
                    except Exception as conn_e:
                        logger.warning(f"双极导联连接性计算失败: {conn_e}")
            else:
                # 单极导联按脑区可视化
                for region_name, channel_indices in REGION_TO_CHANNEL_IDX.items():
                    # 获取该脑区的数据
                    region_data = seizure_segment[channel_indices, :]
                    region_ch_names = REGION_CHANNEL_NAMES[region_name]
                    
                    # 检查是否是SOZ脑区
                    is_soz = region_name in soz_regions
                    soz_suffix = "_soz" if is_soz else ""
                    soz_marker = " [SOZ]" if is_soz else ""
                    
                    title_base = f"{pt_id} - {fn} - SZ{sz_idx+1} ({sz_start:.1f}s - {sz_end:.1f}s)"
                    title_region = f"{title_base} - {region_name.capitalize()}{soz_marker}"
                    
                    # 波形图（使用全局归一化）
                    waveform_path = patient_dir / f"{region_name}{soz_suffix}_waveform.png"
                    plot_waveform(
                        region_data,
                        region_ch_names,
                        fs,
                        f"{title_region} - Waveform",
                        str(waveform_path),
                        global_min=global_min,
                        global_max=global_max
                    )
                    
                    # 热力图（使用全局归一化）
                    heatmap_path = patient_dir / f"{region_name}{soz_suffix}_heatmap.png"
                    plot_heatmap(
                        region_data,
                        region_ch_names,
                        fs,
                        f"{title_region} - Heatmap",
                        str(heatmap_path),
                        global_min=global_min,
                        global_max=global_max
                    )
            
            # ===============================================================
            # 脑网络连接性特征计算与可视化
            # ===============================================================
            if not use_bipolar and seizure_segment.shape[0] == 19:
                logger.info("计算脑网络连接性特征...")
                
                try:
                    # 计算所有连接性特征
                    connectivity_dict = compute_all_connectivity(
                        seizure_segment, 
                        fs=fs,
                        freq_band=(8, 13),  # Alpha频段
                        include_directed=True
                    )
                    
                    # 创建连接性可视化子目录
                    connectivity_dir = patient_dir / "connectivity"
                    
                    # 可视化所有连接性特征（每个特征生成矩阵图+圆形图）
                    # 使用百分位阈值过滤弱连接，z-score过滤异常值
                    visualize_all_connectivity(
                        connectivity_dict,
                        output_dir=str(connectivity_dir),
                        prefix=f"SZ{sz_idx+1}",
                        # 过滤参数
                        percentile=70.0,          # 保留前30%最强连接
                        zscore_threshold=3.0,     # 过滤z-score>3的异常值
                        remove_outliers=True,
                        exclude_diagonal=True,
                        # 可视化参数
                        max_lines=80,             # 圆形图最多显示80条连接
                        line_cmap='Reds'
                    )
                    
                    logger.info(f"连接性可视化完成: {connectivity_dir}")
                    
                except Exception as conn_e:
                    logger.warning(f"连接性计算失败: {conn_e}")
                    
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"处理失败 {row.get('fn')}: {e}")


def main():
    parser = argparse.ArgumentParser(description='EEG脑区可视化 (支持.edf和.set格式)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='数据根目录列表')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认创建时间戳目录）')
    parser.add_argument('--max-records', type=int, default=None,
                        help='最大处理记录数（调试用）')
    parser.add_argument('--bipolar', type=bool, default=True,
                        help='使用TCP双极导联')
    parser.add_argument('--format', type=str, default='set',
                        choices=['edf', 'set'],
                        help='数据文件格式: edf 或 set (默认edf)')
    args = parser.parse_args()
    
    # 加载配置
    config = get_config()
    
    # 覆盖配置
    if args.manifest:
        config.data.manifest_path = args.manifest
    if args.data_roots:
        config.data.edf_data_roots = args.data_roots
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.bipolar:
            output_dir = Path(__file__).parent / "visualizations" / f"{timestamp}_{args.format}_bipolar"
        else:
            output_dir = Path(__file__).parent / "visualizations" / f"{timestamp}_{args.format}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"数据格式: {args.format.upper()}")
    
    if args.format == 'set':
        logger.info("使用.set格式配置:")
        logger.info("  - 数据根目录后缀: 'processed'")
        logger.info("  - 文件名后缀: '_filtered_3_45_postICA_eye'")
        logger.info("  - 滤波范围: 3-45Hz")
    
    # 加载manifest
    df = pd.read_csv(config.data.manifest_path)
    logger.info(f"加载manifest: {len(df)} 条记录")
    
    if args.max_records:
        df = df.head(args.max_records)
        logger.info(f"限制处理前 {args.max_records} 条记录")
    
    # 处理每条记录
    for idx, row in df.iterrows():
        pt_id = row.get('pt_id')
        if pt_id not in ["周良贵",]:
            continue
        logger.info(f"\n处理记录 {idx + 1}/{len(df)} pt_id:{pt_id}")
        process_and_visualize(
            row, config.data, output_dir,
            file_format=args.format, 
            use_bipolar=args.bipolar
        )


    
    logger.info(f"\n可视化完成！输出目录: {output_dir}")


if __name__ == '__main__':
    main()
