#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
私有数据集数据加载器

支持三种标签粒度：
1. onset_zone - 脑区级别分类
2. hemi - 半球级别分类  
3. channel - 通道级别分类

参考DeepSOZ数据处理流程:
1. 读取EDF文件
2. 提取标准19通道
3. 带通滤波 (1.6-30 Hz)
4. 幅值裁剪 (±2 std)
5. 重采样到 200 Hz
6. 分割成1秒窗口
7. Z-score标准化
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import warnings
import logging

# 信号处理
from scipy import signal as sig
from scipy.signal import resample

# EDF读取
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False

from config import DataConfig, get_config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# 标准通道定义
# ==============================================================================

STANDARD_19_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2'
]

# 21电极通道列表（标准19 + Sph-L + Sph-R蝶骨电极）
STANDARD_21_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2',
    'SPHL', 'SPHR'  # 蝶骨电极：Sph-L在F7和T3之间，Sph-R在F8和T4之间
]

# 通道名映射（处理各种变体）
CHANNEL_NAME_MAP = {
    # 大小写变体
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ', 'Oz': 'OZ',
    # 10-10到10-20系统映射
    'T7': 'T3', 'T8': 'T4',
    'P7': 'T5', 'P8': 'T6',
    # 蝶骨电极变体名称
    'Sph-L': 'SPHL', 'SPH-L': 'SPHL', 'SphL': 'SPHL', 'sph-l': 'SPHL', 'sph_l': 'SPHL',
    'Sph-R': 'SPHR', 'SPH-R': 'SPHR', 'SphR': 'SPHR', 'sph-r': 'SPHR', 'sph_r': 'SPHR',
    'Sph1': 'SPHL', 'Sph2': 'SPHR',  # 可能的别名
    # 带参考的通道名
    'EEG FP1': 'FP1', 'EEG FP2': 'FP2',
    'EEG F7': 'F7', 'EEG F3': 'F3', 'EEG FZ': 'FZ', 'EEG F4': 'F4', 'EEG F8': 'F8',
    'EEG T3': 'T3', 'EEG C3': 'C3', 'EEG CZ': 'CZ', 'EEG C4': 'C4', 'EEG T4': 'T4',
    'EEG T5': 'T5', 'EEG P3': 'P3', 'EEG PZ': 'PZ', 'EEG P4': 'P4', 'EEG T6': 'T6',
    'EEG O1': 'O1', 'EEG O2': 'O2',
}

# 脑区映射（19电极）
BRAIN_REGION_MAP = {
    'frontal': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'central': ['C3', 'C4', 'CZ'],
    'parietal': ['P3', 'P4', 'PZ'],
    'occipital': ['O1', 'O2']
}

# 5脑区映射（21电极）- 用户定义
BRAIN_REGION_MAP_21 = {
    'left_frontal': ['FP1', 'F7', 'F3', 'FZ'],
    'left_temporal': ['F7', 'SPHL', 'T3', 'T5', 'O1', 'C3', 'P3'],
    'parietal': ['FZ', 'CZ', 'C3', 'C4', 'P3', 'PZ', 'P4'],
    'right_frontal': ['FP2', 'F4', 'F8', 'FZ'],
    'right_temporal': ['F8', 'SPHR', 'T4', 'T6', 'O2', 'C4', 'P4'],
}

# 半球映射
HEMISPHERE_MAP = {
    'L': ['FP1', 'F3', 'F7', 'T3', 'C3', 'T5', 'P3', 'O1'],
    'R': ['FP2', 'F4', 'F8', 'T4', 'C4', 'T6', 'P4', 'O2'],
    'M': ['FZ', 'CZ', 'PZ']
}

# 21电极半球映射
HEMISPHERE_MAP_21 = {
    'L': ['FP1', 'F3', 'F7', 'T3', 'C3', 'T5', 'P3', 'O1', 'SPHL'],
    'R': ['FP2', 'F4', 'F8', 'T4', 'C4', 'T6', 'P4', 'O2', 'SPHR'],
    'M': ['FZ', 'CZ', 'PZ']
}


# ==============================================================================
# 信号处理函数
# ==============================================================================

# 标准18导联TCP双极导联对（19电极）
# 参考临床EEG标准双极纵向蒙太奇
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

# 双极导联通道名称（18导联）
BIPOLAR_CHANNEL_NAMES = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS_18]

# ==============================================================================
# 21电极双极导联定义（26导联，5脑区）
# ==============================================================================
BIPOLAR_PAIRS_26 = [
    # 左额 (4对)
    ('FP1', 'F7'), ('FP1', 'F3'), ('F7', 'F3'), ('F3', 'FZ'),
    # 左颞 (6对)
    ('F7', 'SPHL'), ('SPHL', 'T3'), ('T3', 'T5'), ('T5', 'O1'), ('T3', 'C3'), ('T5', 'P3'),
    # 顶叶 (6对)
    ('FZ', 'CZ'), ('C3', 'CZ'), ('P3', 'PZ'), ('CZ', 'PZ'), ('CZ', 'C4'), ('PZ', 'P4'),
    # 右额 (4对)
    ('FP2', 'F4'), ('FP2', 'F8'), ('F4', 'F8'), ('FZ', 'F4'),
    # 右颞 (6对)
    ('F8', 'SPHR'), ('SPHR', 'T4'), ('C4', 'T4'), ('T4', 'T6'), ('P4', 'T6'), ('T6', 'O2'),
]

# 双极导联通道名称（26导联）
BIPOLAR_CHANNEL_NAMES_26 = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS_26]

# 5脑区双极导联映射
BIPOLAR_REGION_DEFINITIONS_21 = {
    'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
    'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
    'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
    'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
    'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
}

# 横向导联排列（用于可视化对称性分析）
TRANSVERSE_MONTAGE = [
    # 前额排
    ('FP1', 'FP2'),
    # 前头部排
    ('F7', 'F3'), ('F3', 'FZ'), ('FZ', 'F4'), ('F4', 'F8'),
    # 中央排
    ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'),
    # 后颞/顶排
    ('T5', 'P3'), ('P3', 'PZ'), ('PZ', 'P4'), ('P4', 'T6'),
    # 枕部排
    ('O1', 'O2'),
]

TRANSVERSE_CHANNEL_NAMES = [f"{a}-{c}" for a, c in TRANSVERSE_MONTAGE]


def convert_to_bipolar(
    data: np.ndarray, 
    ch_names: List[str], 
    bipolar_pairs: List[Tuple[str, str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    将单极参考导联转换为TCP双极导联
    
    双极导联通过相邻电极电位相减获得，对局部电位变化最敏感，
    能极大地锐化病灶信号，让起始通道特征在背景中更突显。
    
    Args:
        data: 单极导联数据 (n_channels, n_samples)
        ch_names: 单极导联通道名称列表
        bipolar_pairs: 双极导联对列表，如 [('FP1', 'F7'), ('F7', 'T3'), ...]
                      如果为None，使用标准18导联配置
    
    Returns:
        bipolar_data: 双极导联数据 (n_bipolar_channels, n_samples)
        bipolar_names: 双极导联通道名称列表，如 ['FP1-F7', 'F7-T3', ...]
    """
    if bipolar_pairs is None:
        bipolar_pairs = BIPOLAR_PAIRS_18
    
    # 建立通道名到索引的映射（大写标准化）
    ch_map = {name.upper().strip(): i for i, name in enumerate(ch_names)}
    
    bipolar_data = []
    bipolar_names = []
    missing_pairs = []
    
    for anode, cathode in bipolar_pairs:
        anode_upper = anode.upper().strip()
        cathode_upper = cathode.upper().strip()
        
        if anode_upper in ch_map and cathode_upper in ch_map:
            idx_anode = ch_map[anode_upper]
            idx_cathode = ch_map[cathode_upper]
            
            # 双极导联 = 阳极 - 阴极
            diff_signal = data[idx_anode, :] - data[idx_cathode, :]
            bipolar_data.append(diff_signal)
            bipolar_names.append(f"{anode_upper}-{cathode_upper}")
        else:
            missing_pairs.append((anode, cathode))
            # 用零信号填充缺失的导联对
            bipolar_data.append(np.zeros(data.shape[1]))
            bipolar_names.append(f"{anode_upper}-{cathode_upper}")
    
    if missing_pairs:
        logger.warning(f"双极导联转换: 缺失以下导联对的原始通道: {missing_pairs}")
    
    return np.array(bipolar_data), bipolar_names

def map_channel_name(name: str) -> str:
    """标准化通道名称"""
    name_upper = name.upper().strip()
    # 先检查直接映射
    if name_upper in STANDARD_19_CHANNELS:
        return name_upper
    # 检查特殊映射
    if name in CHANNEL_NAME_MAP:
        return CHANNEL_NAME_MAP[name]
    if name_upper in CHANNEL_NAME_MAP:
        return CHANNEL_NAME_MAP[name_upper]
    # 去除前缀后检查
    for prefix in ['EEG ', 'EEG-', 'REF-']:
        if name_upper.startswith(prefix):
            stripped = name_upper[len(prefix):]
            if stripped in STANDARD_19_CHANNELS:
                return stripped
    return name_upper


def apply_lowpass(x: np.ndarray, fs: float, fc: float = 30, N: int = 4) -> np.ndarray:
    """低通滤波"""
    wc = fc / (fs / 2)
    if wc >= 1:
        wc = 0.99
    b, a = sig.butter(N, wc)
    return sig.filtfilt(b, a, x, method='gust')


def apply_highpass(x: np.ndarray, fs: float, fc: float = 1.6, N: int = 4) -> np.ndarray:
    """高通滤波"""
    wc = fc / (fs / 2)
    if wc >= 1:
        wc = 0.99
    b, a = sig.butter(N, wc, btype='highpass')
    return sig.filtfilt(b, a, x, method='gust')


def bandpass_filter(x: np.ndarray, fs: float, 
                    f_low: float = 1.6, f_high: float = 30) -> np.ndarray:
    """带通滤波"""
    x_high = apply_highpass(x, fs, fc=f_low)
    x_band = apply_lowpass(x_high, fs, fc=f_high)
    return x_band


def clip_amplitude(x: np.ndarray, n_std: float = 2) -> np.ndarray:
    """幅值裁剪"""
    mean = np.mean(x)
    std = np.std(x)
    return np.clip(x, mean - n_std * std, mean + n_std * std)


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """重采样"""
    if orig_fs == target_fs:
        return x
    num_samples = int(len(x) * target_fs / orig_fs)
    return resample(x, num_samples)


def apply_windows(x: np.ndarray, fs: float, window_len: float = 1.0, 
                  overlap: float = 0.0) -> np.ndarray:
    """分割成窗口"""
    samples_per_window = int(window_len * fs)
    step = int(samples_per_window * (1 - overlap))
    
    n_samples = x.shape[-1]
    n_windows = (n_samples - samples_per_window) // step + 1
    
    if len(x.shape) == 1:
        # 单通道
        windows = np.zeros((n_windows, samples_per_window))
        for i in range(n_windows):
            start = i * step
            windows[i] = x[start:start + samples_per_window]
    else:
        # 多通道 (n_channels, n_samples)
        n_channels = x.shape[0]
        windows = np.zeros((n_windows, n_channels, samples_per_window))
        for i in range(n_windows):
            start = i * step
            windows[i] = x[:, start:start + samples_per_window]
    
    return windows


def adaptive_apply_windows(
    x: np.ndarray, 
    fs: float, 
    window_len: float = 1.0, 
    overlap: float = 0.0,
    min_windows: int = 2
) -> np.ndarray:
    """
    自适应窗口划分 - 处理数据不足的情况
    
    当剩余数据不足以按标准重叠策略划分时，采用"首尾两端"策略：
    - 如果数据 >= 2 * window_len: 使用标准重叠策略
    - 如果 window_len < 数据 < 2 * window_len: 取首尾两个窗口
    - 如果数据 <= window_len: 只取一个窗口（可能需要填充）
    
    例如：29s数据，20s窗口
    -> 取 [0-20s] 和 [9-29s] 两个窗口
    
    Args:
        x: 输入数据 (n_channels, n_samples) 或 (n_samples,)
        fs: 采样率
        window_len: 窗口长度（秒）
        overlap: 标准重叠比例
        min_windows: 最少生成的窗口数
    
    Returns:
        windows: (n_windows, n_channels, samples_per_window) 或 (n_windows, samples_per_window)
    """
    samples_per_window = int(window_len * fs)
    step = int(samples_per_window * (1 - overlap))
    
    n_samples = x.shape[-1]
    data_duration = n_samples / fs
    
    is_multichannel = len(x.shape) > 1
    n_channels = x.shape[0] if is_multichannel else 1
    
    # 计算按标准策略可以划分的窗口数
    standard_n_windows = max(0, (n_samples - samples_per_window) // step + 1)
    
    # 情况1: 数据足够，使用标准策略
    if standard_n_windows >= min_windows:
        return apply_windows(x, fs, window_len, overlap)
    
    # 情况2: 数据刚好够2个窗口，但重叠不够 -> 首尾策略
    if n_samples >= samples_per_window and n_samples < 2 * samples_per_window:
        # 取首窗口 [0, window_len] 和尾窗口 [end-window_len, end]
        if is_multichannel:
            windows = np.zeros((2, n_channels, samples_per_window))
            windows[0] = x[:, :samples_per_window]  # 首窗口
            windows[1] = x[:, -samples_per_window:]  # 尾窗口
        else:
            windows = np.zeros((2, samples_per_window))
            windows[0] = x[:samples_per_window]
            windows[1] = x[-samples_per_window:]
        
        # 计算实际重叠
        actual_overlap_samples = 2 * samples_per_window - n_samples
        actual_overlap_ratio = actual_overlap_samples / samples_per_window
        logger.debug(
            f"自适应窗口划分: {data_duration:.1f}s数据, {window_len}s窗口 -> "
            f"2个窗口, 实际重叠{actual_overlap_ratio*100:.1f}%"
        )
        return windows
    
    # 情况3: 数据够2个窗口以上，但按标准策略只有1个 -> 首尾策略
    if n_samples >= 2 * samples_per_window:
        if is_multichannel:
            windows = np.zeros((2, n_channels, samples_per_window))
            windows[0] = x[:, :samples_per_window]
            windows[1] = x[:, -samples_per_window:]
        else:
            windows = np.zeros((2, samples_per_window))
            windows[0] = x[:samples_per_window]
            windows[1] = x[-samples_per_window:]
        return windows
    
    # 情况4: 数据不够一个完整窗口 -> 填充到一个窗口
    if n_samples < samples_per_window:
        if is_multichannel:
            windows = np.zeros((1, n_channels, samples_per_window))
            windows[0, :, :n_samples] = x
        else:
            windows = np.zeros((1, samples_per_window))
            windows[0, :n_samples] = x
        
        logger.warning(
            f"数据不足: {data_duration:.1f}s < {window_len}s窗口, 使用零填充"
        )
        return windows
    
    # 默认回退到标准策略
    return apply_windows(x, fs, window_len, overlap)


# ==============================================================================
# EDF文件读取
# ==============================================================================

def read_edf_pyedflib(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """使用pyedflib读取EDF"""
    f = pyedflib.EdfReader(filepath)
    try:
        n_channels = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs_list = [f.getSampleFrequency(i) for i in range(n_channels)]
        fs = fs_list[0]
        
        n_samples = f.getNSamples()[0]
        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            data[i] = f.readSignal(i)
        
        return data, fs, ch_names
    finally:
        f._close()


def read_edf_mne(filepath: str, encoding: str = 'utf-8') -> Tuple[np.ndarray, float, List[str]]:
    """使用MNE读取EDF"""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR', encoding=encoding)
    data = raw.get_data()
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    return data, fs, ch_names


def read_edf(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """读取EDF文件（多种方法尝试）"""
    errors = []
    
    # 方法1: pyedflib
    if HAS_PYEDFLIB:
        try:
            return read_edf_pyedflib(filepath)
        except Exception as e:
            errors.append(f"pyedflib: {e}")
    
    # 方法2: MNE with utf-8
    if HAS_MNE:
        try:
            return read_edf_mne(filepath, encoding='utf-8')
        except Exception as e:
            errors.append(f"mne(utf-8): {e}")
        
        # 方法3: MNE with latin-1
        try:
            return read_edf_mne(filepath, encoding='latin-1')
        except Exception as e:
            errors.append(f"mne(latin-1): {e}")
    
    raise RuntimeError(f"无法读取EDF文件 {filepath}: {errors}")


def extract_standard_channels(data: np.ndarray, ch_names: List[str],
                             target_channels: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """提取标准19通道"""
    if target_channels is None:
        target_channels = STANDARD_19_CHANNELS
    
    # 映射通道名
    ch_map = {map_channel_name(ch): i for i, ch in enumerate(ch_names)}
    
    # 提取匹配的通道
    output_data = np.zeros((len(target_channels), data.shape[1]))
    found_channels = []
    missing_channels = []
    
    for i, target_ch in enumerate(target_channels):
        if target_ch in ch_map:
            output_data[i] = data[ch_map[target_ch]]
            found_channels.append(target_ch)
        else:
            missing_channels.append(target_ch)
            # 用零填充缺失通道
    
    if missing_channels:
        logger.warning(f"缺失通道: {missing_channels}")
    
    return output_data, found_channels


# ==============================================================================
# 标签处理
# ==============================================================================

def parse_onset_zone_label(onset_zone: str) -> np.ndarray:
    """
    解析onset_zone标签为多标签向量
    
    Args:
        onset_zone: 逗号分隔的脑区字符串，如 "frontal,temporal"
    
    Returns:
        5维向量 [frontal, temporal, central, parietal, occipital]
    """
    regions = ['frontal', 'temporal', 'central', 'parietal', 'occipital']
    label = np.zeros(len(regions), dtype=np.float32)
    
    if pd.isna(onset_zone) or onset_zone == '':
        return label
    
    for part in str(onset_zone).lower().split(','):
        part = part.strip()
        if part in regions:
            label[regions.index(part)] = 1.0
    
    return label


def parse_hemi_label(hemi: str) -> np.ndarray:
    """
    解析hemi标签为分类向量
    
    Args:
        hemi: 半球标签 'L', 'R', 'B' (bilateral), 'U' (unknown)
    
    Returns:
        4维one-hot向量 [L, R, B, U]
    """
    classes = ['L', 'R', 'B', 'U']
    label = np.zeros(len(classes), dtype=np.float32)
    
    if pd.isna(hemi) or hemi == '':
        label[3] = 1.0  # Unknown
        return label
    
    hemi = str(hemi).upper().strip()
    if hemi in classes:
        label[classes.index(hemi)] = 1.0
    else:
        label[3] = 1.0  # Unknown
    
    return label


def parse_channel_labels(row: pd.Series, channel_columns: List[str] = None) -> np.ndarray:
    """
    解析通道级别标签
    
    Args:
        row: DataFrame行
        channel_columns: 通道列名列表
    
    Returns:
        19维向量，每个通道的SOZ标签 (0/1)
    """
    if channel_columns is None:
        channel_columns = [ch.lower() for ch in STANDARD_19_CHANNELS]
    
    label = np.zeros(len(channel_columns), dtype=np.float32)
    
    for i, col in enumerate(channel_columns):
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    label[i] = float(val)
                except (ValueError, TypeError):
                    pass
    
    return label


def parse_baseline(base_line: str) -> Tuple[float, float]:
    """
    解析baseline时间范围
    
    Args:
        base_line: 格式 "start,end" 如 "740.0,750.0"
    
    Returns:
        (start_time, end_time)
    """
    if pd.isna(base_line) or base_line == '':
        return None, None
    
    try:
        parts = str(base_line).split(',')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, TypeError):
        pass
    
    return None, None


def parse_seizure_times(sz_starts: str, sz_ends: str) -> List[Tuple[float, float]]:
    """
    解析癫痫发作时间
    
    Args:
        sz_starts: 分号分隔的起始时间
        sz_ends: 分号分隔的结束时间
    
    Returns:
        [(start1, end1), (start2, end2), ...]
    """
    if pd.isna(sz_starts) or pd.isna(sz_ends):
        return []
    
    try:
        starts = [float(t.strip()) for t in str(sz_starts).split(';') if t.strip()]
        ends = [float(t.strip()) for t in str(sz_ends).split(';') if t.strip()]
        return list(zip(starts, ends))
    except (ValueError, TypeError):
        return []


def parse_mask_segments(mask_segments: str) -> List[Tuple[float, float]]:
    """
    解析mask_segments字段（坏段时间范围）
    
    Args:
        mask_segments: JSON格式的时间范围列表，如 "[[226.0, 228.0],[888,891]]"
    
    Returns:
        [(start1, end1), (start2, end2), ...] 坏段时间范围列表
    """
    if pd.isna(mask_segments) or mask_segments == '' or mask_segments is None:
        return []
    
    try:
        import json
        # 尝试解析JSON格式
        segments = json.loads(str(mask_segments))
        
        result = []
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                result.append((float(seg[0]), float(seg[1])))
        
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"解析mask_segments失败: {mask_segments}, 错误: {e}")
        return []


def apply_mask_segments(
    data: np.ndarray,
    fs: float,
    mask_segments: List[Tuple[float, float]],
    fill_value: float = 0.0
) -> np.ndarray:
    """
    应用掩码，将坏段数据置为指定值
    
    此函数应在滤波之前调用，以避免坏段数据影响滤波结果。
    
    Args:
        data: EEG数据 (n_channels, n_samples)
        fs: 采样率
        mask_segments: 坏段时间范围列表 [(start1, end1), ...]
        fill_value: 填充值，默认为0
    
    Returns:
        处理后的数据（坏段被填充为指定值）
    """
    if not mask_segments:
        return data
    
    masked_data = data.copy()
    n_samples = data.shape[-1]
    
    for start_time, end_time in mask_segments:
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # 边界检查
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        if start_sample < end_sample:
            masked_data[:, start_sample:end_sample] = fill_value
            logger.debug(f"掩码应用: [{start_time:.2f}s, {end_time:.2f}s] -> [{start_sample}, {end_sample}] samples")
    
    return masked_data


def interpolate_mask_segments(
    data: np.ndarray,
    fs: float,
    mask_segments: List[Tuple[float, float]]
) -> np.ndarray:
    """
    使用线性插值替换坏段数据
    
    相比直接置零，插值可以减少滤波时的边界效应。
    
    Args:
        data: EEG数据 (n_channels, n_samples)
        fs: 采样率
        mask_segments: 坏段时间范围列表 [(start1, end1), ...]
    
    Returns:
        处理后的数据（坏段被线性插值替换）
    """
    if not mask_segments:
        return data
    
    interpolated_data = data.copy()
    n_channels, n_samples = data.shape
    
    for start_time, end_time in mask_segments:
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # 边界检查
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        if start_sample < end_sample and end_sample > start_sample:
            # 对每个通道进行线性插值
            for ch in range(n_channels):
                # 获取边界值
                left_val = data[ch, max(0, start_sample - 1)] if start_sample > 0 else 0
                right_val = data[ch, min(n_samples - 1, end_sample)] if end_sample < n_samples else 0
                
                # 线性插值
                n_interp = end_sample - start_sample
                interpolated_data[ch, start_sample:end_sample] = np.linspace(
                    left_val, right_val, n_interp
                )
    
    return interpolated_data


def remove_mask_segments(
    data: np.ndarray,
    fs: float,
    mask_segments: List[Tuple[float, float]]
) -> Tuple[np.ndarray, float]:
    """
    移除坏段数据，将剩余数据拼接

    直接剔除坏段对应的时间段数据，然后将前后的数据拼接起来。

    Args:
        data: EEG数据 (n_channels, n_samples)
        fs: 采样率
        mask_segments: 坏段时间范围列表 [(start1, end1), ...]

    Returns:
        cleaned_data: 移除坏段后的数据 (n_channels, new_n_samples)
        new_duration: 移除坏段后的新时长(秒)
    """
    if not mask_segments:
        return data, data.shape[1] / fs

    n_channels, n_samples = data.shape

    # 将mask_segments转换为采样点，并按起始时间排序
    mask_samples = []
    for start_time, end_time in mask_segments:
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)

        # 边界检查
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)

        if start_sample < end_sample:
            mask_samples.append((start_sample, end_sample))

    # 按起始位置排序
    mask_samples.sort(key=lambda x: x[0])

    # 合并重叠的坏段
    merged_masks = []
    for start, end in mask_samples:
        if merged_masks and start <= merged_masks[-1][1]:
            # 与前一个坏段重叠，合并
            merged_masks[-1] = (merged_masks[-1][0], max(merged_masks[-1][1], end))
        else:
            merged_masks.append((start, end))

    # 计算要保留的数据段
    good_segments = []
    prev_end = 0
    for start, end in merged_masks:
        if prev_end < start:
            good_segments.append((prev_end, start))
        prev_end = end

    # 添加最后一段
    if prev_end < n_samples:
        good_segments.append((prev_end, n_samples))

    # 拼接好的数据段
    if not good_segments:
        # 全是坏段，返回空数据
        logger.warning("所有数据都是坏段，返回空数组")
        return np.zeros((n_channels, 0)), 0.0

    cleaned_parts = [data[:, start:end] for start, end in good_segments]
    cleaned_data = np.concatenate(cleaned_parts, axis=1)

    new_duration = cleaned_data.shape[1] / fs

    # 计算移除的时长
    removed_samples = n_samples - cleaned_data.shape[1]
    removed_duration = removed_samples / fs
    logger.info(f"移除了 {len(merged_masks)} 个坏段, 共 {removed_duration:.2f}s, 剩余 {new_duration:.2f}s")

    return cleaned_data, new_duration


def filter_mask_segments_for_range(
    mask_segments: List[Tuple[float, float]],
    range_start: float,
    range_end: float
) -> List[Tuple[float, float]]:
    """
    筛选并转换mask_segments到指定时间范围的相对时间戳
    
    mask_segments使用的是相对于原始数据的绝对时间戳，
    而我们需要提取的是[range_start, range_end]范围内的片段。
    
    此函数会：
    1. 筛选出与[range_start, range_end]重叠的坏段
    2. 裁剪坏段到范围边界
    3. 转换为相对于range_start的时间戳
    
    例如：
        mask_segments = [[2041.5, 2042.5], [2045.0, 2048.5], [1330, 1331.5]]
        range_start = 2040, range_end = 2075
        
        输出: [[1.5, 2.5], [5.0, 8.5]]  (相对于2040的时间戳)
    
    Args:
        mask_segments: 绝对时间戳的坏段列表 [(abs_start1, abs_end1), ...]
        range_start: 目标范围起始时间（秒）
        range_end: 目标范围结束时间（秒）
    
    Returns:
        相对时间戳的坏段列表 [(rel_start1, rel_end1), ...]
    """
    if not mask_segments:
        return []
    
    relative_segments = []
    
    for abs_start, abs_end in mask_segments:
        # 检查是否与目标范围重叠
        if abs_end <= range_start or abs_start >= range_end:
            # 不重叠，跳过
            continue
        
        # 裁剪到范围边界
        clipped_start = max(abs_start, range_start)
        clipped_end = min(abs_end, range_end)
        
        # 转换为相对时间戳
        rel_start = clipped_start - range_start
        rel_end = clipped_end - range_start
        
        if rel_end > rel_start:
            relative_segments.append((rel_start, rel_end))
    
    # 按起始时间排序
    relative_segments.sort(key=lambda x: x[0])
    
    return relative_segments


def calculate_clean_duration(
    range_start: float,
    range_end: float,
    mask_segments: List[Tuple[float, float]]
) -> float:
    """
    计算指定时间范围内移除坏段后的有效时长
    
    用于在构建样本阶段预先计算清洗后的数据时长，
    以便正确划分滑动窗口。
    
    例如：
        range_start = 2040, range_end = 2075 (35s)
        mask_segments = [[2041.5, 2042.5], [2045.0, 2048.5], [1330, 1331.5]]
        
        范围内的坏段: [2041.5-2042.5] (1s) + [2045.0-2048.5] (3.5s) = 4.5s
        有效时长: 35s - 4.5s = 30.5s
    
    Args:
        range_start: 目标范围起始时间（秒）
        range_end: 目标范围结束时间（秒）
        mask_segments: 绝对时间戳的坏段列表 [(abs_start, abs_end), ...]
    
    Returns:
        clean_duration: 移除坏段后的有效时长（秒）
    """
    if not mask_segments:
        return range_end - range_start
    
    total_bad_duration = 0.0
    
    for abs_start, abs_end in mask_segments:
        # 检查是否与目标范围重叠
        if abs_end <= range_start or abs_start >= range_end:
            # 不重叠，跳过
            continue
        
        # 裁剪到范围边界
        clipped_start = max(abs_start, range_start)
        clipped_end = min(abs_end, range_end)
        
        bad_duration = clipped_end - clipped_start
        if bad_duration > 0:
            total_bad_duration += bad_duration
    
    clean_duration = (range_end - range_start) - total_bad_duration
    return max(0.0, clean_duration)


def extract_segment_with_mask_removal(
    data: np.ndarray,
    fs: float,
    segment_start: float,
    segment_end: float,
    mask_segments: List[Tuple[float, float]]
) -> Tuple[np.ndarray, float]:
    """
    提取指定时间范围的片段，并移除范围内的坏段数据
    
    这是处理mask_segments的正确方式：
    1. 首先提取[segment_start, segment_end]范围的数据
    2. 筛选并转换mask_segments为相对时间戳
    3. 移除坏段，拼接好的数据
    
    Args:
        data: 完整的EEG数据 (n_channels, total_samples)
        fs: 采样率
        segment_start: 片段起始时间（绝对时间戳，秒）
        segment_end: 片段结束时间（绝对时间戳，秒）
        mask_segments: 绝对时间戳的坏段列表 [(abs_start, abs_end), ...]
    
    Returns:
        cleaned_segment: 移除坏段后的片段数据
        clean_duration: 清洗后的时长（秒）
    """
    n_channels, total_samples = data.shape
    
    # 1. 提取片段
    start_sample = int(segment_start * fs)
    end_sample = int(segment_end * fs)
    
    # 边界检查
    start_sample = max(0, start_sample)
    end_sample = min(total_samples, end_sample)
    
    if end_sample <= start_sample:
        logger.warning(f"无效的片段范围: {segment_start}s - {segment_end}s")
        return np.zeros((n_channels, 0)), 0.0
    
    segment = data[:, start_sample:end_sample]
    segment_duration = segment.shape[1] / fs
    
    # 2. 筛选并转换mask_segments为相对时间戳
    relative_masks = filter_mask_segments_for_range(
        mask_segments, segment_start, segment_end
    )
    
    if not relative_masks:
        # 没有需要移除的坏段
        return segment, segment_duration
    
    # 3. 移除坏段
    cleaned_segment, clean_duration = remove_mask_segments(
        segment, fs, relative_masks
    )
    
    logger.debug(
        f"片段提取: [{segment_start:.1f}s, {segment_end:.1f}s] -> "
        f"原始{segment_duration:.1f}s, 移除坏段后{clean_duration:.1f}s"
    )
    
    return cleaned_segment, clean_duration


# ==============================================================================
# 数据集类
# ==============================================================================

class PrivateEEGDataset(Dataset):
    """
    私有EEG数据集
    
    支持三种标签粒度:
    - 'onset_zone': 脑区级别多标签分类
    - 'hemi': 半球级别分类
    - 'channel': 通道级别多标签分类
    """
    
    def __init__(
        self,
        manifest_path: str,
        data_roots: List[str],
        label_type: str = 'channel',  # 'onset_zone', 'hemi', 'channel'
        patient_ids: List[str] = None,  # 指定患者列表（用于交叉验证）
        config: DataConfig = None,
        transform=None,
        max_seizures: int = 10,
        include_baseline: bool = True,
        window_before_onset: float = 0.0,  # 发作前多少秒
        window_after_onset: float = 0.0,   # 发作后多少秒
    ):
        """
        Args:
            manifest_path: CSV manifest文件路径
            data_roots: EDF数据根目录列表
            label_type: 标签类型
            patient_ids: 患者ID列表（用于数据划分）
            config: 数据配置
            transform: 数据增强变换
            max_seizures: 每条记录最大发作数
            include_baseline: 是否包含baseline窗口
            window_before_onset: 发作前窗口时间
            window_after_onset: 发作后窗口时间
        """
        self.manifest_path = manifest_path
        self.data_roots = data_roots if isinstance(data_roots, list) else [data_roots]
        self.label_type = label_type
        self.transform = transform
        self.max_seizures = max_seizures
        self.include_baseline = include_baseline
        self.window_before = window_before_onset
        self.window_after = window_after_onset
        
        # 加载配置
        self.config = config if config is not None else DataConfig()
        
        # 加载manifest
        self.df = pd.read_csv(manifest_path)
        
        # 通道列名
        self.channel_columns = [ch.lower() for ch in STANDARD_19_CHANNELS]
        
        # 过滤患者
        if patient_ids is not None:
            self.df = self.df[self.df['pt_id'].isin(patient_ids)].reset_index(drop=True)
        
        # 构建样本列表（每个发作一个样本）
        self.samples = self._build_samples()
        
        logger.info(f"加载数据集: {len(self.samples)} 个样本, 标签类型: {label_type}")
    
    def _build_samples(self) -> List[Dict]:
        """构建样本列表"""
        samples = []
        
        for idx, row in self.df.iterrows():
            # 解析发作时间
            seizure_times = parse_seizure_times(row.get('sz_starts'), row.get('sz_ends'))
            
            if not seizure_times:
                continue
            
            # 限制发作数量
            seizure_times = seizure_times[:self.max_seizures]
            
            # 查找EDF文件
            edf_path = self._find_edf_file(row)
            if edf_path is None:
                logger.warning(f"未找到EDF文件: {row.get('loc', row.get('fn'))}")
                continue
            
            # 解析标签
            labels = self._parse_labels(row)
            
            # 解析baseline
            baseline = parse_baseline(row.get('base_line'))
            
            # 解析mask_segments（坏段）
            mask_segments = parse_mask_segments(row.get('mask_segments'))
            
            # 为每个发作创建样本
            for sz_idx, (sz_start, sz_end) in enumerate(seizure_times):
                sample = {
                    'row_idx': idx,
                    'pt_id': row.get('pt_id'),
                    'fn': row.get('fn'),
                    'edf_path': edf_path,
                    'sz_idx': sz_idx,
                    'sz_start': sz_start,
                    'sz_end': sz_end,
                    'baseline': baseline,
                    'mask_segments': mask_segments,  # 新增: 坏段时间范围
                    'labels': labels,
                    'duration': row.get('duration'),
                }
                samples.append(sample)
        
        return samples
    
    def _find_edf_file(self, row: pd.Series) -> Optional[str]:
        """查找EDF文件路径"""
        loc = row.get('loc', '')
        
        if pd.notna(loc) and loc:
            # 尝试在各数据根目录下查找
            for root in self.data_roots:
                full_path = Path(root) / loc
                if full_path.exists():
                    return str(full_path)
                
                # 尝试仅文件名
                if '\\' in loc or '/' in loc:
                    fname = Path(loc).name
                    for edf_file in Path(root).rglob(fname):
                        return str(edf_file)
        
        # 根据pt_id和fn构造路径
        pt_id = row.get('pt_id', '')
        fn = row.get('fn', '')
        
        for root in self.data_roots:
            root_path = Path(root)
            
            # 搜索包含pt_id的目录
            for patient_dir in root_path.rglob(f"*{pt_id}*"):
                if patient_dir.is_dir():
                    # 从fn提取SZ编号
                    import re
                    match = re.search(r'SZ(\d+)', fn, re.IGNORECASE)
                    if match:
                        sz_num = match.group(1)
                        for edf_file in patient_dir.glob(f"*SZ{sz_num}*.edf"):
                            return str(edf_file)
                        for edf_file in patient_dir.glob(f"*sz{sz_num}*.edf"):
                            return str(edf_file)
        
        return None
    
    def _parse_labels(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """解析所有类型的标签"""
        return {
            'onset_zone': parse_onset_zone_label(row.get('onset_zone')),
            'hemi': parse_hemi_label(row.get('hemi')),
            'channel': parse_channel_labels(row, self.channel_columns),
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 加载和预处理EDF数据
        try:
            data = self._load_and_preprocess(sample)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"加载数据失败 {sample['fn']}: {e}")
            # 返回零数据
            data = np.zeros((self.config.n_windows, 19, int(self.config.target_fs)))
        
        # 获取标签
        labels = sample['labels'][self.label_type]
        
        # 转换为张量
        data_tensor = torch.FloatTensor(data)  # (T, C, L)
        labels_tensor = torch.FloatTensor(labels)
        
        # 数据增强
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)
        
        return {
            'data': data_tensor,
            'labels': labels_tensor,
            'pt_id': sample['pt_id'],
            'fn': sample['fn'],
            'sz_idx': sample['sz_idx'],
        }
    
    def _load_and_preprocess(self, sample: Dict) -> np.ndarray:
        """
        加载并预处理EDF数据
        
        处理流程优化：
        1. 读取完整EDF数据
        2. 提取标准通道 + 可选双极转换
        3. 预处理（滤波、幅值裁剪）- 在完整数据上进行
        4. 重采样到目标采样率
        5. 提取baseline片段用于标准化
        6. 提取发作片段 + 移除范围内的坏段（正确处理时间偏移）
        7. 窗口划分
        8. 标准化
        """
        edf_path = sample['edf_path']
        sz_start = sample['sz_start']
        sz_end = sample['sz_end']
        base_line_start, base_line_end = sample['baseline']
        mask_segments = sample.get('mask_segments', [])  # 绝对时间戳的坏段
        
        # 1. 读取EDF
        raw_data, fs, ch_names = read_edf(edf_path)
        
        # 2. 提取标准19通道
        data, found_channels = extract_standard_channels(raw_data, ch_names)
        
        # 2.5. 可选：转换为TCP双极导联
        if hasattr(self.config, 'use_bipolar') and self.config.use_bipolar:
            bipolar_pairs = getattr(self.config, 'bipolar_pairs', BIPOLAR_PAIRS_18)
            data, found_channels = convert_to_bipolar(data, found_channels, bipolar_pairs)
            logger.debug(f"双极导联转换完成: {len(found_channels)} 通道")
        
        # 3. 预处理每个通道（在完整数据上进行）
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            # 带通滤波
            filtered = bandpass_filter(
                data[i], fs,
                f_low=self.config.filter_low,
                f_high=self.config.filter_high
            )
            # 幅值裁剪
            clipped = clip_amplitude(filtered, self.config.clip_std)
            processed_data[i] = clipped
        
        # 4. 重采样
        if fs != self.config.target_fs:
            n_samples_new = int(processed_data.shape[1] * self.config.target_fs / fs)
            resampled_data = np.zeros((processed_data.shape[0], n_samples_new))
            for i in range(processed_data.shape[0]):
                resampled_data[i] = resample_signal(
                    processed_data[i], fs, self.config.target_fs
                )
            processed_data = resampled_data
            
            # 重采样后需要调整mask_segments的时间戳（采样率变化）
            # 但由于我们使用的是秒为单位的时间戳，不需要调整
            fs = self.config.target_fs
        
        # 5. 提取baseline片段用于标准化（baseline通常不需要移除坏段）
        if base_line_start is not None and base_line_end is not None:
            bl_start_sample = int(base_line_start * fs)
            bl_end_sample = int(base_line_end * fs)
            bl_start_sample = max(0, bl_start_sample)
            bl_end_sample = min(processed_data.shape[1], bl_end_sample)
            baseline_segment = processed_data[:, bl_start_sample:bl_end_sample]
        else:
            baseline_segment = None
        
        # 6. 提取发作片段 + 移除范围内的坏段
        # 计算需要提取的时间范围（包含发作前后的窗口）
        segment_start = max(0, sz_start - self.window_before)
        segment_end = sz_end + self.window_after
        
        # 使用extract_segment_with_mask_removal正确处理坏段
        # 这会筛选范围内的坏段，转换为相对时间戳，然后移除
        segment, clean_duration = extract_segment_with_mask_removal(
            processed_data, fs, 
            segment_start, segment_end,
            mask_segments
        )
        
        # 检查清洗后的数据是否足够
        if segment.shape[1] == 0:
            logger.warning(f"片段 {sample['fn']} SZ{sample['sz_idx']} 清洗后没有数据")
            # 返回空窗口
            n_channels = processed_data.shape[0]
            n_samples = int(self.config.window_length * fs)
            return np.zeros((self.config.n_windows, n_channels, n_samples))
        
        # 7. 分割成窗口 - 使用自适应策略处理数据不足情况
        windows = adaptive_apply_windows(
            segment, fs,
            window_len=self.config.window_length,
            overlap=self.config.window_overlap,
            min_windows=2
        )  # (n_windows, n_channels, n_samples)

        # 8. 标准化
        if self.config.normalize:
            if baseline_segment is not None and baseline_segment.size > 0:
                # 基线标准化
                windows = (windows - np.mean(baseline_segment)) / (np.std(baseline_segment) + 1e-16)
            else:
                # 使用片段自身进行标准化
                windows = (windows - np.mean(segment)) / (np.std(segment) + 1e-16)
        
        # 9. 确保窗口数量（填充或截断到目标窗口数）
        target_windows = self.config.n_windows
        if windows.shape[0] < target_windows:
            # 填充
            pad_windows = target_windows - windows.shape[0]
            padding = np.zeros((pad_windows, windows.shape[1], windows.shape[2]))
            windows = np.concatenate([windows, padding], axis=0)
        elif windows.shape[0] > target_windows:
            # 截断（以发作onset为中心）
            sz_onset_window = int(self.window_before)  # 发作onset在第15个窗口附近
            start_idx = max(0, sz_onset_window - target_windows // 2)
            windows = windows[start_idx:start_idx + target_windows]
            if windows.shape[0] < target_windows:
                pad_windows = target_windows - windows.shape[0]
                padding = np.zeros((pad_windows, windows.shape[1], windows.shape[2]))
                windows = np.concatenate([windows, padding], axis=0)
        
        return windows


class MultiLabelEEGDataset(PrivateEEGDataset):
    """
    多标签EEG数据集 - 同时返回所有标签类型
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本（包含所有标签）"""
        sample = self.samples[idx]
        
        # 加载和预处理EDF数据
        try:
            data = self._load_and_preprocess(sample)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"加载数据失败 {sample['fn']}: {e}")
            data = np.zeros((self.config.n_windows, 19, int(self.config.target_fs)))
        
        # 转换为张量
        data_tensor = torch.FloatTensor(data)
        
        # 所有标签
        onset_zone_labels = torch.FloatTensor(sample['labels']['onset_zone'])
        hemi_labels = torch.FloatTensor(sample['labels']['hemi'])
        channel_labels = torch.FloatTensor(sample['labels']['channel'])
        
        # 数据增强
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)
        
        return {
            'data': data_tensor,
            'onset_zone_labels': onset_zone_labels,
            'hemi_labels': hemi_labels,
            'channel_labels': channel_labels,
            'pt_id': sample['pt_id'],
            'fn': sample['fn'],
            'sz_idx': sample['sz_idx'],
        }


# ==============================================================================
# 数据划分工具
# ==============================================================================

def get_patient_ids(manifest_path: str) -> List[str]:
    """获取所有患者ID"""
    df = pd.read_csv(manifest_path)
    return df['pt_id'].unique().tolist()


def create_cross_validation_splits(
    manifest_path: str,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    创建交叉验证划分（按患者划分）
    
    Returns:
        List of (train_patient_ids, val_patient_ids)
    """
    np.random.seed(seed)
    
    patient_ids = get_patient_ids(manifest_path)
    np.random.shuffle(patient_ids)
    
    splits = []
    fold_size = len(patient_ids) // n_folds
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(patient_ids)
        
        val_ids = patient_ids[start_idx:end_idx]
        train_ids = [pid for pid in patient_ids if pid not in val_ids]
        
        splits.append((train_ids, val_ids))
    
    return splits


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    config = get_config()
    
    print("测试数据加载器...")
    print(f"Manifest路径: {config.data.manifest_path}")
    
    # 测试数据集
    try:
        dataset = PrivateEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type='channel',
            config=config.data
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"数据形状: {sample['data'].shape}")
            print(f"标签形状: {sample['labels'].shape}")
            print(f"患者ID: {sample['pt_id']}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
