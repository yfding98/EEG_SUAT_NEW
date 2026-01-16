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

# 通道名映射（处理各种变体）
CHANNEL_NAME_MAP = {
    # 大小写变体
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ', 'Oz': 'OZ',
    # 10-10到10-20系统映射
    'T7': 'T3', 'T8': 'T4',
    'P7': 'T5', 'P8': 'T6',
    # 带参考的通道名
    'EEG FP1': 'FP1', 'EEG FP2': 'FP2',
    'EEG F7': 'F7', 'EEG F3': 'F3', 'EEG FZ': 'FZ', 'EEG F4': 'F4', 'EEG F8': 'F8',
    'EEG T3': 'T3', 'EEG C3': 'C3', 'EEG CZ': 'CZ', 'EEG C4': 'C4', 'EEG T4': 'T4',
    'EEG T5': 'T5', 'EEG P3': 'P3', 'EEG PZ': 'PZ', 'EEG P4': 'P4', 'EEG T6': 'T6',
    'EEG O1': 'O1', 'EEG O2': 'O2',
}

# 脑区映射
BRAIN_REGION_MAP = {
    'frontal': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'central': ['C3', 'C4', 'CZ'],
    'parietal': ['P3', 'P4', 'PZ'],
    'occipital': ['O1', 'O2']
}

# 半球映射
HEMISPHERE_MAP = {
    'L': ['FP1', 'F3', 'F7', 'T3', 'C3', 'T5', 'P3', 'O1'],
    'R': ['FP2', 'F4', 'F8', 'T4', 'C4', 'T6', 'P4', 'O2'],
    'M': ['FZ', 'CZ', 'PZ']
}


# ==============================================================================
# 信号处理函数
# ==============================================================================

# 标准18导联TCP双极导联对
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

# 双极导联通道名称
BIPOLAR_CHANNEL_NAMES = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS_18]


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
        """加载并预处理EDF数据"""
        edf_path = sample['edf_path']
        sz_start = sample['sz_start']
        sz_end = sample['sz_end']
        base_line_start, base_line_end = sample['baseline']
        
        # 1. 读取EDF
        raw_data, fs, ch_names = read_edf(edf_path)
        
        # 2. 提取标准19通道
        data, found_channels = extract_standard_channels(raw_data, ch_names)
        
        # 2.5. 可选：转换为TCP双极导联
        if hasattr(self.config, 'use_bipolar') and self.config.use_bipolar:
            bipolar_pairs = getattr(self.config, 'bipolar_pairs', BIPOLAR_PAIRS_18)
            data, found_channels = convert_to_bipolar(data, found_channels, bipolar_pairs)
            logger.debug(f"双极导联转换完成: {len(found_channels)} 通道")
        
        # 3. 预处理每个通道
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
            fs = self.config.target_fs
        
        # 5. 提取发作片段（发作前后的窗口）
        start_sample = max(0, int((sz_start - self.window_before) * fs))
        end_sample = min(processed_data.shape[1], int((sz_end + self.window_after) * fs))
        
        segment = processed_data[:, start_sample:end_sample]
        baseline_segment = processed_data[:,int(base_line_start * fs):int(base_line_end * fs)]
        
        # 6. 分割成窗口
        windows = apply_windows(
            segment, fs,
            window_len=self.config.window_length,
            overlap=self.config.window_overlap
        )  # (n_windows, 19, 200)


        # 7. 标准化
        if self.config.normalize:
            # # 按窗口的标准化
            # windows = (windows - np.mean(windows)) / (np.std(windows) + 1e-16)
            # 基线标准化
            windows = (windows - np.mean(baseline_segment)) / (np.std(baseline_segment) + 1e-16)
        
        # 8. 确保窗口数量（填充或截断到45窗口）
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
