#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUSZ EDF Data Loader

加载TUSZ数据集的EDF文件，并进行预处理：
1. 读取EDF文件
2. 转换为TCP双极导联
3. 带通滤波 (1.6-30 Hz)
4. 幅值裁剪 (±2 std)
5. 重采样到目标采样率 (200 Hz)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings

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

from config import (
    TUSZ_CONFIG,
    BIPOLAR_PAIRS,
    BIPOLAR_PAIRS_18,
    BIPOLAR_CHANNELS,
    BIPOLAR_CHANNELS_18,
    STANDARD_19_CHANNELS,
    normalize_channel_name,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# EDF文件读取
# ==============================================================================

def read_edf_pyedflib(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    使用pyedflib读取EDF文件
    
    Args:
        filepath: EDF文件路径
        
    Returns:
        (data, fs, ch_names)
        - data: (n_channels, n_samples)
        - fs: 采样率
        - ch_names: 通道名列表
    """
    if not HAS_PYEDFLIB:
        raise ImportError("pyedflib未安装")
    
    f = pyedflib.EdfReader(filepath)
    try:
        n_channels = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs_list = [f.getSampleFrequency(i) for i in range(n_channels)]
        fs = fs_list[0]  # 假设所有通道采样率相同
        
        n_samples = f.getNSamples()[0]
        data = np.zeros((n_channels, n_samples))
        
        for i in range(n_channels):
            data[i] = f.readSignal(i)
        
        return data, fs, ch_names
    finally:
        f._close()


def read_edf_mne(filepath: str, encoding: str = 'utf-8') -> Tuple[np.ndarray, float, List[str]]:
    """
    使用MNE读取EDF文件
    
    Args:
        filepath: EDF文件路径
        encoding: 字符编码
        
    Returns:
        (data, fs, ch_names)
    """
    if not HAS_MNE:
        raise ImportError("mne未安装")
    
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR', encoding=encoding)
    data = raw.get_data()
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    return data, fs, ch_names


def read_edf(filepath: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    读取EDF文件（自动尝试多种方法）
    
    Args:
        filepath: EDF文件路径
        
    Returns:
        (data, fs, ch_names)
    """
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


# ==============================================================================
# 信号处理
# ==============================================================================

def apply_lowpass(x: np.ndarray, fs: float, fc: float = 30, order: int = 4) -> np.ndarray:
    """
    低通滤波
    
    Args:
        x: 输入信号 (n_channels, n_samples) 或 (n_samples,)
        fs: 采样率
        fc: 截止频率
        order: 滤波器阶数
        
    Returns:
        滤波后的信号
    """
    wc = fc / (fs / 2)
    if wc >= 1:
        wc = 0.99
    b, a = sig.butter(order, wc, btype='low')
    
    if x.ndim == 1:
        return sig.filtfilt(b, a, x)
    else:
        return np.array([sig.filtfilt(b, a, ch) for ch in x])


def apply_highpass(x: np.ndarray, fs: float, fc: float = 1.6, order: int = 4) -> np.ndarray:
    """
    高通滤波
    
    Args:
        x: 输入信号
        fs: 采样率
        fc: 截止频率
        order: 滤波器阶数
        
    Returns:
        滤波后的信号
    """
    wc = fc / (fs / 2)
    if wc >= 1:
        wc = 0.99
    b, a = sig.butter(order, wc, btype='high')
    
    if x.ndim == 1:
        return sig.filtfilt(b, a, x)
    else:
        return np.array([sig.filtfilt(b, a, ch) for ch in x])


def bandpass_filter(
    x: np.ndarray, 
    fs: float, 
    f_low: float = None, 
    f_high: float = None
) -> np.ndarray:
    """
    带通滤波
    
    Args:
        x: 输入信号
        fs: 采样率
        f_low: 低截止频率
        f_high: 高截止频率
        
    Returns:
        滤波后的信号
    """
    if f_low is None:
        f_low = TUSZ_CONFIG['filter_low']
    if f_high is None:
        f_high = TUSZ_CONFIG['filter_high']
    
    x_high = apply_highpass(x, fs, fc=f_low)
    x_band = apply_lowpass(x_high, fs, fc=f_high)
    return x_band


def clip_amplitude(x: np.ndarray, n_std: float = None) -> np.ndarray:
    """
    幅值裁剪（按标准差）
    
    Args:
        x: 输入信号
        n_std: 裁剪的标准差倍数
        
    Returns:
        裁剪后的信号
    """
    if n_std is None:
        n_std = TUSZ_CONFIG['clip_std']
    
    if x.ndim == 1:
        mean = np.mean(x)
        std = np.std(x)
        return np.clip(x, mean - n_std * std, mean + n_std * std)
    else:
        result = np.zeros_like(x)
        for i, ch in enumerate(x):
            mean = np.mean(ch)
            std = np.std(ch)
            if std > 0:
                result[i] = np.clip(ch, mean - n_std * std, mean + n_std * std)
            else:
                result[i] = ch
        return result


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float = None) -> np.ndarray:
    """
    重采样信号
    
    Args:
        x: 输入信号 (n_channels, n_samples) 或 (n_samples,)
        orig_fs: 原始采样率
        target_fs: 目标采样率
        
    Returns:
        重采样后的信号
    """
    if target_fs is None:
        target_fs = TUSZ_CONFIG['target_fs']
    
    if orig_fs == target_fs:
        return x
    
    ratio = target_fs / orig_fs
    
    if x.ndim == 1:
        num_samples = int(len(x) * ratio)
        return resample(x, num_samples)
    else:
        n_channels, n_samples = x.shape
        new_n_samples = int(n_samples * ratio)
        result = np.zeros((n_channels, new_n_samples))
        for i in range(n_channels):
            result[i] = resample(x[i], new_n_samples)
        return result


def normalize_zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Z-score标准化
    
    Args:
        x: 输入信号
        axis: 标准化的轴
        
    Returns:
        标准化后的信号
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)  # 避免除以0
    return (x - mean) / std


# ==============================================================================
# TCP双极导联转换
# ==============================================================================

def convert_to_bipolar(
    data: np.ndarray,
    ch_names: List[str],
    bipolar_pairs: List[Tuple[str, str]] = None,
    use_18_channels: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    将单极参考导联转换为TCP双极导联
    
    Args:
        data: 单极导联数据 (n_channels, n_samples)
        ch_names: 通道名列表
        bipolar_pairs: 双极导联对列表，None则使用默认配置
        use_18_channels: 是否使用18通道配置
        
    Returns:
        (bipolar_data, bipolar_names)
        - bipolar_data: 双极导联数据
        - bipolar_names: 双极导联通道名
    """
    if bipolar_pairs is None:
        if use_18_channels:
            bipolar_pairs = BIPOLAR_PAIRS_18
        else:
            bipolar_pairs = BIPOLAR_PAIRS
    
    # 建立标准化通道名到索引的映射
    ch_map = {}
    for i, name in enumerate(ch_names):
        normalized = normalize_channel_name(name)
        ch_map[normalized] = i
    
    bipolar_data = []
    bipolar_names = []
    missing_pairs = []
    
    for anode, cathode in bipolar_pairs:
        anode_norm = normalize_channel_name(anode)
        cathode_norm = normalize_channel_name(cathode)
        
        if anode_norm in ch_map and cathode_norm in ch_map:
            idx_anode = ch_map[anode_norm]
            idx_cathode = ch_map[cathode_norm]
            
            # 双极导联 = 阳极 - 阴极
            diff_signal = data[idx_anode, :] - data[idx_cathode, :]
            bipolar_data.append(diff_signal)
            bipolar_names.append(f"{anode_norm}-{cathode_norm}")
        else:
            missing_pairs.append((anode, cathode))
            # 用零填充缺失的导联
            bipolar_data.append(np.zeros(data.shape[1]))
            bipolar_names.append(f"{anode_norm}-{cathode_norm}")
    
    if missing_pairs:
        logger.warning(f"双极导联转换: 缺失导联对 {missing_pairs}")
    
    return np.array(bipolar_data), bipolar_names


def detect_montage_type(edf_path: str) -> str:
    """
    从文件路径检测montage类型
    
    TUSZ数据集的montage类型编码在目录名中:
    - 01_tcp_ar: TCP Averaged Reference
    - 02_tcp_le: TCP Linked Ears
    - 03_tcp_ar_a: TCP AR (alternate)
    - 04_tcp_le_a: TCP LE (alternate)
    
    Args:
        edf_path: EDF文件路径
        
    Returns:
        montage类型字符串
    """
    path = Path(edf_path)
    
    # 检查路径中的目录名
    for part in path.parts:
        if part.startswith('01_tcp_ar'):
            return 'tcp_ar'
        elif part.startswith('02_tcp_le'):
            return 'tcp_le'
        elif part.startswith('03_tcp_ar_a'):
            return 'tcp_ar_a'
        elif part.startswith('04_tcp_le_a'):
            return 'tcp_le_a'
    
    return 'unknown'


# ==============================================================================
# 窗口处理
# ==============================================================================

def apply_windows(
    x: np.ndarray,
    fs: float,
    window_len: float = None,
    overlap: float = None
) -> np.ndarray:
    """
    将信号分割成固定长度的窗口
    
    Args:
        x: 输入信号 (n_channels, n_samples)
        fs: 采样率
        window_len: 窗口长度（秒）
        overlap: 重叠比例 (0-1)
        
    Returns:
        windows: (n_windows, n_channels, samples_per_window)
    """
    if window_len is None:
        window_len = TUSZ_CONFIG['window_len']
    if overlap is None:
        overlap = TUSZ_CONFIG['window_overlap']
    
    samples_per_window = int(window_len * fs)
    step = int(samples_per_window * (1 - overlap))
    
    n_channels, n_samples = x.shape
    n_windows = max(1, (n_samples - samples_per_window) // step + 1)
    
    windows = np.zeros((n_windows, n_channels, samples_per_window))
    
    for i in range(n_windows):
        start = i * step
        end = start + samples_per_window
        if end <= n_samples:
            windows[i] = x[:, start:end]
        else:
            # 边界情况：填充
            valid_len = n_samples - start
            windows[i, :, :valid_len] = x[:, start:n_samples]
    
    return windows


def extract_segment(
    x: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float
) -> np.ndarray:
    """
    提取指定时间段的数据
    
    Args:
        x: 输入信号 (n_channels, n_samples)
        fs: 采样率
        start_time: 起始时间（秒）
        end_time: 结束时间（秒）
        
    Returns:
        segment: (n_channels, segment_samples)
    """
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    
    # 边界检查
    start_sample = max(0, start_sample)
    end_sample = min(x.shape[1], end_sample)
    
    return x[:, start_sample:end_sample]


# ==============================================================================
# 数据加载器类
# ==============================================================================

class TUSZDataLoader:
    """
    TUSZ EDF数据加载器
    
    负责加载EDF文件并进行预处理。
    
    使用示例:
        loader = TUSZDataLoader()
        data, fs, ch_names = loader.load_and_preprocess('path/to/file.edf')
    """
    
    def __init__(
        self,
        target_fs: float = None,
        filter_low: float = None,
        filter_high: float = None,
        clip_std: float = None,
        use_18_channels: bool = True,
        normalize: bool = True
    ):
        """
        初始化数据加载器
        
        Args:
            target_fs: 目标采样率
            filter_low: 高通截止频率
            filter_high: 低通截止频率
            clip_std: 幅值裁剪标准差倍数
            use_18_channels: 是否使用18通道TCP配置
            normalize: 是否进行Z-score标准化
        """
        self.target_fs = target_fs or TUSZ_CONFIG['target_fs']
        self.filter_low = filter_low or TUSZ_CONFIG['filter_low']
        self.filter_high = filter_high or TUSZ_CONFIG['filter_high']
        self.clip_std = clip_std or TUSZ_CONFIG['clip_std']
        self.use_18_channels = use_18_channels
        self.normalize = normalize
        
        if use_18_channels:
            self.bipolar_pairs = BIPOLAR_PAIRS_18
            self.bipolar_channels = BIPOLAR_CHANNELS_18
        else:
            self.bipolar_pairs = BIPOLAR_PAIRS
            self.bipolar_channels = BIPOLAR_CHANNELS
    
    def load_edf(self, edf_path: str) -> Tuple[np.ndarray, float, List[str]]:
        """
        加载EDF文件（原始数据）
        
        Args:
            edf_path: EDF文件路径
            
        Returns:
            (data, fs, ch_names)
        """
        return read_edf(edf_path)
    
    def preprocess(
        self,
        data: np.ndarray,
        fs: float,
        do_filter: bool = True,
        do_clip: bool = True,
        do_resample: bool = True,
        do_normalize: bool = None
    ) -> Tuple[np.ndarray, float]:
        """
        预处理信号
        
        Args:
            data: 输入数据 (n_channels, n_samples)
            fs: 采样率
            do_filter: 是否进行带通滤波
            do_clip: 是否进行幅值裁剪
            do_resample: 是否重采样
            do_normalize: 是否标准化
            
        Returns:
            (processed_data, output_fs)
        """
        if do_normalize is None:
            do_normalize = self.normalize
        
        # 1. 带通滤波
        if do_filter:
            data = bandpass_filter(data, fs, self.filter_low, self.filter_high)
        
        # 2. 幅值裁剪
        if do_clip:
            data = clip_amplitude(data, self.clip_std)
        
        # 3. 重采样
        if do_resample and fs != self.target_fs:
            data = resample_signal(data, fs, self.target_fs)
            fs = self.target_fs
        
        # 4. Z-score标准化
        if do_normalize:
            data = normalize_zscore(data)
        
        return data, fs
    
    def load_and_preprocess(
        self,
        edf_path: str,
        convert_bipolar: bool = True,
        start_time: float = None,
        end_time: float = None
    ) -> Tuple[np.ndarray, float, List[str]]:
        """
        加载EDF文件并进行完整预处理
        
        Args:
            edf_path: EDF文件路径
            convert_bipolar: 是否转换为双极导联
            start_time: 起始时间（秒），None表示从头开始
            end_time: 结束时间（秒），None表示到结尾
            
        Returns:
            (data, fs, ch_names)
        """
        # 加载原始数据
        data, fs, ch_names = self.load_edf(edf_path)
        
        # 提取时间段
        if start_time is not None or end_time is not None:
            if start_time is None:
                start_time = 0
            if end_time is None:
                end_time = data.shape[1] / fs
            data = extract_segment(data, fs, start_time, end_time)
        
        # 转换为双极导联
        if convert_bipolar:
            data, ch_names = convert_to_bipolar(
                data, ch_names, 
                bipolar_pairs=self.bipolar_pairs,
                use_18_channels=self.use_18_channels
            )
        
        # 预处理
        data, fs = self.preprocess(data, fs)
        
        return data, fs, ch_names
    
    def load_seizure_segment(
        self,
        edf_path: str,
        seizure_start: float,
        seizure_end: float,
        pre_buffer: float = None,
        post_buffer: float = None
    ) -> Tuple[np.ndarray, float, List[str], Dict]:
        """
        加载癫痫发作段及其前后缓冲区
        
        Args:
            edf_path: EDF文件路径
            seizure_start: 发作开始时间
            seizure_end: 发作结束时间
            pre_buffer: 发作前缓冲时间
            post_buffer: 发作后缓冲时间
            
        Returns:
            (data, fs, ch_names, segment_info)
        """
        if pre_buffer is None:
            pre_buffer = TUSZ_CONFIG['pre_seizure_buffer']
        if post_buffer is None:
            post_buffer = TUSZ_CONFIG['post_seizure_buffer']
        
        # 计算实际提取范围
        segment_start = max(0, seizure_start - pre_buffer)
        segment_end = seizure_end + post_buffer
        
        # 加载并预处理
        data, fs, ch_names = self.load_and_preprocess(
            edf_path,
            start_time=segment_start,
            end_time=segment_end
        )
        
        # 计算相对于segment的发作时间
        segment_info = {
            'segment_start': segment_start,
            'segment_end': segment_end,
            'seizure_start_relative': seizure_start - segment_start,
            'seizure_end_relative': seizure_end - segment_start,
            'pre_buffer': pre_buffer,
            'post_buffer': post_buffer,
        }
        
        return data, fs, ch_names, segment_info


# ==============================================================================
# 便捷函数
# ==============================================================================

def load_tusz_edf(
    edf_path: str,
    preprocess: bool = True,
    use_18_channels: bool = True
) -> Tuple[np.ndarray, float, List[str]]:
    """
    便捷函数：加载TUSZ EDF文件
    
    Args:
        edf_path: EDF文件路径
        preprocess: 是否进行预处理
        use_18_channels: 是否使用18通道配置
        
    Returns:
        (data, fs, ch_names)
    """
    loader = TUSZDataLoader(use_18_channels=use_18_channels)
    
    if preprocess:
        return loader.load_and_preprocess(edf_path)
    else:
        data, fs, ch_names = loader.load_edf(edf_path)
        data, ch_names = convert_to_bipolar(
            data, ch_names,
            use_18_channels=use_18_channels
        )
        return data, fs, ch_names


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_edf = "F:/dataset/TUSZ/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf"
        
        print("=" * 60)
        print("测试TUSZ EDF数据加载")
        print("=" * 60)
        
        if Path(test_edf).exists():
            # 测试原始加载
            print("\n1. 加载原始数据...")
            data, fs, ch_names = read_edf(test_edf)
            print(f"   原始数据形状: {data.shape}")
            print(f"   采样率: {fs} Hz")
            print(f"   通道数: {len(ch_names)}")
            print(f"   时长: {data.shape[1]/fs:.2f}s")
            print(f"   前5个通道: {ch_names[:5]}")
            
            # 测试双极转换
            print("\n2. 转换为双极导联...")
            bipolar_data, bipolar_names = convert_to_bipolar(data, ch_names, use_18_channels=True)
            print(f"   双极数据形状: {bipolar_data.shape}")
            print(f"   双极通道: {bipolar_names}")
            
            # 测试完整加载
            print("\n3. 完整加载（含预处理）...")
            loader = TUSZDataLoader(use_18_channels=True)
            proc_data, proc_fs, proc_names = loader.load_and_preprocess(test_edf)
            print(f"   处理后数据形状: {proc_data.shape}")
            print(f"   处理后采样率: {proc_fs} Hz")
            print(f"   时长: {proc_data.shape[1]/proc_fs:.2f}s")
            print(f"   数据范围: [{proc_data.min():.3f}, {proc_data.max():.3f}]")
            print(f"   数据均值: {proc_data.mean():.6f}")
            print(f"   数据标准差: {proc_data.std():.6f}")
            
            # 测试窗口分割
            print("\n4. 窗口分割...")
            windows = apply_windows(proc_data, proc_fs, window_len=20.0, overlap=0.5)
            print(f"   窗口形状: {windows.shape}")
            print(f"   窗口数: {windows.shape[0]}")
            print(f"   每窗口样本数: {windows.shape[2]}")
            
            # 测试montage检测
            print("\n5. Montage类型检测...")
            montage = detect_montage_type(test_edf)
            print(f"   检测到montage类型: {montage}")
            
            print("\n测试完成!")
        else:
            print(f"测试文件不存在: {test_edf}")
    else:
        print("用法: python -m TUSZ.data_loader --test")
