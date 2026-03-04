#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态EEG数据集

支持滑动窗口数据增强，同时返回:
1. 原始EEG波形
2. 连接性矩阵 (PLV, wPLI, AEC, Pearson, Granger, TE)
3. 图网络指标 (degree, strength, clustering, betweenness, eigenvector)

关键特性:
- 可配置的滑动窗口 (支持重叠以增加数据量)
- 在线或离线计算连接性特征
- 与现有dataset.py兼容
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import warnings
import logging

from scipy import signal as sig
from scipy.signal import resample

# 导入现有模块
from dataset import (
    PrivateEEGDataset, 
    read_edf, 
    extract_standard_channels,
    bandpass_filter,
    clip_amplitude,
    resample_signal,
    apply_windows,
    adaptive_apply_windows,
    parse_onset_zone_label,
    parse_hemi_label,
    parse_channel_labels,
    parse_baseline,
    parse_seizure_times,
    parse_mask_segments,
    filter_mask_segments_for_range,
    calculate_clean_duration,  # 计算清洗后有效时长
    extract_segment_with_mask_removal,
    convert_to_bipolar,
    STANDARD_19_CHANNELS,
    BIPOLAR_PAIRS_18,
    BIPOLAR_PAIRS_26,
    STANDARD_21_CHANNELS,
)

from connectivity import (
    compute_plv,
    compute_wpli,
    compute_pearson_corr,
    compute_aec,
    compute_granger_causality,
    compute_transfer_entropy,
    compute_all_connectivity
)

from config import DataConfig, get_config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# 5脑区双极导联标签映射 (region_5)
# ==============================================================================

# 5脑区双极导联映射
REGION_5_BIPOLAR_LEADS = {
    'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
    'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
    'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
    'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
    'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
}

# 双极导联到所涉及电极的映射
BIPOLAR_LEAD_ELECTRODES = {
    # 左额
    'FP1-F7': ['fp1', 'f7'], 'FP1-F3': ['fp1', 'f3'], 
    'F7-F3': ['f7', 'f3'], 'F3-FZ': ['f3', 'fz'],
    # 左颞
    'F7-SPHL': ['f7', 'sphl'], 'SPHL-T3': ['sphl', 't3'],
    'T3-T5': ['t3', 't5'], 'T5-O1': ['t5', 'o1'],
    'T3-C3': ['t3', 'c3'], 'T5-P3': ['t5', 'p3'],
    # 顶叶
    'FZ-CZ': ['fz', 'cz'], 'C3-CZ': ['c3', 'cz'],
    'P3-PZ': ['p3', 'pz'], 'CZ-PZ': ['cz', 'pz'],
    'CZ-C4': ['cz', 'c4'], 'PZ-P4': ['pz', 'p4'],
    # 右额
    'FP2-F4': ['fp2', 'f4'], 'FP2-F8': ['fp2', 'f8'],
    'F4-F8': ['f4', 'f8'], 'FZ-F4': ['fz', 'f4'],
    # 右颞
    'F8-SPHR': ['f8', 'sphr'], 'SPHR-T4': ['sphr', 't4'],
    'C4-T4': ['c4', 't4'], 'T4-T6': ['t4', 't6'],
    'P4-T6': ['p4', 't6'], 'T6-O2': ['t6', 'o2'],
}

# chain标签的电极映射
CHAIN_ELECTRODES = {
    'left_temporal': ['fp1', 'f7', 't3', 't5', 'o1'],
    'right_temporal': ['fp2', 'f8', 't4', 't6', 'o2'],
    'left_parasagittal': ['fp1', 'f3', 'c3', 'p3', 'o1'],
    'right_parasagittal': ['fp2', 'f4', 'c4', 'p4', 'o2'],
    'midline': ['fz', 'cz', 'pz'],
}


def parse_region5_label(row: pd.Series) -> np.ndarray:
    """
    解析5脑区(双极导联)标签
    
    根据通道级别SOZ标注，推断出每个脑区是否为SOZ
    判断逻辑：如果某双极导联的任一组成电极是SOZ，则该导联是SOZ
              如果某脑区的任一双极导联是SOZ，则该脑区是SOZ
    
    Args:
        row: DataFrame行，包含通道级别标签
    
    Returns:
        5维向量 [left_frontal, left_temporal, parietal, right_frontal, right_temporal]
    """
    region_names = ['left_frontal', 'left_temporal', 'parietal', 
                    'right_frontal', 'right_temporal']
    region_labels = np.zeros(len(region_names), dtype=np.float32)
    
    # 读取通道级别SOZ（包括21电极）
    all_channel_names = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
                         't3', 'c3', 'cz', 'c4', 't4',
                         't5', 'p3', 'pz', 'p4', 't6',
                         'o1', 'o2', 'sphl', 'sphr']
    
    soz_channels = set()
    for ch in all_channel_names:
        col = ch if ch in row.index else ch.upper()
        if col in row.index:
            try:
                if pd.notna(row[col]) and int(row[col]) == 1:
                    soz_channels.add(ch.lower())
            except:
                pass
    
    # 根据SOZ通道计算每个脑区的标签
    for region_idx, region_name in enumerate(region_names):
        if region_name not in REGION_5_BIPOLAR_LEADS:
            continue
        
        region_bipolar_leads = REGION_5_BIPOLAR_LEADS[region_name]
        region_is_soz = False
        
        for lead in region_bipolar_leads:
            if lead in BIPOLAR_LEAD_ELECTRODES:
                electrodes = BIPOLAR_LEAD_ELECTRODES[lead]
                # 如果任一电极是SOZ，则该导联是SOZ
                if any(elec.lower() in soz_channels for elec in electrodes):
                    region_is_soz = True
                    break
        
        if region_is_soz:
            region_labels[region_idx] = 1.0
    
    return region_labels


def parse_chain_label(row: pd.Series) -> np.ndarray:
    """
    解析五链分区标签
    
    Args:
        row: DataFrame行，包含通道级别标签
    
    Returns:
        5维向量 [left_temporal, right_temporal, left_parasagittal, right_parasagittal, midline]
    """
    chain_names = ['left_temporal', 'right_temporal', 'left_parasagittal', 
                   'right_parasagittal', 'midline']
    chain_labels = np.zeros(len(chain_names), dtype=np.float32)
    
    # 读取通道级别SOZ
    channel_names = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
                     't3', 'c3', 'cz', 'c4', 't4',
                     't5', 'p3', 'pz', 'p4', 't6',
                     'o1', 'o2']
    
    soz_channels = set()
    for ch in channel_names:
        col = ch if ch in row.index else ch.upper()
        if col in row.index:
            try:
                if pd.notna(row[col]) and int(row[col]) == 1:
                    soz_channels.add(ch.lower())
            except:
                pass
    
    # 根据SOZ通道计算chain标签
    for chain_idx, chain_name in enumerate(chain_names):
        if chain_name not in CHAIN_ELECTRODES:
            continue
        chain_electrodes = CHAIN_ELECTRODES[chain_name]
        # 如果该链中任何电极是SOZ，则该链为正
        if any(elec.lower() in soz_channels for elec in chain_electrodes):
            chain_labels[chain_idx] = 1.0
    
    return chain_labels


# ==============================================================================
# 图网络指标计算 (简化版，用于在线计算)
# ==============================================================================

def compute_graph_metrics_from_connectivity(
    connectivity_matrix: np.ndarray,
    metric_names: List[str] = None
) -> np.ndarray:
    """
    从连接矩阵计算节点级别图网络指标
    
    Args:
        connectivity_matrix: (n_channels, n_channels) 连接矩阵
        metric_names: 要计算的指标列表
    
    Returns:
        metrics: (n_channels, n_metrics) 图指标
    """
    if metric_names is None:
        metric_names = ['degree', 'strength', 'clustering', 'betweenness', 'eigenvector']
    
    n_channels = connectivity_matrix.shape[0]
    metrics = []
    
    # 取绝对值并去除对角线
    adj = np.abs(connectivity_matrix.copy())
    np.fill_diagonal(adj, 0)
    
    # 阈值化 (保留top 20%的连接)
    threshold = np.percentile(adj.flatten(), 80)
    adj_binary = (adj >= threshold).astype(float)
    
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        G_binary = nx.from_numpy_array(adj_binary)
        
        for metric in metric_names:
            if metric == 'degree':
                # 度数 (二值邻接矩阵)
                degrees = dict(G_binary.degree())
                values = np.array([degrees[i] for i in range(n_channels)])
            elif metric == 'strength':
                # 节点强度 (加权度数)
                strengths = dict(G.degree(weight='weight'))
                values = np.array([strengths[i] for i in range(n_channels)])
            elif metric == 'clustering':
                # 聚类系数
                clustering = nx.clustering(G, weight='weight')
                values = np.array([clustering[i] for i in range(n_channels)])
            elif metric == 'betweenness':
                # 中介中心性
                if G.number_of_edges() > 0:
                    betweenness = nx.betweenness_centrality(G, weight='weight')
                    values = np.array([betweenness[i] for i in range(n_channels)])
                else:
                    values = np.zeros(n_channels)
            elif metric == 'eigenvector':
                # 特征向量中心性
                try:
                    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                    values = np.array([eigenvector[i] for i in range(n_channels)])
                except:
                    values = np.zeros(n_channels)
            else:
                values = np.zeros(n_channels)
            
            metrics.append(values)
        
    except Exception as e:
        logger.warning(f"图指标计算失败: {e}")
        metrics = [np.zeros(n_channels) for _ in metric_names]
    
    return np.stack(metrics, axis=-1)  # (n_channels, n_metrics)


# ==============================================================================
# 滑动窗口采样器
# ==============================================================================

class SlidingWindowSampler:
    """
    滑动窗口采样器
    
    将长片段分割成多个重叠的子片段以增加数据量
    """
    
    def __init__(
        self,
        window_length: float = 20.0,  # 窗口长度(秒)
        overlap: float = 0.5,          # 重叠比例
        min_windows: int = 1           # 最少生成的窗口数
    ):
        self.window_length = window_length
        self.overlap = overlap
        self.min_windows = min_windows
    
    def get_windows(
        self, 
        total_length: float,
        fs: float = 200.0
    ) -> List[Tuple[int, int]]:
        """
        获取滑动窗口的起止采样点
        
        Args:
            total_length: 总时长(秒)
            fs: 采样率
        
        Returns:
            windows: [(start_sample, end_sample), ...]
        """
        window_samples = int(self.window_length * fs)
        step_samples = int(window_samples * (1 - self.overlap))
        total_samples = int(total_length * fs)
        
        windows = []
        start = 0
        
        while start + window_samples <= total_samples:
            end = start + window_samples
            windows.append((start, end))
            start += step_samples
        
        # 确保至少有min_windows个窗口
        if len(windows) < self.min_windows and total_samples >= window_samples:
            windows = [(0, window_samples)]
        elif len(windows) == 0 and total_samples > 0:
            # 数据不够一个完整窗口，使用全部数据
            windows = [(0, total_samples)]
        
        return windows


# ==============================================================================
# 多模态EEG数据集
# ==============================================================================

class MultiModalEEGDataset(Dataset):
    """
    多模态EEG数据集
    
    返回:
        - data: 原始EEG波形 (T, C, L)
        - connectivity: 连接性矩阵 (M, C, C)
        - graph_metrics: 图网络指标 (C, K)
        - labels: 标签
    
    关键特性:
        - 滑动窗口数据增强
        - 支持离线预计算或在线计算连接性特征
    """
    
    # 连接性类型
    CONNECTIVITY_TYPES = ['plv', 'wpli', 'aec', 'pearson', 'granger', 'transfer_entropy']
    
    # 图指标类型
    GRAPH_METRIC_TYPES = ['degree', 'strength', 'clustering', 'betweenness', 'eigenvector']
    
    def __init__(
        self,
        manifest_path: str,
        data_roots: List[str],
        label_type: str = 'onset_zone',
        patient_ids: List[str] = None,
        config: DataConfig = None,
        # 滑动窗口参数
        segment_length: float = 20.0,      # 片段长度(秒)
        segment_overlap: float = 0.5,       # 片段重叠比例
        # 内部窗口参数 (用于EEGNet)
        window_length: float = 1.0,         # 内部窗口长度(秒)
        window_overlap: float = 0.0,        # 内部窗口重叠
        # 连接性参数
        connectivity_types: List[str] = None,
        connectivity_freq_band: Tuple[float, float] = (8, 30),  # Alpha+Beta
        precomputed_features_dir: str = None,  # 预计算特征目录
        compute_online: bool = True,        # 是否在线计算
        include_directed: bool = True,      # 是否包含有向指标 (Granger, TE)
        # 其他
        transform = None,
        max_seizures: int = 10
    ):
        """
        Args:
            manifest_path: CSV manifest文件路径
            data_roots: EDF数据根目录列表
            label_type: 标签类型 ('onset_zone', 'hemi', 'channel')
            patient_ids: 患者ID列表（用于交叉验证划分）
            config: 数据配置
            segment_length: 滑动窗口片段长度(秒)
            segment_overlap: 滑动窗口重叠比例
            window_length: EEGNet内部窗口长度(秒)
            window_overlap: EEGNet内部窗口重叠
            connectivity_types: 要计算的连接性类型
            connectivity_freq_band: 连接性计算的频带
            precomputed_features_dir: 预计算特征目录
            compute_online: 是否在线计算连接性特征
            include_directed: 是否包含有向指标
            transform: 数据增强
            max_seizures: 每条记录最大发作数
        """
        self.manifest_path = manifest_path
        self.data_roots = data_roots if isinstance(data_roots, list) else [data_roots]
        self.label_type = label_type
        self.transform = transform
        self.max_seizures = max_seizures
        
        # 配置
        self.config = config if config is not None else DataConfig()
        
        # 窗口参数
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.window_length = window_length
        self.window_overlap = window_overlap
        
        # 连接性参数
        if connectivity_types is None:
            if include_directed:
                connectivity_types = self.CONNECTIVITY_TYPES
            else:
                connectivity_types = ['plv', 'wpli', 'aec', 'pearson']
        self.connectivity_types = connectivity_types
        self.connectivity_freq_band = connectivity_freq_band
        self.include_directed = include_directed
        
        # 预计算特征
        self.precomputed_features_dir = precomputed_features_dir
        self.compute_online = compute_online
        
        # 滑动窗口采样器
        self.sampler = SlidingWindowSampler(
            window_length=segment_length,
            overlap=segment_overlap
        )
        
        # 加载manifest
        self.df = pd.read_csv(manifest_path)
        
        # 过滤患者
        if patient_ids is not None:
            self.df = self.df[self.df['pt_id'].isin(patient_ids)].reset_index(drop=True)
        
        # 构建样本列表 (使用滑动窗口)
        self.samples = self._build_samples_with_sliding_window()
        
        logger.info(f"多模态数据集加载完成: {len(self.samples)} 个样本")
        logger.info(f"  片段设置: {segment_length}s, 重叠{segment_overlap*100:.0f}%")
        logger.info(f"  连接性类型: {connectivity_types}")
    
    def _find_edf_file(self, row: pd.Series) -> Optional[str]:
        """查找EDF文件路径（复用现有逻辑）"""
        loc = row.get('loc', '')
        
        if pd.notna(loc) and loc:
            for root in self.data_roots:
                full_path = Path(root) / loc
                if full_path.exists():
                    return str(full_path)
                
                if '\\' in loc or '/' in loc:
                    fname = Path(loc).name
                    for edf_file in Path(root).rglob(fname):
                        return str(edf_file)
        
        pt_id = row.get('pt_id', '')
        fn = row.get('fn', '')
        
        for root in self.data_roots:
            root_path = Path(root)
            for patient_dir in root_path.rglob(f"*{pt_id}*"):
                if patient_dir.is_dir():
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
        channel_columns = [ch.lower() for ch in STANDARD_19_CHANNELS]
        return {
            'onset_zone': parse_onset_zone_label(row.get('onset_zone')),
            'hemi': parse_hemi_label(row.get('hemi')),
            'channel': parse_channel_labels(row, channel_columns),
            'chain': parse_chain_label(row),
            'region_5': parse_region5_label(row),
        }
    
    def _build_samples_with_sliding_window(self) -> List[Dict]:
        """
        使用滑动窗口构建样本列表
        
        关键改进: 基于清洗后的有效时长来划分窗口，
        而不是原始时长，这样可以确保每个样本都有足够的数据。
        """
        samples = []
        
        for idx, row in self.df.iterrows():
            # 解析发作时间
            seizure_times = parse_seizure_times(row.get('sz_starts'), row.get('sz_ends'))
            
            if not seizure_times:
                continue
            
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
            
            # 获取总时长
            duration = row.get('duration', 30.0)
            if pd.isna(duration):
                duration = 30.0
            
            # 为每个发作生成样本
            for sz_idx, (sz_start, sz_end) in enumerate(seizure_times):
                # 计算原始时间范围
                raw_duration = sz_end - sz_start
                start_time = sz_start
                end_time = sz_end
                
                # 如果发作时间不够，扩展到发作前后
                if raw_duration < self.segment_length:
                    start_time = max(0, sz_start - (self.segment_length - raw_duration) / 2)
                    end_time = min(duration, start_time + self.segment_length)
                
                # **关键改进**: 计算清洗后的有效时长
                clean_duration = calculate_clean_duration(
                    start_time, end_time, mask_segments
                )
                
                # 检查清洗后的时长是否足够
                if clean_duration < self.segment_length:
                    # 数据不足一个完整窗口，依然创建一个样本，但标记为较短
                    # 后续在_load_and_preprocess中会使用adaptive_apply_windows处理
                    logger.debug(
                        f"{row.get('fn')} SZ{sz_idx}: 清洗后{clean_duration:.1f}s < {self.segment_length}s"
                    )
                
                # 基于清洗后的时长生成滑动窗口
                # 注意：这里的窗口位置是相对于清洗后数据的
                windows = self.sampler.get_windows(clean_duration, self.config.target_fs)
                
                if len(windows) == 0:
                    # 数据太少，至少创建一个样本
                    windows = [(0, int(min(clean_duration, self.segment_length) * self.config.target_fs))]
                
                for win_idx, (win_start_sample, win_end_sample) in enumerate(windows):
                    sample = {
                        'row_idx': idx,
                        'pt_id': row.get('pt_id'),
                        'fn': row.get('fn'),
                        'edf_path': edf_path,
                        'sz_idx': sz_idx,
                        'win_idx': win_idx,
                        'sz_start': sz_start,
                        'sz_end': sz_end,
                        # 存储原始的绝对时间范围，不是窗口切片
                        'segment_start_time': start_time,
                        'segment_end_time': end_time,
                        # 存储清洗后数据上的窗口位置（相对时间）
                        'clean_win_start': win_start_sample / self.config.target_fs,
                        'clean_win_end': win_end_sample / self.config.target_fs,
                        'clean_duration': clean_duration,
                        'baseline': baseline,
                        'mask_segments': mask_segments,
                        'labels': labels,
                        'duration': duration,
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        bipolar_eeg_data = None  # 双极导联数据
        bipolar_connectivity = None  # 双极导联连接性
        bipolar_graph_metrics = None  # 双极导联图指标
        
        try:
            # 加载和预处理数据
            result = self._load_and_preprocess(sample)
            eeg_data = result['eeg_data']  # 原始单极数据
            connectivity = result['connectivity']  # 基于原始数据
            graph_metrics = result['graph_metrics']  # 基于原始数据
            # 获取双极导联相关数据（如果有）
            if 'bipolar_eeg_data' in result:
                bipolar_eeg_data = result['bipolar_eeg_data']
            if 'bipolar_connectivity' in result:
                bipolar_connectivity = result['bipolar_connectivity']
            if 'bipolar_graph_metrics' in result:
                bipolar_graph_metrics = result['bipolar_graph_metrics']
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"加载数据失败 {sample['fn']}: {e}")
            
            # 返回零数据
            n_windows = int(self.segment_length / self.window_length)
            use_21_channels = getattr(self.config, 'use_21_channels', False)
            use_bipolar = getattr(self.config, 'use_bipolar', False)
            
            # 原始单极通道数
            n_raw_channels = 21 if use_21_channels else 19
            # 双极导联通道数
            n_bipolar_channels = 26 if use_21_channels else 18
            
            n_samples = int(self.config.target_fs * self.window_length)
            
            # eeg_data 始终是原始单极数据
            eeg_data = np.zeros((n_windows, n_raw_channels, n_samples))
            # connectivity 和 graph_metrics 基于原始数据
            connectivity = np.zeros((len(self.connectivity_types), n_raw_channels, n_raw_channels))
            graph_metrics = np.zeros((n_raw_channels, len(self.GRAPH_METRIC_TYPES)))
            
            # 如果启用了双极导联，创建空的双极导联数据
            if use_bipolar:
                bipolar_eeg_data = np.zeros((n_windows, n_bipolar_channels, n_samples))
                bipolar_connectivity = np.zeros((len(self.connectivity_types), n_bipolar_channels, n_bipolar_channels))
                bipolar_graph_metrics = np.zeros((n_bipolar_channels, len(self.GRAPH_METRIC_TYPES)))
        
        # 获取标签
        labels = sample['labels'][self.label_type]
        
        # 转换为张量
        eeg_tensor = torch.FloatTensor(eeg_data)
        connectivity_tensor = torch.FloatTensor(connectivity)
        graph_metrics_tensor = torch.FloatTensor(graph_metrics)
        labels_tensor = torch.FloatTensor(labels)
        
        # 数据增强
        if self.transform is not None:
            eeg_tensor = self.transform(eeg_tensor)
        
        result_dict = {
            'eeg_data': eeg_tensor,  # 原始单极数据
            'connectivity': connectivity_tensor,  # 基于原始数据
            'graph_metrics': graph_metrics_tensor,  # 基于原始数据
            'labels': labels_tensor,
            'pt_id': sample['pt_id'],
            'fn': sample['fn'],
            'sz_idx': sample['sz_idx'],
            'win_idx': sample['win_idx'],
        }
        
        # 如果有双极导联相关数据，也返回
        if bipolar_eeg_data is not None:
            result_dict['bipolar_eeg_data'] = torch.FloatTensor(bipolar_eeg_data)
        if bipolar_connectivity is not None:
            result_dict['bipolar_connectivity'] = torch.FloatTensor(bipolar_connectivity)
        if bipolar_graph_metrics is not None:
            result_dict['bipolar_graph_metrics'] = torch.FloatTensor(bipolar_graph_metrics)
        
        return result_dict
    
    def _load_and_preprocess(self, sample: Dict) -> Dict[str, np.ndarray]:
        """
        加载并预处理数据
        
        使用在构建样本阶段预先计算的窗口位置，确保每个样本
        都基于清洗后的正确时长来提取数据。
        """
        edf_path = sample['edf_path']
        segment_start = sample['segment_start_time']
        segment_end = sample['segment_end_time']
        base_line_start, base_line_end = sample['baseline']
        mask_segments = sample.get('mask_segments', [])
        
        # 获取预计算的窗口位置（相对于清洗后数据的时间）
        clean_win_start = sample.get('clean_win_start', 0)
        clean_win_end = sample.get('clean_win_end', self.segment_length)
        
        # 1. 读取EDF
        raw_data, fs, ch_names = read_edf(edf_path)
        original_fs = fs  # 保存原始采样率，用于后续处理原始单极数据
        
        # 2. 根据配置提取标准通道（19或21）
        use_21_channels = getattr(self.config, 'use_21_channels', False)
        if use_21_channels:
            target_channels = STANDARD_21_CHANNELS
        else:
            target_channels = STANDARD_19_CHANNELS
        data, found_channels = extract_standard_channels(raw_data, ch_names, target_channels)
        n_channels_raw = data.shape[0]
        
        # 保存原始单极数据（在双极转换之前）- 注意：此时还是原始采样率
        raw_unipolar_data = data.copy()
        raw_channel_names = found_channels.copy()
        n_channels_raw = data.shape[0]
        
        # 2.5. 可选：转换为TCP双极导联
        # 注意：双极导联数据和原始数据都会继续处理，不互相覆盖
        bipolar_data = None
        bipolar_channel_names = None
        use_bipolar = hasattr(self.config, 'use_bipolar') and self.config.use_bipolar
        
        if use_bipolar:
            # 根据use_21_channels选择双极导联对
            use_21_channels = getattr(self.config, 'use_21_channels', False)
            if use_21_channels:
                bipolar_pairs = BIPOLAR_PAIRS_26
                logger.debug("使用26通道双极导联 (21电极)")
            else:
                bipolar_pairs = BIPOLAR_PAIRS_18
                logger.debug("使用18通道双极导联 (19电极)")
            bipolar_data, bipolar_channel_names = convert_to_bipolar(data, found_channels, bipolar_pairs)
            n_channels_bipolar = bipolar_data.shape[0]
        
        # 原始单极数据的通道数
        n_channels = data.shape[0]
        
        # ========== 3. 预处理原始单极数据 ==========
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered = bandpass_filter(
                data[i], fs,
                f_low=self.config.filter_low,
                f_high=self.config.filter_high
            )
            clipped = clip_amplitude(filtered, self.config.clip_std)
            processed_data[i] = clipped
        
        # ========== 3b. 预处理双极导联数据（如果启用） ==========
        processed_bipolar = None
        if use_bipolar and bipolar_data is not None:
            processed_bipolar = np.zeros_like(bipolar_data)
            for i in range(bipolar_data.shape[0]):
                filtered = bandpass_filter(
                    bipolar_data[i], fs,
                    f_low=self.config.filter_low,
                    f_high=self.config.filter_high
                )
                clipped = clip_amplitude(filtered, self.config.clip_std)
                processed_bipolar[i] = clipped
        
        # ========== 4. 重采样 ==========
        target_fs = self.config.target_fs
        if fs != target_fs:
            # 重采样原始单极数据
            n_samples_new = int(processed_data.shape[1] * target_fs / fs)
            resampled_data = np.zeros((processed_data.shape[0], n_samples_new))
            for i in range(processed_data.shape[0]):
                resampled_data[i] = resample_signal(processed_data[i], fs, target_fs)
            processed_data = resampled_data
            
            # 重采样双极导联数据
            if processed_bipolar is not None:
                n_samples_new_bipolar = int(processed_bipolar.shape[1] * target_fs / fs)
                resampled_bipolar = np.zeros((processed_bipolar.shape[0], n_samples_new_bipolar))
                for i in range(processed_bipolar.shape[0]):
                    resampled_bipolar[i] = resample_signal(processed_bipolar[i], fs, target_fs)
                processed_bipolar = resampled_bipolar
            
            fs = target_fs
        
        # ========== 5. 提取片段 + 移除坏段 ==========
        # 原始单极数据
        clean_segment, actual_clean_duration = extract_segment_with_mask_removal(
            processed_data, fs,
            segment_start, segment_end,
            mask_segments
        )
        
        # 双极导联数据
        clean_bipolar_segment = None
        if processed_bipolar is not None:
            clean_bipolar_segment, _ = extract_segment_with_mask_removal(
                processed_bipolar, fs,
                segment_start, segment_end,
                mask_segments
            )
        
        # 检查清洗后的数据是否足够
        if clean_segment.shape[1] == 0:
            logger.warning(f"片段清洗后没有数据")
            n_samples = int(self.config.target_fs * self.window_length)
            # 始终返回基于原始数据的特征
            result = {
                'eeg_data': np.zeros((1, n_channels, n_samples)),
                'connectivity': np.zeros((len(self.connectivity_types), n_channels, n_channels)),
                'graph_metrics': np.zeros((n_channels, len(self.GRAPH_METRIC_TYPES)))
            }
            # 如果启用了双极导联，额外返回双极导联相关数据
            if use_bipolar:
                result['bipolar_eeg_data'] = np.zeros((1, n_channels_bipolar, n_samples))
                result['bipolar_connectivity'] = np.zeros((len(self.connectivity_types), n_channels_bipolar, n_channels_bipolar))
                result['bipolar_graph_metrics'] = np.zeros((n_channels_bipolar, len(self.GRAPH_METRIC_TYPES)))
            return result
        
        # ========== 6. 提取窗口数据 ==========
        win_start_sample = int(clean_win_start * fs)
        win_end_sample = int(clean_win_end * fs)
        win_start_sample = max(0, win_start_sample)
        win_end_sample = min(clean_segment.shape[1], win_end_sample)
        
        expected_samples = int(self.segment_length * fs)
        actual_samples = win_end_sample - win_start_sample
        
        # 原始单极数据窗口
        if actual_samples < expected_samples:
            window_data = np.zeros((n_channels, expected_samples))
            window_data[:, :actual_samples] = clean_segment[:, win_start_sample:win_end_sample]
        else:
            window_data = clean_segment[:, win_start_sample:win_start_sample + expected_samples]
        
        # 双极导联数据窗口
        bipolar_window_data = None
        if clean_bipolar_segment is not None:
            bipolar_win_end = min(clean_bipolar_segment.shape[1], win_end_sample)
            bipolar_actual = bipolar_win_end - win_start_sample
            if bipolar_actual < expected_samples:
                bipolar_window_data = np.zeros((n_channels_bipolar, expected_samples))
                bipolar_window_data[:, :bipolar_actual] = clean_bipolar_segment[:, win_start_sample:bipolar_win_end]
            else:
                bipolar_window_data = clean_bipolar_segment[:, win_start_sample:win_start_sample + expected_samples]
        
        # ========== 7. 标准化 ==========
        if base_line_start is not None and base_line_end is not None:
            bl_start = int(base_line_start * fs)
            bl_end = int(base_line_end * fs)
            if bl_end <= processed_data.shape[1]:
                baseline_segment = processed_data[:, bl_start:bl_end]
                window_data = (window_data - np.mean(baseline_segment)) / (np.std(baseline_segment) + 1e-16)
                if bipolar_window_data is not None and processed_bipolar is not None:
                    bipolar_baseline = processed_bipolar[:, bl_start:bl_end]
                    bipolar_window_data = (bipolar_window_data - np.mean(bipolar_baseline)) / (np.std(bipolar_baseline) + 1e-16)
            else:
                window_data = (window_data - np.mean(window_data)) / (np.std(window_data) + 1e-16)
                if bipolar_window_data is not None:
                    bipolar_window_data = (bipolar_window_data - np.mean(bipolar_window_data)) / (np.std(bipolar_window_data) + 1e-16)
        else:
            window_data = (window_data - np.mean(window_data)) / (np.std(window_data) + 1e-16)
            if bipolar_window_data is not None:
                bipolar_window_data = (bipolar_window_data - np.mean(bipolar_window_data)) / (np.std(bipolar_window_data) + 1e-16)
        
        # ========== 8. 窗口划分 ==========
        # 原始单极数据窗口划分
        windows = adaptive_apply_windows(
            window_data, fs,
            window_len=self.window_length,
            overlap=self.window_overlap,
            min_windows=1
        )  # (n_windows, n_channels, n_samples_per_window)
        
        # 双极导联数据窗口划分
        bipolar_windows = None
        if bipolar_window_data is not None:
            bipolar_windows = adaptive_apply_windows(
                bipolar_window_data, fs,
                window_len=self.window_length,
                overlap=self.window_overlap,
                min_windows=1
            )
            logger.debug(f"双极导联数据: {bipolar_windows.shape} (n_windows, {n_channels_bipolar}通道, n_samples)")
        
        # ========== 9. 计算连接性特征 ==========
        # 始终基于原始单极数据计算 connectivity
        if self.compute_online:
            connectivity = self._compute_connectivity_online(clean_segment, fs)
        else:
            connectivity = self._load_precomputed_connectivity(sample)
        
        # 如果启用了双极导联，额外计算基于双极导联数据的 connectivity
        bipolar_connectivity = None
        if use_bipolar and clean_bipolar_segment is not None:
            if self.compute_online:
                bipolar_connectivity = self._compute_connectivity_online(clean_bipolar_segment, fs)
            # 预计算模式暂不支持双极导联，使用在线计算
        
        # ========== 10. 计算图网络指标 ==========
        # 基于原始数据的 graph_metrics
        if connectivity.shape[0] > 0:
            avg_connectivity = connectivity.mean(axis=0)
            graph_metrics = compute_graph_metrics_from_connectivity(
                avg_connectivity, 
                self.GRAPH_METRIC_TYPES
            )
        else:
            graph_metrics = np.zeros((n_channels, len(self.GRAPH_METRIC_TYPES)))
        
        # 基于双极导联数据的 graph_metrics
        bipolar_graph_metrics = None
        if bipolar_connectivity is not None and bipolar_connectivity.shape[0] > 0:
            avg_bipolar_conn = bipolar_connectivity.mean(axis=0)
            bipolar_graph_metrics = compute_graph_metrics_from_connectivity(
                avg_bipolar_conn,
                self.GRAPH_METRIC_TYPES
            )
        
        # ========== 构建返回结果 ==========
        # eeg_data 始终保存原始单极数据（方便统一处理）
        # connectivity 和 graph_metrics 始终基于原始单极数据
        result = {
            'eeg_data': windows,  # 原始单极数据
            'connectivity': connectivity,  # 基于原始数据
            'graph_metrics': graph_metrics  # 基于原始数据
        }
        
        # 如果启用了双极导联，添加双极导联相关数据
        if bipolar_windows is not None:
            result['bipolar_eeg_data'] = bipolar_windows
        if bipolar_connectivity is not None:
            result['bipolar_connectivity'] = bipolar_connectivity
        if bipolar_graph_metrics is not None:
            result['bipolar_graph_metrics'] = bipolar_graph_metrics
        
        return result
    
    def _compute_connectivity_online(
        self, 
        segment: np.ndarray, 
        fs: float
    ) -> np.ndarray:
        """在线计算连接性矩阵"""
        n_channels = segment.shape[0]
        n_types = len(self.connectivity_types)
        connectivity = np.zeros((n_types, n_channels, n_channels))
        
        freq_band = self.connectivity_freq_band
        
        for i, conn_type in enumerate(self.connectivity_types):
            try:
                if conn_type == 'plv':
                    matrix = compute_plv(segment, fs, freq_band)
                elif conn_type == 'wpli':
                    matrix = compute_wpli(segment, fs, freq_band)
                elif conn_type == 'aec':
                    matrix = compute_aec(segment, fs, freq_band)
                elif conn_type == 'pearson':
                    matrix = compute_pearson_corr(segment)
                elif conn_type == 'granger':
                    if self.include_directed:
                        matrix = compute_granger_causality(segment, fs)
                    else:
                        matrix = np.zeros((n_channels, n_channels))
                elif conn_type == 'transfer_entropy':
                    if self.include_directed:
                        matrix = compute_transfer_entropy(segment, fs)
                    else:
                        matrix = np.zeros((n_channels, n_channels))
                else:
                    matrix = np.zeros((n_channels, n_channels))
                
                connectivity[i] = matrix
                
            except Exception as e:
                logger.warning(f"连接性计算失败 ({conn_type}): {e}")
                connectivity[i] = np.zeros((n_channels, n_channels))
        
        return connectivity
    
    def _load_precomputed_connectivity(self, sample: Dict) -> np.ndarray:
        """加载预计算的连接性特征"""
        if self.precomputed_features_dir is None:
            logger.warning("未指定预计算特征目录，使用在线计算")
            return self._compute_connectivity_online(
                np.zeros((19, int(self.segment_length * self.config.target_fs))),
                self.config.target_fs
            )
        
        # 构建特征文件路径
        pt_id = sample['pt_id']
        fn = sample['fn']
        sz_idx = sample['sz_idx']
        win_idx = sample['win_idx']
        
        feature_path = Path(self.precomputed_features_dir) / f"{pt_id}_{fn}_sz{sz_idx}_win{win_idx}_connectivity.npz"
        
        if feature_path.exists():
            data = np.load(feature_path)
            connectivity = data['connectivity']
            return connectivity
        else:
            logger.warning(f"预计算特征不存在: {feature_path}")
            return np.zeros((len(self.connectivity_types), 19, 19))


def create_multimodal_dataloader(
    dataset: MultiModalEEGDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """创建多模态数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试多模态EEG数据集...")
    
    # 测试滑动窗口采样器
    print("\n1. 测试滑动窗口采样器:")
    sampler = SlidingWindowSampler(window_length=20.0, overlap=0.5)
    windows = sampler.get_windows(total_length=30.0, fs=200.0)
    print(f"   30秒片段, 20秒窗口, 50%重叠 -> {len(windows)} 个窗口")
    for i, (start, end) in enumerate(windows):
        print(f"     窗口{i}: [{start/200:.1f}s, {end/200:.1f}s]")
    
    # 测试图指标计算
    print("\n2. 测试图指标计算:")
    test_matrix = np.random.rand(19, 19)
    test_matrix = (test_matrix + test_matrix.T) / 2
    np.fill_diagonal(test_matrix, 1.0)
    
    metrics = compute_graph_metrics_from_connectivity(test_matrix)
    print(f"   输入: {test_matrix.shape}")
    print(f"   输出: {metrics.shape}")
    print(f"   指标: degree, strength, clustering, betweenness, eigenvector")
    
    print("\n测试完成!")
