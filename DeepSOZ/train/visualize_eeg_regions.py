#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG脑区可视化脚本

功能：
1. 加载与train_onset_zone.py相同的manifest文件和数据处理
2. 按癫痫发作片段筛取数据
3. 按脑区可视化通道数据：
   - 原始波形图
   - RGB热力图
4. SOZ脑区在文件名上添加"_soz"标记
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
    read_edf, extract_standard_channels, bandpass_filter, 
    clip_amplitude, resample_signal, parse_seizure_times,
    parse_onset_zone_label, STANDARD_19_CHANNELS, BRAIN_REGION_MAP,
    convert_to_bipolar, BIPOLAR_PAIRS_18,
    parse_mask_segments, extract_segment_with_mask_removal  # mask_segments支持
)
from connectivity import compute_all_connectivity
from connectivity_viz import visualize_all_connectivity

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
    global_max: float = None
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
    """
    n_channels, n_samples = region_data.shape
    time = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        # 使用全局归一化
        normalized_data = normalize_data(region_data[i], global_min, global_max)
        ax.plot(time, normalized_data, 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name, fontsize=10, rotation=0, ha='right', va='center')
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
    global_max: float = None
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
    """
    n_channels, n_samples = region_data.shape
    
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
        ax.set_ylabel(ch_name, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.set_xlim(time_extent)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"保存热力图: {save_path}")


def find_edf_file(row: pd.Series, data_roots: List[str]) -> Optional[str]:
    """查找EDF文件路径"""
    import re
    loc = row.get('loc', '')
    
    if pd.notna(loc) and loc:
        for root in data_roots:
            full_path = Path(root) / loc
            if full_path.exists():
                return str(full_path)
            
            if '\\' in loc or '/' in loc:
                fname = Path(loc).name
                for edf_file in Path(root).rglob(fname):
                    return str(edf_file)
    
    pt_id = row.get('pt_id', '')
    fn = row.get('fn', '')
    
    for root in data_roots:
        root_path = Path(root)
        for patient_dir in root_path.rglob(f"*{pt_id}*"):
            if patient_dir.is_dir():
                match = re.search(r'SZ(\d+)', fn, re.IGNORECASE)
                if match:
                    sz_num = match.group(1)
                    for edf_file in patient_dir.glob(f"*SZ{sz_num}*.edf"):
                        return str(edf_file)
                    for edf_file in patient_dir.glob(f"*sz{sz_num}*.edf"):
                        return str(edf_file)
    
    return None


def process_and_visualize(
    row: pd.Series,
    config: DataConfig,
    output_dir: Path,
    use_bipolar: bool = False
):
    """
    处理单条记录并生成可视化
    
    Args:
        row: CSV中的一行数据
        config: 数据配置
        output_dir: 输出目录
    """
    # 查找EDF文件
    edf_path = find_edf_file(row, config.edf_data_roots)
    if edf_path is None:
        logger.warning(f"未找到EDF文件: {row.get('fn', 'unknown')}")
        return
    
    logger.info(f"处理: {row.get('fn', 'unknown')}")
    
    # 解析发作时间
    seizure_times = parse_seizure_times(row.get('sz_starts'), row.get('sz_ends'))
    if not seizure_times:
        logger.warning(f"无发作时间: {row.get('fn')}")
        return
    
    # 解析SOZ脑区
    onset_zone_str = row.get('onset_zone', '')
    soz_regions = set()
    if pd.notna(onset_zone_str) and onset_zone_str:
        for part in str(onset_zone_str).lower().split(','):
            part = part.strip()
            if part in REGION_TO_CHANNEL_IDX:
                soz_regions.add(part)
    
    try:
        # 1. 读取EDF
        raw_data, fs, ch_names = read_edf(edf_path)
        
        # 2. 提取标准19通道
        data, found_channels = extract_standard_channels(raw_data, ch_names)
        
        # 3. 可选：转换为双极导联
        if use_bipolar:
            data, found_channels = convert_to_bipolar(data, found_channels, BIPOLAR_PAIRS_18)
            logger.info(f"使用双极导联: {len(found_channels)} 通道")
        
        # 3.5. 解析mask_segments（后续在提取片段时使用）
        mask_segments = parse_mask_segments(row.get('mask_segments'))
        
        # 4. 预处理每个通道
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered = bandpass_filter(
                data[i], fs,
                f_low=config.filter_low,
                f_high=config.filter_high
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
            logger.info(f"全局归一化范围: [{global_min:.4f}, {global_max:.4f}]")
            
            # 创建患者/发作目录
            patient_dir = output_dir / f"{pt_id}_{fn}_{sz_start}-{sz_end}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # 按脑区可视化
            if use_bipolar:
                # 双极导联暂时跳过脑区划分，统一可视化
                title_base = f"{pt_id} - {fn} - SZ{sz_idx+1} ({sz_start:.1f}s - {sz_end:.1f}s)"
                
                # 波形图
                waveform_path = patient_dir / "all_channels_waveform.png"
                plot_waveform(
                    seizure_segment,
                    found_channels,
                    fs,
                    f"{title_base} - All Channels Waveform",
                    str(waveform_path),
                    global_min=global_min,
                    global_max=global_max
                )
                
                # 热力图
                heatmap_path = patient_dir / "all_channels_heatmap.png"
                plot_heatmap(
                    seizure_segment,
                    found_channels,
                    fs,
                    f"{title_base} - All Channels Heatmap",
                    str(heatmap_path),
                    global_min=global_min,
                    global_max=global_max
                )
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
    parser = argparse.ArgumentParser(description='EEG脑区可视化')
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认创建时间戳目录）')
    parser.add_argument('--max-records', type=int, default=None,
                        help='最大处理记录数（调试用）')
    parser.add_argument('--bipolar', action='store_true',
                        help='使用TCP双极导联')
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
        output_dir = Path(__file__).parent / "visualizations" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 加载manifest
    df = pd.read_csv(config.data.manifest_path)
    logger.info(f"加载manifest: {len(df)} 条记录")
    
    if args.max_records:
        df = df.head(args.max_records)
        logger.info(f"限制处理前 {args.max_records} 条记录")
    
    # 处理每条记录
    for idx, row in df.iterrows():
        logger.info(f"\n处理记录 {idx + 1}/{len(df)}")
        process_and_visualize(row, config.data, output_dir, use_bipolar=args.bipolar)
    
    logger.info(f"\n可视化完成！输出目录: {output_dir}")


if __name__ == '__main__':
    main()
