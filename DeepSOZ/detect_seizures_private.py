#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
癫痫发作检测脚本 - 私有数据集 (21通道版本)

使用DeepSOZ预训练模型分析EDF文件，检测癫痫发作时间段，
更新converted_manifest.csv中的sz_starts、sz_ends和nsz字段。

功能：
1. 读取converted_manifest.csv中的EDF文件路径
2. 使用DeepSOZ预训练模型进行癫痫发作检测
3. 识别发作起始时间和结束时间
4. 更新manifest中的相关字段

通道说明：
- 标准19通道 + 蝶骨电极 (Sph-L, Sph-R)
- 模型推理使用19通道，蝶骨电极单独处理

Author: EEG_SUAT_NEW Project
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('seizure_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入推理模块
import sys
sys.path.insert(0, str(Path(__file__).parent))

# 尝试从本地导入，如果失败则从原始路径导入
try:
    from inference_edf import (
        read_edf,
        extract_standard_channels,
        preprocess_channel,
        resample_signal,
        apply_windows,
        prepare_for_model,
        TARGET_FS,
        FILTER_LOW,
        FILTER_HIGH,
        CLIP_STD,
        MAX_SEGMENT_WINDOWS,
        MIN_SEGMENT_WINDOWS
    )
except ImportError:
    # 添加原始DeepSOZ路径
    original_deepsoz_path = Path(__file__).parent.parent.parent / 'DeepSOZ' / 'code' / 'inference'
    sys.path.insert(0, str(original_deepsoz_path))
    from inference_edf import (
        read_edf,
        extract_standard_channels,
        preprocess_channel,
        resample_signal,
        apply_windows,
        prepare_for_model,
        TARGET_FS,
        FILTER_LOW,
        FILTER_HIGH,
        CLIP_STD,
        MAX_SEGMENT_WINDOWS,
        MIN_SEGMENT_WINDOWS
    )

try:
    from inference_private_data import (
        txlstm_szpool,
        load_deepsoz_model,
        DEEPSOZ_CHANNELS
    )
except ImportError:
    original_deepsoz_path = Path(__file__).parent.parent.parent / 'DeepSOZ' / 'code' / 'inference'
    sys.path.insert(0, str(original_deepsoz_path))
    from inference_private_data import (
        txlstm_szpool,
        load_deepsoz_model,
        DEEPSOZ_CHANNELS
    )


# ==============================================================================
# 通道配置 (21通道：19标准 + 2蝶骨)
# ==============================================================================

# 标准19通道
STANDARD_19_CHANNELS = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
    't3', 'c3', 'cz', 'c4', 't4',
    't5', 'p3', 'pz', 'p4', 't6',
    'o1', 'o2'
]

# 蝶骨电极
EXTRA_CHANNELS = ['sph-l', 'sph-r']

# 全部21通道
ALL_21_CHANNELS = STANDARD_19_CHANNELS + EXTRA_CHANNELS

# 蝶骨电极别名映射
SPHENOIDAL_ALIASES = {
    'sp1': 'sph-l',
    'sp-l': 'sph-l',
    'spl': 'sph-l',
    'sp2': 'sph-r',
    'sp-r': 'sph-r',
    'spr': 'sph-r',
}

# ==============================================================================
# 配置参数
# ==============================================================================

# 发作检测阈值
SEIZURE_THRESHOLD = 0.5  # 癫痫发作概率阈值
MIN_SEIZURE_DURATION = 3  # 最小发作持续时间(秒)
MERGE_GAP = 5  # 合并间隔小于此秒数的发作段

# 模型路径配置
DEFAULT_MODEL_PATHS = [
    # 尝试多个模型路径
    Path(__file__).parent.parent.parent / 'DeepSOZ' / 'final_models' / 'fold0' / 'txlstm_szpool_finetuned_cv0_0.0001.pth.tar',
    Path(__file__).parent.parent.parent / 'DeepSOZ' / 'final_models' / 'fold5' / 'txlstm_szpool_finetuned_cv5_0.0001.pth.tar',
]


# ==============================================================================
# 通道提取辅助函数
# ==============================================================================

def normalize_channel_name(name: str) -> str:
    """
    标准化通道名称
    
    Args:
        name: 原始通道名称
        
    Returns:
        标准化后的通道名称
    """
    name_lower = name.lower().strip()
    # 移除常见前缀/后缀
    name_lower = name_lower.replace('eeg ', '').replace(' eeg', '')
    name_lower = name_lower.replace('-ref', '').replace('-le', '').replace('-ar', '')
    name_lower = name_lower.strip()
    
    # 检查是否是蝶骨电极别名
    if name_lower in SPHENOIDAL_ALIASES:
        return SPHENOIDAL_ALIASES[name_lower]
    
    return name_lower


def extract_21_channels(raw_data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    从原始EDF数据中提取21通道（19标准 + 2蝶骨）
    
    Args:
        raw_data: (n_channels, n_samples) 原始数据
        ch_names: 通道名称列表
        
    Returns:
        data_19ch: (19, n_samples) 标准19通道数据
        found_19_channels: 找到的19通道名称列表
        data_extra: (2, n_samples) 蝶骨电极数据 (可能包含NaN)
        found_extra_channels: 找到的蝶骨通道名称列表
    """
    n_samples = raw_data.shape[1]
    
    # 创建通道名映射（标准化后的名称 -> 原始索引）
    ch_name_map = {}
    for idx, name in enumerate(ch_names):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        normalized = normalize_channel_name(name)
        ch_name_map[normalized] = idx
    
    # 提取标准19通道
    data_19ch = np.zeros((19, n_samples))
    found_19_channels = []
    
    for i, ch in enumerate(STANDARD_19_CHANNELS):
        if ch in ch_name_map:
            data_19ch[i] = raw_data[ch_name_map[ch]]
            found_19_channels.append(ch)
        else:
            # logger.warning(f"未找到标准通道: {ch}")
            found_19_channels.append(f"{ch}(missing)")
    
    # 提取蝶骨电极
    data_extra = np.full((2, n_samples), np.nan)  # 默认为NaN
    found_extra_channels = []
    
    for i, ch in enumerate(EXTRA_CHANNELS):
        if ch in ch_name_map:
            data_extra[i] = raw_data[ch_name_map[ch]]
            found_extra_channels.append(ch)
        else:
            # 蝶骨电极可选，不警告
            found_extra_channels.append(f"{ch}(missing)")
    
    return data_19ch, found_19_channels, data_extra, found_extra_channels


# ==============================================================================
# 核心检测函数
# ==============================================================================

def detect_seizure_regions(seizure_probs: np.ndarray, 
                           threshold: float = SEIZURE_THRESHOLD,
                           min_duration: int = MIN_SEIZURE_DURATION,
                           merge_gap: int = MERGE_GAP) -> List[Tuple[int, int]]:
    """
    从癫痫发作概率序列中检测发作区域
    
    Args:
        seizure_probs: (T,) 每个时间窗口的癫痫发作概率
        threshold: 发作检测阈值
        min_duration: 最小发作持续时间(秒)
        merge_gap: 合并间隔小于此秒数的发作段
        
    Returns:
        List of (start_sec, end_sec) tuples
    """
    # 二值化
    seizure_mask = seizure_probs > threshold
    
    # 检测发作区域
    regions = []
    in_seizure = False
    start_idx = 0
    
    for i, is_sz in enumerate(seizure_mask):
        if is_sz and not in_seizure:
            start_idx = i
            in_seizure = True
        elif not is_sz and in_seizure:
            regions.append((start_idx, i))
            in_seizure = False
    
    # 处理最后一个区域
    if in_seizure:
        regions.append((start_idx, len(seizure_mask)))
    
    if not regions:
        return []
    
    # 合并相邻区域
    merged = []
    current_start, current_end = regions[0]
    
    for start, end in regions[1:]:
        if start - current_end <= merge_gap:
            # 合并
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    
    # 过滤短发作
    filtered = [(s, e) for s, e in merged if e - s >= min_duration]
    
    return filtered


def process_edf_for_detection(edf_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    处理EDF文件，返回预处理后的窗口数据
    
    Args:
        edf_path: EDF文件路径
        
    Returns:
        windowed_19ch: (n_windows, 19, 200) 标准19通道预处理后的数据
        windowed_extra: (n_windows, 2, 200) 蝶骨电极预处理后的数据 (可能包含NaN)
        info: 处理信息字典
    """
    # 读取EDF文件
    raw_data, original_fs, ch_names = read_edf(edf_path)
    duration_sec = raw_data.shape[1] / original_fs
    
    # 提取21通道
    data_19ch, found_19_channels, data_extra, found_extra_channels = extract_21_channels(raw_data, ch_names)

    if len([ch for ch in found_19_channels if '(missing)' not in ch]) < 19:
        missing_count = len([ch for ch in found_19_channels if '(missing)' in ch])
        logger.warning(f"EDF文件 {edf_path} 缺少 {missing_count} 个标准通道")
        raise ValueError(f"EDF file {edf_path} does not contain 19 standard channels")
    
    # 预处理标准19通道
    for i in range(19):
        data_19ch[i] = preprocess_channel(
            data_19ch[i], original_fs, 
            FILTER_LOW, FILTER_HIGH, 4, CLIP_STD
        )
    
    # 预处理蝶骨电极（如果存在）
    for i in range(2):
        if not np.isnan(data_extra[i]).all():
            data_extra[i] = preprocess_channel(
                data_extra[i], original_fs, 
                FILTER_LOW, FILTER_HIGH, 4, CLIP_STD
            )
    
    # 重采样到目标频率
    if original_fs != TARGET_FS:
        n_samples_new = int(data_19ch.shape[1] * TARGET_FS / original_fs)
        
        # 重采样19通道
        data_19ch_resampled = np.zeros((19, n_samples_new))
        for i in range(19):
            data_19ch_resampled[i] = resample_signal(data_19ch[i], original_fs, TARGET_FS)
        data_19ch = data_19ch_resampled
        
        # 重采样蝶骨电极
        data_extra_resampled = np.full((2, n_samples_new), np.nan)
        for i in range(2):
            if not np.isnan(data_extra[i]).all():
                data_extra_resampled[i] = resample_signal(data_extra[i], original_fs, TARGET_FS)
        data_extra = data_extra_resampled
    
    # 窗口化
    windowed_19ch = apply_windows(data_19ch, TARGET_FS, 1.0, 0.0)
    
    # 窗口化蝶骨电极
    n_windows = windowed_19ch.shape[0]
    window_size = windowed_19ch.shape[2]
    windowed_extra = np.full((n_windows, 2, window_size), np.nan)
    
    for i in range(2):
        if not np.isnan(data_extra[i]).all():
            temp = apply_windows(data_extra[i:i+1], TARGET_FS, 1.0, 0.0)
            windowed_extra[:, i, :] = temp[:, 0, :]
    
    info = {
        'original_fs': original_fs,
        'duration_sec': duration_sec,
        'n_windows': windowed_19ch.shape[0],
        'found_19_channels': found_19_channels,
        'found_extra_channels': found_extra_channels,
        'has_sphenoidal': any('(missing)' not in ch for ch in found_extra_channels)
    }
    
    return windowed_19ch, windowed_extra, info


def run_seizure_detection(model: txlstm_szpool, windowed: np.ndarray,
                          device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    运行癫痫发作检测
    
    Args:
        model: DeepSOZ模型
        windowed: (n_windows, 19, 200) 预处理后的数据
        device: 计算设备
        
    Returns:
        soz_probs: (19,) SOZ通道概率
        seizure_probs: (n_windows,) 每个窗口的发作概率
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 准备数据段
    segments = prepare_for_model(windowed, MAX_SEGMENT_WINDOWS, MIN_SEGMENT_WINDOWS)
    
    all_soz_probs = []
    all_seizure_probs = []
    
    for segment in segments:
        x = torch.from_numpy(segment).to(device)
        
        with torch.no_grad():
            sz_output, soz_output, _, _ = model(x)
            
            # SOZ概率
            soz_probs = soz_output.squeeze().cpu().numpy()
            all_soz_probs.append(soz_probs)
            
            # 发作概率
            seizure_probs = F.softmax(sz_output.squeeze(), dim=-1)[:, 1].cpu().numpy()
            all_seizure_probs.append(seizure_probs)
    
    # 聚合SOZ概率
    weights = [len(sp) for sp in all_seizure_probs]
    total_weight = sum(weights)
    
    aggregated_soz = np.zeros(19)
    for soz, w in zip(all_soz_probs, weights):
        aggregated_soz += soz * w / total_weight
    
    # 拼接发作概率
    all_seizure = np.concatenate(all_seizure_probs)
    
    return aggregated_soz, all_seizure


def estimate_sphenoidal_soz(soz_probs_19: np.ndarray, windowed_extra: np.ndarray) -> np.ndarray:
    """
    估算蝶骨电极的SOZ概率
    
    由于模型只支持19通道，蝶骨电极的SOZ概率基于：
    1. 如果蝶骨电极数据存在，使用与颞叶通道的相关性估算
    2. 如果不存在，返回NaN
    
    Args:
        soz_probs_19: (19,) 标准19通道的SOZ概率
        windowed_extra: (n_windows, 2, 200) 蝶骨电极数据
        
    Returns:
        soz_probs_extra: (2,) 蝶骨电极SOZ概率
    """
    soz_probs_extra = np.full(2, np.nan)
    
    # 颞叶通道索引 (t3, t4, t5, t6)
    temporal_indices = [7, 11, 12, 16]  # t3, t4, t5, t6 在19通道中的索引
    temporal_soz_mean = np.mean(soz_probs_19[temporal_indices])
    
    for i in range(2):
        if not np.isnan(windowed_extra[:, i, :]).all():
            # 蝶骨电极存在时，使用颞叶通道的平均SOZ概率作为估算
            # 可以根据需要调整这个估算方法
            soz_probs_extra[i] = temporal_soz_mean
    
    return soz_probs_extra


def detect_seizures_in_file(edf_path: str, model: txlstm_szpool,
                            device: str = 'cuda',
                            threshold: float = SEIZURE_THRESHOLD) -> Dict:
    """
    对单个EDF文件进行癫痫发作检测
    
    Args:
        edf_path: EDF文件路径
        model: DeepSOZ模型
        device: 计算设备
        threshold: 发作检测阈值
        
    Returns:
        检测结果字典
    """
    # 处理EDF文件
    windowed_19ch, windowed_extra, info = process_edf_for_detection(edf_path)
    
    # 运行检测（使用19通道）
    soz_probs_19, seizure_probs = run_seizure_detection(model, windowed_19ch, device)
    
    # 估算蝶骨电极SOZ概率
    soz_probs_extra = estimate_sphenoidal_soz(soz_probs_19, windowed_extra)
    
    # 合并为21通道SOZ概率
    soz_probs_21 = np.concatenate([soz_probs_19, soz_probs_extra])
    
    # 检测发作区域
    seizure_regions = detect_seizure_regions(
        seizure_probs, threshold, MIN_SEIZURE_DURATION, MERGE_GAP
    )
    
    # 构建结果
    sz_starts = [str(s) for s, e in seizure_regions]
    sz_ends = [str(e) for s, e in seizure_regions]
    
    result = {
        'duration_sec': info['duration_sec'],
        'n_windows': info['n_windows'],
        'found_19_channels': info['found_19_channels'],
        'found_extra_channels': info['found_extra_channels'],
        'has_sphenoidal': info['has_sphenoidal'],
        'soz_probs_19': soz_probs_19,
        'soz_probs_21': soz_probs_21,
        'seizure_probs': seizure_probs,
        'seizure_regions': seizure_regions,
        'nsz': len(seizure_regions),
        'sz_starts': ';'.join(sz_starts) if sz_starts else '',
        'sz_ends': ';'.join(sz_ends) if sz_ends else '',
        'nchns': 21 if info['has_sphenoidal'] else 19,
    }
    
    return result


# ==============================================================================
# 批量处理函数
# ==============================================================================

def find_edf_file(loc: str, data_roots: List[str]) -> Optional[str]:
    """
    根据loc字段查找EDF文件的完整路径
    
    Args:
        loc: CSV中的loc字段值
        data_roots: 可能的数据根目录列表
        
    Returns:
        完整的EDF文件路径，找不到则返回None
    """
    if pd.isna(loc) or not loc:
        return None
    
    loc = str(loc).strip()
    
    # 如果loc已经是绝对路径且存在
    if Path(loc).is_absolute() and Path(loc).exists():
        return loc
    
    # 在各个数据根目录下搜索
    for root in data_roots:
        full_path = Path(root) / loc
        if full_path.exists():
            return str(full_path)
        
        # 尝试替换路径分隔符
        loc_normalized = loc.replace('\\', '/').replace('/', Path('/').as_posix())
        full_path = Path(root) / loc_normalized
        if full_path.exists():
            return str(full_path)
    
    return None


def process_manifest(manifest_path: str,
                     model_path: str,
                     data_roots: List[str],
                     output_path: Optional[str] = None,
                     device: str = 'cuda',
                     threshold: float = SEIZURE_THRESHOLD) -> pd.DataFrame:
    """
    处理manifest CSV文件，检测癫痫发作并更新字段
    
    Args:
        manifest_path: manifest CSV文件路径
        model_path: 模型检查点路径
        data_roots: EDF数据根目录列表
        output_path: 输出CSV路径
        device: 计算设备
        threshold: 发作检测阈值
        
    Returns:
        更新后的DataFrame
    """
    logger.info(f"读取manifest文件: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    logger.info(f"共有 {len(df)} 条记录")
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = load_deepsoz_model(model_path, device)
    logger.info("模型加载成功")
    
    # 确保蝶骨电极列存在
    for ch in EXTRA_CHANNELS:
        if ch not in df.columns:
            df[ch] = 0
    
    # 记录统计信息
    processed_count = 0
    error_count = 0
    not_found_count = 0
    sphenoidal_count = 0
    
    # 处理每条记录
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理EDF文件"):
        loc = row.get('loc', '')
        pt_id = row.get('pt_id', '')
        fn = row.get('fn', '')
        
        # 查找EDF文件
        edf_path = find_edf_file(loc, data_roots)
        
        if edf_path is None:
            logger.warning(f"找不到EDF文件: pt_id={pt_id}, fn={fn}, loc={loc}")
            not_found_count += 1
            continue
        
        try:
            logger.debug(f"处理: {edf_path}")
            
            # 检测癫痫发作
            result = detect_seizures_in_file(edf_path, model, device, threshold)
            
            # 更新DataFrame
            df.at[idx, 'nsz'] = result['nsz']
            df.at[idx, 'sz_starts'] = result['sz_starts']
            df.at[idx, 'sz_ends'] = result['sz_ends']
            df.at[idx, 'nchns'] = result['nchns']
            
            # 可选: 更新duration和fs字段
            if pd.isna(row.get('duration', None)) or row.get('duration', '') == '':
                df.at[idx, 'duration'] = result['duration_sec']
            
            if result['has_sphenoidal']:
                sphenoidal_count += 1
            
            processed_count += 1
            
            # 输出检测结果
            if result['nsz'] > 0:
                logger.info(f"检测到发作 [{pt_id}_{fn}]: "
                           f"发作次数={result['nsz']}, "
                           f"时间段={list(result['seizure_regions'])}, "
                           f"蝶骨电极={'有' if result['has_sphenoidal'] else '无'}")
            
        except Exception as e:
            logger.error(f"处理失败 [{pt_id}_{fn}]: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    # 保存结果
    if output_path is None:
        # 默认保存到原文件同目录下，添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(manifest_path).parent / f"detected_manifest_{timestamp}.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 输出统计
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  包含蝶骨电极: {sphenoidal_count}")
    logger.info(f"  文件未找到: {not_found_count}")
    logger.info(f"  处理失败: {error_count}")
    logger.info(f"  结果保存至: {output_path}")
    logger.info("=" * 60)
    
    return df


# ==============================================================================
# Ensemble检测 (使用多个模型)
# ==============================================================================

def ensemble_detection(edf_path: str, model_paths: List[str],
                       device: str = 'cuda',
                       threshold: float = SEIZURE_THRESHOLD) -> Dict:
    """
    使用多个模型进行集成检测
    
    Args:
        edf_path: EDF文件路径
        model_paths: 模型路径列表
        device: 计算设备
        threshold: 阈值
        
    Returns:
        集成检测结果
    """
    all_seizure_probs = []
    all_soz_probs = []
    info = None
    windowed_extra = None
    
    # 处理EDF文件（只需处理一次）
    windowed_19ch, windowed_extra, info = process_edf_for_detection(edf_path)
    
    # 使用每个模型进行检测
    for model_path in model_paths:
        if not Path(model_path).exists():
            continue
            
        model = load_deepsoz_model(model_path, device)
        soz_probs, seizure_probs = run_seizure_detection(model, windowed_19ch, device)
        
        all_soz_probs.append(soz_probs)
        all_seizure_probs.append(seizure_probs)
        
        del model
        torch.cuda.empty_cache()
    
    if not all_seizure_probs:
        raise ValueError("没有可用的模型")
    
    # 集成: 平均概率
    ensemble_seizure_probs = np.mean(all_seizure_probs, axis=0)
    ensemble_soz_probs_19 = np.mean(all_soz_probs, axis=0)
    
    # 估算蝶骨电极SOZ概率
    soz_probs_extra = estimate_sphenoidal_soz(ensemble_soz_probs_19, windowed_extra)
    ensemble_soz_probs_21 = np.concatenate([ensemble_soz_probs_19, soz_probs_extra])
    
    # 检测发作区域
    seizure_regions = detect_seizure_regions(
        ensemble_seizure_probs, threshold, MIN_SEIZURE_DURATION, MERGE_GAP
    )
    
    sz_starts = [str(s) for s, e in seizure_regions]
    sz_ends = [str(e) for s, e in seizure_regions]
    
    result = {
        'duration_sec': info['duration_sec'],
        'n_windows': info['n_windows'],
        'n_models': len(all_seizure_probs),
        'has_sphenoidal': info['has_sphenoidal'],
        'seizure_regions': seizure_regions,
        'nsz': len(seizure_regions),
        'sz_starts': ';'.join(sz_starts) if sz_starts else '',
        'sz_ends': ';'.join(sz_ends) if sz_ends else '',
        'soz_probs_19': ensemble_soz_probs_19,
        'soz_probs_21': ensemble_soz_probs_21,
        'nchns': 21 if info['has_sphenoidal'] else 19,
    }
    
    return result


def process_manifest_ensemble(manifest_path: str,
                              model_dir: str,
                              data_roots: List[str],
                              output_path: Optional[str] = None,
                              device: str = 'cuda',
                              threshold: float = SEIZURE_THRESHOLD,
                              n_folds: int = 5) -> pd.DataFrame:
    """
    使用集成模型处理manifest
    
    Args:
        manifest_path: manifest CSV文件路径
        model_dir: 模型目录 (包含fold0, fold1, ...)
        data_roots: EDF数据根目录列表
        output_path: 输出CSV路径
        device: 计算设备
        threshold: 阈值
        n_folds: 使用的fold数量
        
    Returns:
        更新后的DataFrame
    """
    # 收集模型路径
    model_paths = []
    model_dir = Path(model_dir)
    
    for i in range(n_folds):
        fold_dir = model_dir / f'fold{i}'
        model_file = fold_dir / f'txlstm_szpool_finetuned_cv{i}_0.0001.pth.tar'
        if model_file.exists():
            model_paths.append(str(model_file))
    
    logger.info(f"找到 {len(model_paths)} 个模型用于集成")
    
    if not model_paths:
        raise ValueError(f"在 {model_dir} 中没有找到模型")
    
    logger.info(f"读取manifest文件: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # 确保蝶骨电极列存在
    for ch in EXTRA_CHANNELS:
        if ch not in df.columns:
            df[ch] = 0
    
    # 处理每条记录
    processed_count = 0
    error_count = 0
    not_found_count = 0
    sphenoidal_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="集成检测"):
        loc = row.get('loc', '')
        pt_id = row.get('pt_id', '')
        fn = row.get('fn', '')
        
        edf_path = find_edf_file(loc, data_roots)
        
        if edf_path is None:
            logger.warning(f"找不到EDF文件: {pt_id}_{fn}")
            not_found_count += 1
            continue
        
        try:
            result = ensemble_detection(edf_path, model_paths, device, threshold)
            
            df.at[idx, 'nsz'] = result['nsz']
            df.at[idx, 'sz_starts'] = result['sz_starts']
            df.at[idx, 'sz_ends'] = result['sz_ends']
            df.at[idx, 'nchns'] = result['nchns']
            
            if pd.isna(row.get('duration', None)) or row.get('duration', '') == '':
                df.at[idx, 'duration'] = result['duration_sec']
            
            if result['has_sphenoidal']:
                sphenoidal_count += 1
            
            processed_count += 1
            
            if result['nsz'] > 0:
                logger.info(f"检测到发作 [{pt_id}_{fn}]: "
                           f"发作次数={result['nsz']}, "
                           f"时间段={list(result['seizure_regions'])}")
            
        except Exception as e:
            logger.error(f"处理失败 [{pt_id}_{fn}]: {e}")
            import traceback
            print(traceback.format_exc())
            error_count += 1
    
    # 保存结果
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(manifest_path).parent / f"detected_ensemble_{timestamp}.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info("=" * 60)
    logger.info("集成检测完成!")
    logger.info(f"  使用模型数: {len(model_paths)}")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  包含蝶骨电极: {sphenoidal_count}")
    logger.info(f"  文件未找到: {not_found_count}")
    logger.info(f"  处理失败: {error_count}")
    logger.info(f"  结果保存至: {output_path}")
    logger.info("=" * 60)
    
    return df


# ==============================================================================
# 命令行入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='使用DeepSOZ模型检测癫痫发作时间段 (支持21通道：19标准+2蝶骨)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型检测
  python detect_seizures_private.py --manifest path/to/converted_manifest.csv --data-root E:/DataSet/EEG
  
  # 集成检测 (使用多个fold的模型)
  python detect_seizures_private.py --manifest path/to/converted_manifest.csv --data-root E:/DataSet/EEG --ensemble
  
  # 指定阈值和输出路径
  python detect_seizures_private.py --manifest path/to/converted_manifest.csv --data-root E:/DataSet/EEG --threshold 0.4 --output detected_results.csv


配置：--manifest E:/code_learn/SUAT/workspace/EEG-projects/EEG_SUAT_NEW/DeepSOZ/converted_manifest.csv --data-root "E:/DataSet/EEG/EEG dataset_SUAT" --ensemble
通道说明:
  - 标准19通道: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
  - 蝶骨电极: Sph-L, Sph-R (可选，如果EDF中存在则自动提取)
        """
    )
    
    parser.add_argument('--manifest', type=str, required=True,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-root', type=str, nargs='+', required=True,
                        help='EDF数据根目录 (可指定多个)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型检查点路径 (默认使用fold0)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型目录 (用于集成检测)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径')
    parser.add_argument('--threshold', type=float, default=SEIZURE_THRESHOLD,
                        help=f'发作检测阈值 (默认: {SEIZURE_THRESHOLD})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备: cuda 或 cpu (默认: cuda)')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成检测 (多个模型)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='集成检测使用的fold数量 (默认: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 确定模型路径
    if args.model:
        model_path = args.model
    else:
        # 使用默认路径
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent.parent / 'DeepSOZ' / 'final_models'
        model_path = model_dir / 'fold0' / 'txlstm_szpool_finetuned_cv0_0.0001.pth.tar'
    
    if args.model_dir:
        model_dir = args.model_dir
    else:
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent.parent / 'DeepSOZ' / 'final_models'
    
    # 运行检测
    if args.ensemble:
        df = process_manifest_ensemble(
            manifest_path=args.manifest,
            model_dir=str(model_dir),
            data_roots=args.data_root,
            output_path=args.output,
            device=args.device,
            threshold=args.threshold,
            n_folds=args.n_folds
        )
    else:
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            logger.info("请使用 --model 参数指定正确的模型路径")
            return
        
        df = process_manifest(
            manifest_path=args.manifest,
            model_path=str(model_path),
            data_roots=args.data_root,
            output_path=args.output,
            device=args.device,
            threshold=args.threshold
        )
    
    # 输出检测统计
    total_seizures = df['nsz'].sum() if 'nsz' in df.columns else 0
    logger.info(f"共检测到 {total_seizures} 次癫痫发作")


if __name__ == '__main__':
    main()
