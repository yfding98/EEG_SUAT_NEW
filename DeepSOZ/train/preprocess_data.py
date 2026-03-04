#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本

将原始EDF数据预处理并保存为.npz文件，包含：
- eeg_data: 预处理后的EEG数据
- connectivity: 连接性矩阵
- graph_metrics: 图网络指标
- labels: 标签
- metadata: 元数据

使用方式：
    python preprocess_data.py --manifest path/to/manifest.csv --output-dir path/to/output
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from dataset_with_connectivity import (
    MultiModalEEGDataset,
    parse_onset_zone_label,
    parse_hemi_label,
    parse_channel_labels,
    parse_baseline,
    parse_seizure_times,
    parse_mask_segments,
    calculate_clean_duration,
)
from dataset import (
    read_edf,
    extract_standard_channels,
    bandpass_filter,
    clip_amplitude,
    resample_signal,
    convert_to_bipolar,
    STANDARD_19_CHANNELS,
    STANDARD_21_CHANNELS,
    BIPOLAR_PAIRS_18,
    BIPOLAR_PAIRS_26,
    BIPOLAR_CHANNEL_NAMES,
    BIPOLAR_CHANNEL_NAMES_26,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_and_save_dataset(
    manifest_path: str,
    data_roots: List[str],
    output_dir: str,
    segment_length: float = 20.0,
    segment_overlap: float = 0.5,
    window_length: float = 1.0,
    connectivity_types: List[str] = None,
    connectivity_freq_band: Tuple[float, float] = (8, 30),
    include_directed: bool = True,
    patient_ids: List[str] = None,
    config: Config = None,
    label_type: str = 'onset_zone',
):
    """
    预处理数据集并保存到磁盘
    
    Args:
        manifest_path: CSV manifest文件路径
        data_roots: EDF数据根目录列表
        output_dir: 输出目录
        segment_length: 片段长度(秒)
        segment_overlap: 重叠比例
        window_length: 窗口长度(秒)
        connectivity_types: 连接性类型列表
        connectivity_freq_band: 连接性频带
        include_directed: 是否包含有向连接性
        patient_ids: 指定患者ID列表(可选,用于增量处理)
        config: 配置对象
        label_type: 标签类型，支持 onset_zone, hemi, channel, chain, region_5
        
    注意:
        当 label_type='region_5' 时，会自动启用：
        - use_21_channels=True (需要SPHL/SPHR电极)
        - use_bipolar=True (使用26通道双极导联)
        这将同时保存双极导联数据和原始单极数据
    """
    # 当使用 region_5 时，强制启用 21 电极和双极导联
    if label_type == 'region_5':
        if config is not None:
            if not getattr(config.data, 'use_21_channels', False):
                logger.info("region_5 标签需要21电极配置，自动启用 use_21_channels=True")
                config.data.use_21_channels = True
            if not getattr(config.data, 'use_bipolar', False):
                logger.info("region_5 标签需要双极导联，自动启用 use_bipolar=True")
                config.data.use_bipolar = True
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = get_config()
    
    if connectivity_types is None:
        if include_directed:
            connectivity_types = ['plv', 'wpli', 'aec', 'pearson', 'granger', 'transfer_entropy']
        else:
            connectivity_types = ['plv', 'wpli', 'aec', 'pearson']
    
    logger.info(f"开始数据预处理...")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  片段长度: {segment_length}s, 重叠: {segment_overlap*100:.0f}%")
    logger.info(f"  连接性类型: {connectivity_types}")
    
    # 创建数据集（用于遍历和预处理）
    dataset = MultiModalEEGDataset(
        manifest_path=manifest_path,
        data_roots=data_roots,
        label_type=label_type,
        patient_ids=patient_ids,
        config=config.data,
        segment_length=segment_length,
        segment_overlap=segment_overlap,
        window_length=window_length,
        connectivity_types=connectivity_types,
        connectivity_freq_band=connectivity_freq_band,
        compute_online=True,
        include_directed=include_directed
    )
    
    logger.info(f"数据集大小: {len(dataset)} 样本")
    
    # 保存配置信息
    config_info = {
        'segment_length': segment_length,
        'segment_overlap': segment_overlap,
        'window_length': window_length,
        'connectivity_types': connectivity_types,
        'connectivity_freq_band': connectivity_freq_band,
        'include_directed': include_directed,
        'target_fs': config.data.target_fs,
        'filter_low': config.data.filter_low,
        'filter_high': config.data.filter_high,
        # 新增：标签类型和通道配置
        'label_type': label_type,
        'use_21_channels': getattr(config.data, 'use_21_channels', False),
        'use_bipolar': getattr(config.data, 'use_bipolar', False),
    }
    np.save(output_path / 'config_info.npy', config_info)
    
    logger.info(f"配置信息:")
    logger.info(f"  标签类型: {label_type}")
    logger.info(f"  使用21电极: {config_info['use_21_channels']}")
    logger.info(f"  使用双极导联: {config_info['use_bipolar']}")
    
    # 按患者分组保存
    df = pd.read_csv(manifest_path)
    all_patient_ids = df['pt_id'].unique().tolist()
    
    if patient_ids is not None:
        all_patient_ids = [p for p in all_patient_ids if p in patient_ids]
    
    # 创建样本索引映射
    sample_to_patient = {}
    for i, sample in enumerate(dataset.samples):
        pt_id = sample.get('pt_id', 'unknown')
        if pt_id not in sample_to_patient:
            sample_to_patient[pt_id] = []
        sample_to_patient[pt_id].append(i)
    
    # 按患者处理并保存
    saved_count = 0
    failed_count = 0
    
    for pt_id in tqdm(all_patient_ids, desc="处理患者"):
        if pt_id not in sample_to_patient:
            continue
        
        sample_indices = sample_to_patient[pt_id]
        patient_data = {
            'eeg_data': [],               # 原始单极数据 (19/21通道)
            'connectivity': [],           # 基于原始数据的连接性
            'graph_metrics': [],          # 基于原始数据的图指标
            'bipolar_eeg_data': [],       # 双极导联数据 (18/26通道)
            'bipolar_connectivity': [],   # 基于双极导联的连接性
            'bipolar_graph_metrics': [],  # 基于双极导联的图指标
            'labels': [],
            'metadata': []
        }
        
        for idx in sample_indices:
            try:
                sample = dataset[idx]
                
                # eeg_data 始终是原始单极数据
                patient_data['eeg_data'].append(sample['eeg_data'].numpy())
                # connectivity 和 graph_metrics 基于原始数据
                patient_data['connectivity'].append(sample['connectivity'].numpy())
                patient_data['graph_metrics'].append(sample['graph_metrics'].numpy())
                
                # 如果数据集返回了双极导联相关数据，保存之
                if 'bipolar_eeg_data' in sample:
                    patient_data['bipolar_eeg_data'].append(sample['bipolar_eeg_data'].numpy())
                if 'bipolar_connectivity' in sample:
                    patient_data['bipolar_connectivity'].append(sample['bipolar_connectivity'].numpy())
                if 'bipolar_graph_metrics' in sample:
                    patient_data['bipolar_graph_metrics'].append(sample['bipolar_graph_metrics'].numpy())
                
                patient_data['labels'].append(sample['labels'].numpy())
                patient_data['metadata'].append({
                    'pt_id': sample['pt_id'],
                    'fn': sample['fn'],
                    'sz_idx': sample['sz_idx'],
                    'win_idx': sample['win_idx'],
                })
                
                saved_count += 1
                
            except Exception as e:
                logger.warning(f"处理样本 {idx} 失败: {e}")
                failed_count += 1
        
        if len(patient_data['eeg_data']) > 0:
            # 保存为.npz文件
            patient_file = output_path / f"{pt_id}.npz"
            
            save_dict = {
                'eeg_data': np.array(patient_data['eeg_data']),  # 原始单极数据
                'connectivity': np.array(patient_data['connectivity']),  # 基于原始数据
                'graph_metrics': np.array(patient_data['graph_metrics']),  # 基于原始数据
                'labels': np.array(patient_data['labels']),
                'metadata': np.array(patient_data['metadata'], dtype=object)
            }
            
            # 如果有双极导联相关数据，也保存
            if len(patient_data['bipolar_eeg_data']) > 0:
                save_dict['bipolar_eeg_data'] = np.array(patient_data['bipolar_eeg_data'])
            if len(patient_data['bipolar_connectivity']) > 0:
                save_dict['bipolar_connectivity'] = np.array(patient_data['bipolar_connectivity'])
            if len(patient_data['bipolar_graph_metrics']) > 0:
                save_dict['bipolar_graph_metrics'] = np.array(patient_data['bipolar_graph_metrics'])
            
            np.savez_compressed(patient_file, **save_dict)
            logger.debug(f"保存 {pt_id}: {len(patient_data['eeg_data'])} 样本")
    
    # 保存患者列表
    np.save(output_path / 'patient_ids.npy', all_patient_ids)
    
    logger.info(f"预处理完成!")
    logger.info(f"  成功: {saved_count} 样本")
    logger.info(f"  失败: {failed_count} 样本")
    logger.info(f"  输出目录: {output_dir}")


class PreprocessedDataset:
    """
    加载预处理好的数据集
    
    从.npz文件加载，避免重复的数据处理
    支持多种标签类型：onset_zone, hemi, channel, chain, region_5
    """
    
    # 21电极单极到5脑区的映射（用于chain标签）
    CHAIN_ELECTRODES = {
        'left_frontal': ['fp1', 'f7', 'f3', 'fz'],
        'left_temporal': ['f7', 'sphl', 't3', 't5', 'o1', 'c3', 'p3'],
        'parietal': ['fz', 'cz', 'c3', 'c4', 'p3', 'pz', 'p4'],
        'right_frontal': ['fp2', 'f4', 'f8', 'fz'],
        'right_temporal': ['f8', 'sphr', 't4', 't6', 'o2', 'c4', 'p4'],
    }
    
    # 5脑区双极导联映射（用于region_5标签）
    # 每个脑区对应的双极导联列表
    REGION_5_BIPOLAR_LEADS = {
        'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
        'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
        'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
        'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
        'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
    }
    
    # 双极导联到所涉及电极的映射（用于从单极通道标签推断双极导联标签）
    # 每个双极导联由两个电极组成：阳极-阴极
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
    
    # 21电极通道名称
    CHANNEL_NAMES = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
                     't3', 'c3', 'cz', 'c4', 't4',
                     't5', 'p3', 'pz', 'p4', 't6',
                     'o1', 'o2',
                     'sphl', 'sphr']
    
    def __init__(
        self,
        preprocessed_dir: str,
        patient_ids: List[str] = None,
        label_type: str = 'onset_zone',
        label_config: Dict = None,
        manifest_path: str = None  # 新增：用于读取准确的chain标签
    ):
        """
        Args:
            preprocessed_dir: 预处理数据目录
            patient_ids: 要加载的患者ID列表，None表示全部
            label_type: 标签类型 (onset_zone, hemi, channel, chain)
            label_config: 标签配置字典，包含classes和n_classes
            manifest_path: manifest CSV路径，用于读取准确的chain/channel标签
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.label_type = label_type
        self.label_config = label_config or {}
        self.manifest_path = manifest_path
        
        # 如果需要chain/channel/hemi/region_5标签，加载manifest
        self.manifest_df = None
        if label_type in ['chain', 'channel', 'hemi', 'region_5'] and manifest_path:
            try:
                self.manifest_df = pd.read_csv(manifest_path)
                logger.info(f"加载manifest用于{label_type}标签: {manifest_path}")
            except Exception as e:
                logger.warning(f"加载manifest失败: {e}")
        
        # 加载配置
        config_path = self.preprocessed_dir / 'config_info.npy'
        if config_path.exists():
            self.config_info = np.load(config_path, allow_pickle=True).item()
        else:
            self.config_info = {}
        
        # 加载患者列表
        all_patients_path = self.preprocessed_dir / 'patient_ids.npy'
        if all_patients_path.exists():
            all_patients = np.load(all_patients_path, allow_pickle=True).tolist()
        else:
            # 从文件名推断
            all_patients = [f.stem for f in self.preprocessed_dir.glob('*.npz')]
        
        if patient_ids is not None:
            self.patient_ids = [p for p in patient_ids if p in all_patients]
        else:
            self.patient_ids = all_patients
        
        # 加载数据
        self._load_data()
    
    def _get_chain_labels_from_manifest(self, pt_id: str, fn: str = None) -> np.ndarray:
        """从manifest读取准确的chain标签"""
        chain_names = ['left_temporal', 'right_temporal', 'left_parasagittal', 
                       'right_parasagittal', 'midline']
        if 'classes' in self.label_config:
            chain_names = self.label_config['classes']
        
        chain_labels = np.zeros(len(chain_names), dtype=np.float32)
        
        if self.manifest_df is None:
            return chain_labels
        
        # 查找患者记录
        row = None
        if fn:
            matches = self.manifest_df[(self.manifest_df['pt_id'] == pt_id) & 
                                       (self.manifest_df['fn'] == fn)]
        else:
            matches = self.manifest_df[self.manifest_df['pt_id'] == pt_id]
        
        if len(matches) == 0:
            return chain_labels
        
        row = matches.iloc[0]
        
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
        for chain_name in chain_names:
            if chain_name not in self.CHAIN_ELECTRODES:
                continue
            chain_electrodes = self.CHAIN_ELECTRODES[chain_name]
            # 如果该链中任何电极是SOZ，则该链为正
            if any(elec.lower() in soz_channels for elec in chain_electrodes):
                chain_labels[chain_names.index(chain_name)] = 1.0
        
        return chain_labels

    def _get_region5_labels_from_manifest(self, pt_id: str, fn: str = None) -> np.ndarray:
        """
        从manifest读取准确的5脑区(双极导联)标签
        
        5脑区基于双极导联定义：
        - left_frontal: FP1-F7, FP1-F3, F7-F3, F3-FZ
        - left_temporal: F7-SPHL, SPHL-T3, T3-T5, T5-O1, T3-C3, T5-P3
        - parietal: FZ-CZ, C3-CZ, P3-PZ, CZ-PZ, CZ-C4, PZ-P4
        - right_frontal: FP2-F4, FP2-F8, F4-F8, FZ-F4
        - right_temporal: F8-SPHR, SPHR-T4, C4-T4, T4-T6, P4-T6, T6-O2
        
        判断逻辑：如果某双极导联的任一组成电极是SOZ，则该导联被标记为SOZ，
        如果某脑区的任一双极导联是SOZ，则该脑区被标记为SOZ
        """
        region_names = ['left_frontal', 'left_temporal', 'parietal', 
                        'right_frontal', 'right_temporal']
        if 'classes' in self.label_config:
            region_names = self.label_config['classes']
        
        region_labels = np.zeros(len(region_names), dtype=np.float32)
        
        if self.manifest_df is None:
            return region_labels
        
        # 查找患者记录
        row = None
        if fn:
            matches = self.manifest_df[(self.manifest_df['pt_id'] == pt_id) & 
                                       (self.manifest_df['fn'] == fn)]
        else:
            matches = self.manifest_df[self.manifest_df['pt_id'] == pt_id]
        
        if len(matches) == 0:
            return region_labels
        
        row = matches.iloc[0]
        
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
        # 逻辑：如果某双极导联的任一组成电极是SOZ，则该导联标记为SOZ
        #       如果某脑区的任一双极导联是SOZ，则该脑区标记为SOZ
        for region_name in region_names:
            if region_name not in self.REGION_5_BIPOLAR_LEADS:
                continue
            
            region_bipolar_leads = self.REGION_5_BIPOLAR_LEADS[region_name]
            region_is_soz = False
            
            for lead in region_bipolar_leads:
                # 获取该双极导联的组成电极
                if lead in self.BIPOLAR_LEAD_ELECTRODES:
                    electrodes = self.BIPOLAR_LEAD_ELECTRODES[lead]
                    # 如果任一电极是SOZ，则该导联是SOZ
                    if any(elec.lower() in soz_channels for elec in electrodes):
                        region_is_soz = True
                        break
            
            if region_is_soz:
                region_labels[region_names.index(region_name)] = 1.0
        
        return region_labels

    def _convert_labels(self, original_labels: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        根据label_type转换标签
        
        原始预处理数据中的labels是onset_zone (5类)
        需要根据label_type转换为相应的标签格式
        """
        if self.label_type == 'onset_zone':
            # 保持原样，但可能需要排除某些类别
            if 'classes' in self.label_config:
                original_classes = ['frontal', 'temporal', 'central', 'parietal', 'occipital']
                target_classes = self.label_config['classes']
                
                # 提取目标类别的索引
                indices = [original_classes.index(c) for c in target_classes if c in original_classes]
                return original_labels[indices]
            return original_labels
        
        elif self.label_type == 'chain':
            # 优先从manifest读取准确的chain标签
            pt_id = metadata.get('pt_id', '')
            fn = metadata.get('fn', '')
            
            if self.manifest_df is not None:
                # 使用manifest中的通道级别SOZ计算准确的chain标签
                chain_labels = self._get_chain_labels_from_manifest(pt_id, fn)
                return chain_labels
            
            # 如果没有manifest，使用近似映射（不推荐）
            chain_names = ['left_temporal', 'right_temporal', 'left_parasagittal', 
                           'right_parasagittal', 'midline']
            if 'classes' in self.label_config:
                chain_names = self.label_config['classes']
            
            chain_labels = np.zeros(len(chain_names), dtype=np.float32)
            
            # 简单映射（不精确，仅作为后备）
            onset_regions = original_labels > 0.5
            if len(onset_regions) >= 5:
                frontal, temporal, central, parietal, occipital = onset_regions[:5]
                
                if frontal or temporal:
                    if 'left_temporal' in chain_names:
                        chain_labels[chain_names.index('left_temporal')] = 1.0
                    if 'right_temporal' in chain_names:
                        chain_labels[chain_names.index('right_temporal')] = 1.0
                if central:
                    if 'left_parasagittal' in chain_names:
                        chain_labels[chain_names.index('left_parasagittal')] = 1.0
                    if 'right_parasagittal' in chain_names:
                        chain_labels[chain_names.index('right_parasagittal')] = 1.0

            return chain_labels
        
        elif self.label_type == 'hemi':
            # 从onset_zone推断半球（不精确）
            # L, R, B, U
            hemi_classes = self.label_config.get('classes', ['L', 'R', 'B', 'U'])
            hemi_labels = np.zeros(len(hemi_classes), dtype=np.float32)
            # 默认返回Unknown，实际应从manifest读取
            if 'U' in hemi_classes:
                hemi_labels[hemi_classes.index('U')] = 1.0
            return hemi_labels
        
        elif self.label_type == 'channel':
            # 通道级别标签（19通道）
            channel_classes = self.label_config.get('classes', self.CHANNEL_NAMES)
            channel_labels = np.zeros(len(channel_classes), dtype=np.float32)
            # 无法从onset_zone准确推断，返回空标签
            return channel_labels
        
        elif self.label_type == 'region_5':
            # 5脑区（双极导联）标签
            # 优先从manifest读取准确的region_5标签
            pt_id = metadata.get('pt_id', '')
            fn = metadata.get('fn', '')
            
            if self.manifest_df is not None:
                # 使用manifest中的通道级别SOZ计算准确的region_5标签
                region_labels = self._get_region5_labels_from_manifest(pt_id, fn)
                return region_labels
            
            # 如果没有manifest，使用onset_zone近似映射（不推荐）
            region_names = ['left_frontal', 'left_temporal', 'parietal', 
                            'right_frontal', 'right_temporal']
            if 'classes' in self.label_config:
                region_names = self.label_config['classes']
            
            region_labels = np.zeros(len(region_names), dtype=np.float32)
            
            # 近似映射（基于onset_zone的5类：frontal, temporal, central, parietal, occipital）
            onset_regions = original_labels > 0.5
            if len(onset_regions) >= 5:
                frontal, temporal, central, parietal, occipital = onset_regions[:5]
                
                # frontal -> left_frontal 和 right_frontal
                if frontal:
                    if 'left_frontal' in region_names:
                        region_labels[region_names.index('left_frontal')] = 1.0
                    if 'right_frontal' in region_names:
                        region_labels[region_names.index('right_frontal')] = 1.0
                
                # temporal -> left_temporal 和 right_temporal
                if temporal:
                    if 'left_temporal' in region_names:
                        region_labels[region_names.index('left_temporal')] = 1.0
                    if 'right_temporal' in region_names:
                        region_labels[region_names.index('right_temporal')] = 1.0
                
                # central + parietal -> parietal
                if central or parietal:
                    if 'parietal' in region_names:
                        region_labels[region_names.index('parietal')] = 1.0
            
            return region_labels
        
        else:
            return original_labels
    
    def _load_data(self):
        """加载所有数据到内存"""
        self.samples = []
        self.has_bipolar_data = False  # 标记是否有双极导联相关数据
        
        for pt_id in tqdm(self.patient_ids, desc="加载预处理数据"):
            patient_file = self.preprocessed_dir / f"{pt_id}.npz"
            if not patient_file.exists():
                logger.warning(f"未找到患者文件: {patient_file}")
                continue
            
            data = np.load(patient_file, allow_pickle=True)
            
            # 检查是否有双极导联相关数据
            has_bipolar_eeg = 'bipolar_eeg_data' in data.files
            has_bipolar_conn = 'bipolar_connectivity' in data.files
            has_bipolar_graph = 'bipolar_graph_metrics' in data.files
            
            if has_bipolar_eeg and not self.has_bipolar_data:
                self.has_bipolar_data = True
                logger.info(f"检测到双极导联数据")
            
            n_samples = len(data['eeg_data'])
            for i in range(n_samples):
                metadata = data['metadata'][i] if 'metadata' in data else {}
                original_labels = data['labels'][i]
                
                # 转换标签
                converted_labels = self._convert_labels(original_labels, metadata)
                
                sample = {
                    'eeg_data': data['eeg_data'][i],  # 原始单极数据
                    'connectivity': data['connectivity'][i],  # 基于原始数据
                    'graph_metrics': data['graph_metrics'][i],  # 基于原始数据
                    'labels': converted_labels,
                    'original_labels': original_labels,  # 保留原始标签
                    'metadata': metadata,
                }
                
                # 如果有双极导联相关数据，也加载
                if has_bipolar_eeg:
                    sample['bipolar_eeg_data'] = data['bipolar_eeg_data'][i]
                if has_bipolar_conn:
                    sample['bipolar_connectivity'] = data['bipolar_connectivity'][i]
                if has_bipolar_graph:
                    sample['bipolar_graph_metrics'] = data['bipolar_graph_metrics'][i]
                
                self.samples.append(sample)
        
        logger.info(f"加载了 {len(self.samples)} 个样本, 标签类型: {self.label_type}")
        if self.label_config:
            logger.info(f"类别数: {self.label_config.get('n_classes', 'unknown')}")
        if self.has_bipolar_data:
            logger.info(f"包含双极导联数据 (eeg_data, connectivity, graph_metrics)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import torch
        
        sample = self.samples[idx]
        metadata = sample.get('metadata', {})
        
        result = {
            'eeg_data': torch.from_numpy(sample['eeg_data']).float(),  # 原始单极数据
            'connectivity': torch.from_numpy(sample['connectivity']).float(),  # 基于原始数据
            'graph_metrics': torch.from_numpy(sample['graph_metrics']).float(),  # 基于原始数据
            'labels': torch.from_numpy(sample['labels']).float(),
            'pt_id': metadata.get('pt_id', 'unknown'),
            'fn': metadata.get('fn', 'unknown'),
            'sz_idx': metadata.get('sz_idx', 0),
            'win_idx': metadata.get('win_idx', 0),
        }
        
        # 如果有双极导联相关数据，也返回
        if 'bipolar_eeg_data' in sample:
            result['bipolar_eeg_data'] = torch.from_numpy(sample['bipolar_eeg_data']).float()
        if 'bipolar_connectivity' in sample:
            result['bipolar_connectivity'] = torch.from_numpy(sample['bipolar_connectivity']).float()
        if 'bipolar_graph_metrics' in sample:
            result['bipolar_graph_metrics'] = torch.from_numpy(sample['bipolar_graph_metrics']).float()
        
        return result



def parse_args():
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest CSV文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='EDF数据根目录列表')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    
    # 处理参数
    parser.add_argument('--segment-length', type=float, default=20.0,
                        help='片段长度(秒)')
    parser.add_argument('--segment-overlap', type=float, default=0.5,
                        help='重叠比例')
    parser.add_argument('--window-length', type=float, default=1.0,
                        help='窗口长度(秒)')
    
    # 连接性参数
    parser.add_argument('--include-directed', action='store_true', default=True,
                        help='包含有向连接性指标')
    parser.add_argument('--no-directed', action='store_false', dest='include_directed',
                        help='不包含有向连接性指标')
    parser.add_argument('--freq-band', type=float, nargs=2, default=[3, 45],
                        help='连接性计算频带 (Hz)')
    
    # 患者过滤
    parser.add_argument('--patient-ids', type=str, nargs='+', default=None,
                        help='指定处理的患者ID')
    
    # 通道配置
    parser.add_argument('--use-21-channels', action='store_true', default=False,
                        help='使用21电极配置（默认19电极）')
    parser.add_argument('--use-bipolar', action='store_true', default=False,
                        help='使用双极导联转换（默认单极）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = get_config()
    
    # 覆盖数据根目录
    if args.data_roots:
        config.data.edf_data_roots = args.data_roots
    # 覆盖配置
    if not args.manifest:
        args.manifest = config.data.manifest_path
    if not args.output_dir:
        args.output_dir = config.data.preprocessed_dir
    
    # 设置通道配置
    if args.use_21_channels:
        config.data.use_21_channels = True
        logger.info("使用21电极配置（SPHL, SPHR）")
    else:
        config.data.use_21_channels = False
    
    if args.use_bipolar:
        config.data.use_bipolar = True
        n_bipolar = 26 if args.use_21_channels else 18
        logger.info(f"使用双极导联转换（{n_bipolar}通道）")
    else:
        config.data.use_bipolar = False
    
    # 使用 region_5 作为默认标签类型
    label_type = "region_5"
    logger.info(f"使用标签类型: {label_type}")
    logger.info("注意: region_5 标签类型将自动启用 use_21_channels 和 use_bipolar")
    logger.info("这将同时保存：双极导联数据(26通道) 和 原始单极数据(21通道)")

    # 预处理并保存
    preprocess_and_save_dataset(
        manifest_path=args.manifest,
        data_roots=config.data.edf_data_roots,
        output_dir=args.output_dir,
        segment_length=args.segment_length,
        label_type=label_type,
        segment_overlap=args.segment_overlap,
        window_length=args.window_length,
        connectivity_freq_band=tuple(args.freq_band),
        include_directed=args.include_directed,
        patient_ids=args.patient_ids,
        config=config
    )


if __name__ == '__main__':
    main()
