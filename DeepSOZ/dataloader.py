"""
数据加载器模块
支持从EDF文件加载EEG数据，并与manifest文件配合使用
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import mne
from pathlib import Path

from utils import (
    read_manifest, load_onset_map, get_region_label, get_region_onehot,
    normalize_eeg, DEEPSOZ_CHANNEL_ORDER, CHANNEL_TO_REGION, REGION_TO_IDX
)


class EDFReader:
    """EDF文件读取器"""
    
    def __init__(self, channel_order=None, target_fs=200):
        """
        Args:
            channel_order: 目标通道顺序
            target_fs: 目标采样率
        """
        self.channel_order = channel_order or DEEPSOZ_CHANNEL_ORDER
        self.target_fs = target_fs
    
    def read(self, filepath, start_time=None, end_time=None):
        """
        读取EDF文件
        
        Args:
            filepath: EDF文件路径
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）
        
        Returns:
            data: EEG数据 [channels, samples]
            fs: 采样率
            channel_names: 实际读取的通道名
        """
        try:
            # 使用MNE读取
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # 获取通道名称
            ch_names = [ch.lower() for ch in raw.ch_names]
            
            # 重采样
            if raw.info['sfreq'] != self.target_fs:
                raw.resample(self.target_fs)
            
            # 选择时间段
            if start_time is not None or end_time is not None:
                start_time = start_time or 0
                end_time = end_time or raw.times[-1]
                raw.crop(tmin=start_time, tmax=end_time)
            
            # 获取数据
            data = raw.get_data()
            
            # 重新排列通道顺序
            reordered_data, matched_channels = self._reorder_channels(
                data, ch_names, self.channel_order
            )
            
            return reordered_data, self.target_fs, matched_channels
            
        except Exception as e:
            print(f"读取EDF文件失败: {filepath}, 错误: {e}")
            return None, None, None
    
    def _reorder_channels(self, data, ch_names, target_order):
        """
        重新排列通道顺序
        
        Args:
            data: 原始数据 [channels, samples]
            ch_names: 原始通道名
            target_order: 目标通道顺序
        
        Returns:
            reordered_data: 重新排列的数据
            matched_channels: 匹配的通道列表
        """
        n_samples = data.shape[1]
        n_target_channels = len(target_order)
        reordered_data = np.zeros((n_target_channels, n_samples))
        matched_channels = []
        
        # 创建通道名映射（处理不同命名方式）
        ch_name_mapping = {}
        for i, name in enumerate(ch_names):
            name_clean = name.lower().replace('-ref', '').replace('-le', '').strip()
            name_clean = name_clean.replace('eeg ', '').replace('eeg', '')
            # 处理T7/T8与T3/T4的映射
            if name_clean == 't7':
                name_clean = 't3'
            elif name_clean == 't8':
                name_clean = 't4'
            elif name_clean == 'p7':
                name_clean = 't5'
            elif name_clean == 'p8':
                name_clean = 't6'
            ch_name_mapping[name_clean] = i
        
        for i, target_ch in enumerate(target_order):
            target_clean = target_ch.lower()
            if target_clean in ch_name_mapping:
                idx = ch_name_mapping[target_clean]
                reordered_data[i, :] = data[idx, :]
                matched_channels.append(target_ch)
            else:
                # 通道不存在，填充零
                matched_channels.append(None)
        
        return reordered_data, matched_channels


class SOZDataset(Dataset):
    """
    SOZ定位数据集
    从manifest和EDF文件加载数据
    """
    
    def __init__(self, data_root, patient_list, manifest,
                 window_size=600, step_size=200,
                 normalize=True, transform=None,
                 channel_order=None, target_fs=200,
                 label_type='channel'):
        """
        Args:
            data_root: 数据根目录
            patient_list: 患者ID列表
            manifest: manifest列表
            window_size: 窗口大小（样本数）
            step_size: 滑动步长
            normalize: 是否标准化
            transform: 数据变换
            channel_order: 通道顺序
            target_fs: 目标采样率
            label_type: 标签类型 'channel' 或 'region'
        """
        self.data_root = data_root
        self.patient_list = patient_list
        self.manifest = manifest
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.transform = transform
        self.channel_order = channel_order or DEEPSOZ_CHANNEL_ORDER
        self.target_fs = target_fs
        self.label_type = label_type
        
        # 过滤manifest，只保留指定患者
        self.filtered_manifest = self._filter_manifest()
        
        # EDF读取器
        self.edf_reader = EDFReader(self.channel_order, self.target_fs)
        
        # 预计算所有样本索引
        self.samples = self._prepare_samples()
    
    def _filter_manifest(self):
        """过滤manifest，只保留指定患者"""
        filtered = []
        for item in self.manifest:
            pt_id = item.get('pt_id', '')
            if pt_id in self.patient_list:
                filtered.append(item)
        return filtered
    
    def _prepare_samples(self):
        """
        准备样本索引
        每个样本包含：(manifest_idx, window_start)
        """
        samples = []
        for idx, item in enumerate(self.filtered_manifest):
            # 这里假设每个发作记录作为一个样本
            # 如果需要滑动窗口，可以在这里添加逻辑
            samples.append({
                'manifest_idx': idx,
                'item': item
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        item = sample_info['item']
        
        # 加载onset map (SOZ标签)
        onset_map = load_onset_map(item, self.channel_order)
        
        # 尝试加载EDF数据
        loc = item.get('loc', '')
        edf_path = os.path.join(self.data_root, loc) if loc else None
        
        if edf_path and os.path.exists(edf_path):
            data, fs, matched_channels = self.edf_reader.read(edf_path)
            if data is not None:
                # 创建时间序列数据
                if self.normalize:
                    data = normalize_eeg(data)
                
                # 转换为PyTorch张量
                data_tensor = torch.from_numpy(data).float()
            else:
                # 无法读取，创建空数据
                data_tensor = torch.zeros(len(self.channel_order), self.window_size)
        else:
            # 没有EDF文件，创建模拟数据用于测试
            data_tensor = torch.randn(len(self.channel_order), self.window_size)
        
        # 准备标签
        if self.label_type == 'channel':
            label = torch.from_numpy(onset_map).float()
        elif self.label_type == 'region':
            region_label = get_region_onehot(item, self.channel_order)
            label = torch.from_numpy(region_label).float()
        else:
            label = torch.from_numpy(onset_map).float()
        
        # 应用变换
        if self.transform:
            data_tensor = self.transform(data_tensor)
        
        return {
            'data': data_tensor,
            'label': label,
            'onset_map': torch.from_numpy(onset_map).float(),
            'patient_id': item.get('pt_id', ''),
            'filename': item.get('fn', ''),
        }


class PreprocessedDataset(Dataset):
    """
    预处理数据集
    用于加载已经预处理好的npy文件（与DeepSOZ原始格式兼容）
    """
    
    def __init__(self, data_root, patient_list, manifest,
                 normalize=True, max_seizures=10,
                 channel_order=None):
        """
        Args:
            data_root: 数据根目录
            patient_list: 患者ID列表
            manifest: manifest列表
            normalize: 是否标准化
            max_seizures: 最大发作数
            channel_order: 通道顺序
        """
        self.data_root = data_root
        self.patient_list = patient_list
        self.normalize = normalize
        self.max_seizures = max_seizures
        self.channel_order = channel_order or DEEPSOZ_CHANNEL_ORDER
        
        # 过滤manifest
        self.filtered_manifest = []
        for item in manifest:
            pt_id = item.get('pt_id', '')
            try:
                pt_id_num = json.loads(pt_id) if isinstance(pt_id, str) else pt_id
            except:
                pt_id_num = pt_id
            
            if pt_id_num in patient_list or pt_id in patient_list:
                self.filtered_manifest.append(item)
        
        # 通道邻居关系 (用于数据增强)
        self.chn_neighbours = {
            0: [1, 2, 3, 4], 
            1: [0, 4, 5, 6], 
            2: [0, 3, 7, 8], 
            3: [0, 2, 4, 8, 9], 
            4: [0, 1, 3, 5, 9], 
            5: [1, 4, 6, 9, 10],
            6: [1, 5, 10, 11], 
            7: [2, 8, 12, 13, 17], 
            8: [2, 3, 7, 9, 12, 13, 14], 
            9: [3, 4, 5, 8, 10, 13, 14, 15], 
            10: [5, 6, 9, 11, 14, 15, 16], 
            11: [6, 10, 15, 16, 18], 
            12: [7, 8, 13, 17], 
            13: [7, 8, 9, 12, 14, 17],
            14: [8, 9, 10, 13, 15, 17, 18],
            15: [9, 10, 11, 14, 16, 18], 
            16: [10, 11, 15, 18], 
            17: [7, 12, 13, 14, 18], 
            18: [11, 14, 15, 16, 17]
        }
    
    def __len__(self):
        return len(self.filtered_manifest)
    
    def __getitem__(self, idx):
        item = self.filtered_manifest[idx]
        fn = item.get('fn', '')
        loc = item.get('loc', '')
        
        # 尝试加载预处理数据
        xloc = os.path.join(self.data_root, loc) if loc else None
        
        if xloc and os.path.exists(xloc):
            X = np.load(xloc)[:self.max_seizures]
            yloc = xloc.replace('.npy', '_label.npy')
            if os.path.exists(yloc):
                Y = np.load(yloc)[:self.max_seizures]
            else:
                Y = np.zeros((X.shape[0], X.shape[1]))
        else:
            # 创建模拟数据
            X = np.random.randn(1, 45, 19, 200)
            Y = np.zeros((1, 45))
        
        # 加载SOZ标签
        soz = self._load_onset_map(item)
        
        # 标准化
        if self.normalize and X.size > 0:
            X = (X - np.mean(X)) / (np.std(X) + 1e-6)
        
        return {
            'patient_numbers': fn,
            'buffers': torch.from_numpy(X).float(),
            'sz_labels': torch.from_numpy(Y).float(),
            'onset_map': torch.from_numpy(soz).float(),
        }
    
    def _load_onset_map(self, item):
        """加载onset map"""
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                   'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        
        for i, chn in enumerate(req_chn):
            value = item.get(chn, '')
            if value != '' and value is not None:
                try:
                    soz[i] = int(float(value))
                except:
                    soz[i] = 0
        
        return soz


class RegionDataset(Dataset):
    """
    脑区级别分类数据集
    将通道级别标签聚合到脑区级别
    """
    
    def __init__(self, data_root, patient_list, manifest,
                 normalize=True, channel_order=None):
        """
        Args:
            data_root: 数据根目录
            patient_list: 患者ID列表
            manifest: manifest列表
            normalize: 是否标准化
            channel_order: 通道顺序
        """
        self.data_root = data_root
        self.channel_order = channel_order or DEEPSOZ_CHANNEL_ORDER
        self.normalize = normalize
        
        # 过滤manifest
        self.filtered_manifest = []
        for item in manifest:
            pt_id = str(item.get('pt_id', ''))
            if pt_id in [str(p) for p in patient_list]:
                self.filtered_manifest.append(item)
        
        # EDF读取器
        self.edf_reader = EDFReader(self.channel_order)
    
    def __len__(self):
        return len(self.filtered_manifest)
    
    def __getitem__(self, idx):
        item = self.filtered_manifest[idx]
        
        # 加载通道级别标签
        onset_map = load_onset_map(item, self.channel_order)
        
        # 转换为脑区级别标签
        region_label = get_region_onehot(item, self.channel_order)
        
        # 获取主要脑区（单标签）
        region_idx = get_region_label(item, self.channel_order)
        if region_idx == -1:
            region_idx = 0  # 默认
        
        # 加载EDF数据
        loc = item.get('loc', '')
        edf_path = os.path.join(self.data_root, loc) if loc else None
        
        if edf_path and os.path.exists(edf_path):
            data, fs, _ = self.edf_reader.read(edf_path)
            if data is not None:
                if self.normalize:
                    data = normalize_eeg(data)
            else:
                data = np.random.randn(len(self.channel_order), 60000)
        else:
            data = np.random.randn(len(self.channel_order), 60000)
        
        return {
            'data': torch.from_numpy(data).float(),
            'region_label_onehot': torch.from_numpy(region_label).float(),
            'region_label': region_idx,
            'onset_map': torch.from_numpy(onset_map).float(),
            'patient_id': item.get('pt_id', ''),
            'filename': item.get('fn', ''),
        }


class ChannelDataset(Dataset):
    """
    通道级别分类数据集
    预测每个通道是否为SOZ
    """
    
    def __init__(self, data_root, patient_list, manifest,
                 normalize=True, channel_order=None):
        """
        Args:
            data_root: 数据根目录
            patient_list: 患者ID列表
            manifest: manifest列表
            normalize: 是否标准化
            channel_order: 通道顺序
        """
        self.data_root = data_root
        self.channel_order = channel_order or DEEPSOZ_CHANNEL_ORDER
        self.normalize = normalize
        
        # 过滤manifest
        self.filtered_manifest = []
        for item in manifest:
            pt_id = str(item.get('pt_id', ''))
            if pt_id in [str(p) for p in patient_list]:
                self.filtered_manifest.append(item)
        
        self.edf_reader = EDFReader(self.channel_order)
    
    def __len__(self):
        return len(self.filtered_manifest)
    
    def __getitem__(self, idx):
        item = self.filtered_manifest[idx]
        
        # 加载通道级别标签
        onset_map = load_onset_map(item, self.channel_order)
        
        # 加载EDF数据
        loc = item.get('loc', '')
        edf_path = os.path.join(self.data_root, loc) if loc else None
        
        if edf_path and os.path.exists(edf_path):
            data, fs, _ = self.edf_reader.read(edf_path)
            if data is not None:
                if self.normalize:
                    data = normalize_eeg(data)
            else:
                data = np.random.randn(len(self.channel_order), 60000)
        else:
            data = np.random.randn(len(self.channel_order), 60000)
        
        return {
            'data': torch.from_numpy(data).float(),
            'channel_label': torch.from_numpy(onset_map).float(),
            'patient_id': item.get('pt_id', ''),
            'filename': item.get('fn', ''),
        }


def create_data_loaders(data_root, manifest_path, batch_size=1,
                       train_ratio=0.7, val_ratio=0.15,
                       label_type='channel', seed=42):
    """
    创建训练、验证、测试数据加载器
    
    Args:
        data_root: 数据根目录
        manifest_path: manifest文件路径
        batch_size: 批大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        label_type: 标签类型 'channel' 或 'region'
        seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from utils import split_patients, get_patient_ids_from_manifest
    
    manifest = read_manifest(manifest_path)
    patient_ids = get_patient_ids_from_manifest(manifest)
    
    train_ids, val_ids, test_ids = split_patients(
        patient_ids, train_ratio, val_ratio, seed
    )
    
    print(f"数据集划分: 训练 {len(train_ids)}, 验证 {len(val_ids)}, 测试 {len(test_ids)}")
    
    if label_type == 'channel':
        DatasetClass = ChannelDataset
    else:
        DatasetClass = RegionDataset
    
    train_dataset = DatasetClass(data_root, train_ids, manifest)
    val_dataset = DatasetClass(data_root, val_ids, manifest)
    test_dataset = DatasetClass(data_root, test_ids, manifest)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
