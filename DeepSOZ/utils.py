"""
通用工具模块
包含数据加载、manifest处理、EDF读取等通用函数
"""

import numpy as np
import torch
import csv
import json
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


def read_manifest(filename, delimiter=','):
    """
    读取manifest CSV文件
    
    Args:
        filename: CSV文件路径
        delimiter: 分隔符
    
    Returns:
        list of dict: manifest条目列表
    """
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        dicts = list(reader)
    return dicts


def write_manifest(manifest_list, filename='manifest.csv', delimiter=','):
    """
    写入manifest CSV文件
    
    Args:
        manifest_list: manifest条目列表
        filename: 输出文件路径
        delimiter: 分隔符
    """
    if len(manifest_list) == 0:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = manifest_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for item in manifest_list:
            writer.writerow(item)


def load_manifest_as_df(filename, delimiter=','):
    """
    加载manifest为pandas DataFrame
    
    Args:
        filename: CSV文件路径
        delimiter: 分隔符
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)


def parse_json_field(value, default=None):
    """
    安全解析可能是JSON格式的字段
    
    Args:
        value: 字段值
        default: 默认值
    
    Returns:
        解析后的值
    """
    if pd.isna(value) or value == '' or value is None:
        return default
    
    if isinstance(value, (int, float)):
        return value
    
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, ValueError):
        return value


def safe_int(value, default=0):
    """安全转换为整数"""
    try:
        if pd.isna(value) or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """安全转换为浮点数"""
    try:
        if pd.isna(value) or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# 标准19通道顺序 (与DeepSOZ保持一致)
STANDARD_19_CHANNELS = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 
    't3', 'c3', 'cz', 'c4', 't4', 
    't5', 'p3', 'pz', 'p4', 't6', 
    'o1', 'o2'
]

# 与DeepSOZ load_onset_map函数使用的顺序一致
DEEPSOZ_CHANNEL_ORDER = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 
    't3', 'c3', 'cz', 'c4', 't4', 
    't5', 'p3', 'pz', 'p4', 't6', 
    'o1', 'o2'
]

# 脑区映射
BRAIN_REGIONS = {
    'frontal': ['fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz'],
    'central': ['c3', 'c4', 'cz'],
    'temporal': ['t3', 't4', 't5', 't6'],
    'parietal': ['p3', 'p4', 'pz'],
    'occipital': ['o1', 'o2']
}

# 反向映射：通道到脑区
CHANNEL_TO_REGION = {}
for region, channels in BRAIN_REGIONS.items():
    for ch in channels:
        CHANNEL_TO_REGION[ch] = region

# 脑区编号映射
REGION_TO_IDX = {
    'frontal': 0,
    'central': 1,
    'temporal': 2,
    'parietal': 3,
    'occipital': 4
}

IDX_TO_REGION = {v: k for k, v in REGION_TO_IDX.items()}


def load_onset_map(mnitem, channel_order=None):
    """
    从manifest条目加载onset map (SOZ标记)
    
    Args:
        mnitem: manifest条目 (dict)
        channel_order: 通道顺序列表，默认使用DEEPSOZ_CHANNEL_ORDER
    
    Returns:
        np.ndarray: onset map (长度为通道数的01向量)
    """
    if channel_order is None:
        channel_order = DEEPSOZ_CHANNEL_ORDER
    
    soz = np.zeros(len(channel_order))
    
    for i, chn in enumerate(channel_order):
        value = mnitem.get(chn, '')
        if value != '' and value is not None:
            try:
                soz[i] = int(float(value))
            except (ValueError, TypeError):
                soz[i] = 0
    
    return soz


def get_region_label(mnitem, channel_order=None):
    """
    根据onset map获取脑区级别的标签
    
    Args:
        mnitem: manifest条目
        channel_order: 通道顺序列表
    
    Returns:
        int: 脑区索引 (0-4)，-1表示未知
    """
    if channel_order is None:
        channel_order = DEEPSOZ_CHANNEL_ORDER
    
    soz = load_onset_map(mnitem, channel_order)
    
    # 统计每个脑区的标记数量
    region_counts = {r: 0 for r in REGION_TO_IDX.keys()}
    
    for i, ch in enumerate(channel_order):
        if soz[i] == 1 and ch in CHANNEL_TO_REGION:
            region = CHANNEL_TO_REGION[ch]
            region_counts[region] += 1
    
    # 找出标记最多的脑区
    max_count = max(region_counts.values())
    if max_count == 0:
        return -1  # 无标记
    
    # 按优先级选择
    priority_order = ['temporal', 'frontal', 'parietal', 'central', 'occipital']
    for region in priority_order:
        if region_counts[region] == max_count:
            return REGION_TO_IDX[region]
    
    return -1


def get_region_onehot(mnitem, channel_order=None):
    """
    获取脑区级别的one-hot标签 (多标签)
    
    Args:
        mnitem: manifest条目
        channel_order: 通道顺序列表
    
    Returns:
        np.ndarray: 5维one-hot向量 (可能有多个1)
    """
    if channel_order is None:
        channel_order = DEEPSOZ_CHANNEL_ORDER
    
    soz = load_onset_map(mnitem, channel_order)
    region_label = np.zeros(len(REGION_TO_IDX))
    
    for i, ch in enumerate(channel_order):
        if soz[i] == 1 and ch in CHANNEL_TO_REGION:
            region = CHANNEL_TO_REGION[ch]
            region_label[REGION_TO_IDX[region]] = 1
    
    return region_label


def normalize_eeg(data, method='zscore'):
    """
    标准化EEG数据
    
    Args:
        data: EEG数据
        method: 'zscore' 或 'minmax'
    
    Returns:
        标准化后的数据
    """
    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return (data - mean) / std
        return data - mean
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 0:
            return (data - min_val) / (max_val - min_val)
        return data - min_val
    else:
        return data


def split_patients(patient_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    划分患者为训练集、验证集、测试集
    
    Args:
        patient_ids: 患者ID列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        tuple: (train_ids, val_ids, test_ids)
    """
    np.random.seed(seed)
    patient_ids = list(set(patient_ids))
    np.random.shuffle(patient_ids)
    
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def get_patient_ids_from_manifest(manifest):
    """
    从manifest中提取唯一的患者ID
    
    Args:
        manifest: manifest列表
    
    Returns:
        list: 患者ID列表
    """
    patient_ids = set()
    for item in manifest:
        pt_id = item.get('pt_id', '')
        if pt_id:
            patient_ids.add(pt_id)
    return list(patient_ids)


def create_kfold_splits(patient_ids, k=5, seed=42):
    """
    创建K折交叉验证的划分
    
    Args:
        patient_ids: 患者ID列表
        k: 折数
        seed: 随机种子
    
    Returns:
        list: [(train_ids, val_ids), ...] 每折的训练和验证ID
    """
    np.random.seed(seed)
    patient_ids = list(set(patient_ids))
    np.random.shuffle(patient_ids)
    
    fold_size = len(patient_ids) // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        if i == k - 1:
            end = len(patient_ids)
        else:
            end = (i + 1) * fold_size
        
        val_ids = patient_ids[start:end]
        train_ids = patient_ids[:start] + patient_ids[end:]
        folds.append((train_ids, val_ids))
    
    return folds


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        filepath: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
        device: 设备
    
    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, delta=0, verbose=False):
        """
        Args:
            patience: 容忍的轮次数
            delta: 最小改进量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss, model, path=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if path:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if path:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
