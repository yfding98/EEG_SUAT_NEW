#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置文件

统一管理STGNN训练的所有超参数和配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # EEG-projects
STGNN_ROOT = PROJECT_ROOT / "EEG_SUAT_NEW" / "STGNN"
DATA_ROOT = PROJECT_ROOT / "EEG_SUAT_NEW" / "DeepSOZ"

# DeepSOZ项目路径（用于参考和模型加载）
DEEPSOZ_ROOT = PROJECT_ROOT / "DeepSOZ"


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    manifest_path: str = str(DATA_ROOT / "detected_ensemble_updated_with_baseline.csv")
    edf_data_roots: List[str] = field(default_factory=lambda: [
        # r"E:\DataSet\EEG\private_data",  # 主数据目录
        r"E:\DataSet\EEG\EEG dataset_SUAT",  # 备用目录
    ])
    
    # 标准19通道（DeepSOZ顺序）
    channel_names: List[str] = field(default_factory=lambda: [
        'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 
        't3', 'c3', 'cz', 'c4', 't4', 
        't5', 'p3', 'pz', 'p4', 't6', 
        'o1', 'o2'
    ])
    
    # 采样和预处理参数
    target_fs: float = 200.0  # DeepSOZ使用200Hz
    original_fs: float = 250.0  # 原始采样率
    window_length: float = 1.0  # 1秒窗口
    window_overlap: float = 0.0  # 无重叠
    
    # 滤波参数
    filter_low: float = 1.6  # 高通截止
    filter_high: float = 30.0  # 低通截止
    clip_std: float = 2.0  # 幅值裁剪（±2 std）
    
    # 数据限制
    max_seizures_per_record: int = 10
    max_windows: int = 600  # 最大10分钟
    min_windows: int = 30  # 最小45秒

    # 统一处理的时间窗口大小
    n_windows:int =30
    
    # baseline窗口配置
    baseline_window_before: float = 10.0  # 发作前10秒baseline
    baseline_include: bool = True  # 是否包含baseline窗口
    
    # 数据增强
    normalize: bool = True
    add_noise: bool = False
    noise_sigma: float = 0.2
    
    # 双极导联配置
    use_bipolar: bool = True  # 是否使用TCP双极导联
    bipolar_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        # 左颞链
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        # 右颞链
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        # 左副矢状链
        ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        # 右副矢状链
        ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        # 中线链
        ('FZ', 'CZ'), ('CZ', 'PZ'),
    ])


@dataclass
class ModelConfig:
    """STGNN模型配置"""
    # 基本参数
    n_channels: int = 19
    n_bands: int = 5  # 频带数（用于特征提取后）
    time_steps: int = 200  # 每窗口200采样点（1秒@200Hz）
    
    # 时间特征提取器
    temporal_hidden_dim: int = 32
    temporal_reduced_steps: int = 64
    
    # 图学习模块
    graph_learning_method: str = 'combined'  # 'parameter', 'feature', 'combined'
    graph_hidden_dim: int = 32
    
    # 图卷积模块
    graph_conv_type: str = 'gcn'  # 'gcn' or 'gat'
    graph_n_layers: int = 2
    
    # 分类头
    classifier_hidden_dim: Optional[int] = None  # None = use graph_hidden_dim
    
    # 正则化
    dropout: float = 0.6
    use_batch_norm: bool = True
    
    # 距离先验（容积传导）
    use_distance_prior: bool = False


@dataclass
class TrainingConfig:
    """训练配置"""
    # 优化器
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adam'  # 'adam' or 'adamw'
    
    # 学习率调度
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 训练参数
    batch_size: int = 4
    num_epochs: int = 100
    early_stopping_patience: int = 20
    
    # 交叉验证
    n_folds: int = 5
    val_ratio: float = 0.2
    
    # 设备
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    # 保存
    checkpoint_dir: str = str(DATA_ROOT / "train" / "checkpoints")
    log_dir: str = str(DATA_ROOT / "train" / "logs")
    save_every: int = 10
    
    # 混合精度训练
    use_amp: bool = True
    
    # 梯度裁剪
    grad_clip_norm: float = 1.0


@dataclass
class LossConfig:
    """损失函数配置"""
    # 分类损失
    classification_loss: str = 'bce'  # 'bce', 'focal', 'dice'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 正负样本权重（针对类别不平衡）
    pos_weight: float = 5.0  # SOZ通道通常较少
    
    # SOZ定位损失权重
    soz_loss_weight: float = 1.0
    
    # Map损失权重（参考DeepSOZ）
    map_loss_pos_weight: float = 1.0
    map_loss_neg_weight: float = 0.5
    map_loss_margin_weight: float = 0.5
    
    # L2正则化
    l2_weight: float = 0.0
    
    # 标签平滑
    label_smoothing: float = 0.1


@dataclass
class OnsetZoneConfig:
    """onset_zone脑区级别分类配置"""
    # 脑区定义（参考DeepSOZ的A/P分区）
    region_mapping: dict = field(default_factory=lambda: {
        'frontal': ['fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz'],
        'temporal': ['t3', 't4', 't5', 't6'],
        'central': ['c3', 'c4', 'cz'],
        'parietal': ['p3', 'p4', 'pz'],
        'occipital': ['o1', 'o2']
    })
    
    # DeepSOZ A/P分区
    ap_mapping: dict = field(default_factory=lambda: {
        'A': ['fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz', 'c3', 'c4', 'cz', 't3', 't4'],
        'P': ['t5', 't6', 'p3', 'p4', 'pz', 'o1', 'o2']
    })
    
    # 分类类别数
    num_classes: int = 5  # frontal, temporal, central, parietal, occipital
    
    # 多标签分类（一个样本可能有多个脑区）
    multi_label: bool = True


@dataclass
class HemiConfig:
    """hemi半球级别分类配置"""
    # 半球定义
    hemi_mapping: dict = field(default_factory=lambda: {
        'L': ['fp1', 'f3', 'f7', 't3', 'c3', 't5', 'p3', 'o1'],
        'R': ['fp2', 'f4', 'f8', 't4', 'c4', 't6', 'p4', 'o2'],
        'M': ['fz', 'cz', 'pz']  # 中线通道
    })
    
    # 分类类别：L, R, Bilateral, Unknown
    classes: List[str] = field(default_factory=lambda: ['L', 'R', 'B', 'U'])
    num_classes: int = 4


@dataclass
class ChannelConfig:
    """通道级别分类配置"""
    # 19通道二分类
    num_channels: int = 19
    
    # 通道邻居关系（用于图学习）
    channel_neighbors: dict = field(default_factory=lambda: {
        0: [1, 2, 3, 4],  # fp1
        1: [0, 4, 5, 6],  # fp2
        2: [0, 3, 7, 8],  # f7
        3: [0, 2, 4, 8, 9],  # f3
        4: [0, 1, 3, 5, 9],  # fz
        5: [1, 4, 6, 9, 10],  # f4
        6: [1, 5, 10, 11],  # f8
        7: [2, 8, 12, 13, 17],  # t3
        8: [2, 3, 7, 9, 12, 13, 14],  # c3
        9: [3, 4, 5, 8, 10, 13, 14, 15],  # cz
        10: [5, 6, 9, 11, 14, 15, 16],  # c4
        11: [6, 10, 15, 16, 18],  # t4
        12: [7, 8, 13, 17],  # t5
        13: [7, 8, 9, 12, 14, 17],  # p3
        14: [8, 9, 10, 13, 15, 17, 18],  # pz
        15: [9, 10, 11, 14, 16, 18],  # p4
        16: [10, 11, 15, 18],  # t6
        17: [7, 12, 13, 14, 18],  # o1
        18: [11, 14, 15, 16, 17]  # o2
    })


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    onset_zone: OnsetZoneConfig = field(default_factory=OnsetZoneConfig)
    hemi: HemiConfig = field(default_factory=HemiConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    
    def __post_init__(self):
        """初始化后检查和创建目录"""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)


def get_config() -> Config:
    """获取默认配置"""
    return Config()


def save_config(config: Config, path: str):
    """保存配置到文件"""
    import json
    from dataclasses import asdict
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> Config:
    """从文件加载配置"""
    import json
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Config(
        data=DataConfig(**data.get('data', {})),
        model=ModelConfig(**data.get('model', {})),
        training=TrainingConfig(**data.get('training', {})),
        loss=LossConfig(**data.get('loss', {})),
        onset_zone=OnsetZoneConfig(**data.get('onset_zone', {})),
        hemi=HemiConfig(**data.get('hemi', {})),
        channel=ChannelConfig(**data.get('channel', {}))
    )


if __name__ == '__main__':
    # 测试配置
    config = get_config()
    print(f"数据路径: {config.data.manifest_path}")
    print(f"通道数: {config.model.n_channels}")
    print(f"学习率: {config.training.learning_rate}")
    print(f"checkpoint目录: {config.training.checkpoint_dir}")
