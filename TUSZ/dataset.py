#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined SOZ Dataset
────────────────────────────────────────────────────────────────────
基于 combined_manifest.csv 的统一 PyTorch Dataset。

combined_manifest 结构（每行 = 一次发作事件）：
  source          : 'tusz' / 'private'
  patient_id      : 患者ID
  edf_path        : EDF 相对路径（TUSZ 相对 tusz_data_root；private 相对 private_data_root）
  split           : train / dev / eval / private
  duration        : 文件时长（秒）
  sz_start        : 本次发作起始时间（秒）
  sz_end          : 本次发作结束时间（秒）
  sz_duration     : 本次发作时长（秒）
  n_seizure_events: 该文件发作总次数
  hemisphere      : L/R/B/M/U
  onset_channels  : 逗号分隔的双极导联名
  soz_bipolar     : 同 onset_channels
  FP1_F7 … T4_A2 : 22 个 TCP 双极导联的 0/1 标签（已预计算）

支持功能：
  - 按 source 过滤（tusz / private / both）
  - 按 split 过滤（train / dev / eval / private）
  - 直接从预计算的 01 列读取标签（无需重新解析通道名）
  - 发作窗口提取：[sz_start - pre_buffer, sz_end + post_buffer]
  - 支持 TUSZ 和私有数据集不同的数据根目录
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

from config import (
    TUSZ_CONFIG,
    BIPOLAR_CHANNELS,     # 22 通道（与 eeg_pipeline.py 一致）
    BIPOLAR_CHANNELS_18,
    BRAIN_REGIONS,
    HEMISPHERES,
)
from data_loader import (
    TUSZDataLoader,
    normalize_zscore,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── TCP 22 通道名称（列名格式，与 combined_manifest.csv 一致）─────────────────
TCP_COL_NAMES = [ch.replace('-', '_') for ch in BIPOLAR_CHANNELS]
# 对应的导联名（用于 onset_channels 解析和元数据）
TCP_CHANNEL_NAMES = BIPOLAR_CHANNELS  # ['FP1-F7', 'F7-T3', ...]


# ==============================================================================
# 主数据集类
# ==============================================================================

class CombinedSOZDataset(Dataset):
    """
    统一 SOZ 定位数据集（TUSZ + 私有数据集）

    manifest 格式: combined_manifest.csv（每行一次发作）

    使用示例:
        # 只用 TUSZ train 集
        ds = CombinedSOZDataset(
            manifest_path='TUSZ/combined_manifest.csv',
            tusz_data_root='F:/dataset/TUSZ/v2.0.3/edf',
            source_filter='tusz',
            split_filter=['train'],
        )

        # TUSZ + 私有，全量
        ds = CombinedSOZDataset(
            manifest_path='TUSZ/combined_manifest.csv',
            tusz_data_root='F:/dataset/TUSZ/v2.0.3/edf',
            private_data_root='E:/DataSet/EEG/EEG dataset_SUAT',
            source_filter='both',
        )

        data, label, meta = ds[0]
        # data:  Tensor (22, n_samples)
        # label: Tensor (22,) — 22 个 TCP 双极导联的 0/1 标签
        # meta:  dict
    """

    def __init__(
        self,
        manifest_path: str,
        tusz_data_root: str = None,
        private_data_root: str = None,
        source_filter: str = 'both',        # 'tusz' / 'private' / 'both'
        split_filter: List[str] = None,     # None = 全部；e.g. ['train', 'dev']
        label_type: str = 'channel',        # 'channel'(22) / 'region' / 'hemi'
        window_len: float = 10.0,           # 窗口长度（秒）
        pre_seizure_buffer: float = 5.0,    # 发作前缓冲（秒）
        post_seizure_buffer: float = 5.0,   # 发作后缓冲（秒）
        target_fs: int = 200,
        normalize: bool = True,
        transform=None,
        cache_data: bool = False,
    ):
        self.manifest_path = manifest_path
        self.tusz_data_root = tusz_data_root or TUSZ_CONFIG['data_root']
        self.private_data_root = private_data_root  # 可为 None（若不使用私有数据）
        self.source_filter = source_filter
        self.split_filter = split_filter
        self.label_type = label_type
        self.window_len = window_len
        self.pre_seizure_buffer = pre_seizure_buffer
        self.post_seizure_buffer = post_seizure_buffer
        self.target_fs = target_fs
        self.normalize = normalize
        self.transform = transform
        self.cache_data = cache_data

        self.samples_per_window = int(window_len * target_fs)

        # 数据加载器（MNE EDF 读取 + 双极转换）
        self.loader = TUSZDataLoader(
            target_fs=target_fs,
            use_18_channels=False,  # 使用完整 22 通道
            normalize=normalize,
        )

        self._cache: Dict = {} if cache_data else None

        # 加载并过滤 manifest
        self._load_manifest()

    # ──────────────────────────────────────────────────────────────────────────
    # Manifest 加载与过滤
    # ──────────────────────────────────────────────────────────────────────────

    def _load_manifest(self):
        logger.info(f"加载 Manifest: {self.manifest_path}")
        df = pd.read_csv(self.manifest_path)
        original_count = len(df)

        # 按 source 过滤
        if self.source_filter != 'both':
            df = df[df['source'] == self.source_filter].reset_index(drop=True)

        # 按 split 过滤
        if self.split_filter:
            df = df[df['split'].isin(self.split_filter)].reset_index(drop=True)

        # 过滤掉 sz_start/sz_end 缺失的行（无法提取发作窗口）
        df = df.dropna(subset=['sz_start', 'sz_end']).reset_index(drop=True)

        # 过滤掉无效时间（sz_end <= sz_start）
        df = df[df['sz_end'] > df['sz_start']].reset_index(drop=True)

        self.manifest = df
        logger.info(
            f"Manifest 过滤: {original_count} → {len(self.manifest)} 行 "
            f"(source={self.source_filter}, split={self.split_filter})"
        )

        # 来源分布
        if len(self.manifest) > 0:
            src_dist = self.manifest['source'].value_counts().to_dict()
            logger.info(f"  来源分布: {src_dist}")

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset 接口
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        row = self.manifest.iloc[idx]

        # ── 标签 ──────────────────────────────────────────────────────────────
        label = self._get_label(row)

        # ── 元数据 ────────────────────────────────────────────────────────────
        metadata = {
            'source':         str(row.get('source', '')),
            'patient_id':     str(row.get('patient_id', '')),
            'edf_path':       str(row.get('edf_path', '')),
            'split':          str(row.get('split', '')),
            'sz_start':       float(row.get('sz_start', 0.0)),
            'sz_end':         float(row.get('sz_end', 0.0)),
            'hemisphere':     str(row.get('hemisphere', 'U')),
            'onset_channels': str(row.get('onset_channels', '')),
            'row_idx':        idx,
        }

        # ── EEG 数据 ──────────────────────────────────────────────────────────
        data = self._load_window(row, idx)

        if self.transform:
            data = self.transform(data)

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            metadata,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_edf_path(self, row: pd.Series) -> Path:
        """根据 source 字段选择对应的数据根目录"""
        source = str(row.get('source', 'tusz'))
        rel_path = str(row['edf_path'])
        if source == 'tusz':
            return Path(self.tusz_data_root) / rel_path
        else:
            if self.private_data_root is None:
                raise RuntimeError(
                    "private_data_root 未设置，无法加载私有数据集的 EDF 文件。"
                    f"请在 CombinedSOZDataset 构造函数中传入 private_data_root 参数。"
                )
            return Path(self.private_data_root) / rel_path

    def _load_window(self, row: pd.Series, idx: int) -> np.ndarray:
        """加载并返回围绕发作时间的 EEG 窗口数据"""
        cache_key = (str(row['patient_id']), str(row['edf_path']),
                     float(row['sz_start']))
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        edf_path = self._resolve_edf_path(row)
        sz_start = float(row['sz_start'])
        sz_end   = float(row['sz_end'])

        # 窗口范围：[sz_start - pre_buffer, sz_end + post_buffer]
        win_start = max(0.0, sz_start - self.pre_seizure_buffer)
        win_end   = sz_end + self.post_seizure_buffer

        n_ch = len(TCP_CHANNEL_NAMES)  # 22

        try:
            seg_data, seg_fs, _ = self.loader.load_and_preprocess(
                str(edf_path),
                start_time=win_start,
                end_time=win_end,
            )
            # seg_data: (n_channels, n_samples)
            # 截取或填充到 samples_per_window
            window = self._fit_to_window(seg_data, self.samples_per_window)
        except Exception as e:
            logger.warning(f"[{idx}] 加载失败 {edf_path}: {e} → 返回零数据")
            window = np.zeros((n_ch, self.samples_per_window), dtype=np.float32)

        if self._cache is not None:
            self._cache[cache_key] = window

        return window

    @staticmethod
    def _fit_to_window(data: np.ndarray, target_len: int) -> np.ndarray:
        """将数据截断或零填充到 target_len 采样点"""
        n_ch, n_samp = data.shape
        if n_samp >= target_len:
            return data[:, :target_len].astype(np.float32)
        padded = np.zeros((n_ch, target_len), dtype=np.float32)
        padded[:, :n_samp] = data
        return padded

    # ──────────────────────────────────────────────────────────────────────────
    # 标签
    # ──────────────────────────────────────────────────────────────────────────

    def _get_label(self, row: pd.Series) -> np.ndarray:
        if self.label_type == 'channel':
            # 直接读取预计算的 22 个 0/1 列 — 最快，无需字符串解析
            label = np.array(
                [int(row.get(col, 0)) for col in TCP_COL_NAMES],
                dtype=np.float32,
            )
            return label

        elif self.label_type == 'region':
            label = np.zeros(len(BRAIN_REGIONS), dtype=np.float32)
            from config import BIPOLAR_TO_REGION
            onset_str = str(row.get('onset_channels', ''))
            onset_chs = [c.strip() for c in onset_str.split(',') if c.strip()]
            active_regions = set()
            for ch in onset_chs:
                region = BIPOLAR_TO_REGION.get(ch.upper(), None)
                if region:
                    active_regions.add(region)
            for i, r in enumerate(BRAIN_REGIONS):
                if r in active_regions:
                    label[i] = 1.0
            return label

        elif self.label_type == 'hemi':
            label = np.zeros(len(HEMISPHERES), dtype=np.float32)
            hemi = str(row.get('hemisphere', 'U')).strip().upper()
            if hemi in HEMISPHERES:
                label[HEMISPHERES.index(hemi)] = 1.0
            else:
                label[-1] = 1.0  # Unknown
            return label

        else:
            raise ValueError(f"未知标签类型: {self.label_type}")

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def get_num_classes(self) -> int:
        if self.label_type == 'channel':
            return len(TCP_COL_NAMES)   # 22
        elif self.label_type == 'region':
            return len(BRAIN_REGIONS)
        elif self.label_type == 'hemi':
            return len(HEMISPHERES)
        return 0

    def get_channel_names(self) -> List[str]:
        """返回 22 个双极导联名称列表"""
        return TCP_CHANNEL_NAMES

    def get_patient_ids(self) -> List[str]:
        return self.manifest['patient_id'].unique().tolist()

    def get_source_mask(self, source: str) -> np.ndarray:
        """返回指定 source 的行布尔掩码（用于分来源评估）"""
        return (self.manifest['source'] == source).values

    def get_split_indices(self, split: str) -> List[int]:
        """返回指定 split 的行索引"""
        return self.manifest.index[self.manifest['split'] == split].tolist()

    def summary(self):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"CombinedSOZDataset: {self.manifest_path}")
        print(f"  总样本数: {len(self.manifest)}")
        print(f"  label_type: {self.label_type} ({self.get_num_classes()} 维)")
        print(f"  window_len: {self.window_len}s  fs={self.target_fs}Hz  "
              f"→ {self.samples_per_window} 采样点")
        print(f"\n  source 分布:")
        for src, cnt in self.manifest['source'].value_counts().items():
            print(f"    {src}: {cnt}")
        print(f"\n  split 分布:")
        for sp, cnt in self.manifest['split'].value_counts().items():
            print(f"    {sp}: {cnt}")
        if self.label_type == 'channel':
            print(f"\n  SOZ 导联频次（Top 10）:")
            col_sums = {
                ch: int(self.manifest[col].sum())
                for ch, col in zip(TCP_CHANNEL_NAMES, TCP_COL_NAMES)
                if col in self.manifest.columns
            }
            for ch, cnt in sorted(col_sums.items(), key=lambda x: -x[1])[:10]:
                print(f"    {ch:12s}: {cnt}")
        print(f"{'='*60}\n")


# ==============================================================================
# 向后兼容：保留旧类名（仅转发给 CombinedSOZDataset）
# ==============================================================================

class TUSZSOZDataset(CombinedSOZDataset):
    """
    向后兼容别名。

    旧代码使用 TUSZSOZDataset(manifest_path=..., data_root=...) 时，
    会自动映射到 CombinedSOZDataset，但 manifest 需使用 combined_manifest 格式。

    若仍需兼容旧 tusz_manifest.csv 格式，请直接使用 CombinedSOZDataset。
    """
    def __init__(self, manifest_path, data_root=None, **kwargs):
        # 旧接口映射
        super().__init__(
            manifest_path=manifest_path,
            tusz_data_root=data_root,
            source_filter=kwargs.pop('source_filter', 'tusz'),
            **{k: v for k, v in kwargs.items()
               if k not in ('use_18_channels', 'has_seizure', 'seizure_only',
                            'use_seizure_windows', 'use_baseline_windows',
                            'baseline_ratio', 'min_seizure_duration',
                            'channel_list', 'cache_data')},
        )


# ==============================================================================
# DataLoader 工厂
# ==============================================================================

def create_dataloader(
    manifest_path: str,
    tusz_data_root: str = None,
    private_data_root: str = None,
    source_filter: str = 'both',
    split_filter: List[str] = None,
    label_type: str = 'channel',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    创建 PyTorch DataLoader

    Args:
        manifest_path:     combined_manifest.csv 路径
        tusz_data_root:    TUSZ EDF 文件根目录
        private_data_root: 私有数据集 EDF 文件根目录（可 None）
        source_filter:     'tusz' / 'private' / 'both'
        split_filter:      ['train'] / ['train','dev'] / None（全部）
        label_type:        'channel' / 'region' / 'hemi'
        batch_size:        批大小
        shuffle:           是否打乱
        num_workers:       工作进程数
    """
    dataset = CombinedSOZDataset(
        manifest_path=manifest_path,
        tusz_data_root=tusz_data_root,
        private_data_root=private_data_root,
        source_filter=source_filter,
        split_filter=split_filter,
        label_type=label_type,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """自定义 collate，处理元数据字典"""
    data   = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    meta   = {}
    for key in batch[0][2].keys():
        meta[key] = [item[2][key] for item in batch]
    return data, labels, meta


# ==============================================================================
# 快速测试（仅验证 manifest 加载和标签读取，不需要实际 EDF 文件）
# ==============================================================================

if __name__ == '__main__':
    import sys

    manifest = r"E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\combined_manifest.csv"

    print("=" * 60)
    print("测试 CombinedSOZDataset（仅 manifest 统计，不加载 EDF）")
    print("=" * 60)

    df = pd.read_csv(manifest)
    print(f"共 {len(df)} 行")
    print(f"source 分布:\n{df['source'].value_counts().to_string()}")
    print(f"\nsplit 分布:\n{df['split'].value_counts().to_string()}")

    # 验证标签列
    from config import BIPOLAR_CHANNELS
    tcp_cols = [ch.replace('-', '_') for ch in BIPOLAR_CHANNELS]
    missing = [c for c in tcp_cols if c not in df.columns]
    if missing:
        print(f"\n[警告] 缺少标签列: {missing}")
    else:
        print(f"\n22 个 TCP 导联标签列均存在 [OK]")
        print(f"\n各导联 SOZ 频次:")
        for ch, col in zip(BIPOLAR_CHANNELS, tcp_cols):
            print(f"  {ch:12s}: {int(df[col].sum()):4d}")

    print("\n[提示] 实际数据加载（EDF）需要传入正确的 tusz_data_root / private_data_root。")
    print("完成！")
