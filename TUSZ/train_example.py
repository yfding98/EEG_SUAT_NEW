#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined SOZ Training Example
────────────────────────────────────────────────────────────────────
基于 combined_manifest.csv 的 SOZ 定位训练示例脚本。

支持:
  - 单独训练 TUSZ 数据（--source tusz）
  - 单独训练私有数据（--source private）
  - 混合训练（--source both，默认）
  - 使用 TUSZ 官方 split（train/dev/eval）
  - 22 通道 TCP 双极导联标签

用法:
    # 混合训练（TUSZ train + private）
    python train_example.py \\
        --manifest TUSZ/combined_manifest.csv \\
        --tusz-data-root F:/dataset/TUSZ/v2.0.3/edf \\
        --private-data-root E:/DataSet/EEG/EEG_dataset_SUAT \\
        --source both --epochs 20

    # 只用 TUSZ train 集
    python train_example.py \\
        --manifest TUSZ/combined_manifest.csv \\
        --tusz-data-root F:/dataset/TUSZ/v2.0.3/edf \\
        --source tusz --split train --epochs 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from dataset import CombinedSOZDataset, collate_fn, TCP_CHANNEL_NAMES
from config import TUSZ_CONFIG, BIPOLAR_CHANNELS, BRAIN_REGIONS, HEMISPHERES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 简单 SOZ 分类模型（演示用）
# ==============================================================================

class SimpleSOZClassifier(nn.Module):
    """
    简单的多标签 SOZ 分类模型

    输入: (batch, n_channels=22, n_samples)
    输出: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 2000,
        n_classes: int = 22,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 时间卷积块
        self.conv1 = nn.Conv1d(n_channels, 64,  kernel_size=25, stride=5,  padding=10)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64,  128, kernel_size=15, stride=3,  padding=5)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=7,  stride=2,  padding=3)
        self.bn3   = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(8)

        self.fc1     = nn.Linear(256 * 8, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(hidden_dim, n_classes)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 22, T)
        x = self.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))


# ==============================================================================
# 训练 / 评估函数
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device, source_filter='both'):
    model.train()
    total_loss, n_batches = 0.0, 0
    for data, labels, meta in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels, meta in loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            total_loss += criterion(out, labels).item()
            all_preds.append(out.cpu())
            all_labels.append(labels.cpu())
            n_batches += 1

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)
    pb     = (preds > 0.5).float()

    tp = ((pb == 1) & (labels == 1)).sum(dim=1).float()
    fp = ((pb == 1) & (labels == 0)).sum(dim=1).float()
    fn = ((pb == 0) & (labels == 1)).sum(dim=1).float()

    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall    = (tp / (tp + fn + 1e-8)).mean().item()
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'loss':      total_loss / max(n_batches, 1),
        'accuracy':  (pb == labels).float().mean().item(),
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
    }


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined SOZ Training Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--manifest', type=str,
        default=r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\combined_manifest.csv',
        help='combined_manifest.csv 路径',
    )
    parser.add_argument(
        '--tusz-data-root', type=str,
        default=TUSZ_CONFIG['data_root'],
        help='TUSZ EDF 文件根目录',
    )
    parser.add_argument(
        '--private-data-root', type=str, default=None,
        help='私有数据集 EDF 文件根目录（仅在 source=private/both 时需要）',
    )
    parser.add_argument(
        '--source', type=str, default='both',
        choices=['tusz', 'private', 'both'],
        help='使用的数据来源',
    )
    parser.add_argument(
        '--split', type=str, nargs='+', default=None,
        help='使用的 split（可多选），例如 --split train dev。'
             'None 表示使用全部（tusz:train+dev+eval，private:private）。'
             '混合训练建议: --split train private',
    )

    # ── 标签 ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--label-type', type=str, default='channel',
        choices=['channel', 'region', 'hemi'],
        help='标签粒度',
    )

    # ── 窗口 ──────────────────────────────────────────────────────────────────
    parser.add_argument('--window-len',  type=float, default=10.0, help='窗口长度（秒）')
    parser.add_argument('--pre-buffer',  type=float, default=5.0,  help='发作前缓冲（秒）')
    parser.add_argument('--post-buffer', type=float, default=5.0,  help='发作后缓冲（秒）')

    # ── 训练 ──────────────────────────────────────────────────────────────────
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--num-workers',type=int,   default=0)

    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    logger.info(f"设备: {device}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    # 训练集：指定 split 或默认按 source 取 train/private
    train_splits = args.split
    if train_splits is None:
        if args.source == 'tusz':
            train_splits = ['train']
        elif args.source == 'private':
            train_splits = ['private']
        else:
            train_splits = ['train', 'private']  # 混合

    train_ds = CombinedSOZDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        private_data_root=args.private_data_root,
        source_filter=args.source,
        split_filter=train_splits,
        label_type=args.label_type,
        window_len=args.window_len,
        pre_seizure_buffer=args.pre_buffer,
        post_seizure_buffer=args.post_buffer,
    )
    train_ds.summary()

    # 验证集：TUSZ dev（若 source 包含 tusz）
    val_ds = None
    if args.source in ('tusz', 'both'):
        val_splits = ['dev']
        val_ds = CombinedSOZDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            private_data_root=args.private_data_root,
            source_filter='tusz',
            split_filter=val_splits,
            label_type=args.label_type,
            window_len=args.window_len,
            pre_seizure_buffer=args.pre_buffer,
            post_seizure_buffer=args.post_buffer,
        )
        logger.info(f"验证集（TUSZ dev）: {len(val_ds)} 样本")

    if len(train_ds) == 0:
        logger.error("训练集为空！请检查 manifest 路径与 source/split 参数。")
        return

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == 'cuda'),
        )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    n_classes = train_ds.get_num_classes()
    n_samples = int(args.window_len * 200)  # 200Hz
    n_channels = 22  # 始终使用 22 通道 TCP

    logger.info(f"模型: n_channels={n_channels}, n_samples={n_samples}, n_classes={n_classes}")
    model = SimpleSOZClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    logger.info(f"开始训练: {args.epochs} epochs")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        log_msg = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f}"

        if val_loader is not None:
            val_m = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_m['loss'])
            log_msg += (
                f" | Val Loss: {val_m['loss']:.4f}"
                f" | Val F1: {val_m['f1']:.4f}"
                f" | Val Recall: {val_m['recall']:.4f}"
            )
            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                # torch.save(model.state_dict(), 'best_model.pt')

        logger.info(log_msg)

    logger.info(f"训练完成！最佳验证 F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
