#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练工具模块

包含训练循环、评估、日志记录等通用功能
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 评估指标
# ==============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    task_type: str = 'multilabel',
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签（二值化后）
        y_prob: 预测概率
        task_type: 'multilabel', 'multiclass', 'binary'
        threshold: 二值化阈值
    
    Returns:
        指标字典
    """
    metrics = {}
    
    if task_type == 'multilabel':
        # 多标签分类
        if y_prob is not None:
            y_pred = (y_prob > threshold).astype(int)
        
        metrics['accuracy'] = accuracy_score(y_true.flatten(), y_pred.flatten())
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if y_prob is not None:
            try:
                # 逐标签计算AUC
                auc_list = []
                for i in range(y_true.shape[1]):
                    if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
                        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                        auc_list.append(auc)
                metrics['auc_mean'] = np.mean(auc_list) if auc_list else 0.0
            except Exception:
                metrics['auc_mean'] = 0.0
    
    elif task_type == 'multiclass':
        # 多分类
        if y_prob is not None:
            y_pred = np.argmax(y_prob, axis=1)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception:
                metrics['auc_ovr'] = 0.0
    
    else:
        # 二分类
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
                metrics['ap'] = average_precision_score(y_true, y_prob)
            except Exception:
                metrics['auc'] = 0.0
                metrics['ap'] = 0.0
    
    return metrics


def compute_channel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    channel_names: List[str] = None,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    计算每个通道的评估指标
    
    Args:
        y_true: (N, 19) 真实标签
        y_pred: (N, 19) 预测标签
        y_prob: (N, 19) 预测概率
        channel_names: 通道名称列表
        threshold: 二值化阈值
    
    Returns:
        每个通道的指标
    """
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(y_true.shape[1])]
    
    if y_prob is not None:
        y_pred = (y_prob > threshold).astype(int)
    
    channel_metrics = {}
    for i, ch_name in enumerate(channel_names):
        true_ch = y_true[:, i]
        pred_ch = y_pred[:, i]
        
        ch_metrics = {
            'accuracy': accuracy_score(true_ch, pred_ch),
            'precision': precision_score(true_ch, pred_ch, zero_division=0),
            'recall': recall_score(true_ch, pred_ch, zero_division=0),
            'f1': f1_score(true_ch, pred_ch, zero_division=0),
            'n_positive': true_ch.sum(),
            'n_predicted': pred_ch.sum()
        }
        
        if y_prob is not None and true_ch.sum() > 0 and true_ch.sum() < len(true_ch):
            try:
                ch_metrics['auc'] = roc_auc_score(true_ch, y_prob[:, i])
            except Exception:
                ch_metrics['auc'] = 0.0
        
        channel_metrics[ch_name] = ch_metrics
    
    return channel_metrics


# ==============================================================================
# 训练器
# ==============================================================================

class Trainer:
    """
    通用训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler=None,
        device: str = 'cuda',
        task_type: str = 'multilabel',
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        experiment_name: str = 'experiment'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task_type = task_type
        self.use_amp = use_amp and device == 'cuda'
        self.grad_clip_norm = grad_clip_norm
        
        # 目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            data = batch['data'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    if isinstance(outputs, dict):
                        outputs = outputs.get(self.task_type.split('_')[0], outputs.get('channel'))
                    loss = self.criterion(outputs, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    outputs = outputs.get(self.task_type.split('_')[0], outputs.get('channel'))
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测
            if self.task_type == 'multiclass':
                probs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
            else:
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs, self.task_type)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        
        for batch in pbar:
            data = batch['data'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(data)
            if isinstance(outputs, dict):
                outputs = outputs.get(self.task_type.split('_')[0], outputs.get('channel'))
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            if self.task_type == 'multiclass':
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
            else:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs, self.task_type)
        
        return avg_loss, metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 20,
        save_every: int = 10,
        metric_for_best: str = 'f1_macro'
    ):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_every: 每N轮保存一次
            metric_for_best: 用于选择最佳模型的指标
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss, train_metrics = self.train_epoch()
            
            # 验证
            val_loss, val_metrics = self.validate()
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            
            # 日志
            logger.info(
                f'Epoch {self.current_epoch}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Train {metric_for_best}: {train_metrics.get(metric_for_best, 0):.4f}, '
                f'Val {metric_for_best}: {val_metrics.get(metric_for_best, 0):.4f}, '
                f'LR: {current_lr:.6f}'
            )
            
            # 检查是否是最佳模型
            current_metric = val_metrics.get(metric_for_best, val_metrics.get('f1_macro', 0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint('best')
                patience_counter = 0
                logger.info(f'New best model! {metric_for_best}: {self.best_metric:.4f}')
            else:
                patience_counter += 1
            
            # 定期保存
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}')
            
            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping at epoch {self.current_epoch}')
                break
        
        # 保存最终模型和历史
        self.save_checkpoint('final')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / f'{self.experiment_name}_{name}.pth'
        torch.save(checkpoint, path)
        logger.info(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f'Checkpoint loaded: {path}')
    
    def save_history(self):
        """保存训练历史"""
        path = self.log_dir / f'{self.experiment_name}_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2, default=float)
        logger.info(f'History saved: {path}')


# ==============================================================================
# 工具函数
# ==============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4
) -> optim.Optimizer:
    """创建优化器"""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f'未知优化器: {optimizer_name}')


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6
):
    """创建学习率调度器"""
    if scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)
    elif scheduler_name.lower() == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        return None


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """获取可用设备"""
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试训练工具...")
    
    # 测试指标计算
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_prob = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.1], [0.6, 0.7, 0.3]])
    
    print("\n多标签分类指标:")
    metrics = compute_metrics(y_true, None, y_prob, 'multilabel')
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试多分类指标
    y_true_mc = np.array([0, 1, 2, 0])
    y_prob_mc = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1]
    ])
    
    print("\n多分类指标:")
    metrics = compute_metrics(y_true_mc, None, y_prob_mc, 'multiclass')
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n测试通过!")
