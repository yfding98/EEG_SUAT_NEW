#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ_new 训练器模块

忠实复现官方两阶段训练流程：

  Stage-1（预训练，对应官方 nested_cv_pretrain / lopofn.py）
    模型  : TransformerLSTM
    任务  : 时序癫痫检测（每 1s 窗口二分类）
    损失  : CrossEntropyLoss(weight=[0.2, 0.8])
    优化  : Adam, lr=1e-5, maxiter=30
    评估  : 逐患者预测后汇总 detection accuracy

  Stage-2（SOZ 定位微调，对应官方 cvszloc_finetune / szloc_train.py）
    模型  : DeepSOZLocator (ctg_11_8)
    任务  : 通道级 SOZ 定位（19 通道多标签）
    损失  : Stage2SOZLoss（CE + MapLoss 组合）
    优化  : Adam, lr=1e-4, maxiter=50
    sz_label_idx : list(range(15)) + list(range(30,45))
    评估  : per-channel F1 / AUC

两阶段之间可选冻结 Stage-1 权重，仅训练 Stage-2 新增参数。
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_detection_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict:
    """
    Stage-1 时序癫痫检测指标

    y_true, y_pred : [N]  0/1 标量
    """
    acc  = float((y_pred == y_true).mean())
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec)


def compute_soz_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Stage-2 SOZ 定位多标签指标

    y_true : [N, C]  0/1
    y_prob : [N, C]  概率
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'f1_macro':   f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'f1_micro':   f1_score(y_true, y_pred, average='micro',  zero_division=0),
        'prec_macro': precision_score(y_true, y_pred, average='macro',  zero_division=0),
        'rec_macro':  recall_score(y_true, y_pred, average='macro',  zero_division=0),
    }

    aucs = []
    for c in range(y_true.shape[1]):
        if 0 < y_true[:, c].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
            except Exception:
                pass
    metrics['auc_macro'] = float(np.mean(aucs)) if aucs else 0.0

    if label_names is not None:
        per_label = {}
        for c, name in enumerate(label_names):
            per_label[name] = {
                'f1':      f1_score(y_true[:, c], y_pred[:, c], zero_division=0),
                'prec':    precision_score(y_true[:, c], y_pred[:, c], zero_division=0),
                'rec':     recall_score(y_true[:, c], y_pred[:, c], zero_division=0),
                'support': int(y_true[:, c].sum()),
            }
        metrics['per_label'] = per_label

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 Trainer（发作检测预训练）
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Trainer:
    """
    Stage-1 预训练器（对应官方 nested_cv_pretrain / lopofn.py）

    官方超参：
      optimizer : Adam, lr=1e-5
      epochs    : 30
      loss      : CrossEntropyLoss(weight=[0.2, 0.8])

    batch 格式（来自 Stage1Dataset / OnlineStage1Dataset）：
      batch['buffers']   : [B, Nsz, T, C, L]   EEG 窗口特征
      batch['sz_labels'] : [B, Nsz, T]          帧级 0/1 发作标签
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        n_epochs: int = 30,
        patience: int = 10,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        device: str = 'cuda',
        ckpt_dir: str = 'checkpoints',
        exp_name: str = 'stage1',
    ):
        # 设备
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            use_amp = False

        self.model = model.to(self.device)
        self.n_epochs  = n_epochs
        self.patience  = patience
        self.grad_clip = grad_clip
        self.use_amp   = use_amp
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.ckpt_dir  = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name  = exp_name

        # 损失（官方 [0.2, 0.8] 类权重）
        w = torch.tensor([0.2, 0.8], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=w)

        # 优化器（官方 Adam lr=1e-5，无 weight decay）
        self.optimizer = Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_metric = 0.0
        self.best_epoch  = 0
        self.history: List[Dict] = []

    # ── 单 epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        all_pred, all_true = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc='train' if train else 'val ', leave=False)
            for batch in pbar:
                X = batch['buffers'].to(self.device)    # [B, Nsz, T, C, L]
                Y = batch['sz_labels'].to(self.device)  # [B, Nsz, T]

                B, Nsz, T, C, L = X.shape

                with autocast(enabled=self.use_amp):
                    logits, _, _ = self.model(X)    # logits: [B, Nsz, T, 2]
                    # 展平所有帧
                    logits_flat = logits.reshape(-1, 2)        # [B*Nsz*T, 2]
                    labels_flat = Y.reshape(-1).long()          # [B*Nsz*T]
                    loss = self.criterion(logits_flat, labels_flat)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                pred = logits_flat.argmax(dim=1).detach().cpu().numpy()
                true = labels_flat.detach().cpu().numpy()
                all_pred.append(pred)
                all_true.append(true)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        metrics = compute_detection_metrics(all_true, all_pred)
        metrics['loss'] = total_loss / max(len(loader), 1)
        return metrics

    # ── 训练主循环 ─────────────────────────────────────────────────────────

    def train(self) -> List[Dict]:
        """
        执行 Stage-1 预训练，返回历史记录。
        最优模型按 val f1 保存。
        """
        logger.info(f'[Stage-1] 开始预训练: {self.exp_name}  device={self.device}')
        total, trainable = count_parameters(self.model)
        logger.info(f'[Stage-1] 参数量: {total:,}  可训练: {trainable:,}')

        no_improve = 0
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, train=True)
            va = self._run_epoch(self.val_loader,   train=False)

            record = {
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in tr.items()},
                **{f'val_{k}': v for k, v in va.items()},
                'time': time.time() - t0,
            }
            self.history.append(record)

            val_metric = va['f1']
            improved   = val_metric > self.best_metric
            if improved:
                self.best_metric = val_metric
                self.best_epoch  = epoch
                no_improve = 0
                self._save_checkpoint('best')
            else:
                no_improve += 1

            if epoch % 5 == 0:
                self._save_checkpoint(f'epoch{epoch:03d}')

            logger.info(
                f'[S1] E{epoch:03d}  '
                f'tr_loss={tr["loss"]:.4f}  tr_f1={tr["f1"]:.4f}  '
                f'va_loss={va["loss"]:.4f}  va_f1={va["f1"]:.4f}  '
                f'{"★" if improved else f"no_imp={no_improve}"}'
            )

            if no_improve >= self.patience:
                logger.info(f'[Stage-1] 早停 (patience={self.patience})')
                break

        self._save_history()
        logger.info(f'[Stage-1] 完成: best_epoch={self.best_epoch}  '
                    f'best_f1={self.best_metric:.4f}')
        return self.history

    def _save_checkpoint(self, tag: str):
        path = self.ckpt_dir / f'{self.exp_name}_{tag}.pth'
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric':          self.best_metric,
            'best_epoch':           self.best_epoch,
            'stage':                1,
        }, path)

    def load_best(self):
        path = self.ckpt_dir / f'{self.exp_name}_best.pth'
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f'[Stage-1] 加载 best checkpoint: {path}')

    def _save_history(self):
        path = self.ckpt_dir / f'{self.exp_name}_history.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)

    def get_pretrained_weights(self) -> Dict:
        """返回 model.state_dict() 供 Stage-2 初始化使用"""
        return self.model.state_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 Trainer（SOZ 定位微调）
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Trainer:
    """
    Stage-2 SOZ 定位微调训练器（对应官方 cvszloc_finetune / szloc_train.py）

    官方超参：
      optimizer  : Adam, lr=1e-4
      epochs     : 50
      loss       : Stage2SOZLoss（CE + MapLoss 组合，见 losses.py）
      sz_label_idx : list(range(15)) + list(range(30,45))

    batch 格式（来自 Stage2Dataset / OnlineStage2Dataset）：
      batch['buffers']   : [B, Nsz, T, C, L]   45 帧窗口（onset 为中心）
      batch['sz_labels'] : [B, Nsz, T]          帧级发作标签
      batch['onset_map'] : [B, 19]              通道级 SOZ 标签

    前向传播输出（DeepSOZLocator.forward）：
      channel_sz_logits [B,T,2]
      h_m               [B,T,2]
      attn_onset_map    [B,C]
      chn_onset_map     [B,C]
    """

    # 官方 sz_label_idx：发作前 15 帧 + 发作后 15 帧
    DEFAULT_SZ_LABEL_IDX = list(range(15)) + list(range(30, 45))

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,          # Stage2SOZLoss 实例
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        n_epochs: int = 50,
        patience: int = 15,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        device: str = 'cuda',
        ckpt_dir: str = 'checkpoints',
        exp_name: str = 'stage2',
        label_names: Optional[List[str]] = None,
        sz_label_idx: Optional[List[int]] = None,
        # Stage-1 checkpoint 路径（可选，加载预训练权重后微调）
        stage1_ckpt: Optional[str] = None,
    ):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            use_amp = False

        self.model     = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.n_epochs  = n_epochs
        self.patience  = patience
        self.grad_clip = grad_clip
        self.use_amp   = use_amp
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.ckpt_dir  = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name  = exp_name
        self.label_names = label_names
        self.sz_label_idx = (sz_label_idx
                             if sz_label_idx is not None
                             else self.DEFAULT_SZ_LABEL_IDX)

        # 可选：从 Stage-1 加载预训练权重（部分权重匹配）
        if stage1_ckpt is not None:
            self._load_stage1_weights(stage1_ckpt)

        # 优化器（官方 Adam lr=1e-4）
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_metric = 0.0
        self.best_epoch  = 0
        self.history: List[Dict] = []

    def _load_stage1_weights(self, ckpt_path: str):
        """
        从 Stage-1 checkpoint 加载预训练权重到 Stage-2 模型。

        Stage-1 (TransformerLSTM) 与 Stage-2 (DeepSOZLocator) 是不同架构，
        官方并不直接迁移参数，而是从头训练 Stage-2。
        此函数预留接口，用于未来研究（如用 Stage-1 特征初始化 Stage-2）。
        目前仅打印日志并跳过。
        """
        logger.info(f'[Stage-2] stage1_ckpt 参数已忽略（两阶段架构不共享权重）: '
                    f'{ckpt_path}')
        # 如果将来 Stage-2 包含 Stage-1 的子模块，可以在此处做部分加载：
        # ckpt = torch.load(ckpt_path, map_location=self.device)
        # partial_state = {k: v for k, v in ckpt['model_state_dict'].items()
        #                  if k in self.model.state_dict()}
        # self.model.load_state_dict(partial_state, strict=False)

    # ── 单 epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[Dict, Dict]:
        """
        返回 (loss_metrics, soz_metrics)

        loss_metrics : dict with 'loss' and loss components
        soz_metrics  : dict with f1_macro, auc_macro, etc.
        """
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        all_attn_probs, all_soz_labels = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc='train' if train else 'val ', leave=False)
            for batch in pbar:
                X   = batch['buffers'].to(self.device)    # [B, Nsz, T, C, L]
                Y   = batch['sz_labels'].to(self.device)  # [B, Nsz, T]
                soz = batch['onset_map'].to(self.device)  # [B, 19]

                B, Nsz, T, C, L = X.shape

                # Stage-2 官方按发作(Nsz)逐个处理，这里把 Nsz 折入 B
                # DeepSOZLocator.forward 接受 [B, Nsz, T, C, L]
                # 返回 [B,T,2], [B,T,2], [B,C], [B,C]
                with autocast(enabled=self.use_amp):
                    ch_sz_logits, tot_sz_logits, attn_map, chn_map = \
                        self.model(X)

                    # sz_labels 按发作取平均（多次发作的标签合并）
                    # 官方以 Nsz=1 批次处理；这里做均值近似
                    sz_labels_2d = Y[:, 0, :]   # [B, T] 取第��次发作

                    loss, comps = self.criterion(
                        chn_sz_logits=ch_sz_logits,
                        tot_sz_logits=tot_sz_logits,
                        attn_onset_map=attn_map,
                        chn_onset_map=chn_map,
                        sz_labels=sz_labels_2d,
                        onset_map_gt=soz,
                        sz_label_idx=self.sz_label_idx,
                        return_components=True,
                    )

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                # 累计各损失分量
                for k, v in comps.items():
                    loss_components[k] = loss_components.get(k, 0.0) + v

                # 收集 SOZ 预测（attn_onset_map 作为概率）
                import torch.nn.functional as F
                attn_prob = F.sigmoid(attn_map).detach().cpu().numpy()  # [B, C]
                all_attn_probs.append(attn_prob)
                all_soz_labels.append(soz.detach().cpu().numpy())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n_batches = max(len(loader), 1)
        loss_dict = {'loss': total_loss / n_batches}
        for k in loss_components:
            loss_dict[k] = loss_components[k] / n_batches

        all_probs  = np.concatenate(all_attn_probs,  axis=0)
        all_labels = np.concatenate(all_soz_labels,  axis=0)
        soz_metrics = compute_soz_metrics(all_labels, all_probs,
                                          label_names=self.label_names)
        return loss_dict, soz_metrics

    # ── 训练主循环 ─────────────────────────────────────────────────────────

    def train(self) -> List[Dict]:
        """
        执行 Stage-2 SOZ 定位微调，返回历史记录。
        最优模型按 val f1_macro 保存。
        """
        logger.info(f'[Stage-2] 开始微调: {self.exp_name}  device={self.device}')
        total, trainable = count_parameters(self.model)
        logger.info(f'[Stage-2] 参数量: {total:,}  可训练: {trainable:,}')
        logger.info(f'[Stage-2] sz_label_idx: {self.sz_label_idx}')

        no_improve = 0
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_soz = self._run_epoch(self.train_loader, train=True)
            va_loss, va_soz = self._run_epoch(self.val_loader,   train=False)

            record = {
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in tr_loss.items()},
                **{f'train_{k}': v for k, v in tr_soz.items()
                   if not isinstance(v, dict)},
                **{f'val_{k}': v for k, v in va_loss.items()},
                **{f'val_{k}': v for k, v in va_soz.items()
                   if not isinstance(v, dict)},
                'time': time.time() - t0,
            }
            self.history.append(record)

            val_metric = va_soz.get('f1_macro', 0.0)
            improved   = val_metric > self.best_metric
            if improved:
                self.best_metric = val_metric
                self.best_epoch  = epoch
                no_improve = 0
                self._save_checkpoint('best')
            else:
                no_improve += 1

            if epoch % 5 == 0:
                self._save_checkpoint(f'epoch{epoch:03d}')

            logger.info(
                f'[S2] E{epoch:03d}  '
                f'tr_loss={tr_loss["loss"]:.4f}  '
                f'tr_f1={tr_soz["f1_macro"]:.4f}  '
                f'va_loss={va_loss["loss"]:.4f}  '
                f'va_f1={va_soz["f1_macro"]:.4f}  '
                f'va_auc={va_soz.get("auc_macro", 0.):.4f}  '
                f'{"★" if improved else f"no_imp={no_improve}"}'
            )

            if no_improve >= self.patience:
                logger.info(f'[Stage-2] 早停 (patience={self.patience})')
                break

        self._save_history()
        logger.info(f'[Stage-2] 完成: best_epoch={self.best_epoch}  '
                    f'best_f1={self.best_metric:.4f}')
        return self.history

    def _save_checkpoint(self, tag: str):
        path = self.ckpt_dir / f'{self.exp_name}_{tag}.pth'
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric':          self.best_metric,
            'best_epoch':           self.best_epoch,
            'stage':                2,
        }, path)

    def load_best(self):
        path = self.ckpt_dir / f'{self.exp_name}_best.pth'
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f'[Stage-2] 加载 best checkpoint: {path}')

    def _save_history(self):
        path = self.ckpt_dir / f'{self.exp_name}_history.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> Dict:
        """完整推理评估（加载 best checkpoint 后调用）"""
        self.model.eval()
        all_probs, all_labels = [], []

        for batch in loader:
            X   = batch['buffers'].to(self.device)
            soz = batch['onset_map'].to(self.device)
            with autocast(enabled=self.use_amp):
                _, _, attn_map, _ = self.model(X)
            import torch.nn.functional as F
            all_probs.append(F.sigmoid(attn_map).cpu().numpy())
            all_labels.append(soz.cpu().numpy())

        all_probs  = np.concatenate(all_probs,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return compute_soz_metrics(all_labels, all_probs,
                                   threshold=threshold,
                                   label_names=self.label_names)


# ─────────────────────────────────────────────────────────────────────────────
# 两阶段 K 折交叉验证（LOPO 或标准 K-Fold）
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageKFoldRunner:
    """
    两阶段 K 折（LOPO）交叉验证辅助器

    用法：
      runner = TwoStageKFoldRunner(config)
      results = runner.run(splits, build_datasets_fn)
    """

    def __init__(self, config: Dict):
        """
        config 键：
          device, seed, output_dir, exp_prefix
          stage1_lr, stage1_epochs, stage1_patience
          stage2_lr, stage2_epochs, stage2_patience
          grad_clip, use_amp
          label_names
        """
        self.cfg = config
        set_seed(config.get('seed', 42))

    def run(
        self,
        splits: List[Tuple[List, List]],
        build_fn,   # callable(train_ids, val_ids) → (s1_train, s1_val, s2_train, s2_val)
    ) -> List[Dict]:
        """
        对每一折执行两阶段训练，返回各折验证指标。

        build_fn(train_ids, val_ids) 须返回四个 DataLoader：
          stage1_train_loader, stage1_val_loader,
          stage2_train_loader, stage2_val_loader
        """
        from deepsoz_model import build_stage1_model, build_stage2_model
        from losses import Stage2SOZLoss

        cfg = self.cfg
        output_dir  = Path(cfg.get('output_dir', 'runs'))
        exp_prefix  = cfg.get('exp_prefix', 'deepsoz')
        device      = cfg.get('device', 'cpu')
        label_names = cfg.get('label_names', None)

        all_results = []

        for fold, (train_ids, val_ids) in enumerate(splits):
            logger.info(f'\n{"="*60}\nFold {fold}  '
                        f'train={len(train_ids)}  val={len(val_ids)}\n{"="*60}')
            set_seed(cfg.get('seed', 42) + fold)

            ckpt_dir = output_dir / f'fold{fold}'

            # ── 构建 DataLoader ──────────────────────────────────────────
            s1_tr, s1_va, s2_tr, s2_va = build_fn(train_ids, val_ids)

            # ── Stage-1 ──────────────────────────────────────────────────
            n_channels = cfg.get('n_channels', 19)
            s1_model = build_stage1_model(
                n_channels=n_channels,
                transformer_dropout=cfg.get('tf_dropout', 0.15),
                device=device
            )
            s1_trainer = Stage1Trainer(
                model=s1_model,
                train_loader=s1_tr,
                val_loader=s1_va,
                lr=cfg.get('stage1_lr', 1e-5),
                n_epochs=cfg.get('stage1_epochs', 30),
                patience=cfg.get('stage1_patience', 10),
                grad_clip=cfg.get('grad_clip', 1.0),
                use_amp=cfg.get('use_amp', True),
                device=device,
                ckpt_dir=str(ckpt_dir),
                exp_name=f'{exp_prefix}_fold{fold}_s1',
            )
            s1_trainer.train()

            # ── Stage-2 ──────────────────────────────────────────────────
            s2_model = build_stage2_model(
                n_channels=n_channels,
                cnn_dropout=cfg.get('cnn_dropout', 0.15),
                gru_dropout=cfg.get('gru_dropout', 0.0),
                transformer_dropout=cfg.get('tf_dropout', 0.15),
            )
            s2_criterion = Stage2SOZLoss()
            s2_trainer = Stage2Trainer(
                model=s2_model,
                criterion=s2_criterion,
                train_loader=s2_tr,
                val_loader=s2_va,
                lr=cfg.get('stage2_lr', 1e-4),
                n_epochs=cfg.get('stage2_epochs', 50),
                patience=cfg.get('stage2_patience', 15),
                grad_clip=cfg.get('grad_clip', 1.0),
                use_amp=cfg.get('use_amp', True),
                device=device,
                ckpt_dir=str(ckpt_dir),
                exp_name=f'{exp_prefix}_fold{fold}_s2',
                label_names=label_names,
            )
            s2_trainer.train()

            # ── 最终评估 ─────────────────────────────────────────────────
            s2_trainer.load_best()
            val_metrics = s2_trainer.evaluate(s2_va)
            logger.info(f'Fold {fold} 最终评估: '
                        f'f1_macro={val_metrics["f1_macro"]:.4f}  '
                        f'auc_macro={val_metrics.get("auc_macro", 0.):.4f}')

            all_results.append({'fold': fold, **val_metrics})

        # ── 汇总 ─────────────────────────────────────────────────────────
        f1s  = [r['f1_macro']              for r in all_results]
        aucs = [r.get('auc_macro', 0.)     for r in all_results]
        logger.info(
            f'\n{"="*60}\n'
            f'K-Fold 汇总  F1(mean±std): '
            f'{np.mean(f1s):.4f}±{np.std(f1s):.4f}  '
            f'AUC(mean): {np.mean(aucs):.4f}\n'
            f'{"="*60}'
        )

        # 保存汇总
        summary_path = output_dir / f'{exp_prefix}_kfold_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'folds': all_results,
                'f1_mean': float(np.mean(f1s)),
                'f1_std':  float(np.std(f1s)),
                'auc_mean': float(np.mean(aucs)),
            }, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f'汇总保存: {summary_path}')

        return all_results
