#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_soz_locator_with_brain_networks.py

端到端 SOZ 定位训练脚本，集成脑网络特征。

流程:
  1. 数据加载 (ManifestSOZDataset)
  2. (可选) 对比学习预训练
  3. 三阶段微调 (冻结骨干 -> 解冻TimeFilter -> 全模型)
  4. 测试评估 + 可解释性报告

支持: DDP / AMP / 断点续训 / TensorBoard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# ── project path ──
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── imports ──
try:
    from models.integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from models.contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from models.brain_network_extractor import MultiScaleBrainNetworkExtractor
    from models.dynamic_network_evolution import DynamicNetworkEvolutionModel
    from models.manifest_dataset import ManifestSOZDataset
except ImportError:
    from .integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from .contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from .brain_network_extractor import MultiScaleBrainNetworkExtractor
    from .dynamic_network_evolution import DynamicNetworkEvolutionModel
    from .manifest_dataset import ManifestSOZDataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

log = logging.getLogger('train_bn')


# =====================================================================
# Helpers
# =====================================================================

def setup_logging(output_dir: Path, rank: int = 0):
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / 'train.log'))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def setup_ddp():
    """Initialise DDP if launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


# =====================================================================
# Metrics
# =====================================================================

def compute_top_k(probs: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
    """For each sample, check if any of the top-k predicted channels is an actual SOZ."""
    correct = 0
    for p, t in zip(probs, targets):
        top_k_idx = p.argsort()[-k:]
        soz_idx = np.where(t > 0.5)[0]
        if len(soz_idx) > 0 and len(set(top_k_idx) & set(soz_idx)) > 0:
            correct += 1
    return correct / len(probs) if len(probs) > 0 else 0.0


def compute_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    try:
        valid = targets.sum(axis=0) > 0
        if valid.sum() < 2:
            return 0.0
        return roc_auc_score(targets[:, valid], probs[:, valid], average='macro')
    except Exception:
        return 0.0


# =====================================================================
# Dataset wrapper (adds onset / window metadata for patching)
# =====================================================================

class SOZBrainNetworkDataset(torch.utils.data.Dataset):
    """
    Wraps ManifestSOZDataset and provides seizure metadata
    needed by SeizureAlignedAdaptivePatching.
    """

    def __init__(self, manifest_ds: ManifestSOZDataset, precomputed_dir: str = None):
        self.ds = manifest_ds
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None

    def __len__(self):
        return len(self.ds)

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if not self.precomputed_dir:
            return None
        row = self.ds.df.iloc[idx]
        edf_rel = Path(str(row.get('edf_path', '')))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        # Keep directory structure: precomputed_dir / dir_of_edf / filename_start_sec.npz
        # Using .with_suffix('') to remove the .edf extension before appending
        rel_path = edf_rel.parent / f"{edf_rel.stem}_w{start_sec:.1f}.npz"
        
        cache_file = self.precomputed_dir / rel_path
        # Ensure the subdirectories exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        return cache_file

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # x = sample['data']                         # [22, 20, 100]
        # label = sample['label']                    # [22] or [19]

        x, label, mask, meta, y_bipolar, y_monopolar = sample

        # flatten to [22, 2000] for patching module
        C, P, L = x.shape
        x_flat = x.reshape(C, P * L)

        # extract onset / start from manifest row
        row = self.ds.df.iloc[idx]
        onset_sec = float(row.get('onset_sec', 5.0))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        ret = {
            'idx': idx,
            'x': x_flat,
            'label': label,
            'bipolar_label': y_bipolar,
            'monopolar_label': y_monopolar,
            'onset_sec': onset_sec,
            'start_sec': start_sec,
            'source': row.get('source', 'unknown'),
            'patient_id': row.get('patient_id', 'unknown'),
            'edf_path': row.get('edf_path', ''),
        }
        
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                data = np.load(str(cache_path))
                ret['brain_nets'] = torch.from_numpy(data['brain_nets'])
                ret['valid_patch_counts'] = torch.tensor(data['valid_patch_counts'])
                ret['rel_time'] = torch.from_numpy(data['rel_time'])
            except Exception as e:
                pass # Fallback to online computation if loading fails
                
        return ret


def collate_fn(batch):
    ret = {
        'idx': [b['idx'] for b in batch],
        'x': torch.stack([b['x'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'bipolar_label': torch.stack([b['bipolar_label'] for b in batch]),
        'monopolar_label': torch.stack([b['monopolar_label'] for b in batch]),
        'onset_sec': torch.tensor([b['onset_sec'] for b in batch]),
        'start_sec': torch.tensor([b['start_sec'] for b in batch]),
        'source': [b['source'] for b in batch],
        'patient_id': [b['patient_id'] for b in batch],
        'edf_path': [b['edf_path'] for b in batch],
    }
    
    if all('brain_nets' in b for b in batch):
        ret['brain_nets'] = torch.stack([b['brain_nets'] for b in batch])
        ret['valid_patch_counts'] = torch.stack([b['valid_patch_counts'] for b in batch])
        ret['rel_time'] = torch.stack([b['rel_time'] for b in batch])
        
    return ret


# =====================================================================
# Training loops
# =====================================================================

def train_one_epoch(
    model, loader, optimizer, scaler, device, epoch, cfg, writer=None,
):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_probs, all_targets = [], []

    for step, batch in enumerate(loader):
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        
        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(
                x, onset, start,
                valid_patch_counts=vp_counts,
                brain_networks=brain_nets,
                rel_time=rel_time,
            )

            # build aux targets
            vm = model.module._build_valid_mask(
                outputs['valid_patch_counts'],
                outputs['transition_probs'].size(1),
            ) if hasattr(model, 'module') else \
                type(model)._build_valid_mask(
                    None,
                    outputs['valid_patch_counts'],
                    outputs['transition_probs'].size(1),
                )

            # use model's internal method
            base = model.module if hasattr(model, 'module') else model
            vm = DynamicNetworkEvolutionModel._build_valid_mask(
                outputs['valid_patch_counts'],
                outputs['transition_probs'].size(1),
            )
            aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
                outputs['seizure_relative_time'], vm,
            )

            loss, losses = base.compute_loss(
                outputs, label,
                transition_targets=aux['transition_targets'].to(device),
                pattern_targets=aux['pattern_targets'].to(device),
            )

        # NaN check
        if torch.isnan(loss):
            log.warning(f"NaN loss at epoch {epoch} step {step}, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(outputs['soz_probs'].detach().cpu().numpy())
        all_targets.append(label.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    top3 = compute_top_k(probs, targets, k=3)
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/top3', top3, epoch)
        writer.add_scalar('train/auc', auc, epoch)

    return avg_loss, top3, auc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        
        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        out = model(
            x, onset, start,
            valid_patch_counts=vp_counts,
            brain_networks=brain_nets,
            rel_time=rel_time,
        )
        all_probs.append(out['soz_probs'].cpu().numpy())
        all_targets.append(label.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    top1 = compute_top_k(probs, targets, k=1)
    top3 = compute_top_k(probs, targets, k=3)
    top5 = compute_top_k(probs, targets, k=5)
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0
    return {'top1': top1, 'top3': top3, 'top5': top5, 'auc': auc,
            'probs': probs, 'targets': targets}


# =====================================================================
# Contrastive pretraining
# =====================================================================

def run_contrastive_pretraining(
    model_pretrain, train_loader, device, args, writer=None,
):
    log.info("=== Contrastive pretraining ===")
    net_ext = MultiScaleBrainNetworkExtractor(
        n_channels=22, patch_len=100, fs=args.fs,
    ).to(device)
    optimizer = torch.optim.AdamW(model_pretrain.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs,
    )

    for epoch in range(args.pretrain_epochs):
        model_pretrain.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            # compute brain networks from raw patches
            B, C, T = x.shape
            P = T // 100
            patches = x.reshape(B, C, P, 100).permute(0, 2, 1, 3)  # [B,P,22,100]
            with torch.no_grad():
                nets = net_ext(patches)['all']                        # [B,P,22,22,4]

            # for contrastive: use same data with augmentation as pos,
            # circularly shifted as neg
            neg = nets.roll(1, dims=0)
            out = model_pretrain(nets, neg)
            loss = out['contrastive_loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_pretrain.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(train_loader), 1)
        if writer:
            writer.add_scalar('pretrain/loss', avg, epoch)
        if epoch % 10 == 0:
            log.info(f"  Pretrain epoch {epoch}/{args.pretrain_epochs}  loss={avg:.4f}")

    save_path = Path(args.output_dir) / 'pretrained_encoder.pt'
    model_pretrain.save_pretrained_encoder(str(save_path))
    log.info(f"Pretrained encoder saved to {save_path}")
    return save_path


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description='SOZ Locator with Brain Networks')

    # data
    p.add_argument('--manifest', required=True, help='combined_manifest.csv')
    p.add_argument('--private-data-root', default='', help='preprocessed data root')
    p.add_argument('--tusz-data-root', default='', help='TUSZ EDF root')
    p.add_argument('--source', default='all', choices=['tusz', 'private', 'all'])

    # model
    p.add_argument('--labram-ckpt', default='', help='LaBraM pretrained weights')
    p.add_argument('--patch-duration', type=float, default=0.5)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--output-mode', default='monopolar', choices=['monopolar', 'bipolar'])

    # brain networks
    p.add_argument('--brain-network-features', default='gc,te,aec,wpli')
    p.add_argument('--use-contrastive', action='store_true')
    p.add_argument('--pretrain-epochs', type=int, default=50)

    # training
    p.add_argument('--finetune-epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--amp', action='store_true', help='mixed precision')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)

    # output
    p.add_argument('--output-dir', default='output/train_bn')
    p.add_argument('--precomputed-dir', default=None, help='Directory with precomputed brain networks')
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    p.add_argument('--save-every', type=int, default=10)

    # validation
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)

    return p.parse_args()


def main():
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Save config ──
    if is_main(rank):
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        log.info(f"Config saved to {output_dir / 'config.json'}")

    # ── 1. Data loading ──
    log.info("=== Step 1: Loading data ===")
    manifest_ds = ManifestSOZDataset(
        manifest_path=args.manifest,
        private_data_root=args.private_data_root,
        tusz_data_root=args.tusz_data_root,
        source_filter=args.source,
    )
    log.info(f"  Manifest loaded: {len(manifest_ds)} samples")

    dataset = SOZBrainNetworkDataset(manifest_ds, precomputed_dir=args.precomputed_dir)

    # split (simple random; for LOPO use external loop)
    n = len(dataset)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    # check sample size
    if n_train < 50:
        log.warning(f"  Small dataset ({n_train} samples) — strong augmentation recommended")

    # loaders
    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
    else:
        train_sampler = RandomSampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    # ── 2. ModelInit ──
    log.info("=== Step 2: Initializing model ===")
    patch_len = int(args.patch_duration * args.fs)
    cfg = IntegrationConfig(
        embed_dim=args.embed_dim,
        patch_len=patch_len,
        fs=args.fs,
        labram_checkpoint=args.labram_ckpt,
        output_mode=args.output_mode,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device)
    log.info(model.summary())

    # ── 3. Contrastive pretraining (optional) ──
    pretrain_encoder_path = None
    if args.use_contrastive:
        log.info("=== Step 3: Contrastive pretraining ===")
        pt_cfg = PretrainConfig(embed_dim=cfg.embed_dim)
        pretrain_model = BrainNetworkContrastivePretrainer(pt_cfg).to(device)
        writer_pt = SummaryWriter(str(output_dir / 'tb_pretrain')) if (_HAS_TB and is_main(rank)) else None
        pretrain_encoder_path = run_contrastive_pretraining(
            pretrain_model, train_loader, device, args, writer_pt,
        )
        if writer_pt:
            writer_pt.close()

    # ── 4. Fine-tuning ──
    log.info("=== Step 4: Fine-tuning ===")
    writer = SummaryWriter(str(output_dir / 'tb')) if (_HAS_TB and is_main(rank)) else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # DDP
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # resume
    start_epoch = 0
    best_top3 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_top3 = ckpt.get('best_top3', 0.0)
        log.info(f"  Resumed from epoch {start_epoch}, best_top3={best_top3:.4f}")

    total_epochs = args.finetune_epochs
    phase1_end = total_epochs // 5       # 20% frozen backbone
    phase2_end = total_epochs * 3 // 5   # next 40% unfreeze timefilter

    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.lr), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2,
    )

    for epoch in range(start_epoch, total_epochs):
        # phase transitions
        if epoch == 0:
            base_model.freeze_backbone()
            log.info("  Phase 1: backbone frozen")
        elif epoch == phase1_end:
            base_model.unfreeze_timefilter()
            optimizer = torch.optim.AdamW(
                base_model.get_param_groups(args.lr * 0.5), weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2,
            )
            log.info("  Phase 2: TimeFilter + network unfrozen")
        elif epoch == phase2_end:
            base_model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                base_model.get_param_groups(args.lr * 0.1), weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2,
            )
            log.info("  Phase 3: full model unfrozen")

        if world > 1:
            train_sampler.set_epoch(epoch)

        # train
        t0 = time.time()
        loss, top3, auc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg, writer,
        )
        scheduler.step()
        dt = time.time() - t0

        # validate
        val_metrics = evaluate(model, val_loader, device)

        if is_main(rank):
            log.info(
                f"Epoch {epoch:3d}/{total_epochs} "
                f"loss={loss:.4f} "
                f"train_top3={top3:.3f} "
                f"val_top3={val_metrics['top3']:.3f} "
                f"val_auc={val_metrics['auc']:.3f} "
                f"({dt:.1f}s)"
            )
            if writer:
                writer.add_scalar('val/top3', val_metrics['top3'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

            # save best
            if val_metrics['top3'] > best_top3:
                best_top3 = val_metrics['top3']
                base_model.save_checkpoint(
                    str(output_dir / 'best_model.pt'),
                    extra={'epoch': epoch, 'best_top3': best_top3,
                           'val_metrics': val_metrics},
                )
                log.info(f"  ** New best: top3={best_top3:.4f}")

            # periodic save
            if (epoch + 1) % args.save_every == 0:
                base_model.save_checkpoint(
                    str(output_dir / f'ckpt_epoch{epoch:03d}.pt'),
                    extra={'epoch': epoch, 'best_top3': best_top3},
                )

    # ── 5. Test evaluation ──
    log.info("=== Step 5: Test evaluation ===")
    # load best
    best_path = output_dir / 'best_model.pt'
    if best_path.exists():
        ckpt_best = torch.load(str(best_path), map_location=device)
        base_model.load_state_dict(ckpt_best['model_state'])

    test_metrics = evaluate(model, test_loader, device)

    if is_main(rank):
        log.info(
            f"\nTest results:\n"
            f"  Top-1: {test_metrics['top1']:.4f}\n"
            f"  Top-3: {test_metrics['top3']:.4f}\n"
            f"  Top-5: {test_metrics['top5']:.4f}\n"
            f"  AUC:   {test_metrics['auc']:.4f}"
        )

        # save test report (markdown)
        report = (
            f"# SOZ Localization Report\n\n"
            f"## Configuration\n"
            f"- Manifest: `{args.manifest}`\n"
            f"- LaBraM checkpoint: `{args.labram_ckpt}`\n"
            f"- Contrastive pretraining: {args.use_contrastive}\n"
            f"- Finetune epochs: {total_epochs}\n"
            f"- Output mode: {args.output_mode}\n\n"
            f"## Test Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Top-1 | {test_metrics['top1']:.4f} |\n"
            f"| Top-3 | {test_metrics['top3']:.4f} |\n"
            f"| Top-5 | {test_metrics['top5']:.4f} |\n"
            f"| AUC   | {test_metrics['auc']:.4f} |\n\n"
            f"## Best validation Top-3: {best_top3:.4f}\n"
        )
        (output_dir / 'report.md').write_text(report, encoding='utf-8')
        log.info(f"Report saved to {output_dir / 'report.md'}")

        # save test predictions
        np.savez(
            str(output_dir / 'test_predictions.npz'),
            probs=test_metrics['probs'],
            targets=test_metrics['targets'],
        )

    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()

    log.info("Done.")
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    sys.exit(main())
