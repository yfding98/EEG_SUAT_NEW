#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone DeepSOZ-style seizure detection evaluation.

Loads a stage-pretrained checkpoint and evaluates with DeepSOZ-style metrics:
  Window-level:  AU-ROC, Sensitivity, Specificity
  Seizure-level: FPR/hr, Sensitivity, Latency (with moving average smoothing)

Usage:
  python evaluate_stage_deepsoz.py \
      --checkpoint output/best_pretrain_ckpt.pth \
      --manifest TUSZ/tusz_manifest.csv \
      --tusz-data-root F:/dataset/TUSZ/v2.0.3/edf \
      --output-dir output/deepsoz_eval
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─── Project path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # EEG_SUAT_NEW
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ'))
sys.path.insert(0, str(PROJECT_ROOT / 'TUSZ' / 'models'))

from models.integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
)
from data_preprocess.eeg_pipeline import PipelineConfig
from tasks.stage_detection import (
    EEGStagePretrainDataset,
    stage_collate_fn,
    summarize_stage_dataset,
)
from tasks.stage_seizure_metrics import run_detailed_evaluation
from models.train_soz_locator_with_brain_networks import evaluate_stage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)


def build_model(args) -> tuple:
    """Load model from checkpoint."""
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' in ckpt:
        cfg = ckpt['config']
        if isinstance(cfg, dict):
            cfg = IntegrationConfig(**cfg)
        log.info('Using config from checkpoint')
    else:
        log.warning('Checkpoint has no config, building from CLI args')
        cfg = IntegrationConfig(
            task_mode='stage_pretrain',
            embed_dim=args.embed_dim,
            patch_len=args.patch_len,
            n_pre_patches=int(np.ceil(args.pre_onset_sec / args.patch_duration)),
            n_post_patches=int(np.ceil(args.post_onset_sec / args.patch_duration)),
            fs=args.fs,
            labram_checkpoint='',
            output_mode=args.output_mode,
        )

    cfg.n_frozen_layers = 0
    cfg.labram_checkpoint = ''
    if hasattr(cfg, 'use_checkpoint'):
        cfg.use_checkpoint = False

    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg)

    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    if not isinstance(state, dict):
        raise KeyError("Checkpoint does not contain a valid model state dict")

    own_state = model.state_dict()
    filtered_state = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            filtered_state[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    log.info('Loaded checkpoint: %s', args.checkpoint)
    log.info('  loaded=%d, missing=%d, unexpected=%d',
             len(filtered_state), len(missing), len(unexpected))

    # Restore stage pretraining configuration
    model.configure_stage_pretraining(train_backbone=False)

    return model, cfg


def main():
    p = argparse.ArgumentParser(description='DeepSOZ-style stage detection evaluation')
    p.add_argument('--checkpoint', required=True, help='Stage pretrain checkpoint (.pth)')
    p.add_argument('--manifest', required=True, help='Path to manifest CSV')
    p.add_argument('--tusz-data-root', default='', help='TUSZ EDF root directory')
    p.add_argument('--split', nargs='+', default=['dev'],
                   help='Split(s) to evaluate (default: dev)')
    p.add_argument('--source', default='tusz', help='Source filter (default: tusz)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output-dir', default='', help='Directory to save results')
    p.add_argument('--smoother-kernel', type=int, default=31,
                   help='Moving average kernel size (default: 31)')
    p.add_argument('--threshold', type=float, default=None,
                   help='Fixed threshold (if not set, auto-search is used)')
    p.add_argument('--max-fpr-per-hour', type=float, default=120.0,
                   help='FPR constraint for threshold search (default: 120)')

    # Model architecture fallbacks (used only if checkpoint has no config)
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--patch-len', type=int, default=200)
    p.add_argument('--patch-duration', type=float, default=1.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--pre-onset-sec', type=float, default=8.0)
    p.add_argument('--post-onset-sec', type=float, default=4.0)
    p.add_argument('--output-mode', default='monopolar')
    p.add_argument('--sample-roles', nargs='+', default=['onset'],
                   help='Window sampling roles (default: onset)')

    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── Build model ──────────────────────────────────────────────────────
    model, cfg = build_model(args)
    model = model.to(device)
    model.eval()

    # ─── Build dataset ────────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        patch_len=int(args.patch_len),
        n_patches=cfg.n_pre_patches + cfg.n_post_patches,
    )

    dataset = EEGStagePretrainDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        pipeline_cfg=pipeline_cfg,
        source_filter=args.source,
        split_filter=args.split,
        roles=args.sample_roles,
        center_jitter_sec=0.0,  # No jitter for evaluation
    )
    summary = summarize_stage_dataset(dataset)
    log.info('Dataset: %s', summary)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=stage_collate_fn,
        pin_memory=True,
    )

    # ─── Run evaluation ───────────────────────────────────────────────────
    log.info('Running evaluation with collect_temporal=True ...')
    val_metrics = evaluate_stage(
        model, loader, device,
        show_progress=True,
        collect_temporal=True,
    )

    if 'temporal_records' not in val_metrics:
        log.error('No temporal records collected. Cannot compute DeepSOZ metrics.')
        return

    patch_duration_sec = float(cfg.patch_len) / float(cfg.fs)
    output_dir = args.output_dir or str(Path(args.checkpoint).parent / 'deepsoz_eval')

    log.info('Computing DeepSOZ-style metrics (smoother_kernel=%d) ...', args.smoother_kernel)
    results = run_detailed_evaluation(
        records=val_metrics['temporal_records'],
        patch_duration_sec=patch_duration_sec,
        smoother_kernel_size=args.smoother_kernel,
        threshold=args.threshold,
        max_fpr_per_hour=args.max_fpr_per_hour,
        output_dir=output_dir,
    )

    metrics = results['metrics']
    per_seizure = results['per_seizure']

    # ─── Print report ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('DeepSOZ-Style Stage Detection Evaluation')
    print('=' * 70)
    print(f'\nCheckpoint: {args.checkpoint}')
    print(f'Split: {args.split}  Source: {args.source}')
    print(f'Smoother kernel: {args.smoother_kernel}  Threshold: {metrics["optimal_threshold"]:.3f}')
    print(f'\n--- Window-level Metrics ---')
    print(f'  AU-ROC:      {metrics["window_auroc"]:.4f}')
    print(f'  Sensitivity: {metrics["window_sensitivity"]:.4f}')
    print(f'  Specificity: {metrics["window_specificity"]:.4f}')
    print(f'\n--- Seizure-level Metrics ---')
    print(f'  Sensitivity: {metrics["seizure_sensitivity"]:.4f}  '
          f'({metrics["n_seizures_detected"]}/{metrics["n_seizures_total"]})')
    print(f'  FPR/hr:      {metrics["fpr_per_hour"]:.1f}')
    print(f'  Mean Latency:   {metrics["mean_latency_sec"]:.2f}s')
    print(f'  Median Latency: {metrics["median_latency_sec"]:.2f}s')
    print(f'\n--- Patch-level Metrics (existing) ---')
    print(f'  AUC:       {val_metrics.get("auc", 0.0):.4f}')
    print(f'  Recall:    {val_metrics.get("recall", 0.0):.4f}')
    print(f'  Precision: {val_metrics.get("precision", 0.0):.4f}')
    print(f'  F1:        {val_metrics.get("f1", 0.0):.4f}')

    if per_seizure:
        n_detected = sum(1 for s in per_seizure if s['detected'])
        n_total = len(per_seizure)
        print(f'\n--- Per-Seizure Summary ({n_detected}/{n_total} detected) ---')
        for i, s in enumerate(per_seizure[:20]):  # Show first 20
            status = 'DETECTED' if s['detected'] else 'MISSED'
            lat_str = f'{s["latency_sec"]:.1f}s' if s['latency_sec'] is not None else 'N/A'
            print(f'  [{i+1:3d}] {s["patient_id"]:12s} '
                  f'sz={s["seizure_start_sec"]:.0f}-{s["seizure_end_sec"]:.0f}s '
                  f'patches={s["n_patches"]:4d} '
                  f'{status:8s} lat={lat_str}')
        if len(per_seizure) > 20:
            print(f'  ... and {len(per_seizure) - 20} more seizures')

    print(f'\nResults saved to: {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
