#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_brain_networks.py

前置计算脑网络特征（GC, TE, AEC, wPLI）并保存到磁盘，
以避免在训练期间进行重复、耗时的在线计算。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from models.integration_model import IntegrationConfig
    from models.seizure_aligned_patching import SeizureAlignedAdaptivePatching
    from models.brain_network_extractor import MultiScaleBrainNetworkExtractor
    from models.manifest_dataset import ManifestSOZDataset
    from models.train_soz_locator_with_brain_networks import SOZBrainNetworkDataset, collate_fn
except ImportError:
    from integration_model import IntegrationConfig
    from seizure_aligned_patching import SeizureAlignedAdaptivePatching
    from brain_network_extractor import MultiScaleBrainNetworkExtractor
    from manifest_dataset import ManifestSOZDataset
    from train_soz_locator_with_brain_networks import SOZBrainNetworkDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('preprocess')

def parse_args():
    p = argparse.ArgumentParser(description='Precompute Brain Networks')
    p.add_argument('--manifest', default=r"E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\combined_manifest.csv" ,help='combined_manifest.csv path')
    p.add_argument('--private-data-root', default=r"F:\dyf\Dataset\DataSet\EEG\EEG dataset_SUAT", help='Preprocessed private data root')
    p.add_argument('--tusz-data-root', default=r'F:\dataset\TUSZ\v2.0.3\edf', help='TUSZ EDF root')
    p.add_argument('--source', default='both', choices=['tusz', 'private', 'both'])
    
    p.add_argument('--output-dir', default=r"F:\dyf\output\EEG_SUAT_NEW", help='Directory to save precomputed networks (.npz files)')
    
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--patch-duration', type=float, default=0.5)
    p.add_argument('--fs', type=float, default=200.0)
    
    # Sequence length configurations
    p.add_argument('--pre-onset-sec', type=float, default=5.0, help='Seconds before onset')
    p.add_argument('--post-onset-sec', type=float, default=5.0, help='Seconds after onset')
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Loading manifest from {args.manifest}")
    
    patch_len = int(args.patch_duration * args.fs)
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    n_patches = n_pre_patches + n_post_patches
    
    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from .data_preprocess.eeg_pipeline import PipelineConfig

    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_patches,
        patch_len=patch_len
    )

    manifest_ds = ManifestSOZDataset(
        manifest_path=args.manifest,
        private_data_root=args.private_data_root,
        tusz_data_root=args.tusz_data_root,
        source_filter=args.source,
        pipeline_cfg=pipeline_cfg,
    )
    # Instantiate without precomputed_dir since we are generating them
    dataset = SOZBrainNetworkDataset(manifest_ds)
    dataset.precomputed_dir = output_dir  # Just to use _get_cache_path method
    
    log.info(f"Found {len(dataset)} samples. Creating DataLoader...")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    
    log.info("Initializing models...")
    cfg = IntegrationConfig(
        patch_len=patch_len, 
        n_pre_patches=n_pre_patches,
        n_post_patches=n_post_patches,
        fs=args.fs
    )
    patching = SeizureAlignedAdaptivePatching(
        n_channels=cfg.n_channels, patch_len=cfg.patch_len,
        n_pre_patches=cfg.n_pre_patches,
        n_post_patches=cfg.n_post_patches, fs=cfg.fs,
    ).to(device)
    patching.eval()
    
    net_extractor = MultiScaleBrainNetworkExtractor(
        n_channels=cfg.n_channels, patch_len=cfg.patch_len,
        fs=cfg.fs, gc_order=cfg.gc_order, te_n_bins=cfg.te_n_bins,
    ).to(device)
    net_extractor.eval()
    
    log.info(f"Starting precomputation, saving to {output_dir}")
    skipped_count = 0
    saved_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precomputing features"):
            # Before moving everything to GPU, check which ones actually need processing
            idxs = batch['idx']
            process_indices = []
            cache_paths = []
            
            for i, idx in enumerate(idxs):
                cache_path = dataset._get_cache_path(idx)
                if cache_path.exists():
                    skipped_count += 1
                    process_indices.append(False)
                else:
                    process_indices.append(True)
                    cache_paths.append(cache_path)
            
            if not any(process_indices):
                continue
                
            # Filter the batch for elements that need processing
            x = batch['x'][process_indices].to(device)
            onset = batch['onset_sec'][process_indices].to(device)
            start = batch['start_sec'][process_indices].to(device)
            
            # Step 1: Patching
            patches, vp_counts, rel_time = patching(x, onset, start)
            
            # Step 2: Brain Network Extraction
            net_result = net_extractor(patches)
            brain_nets = net_result['all']  # [B, P, 22, 22, 4]
            
            # Move results to CPU for saving
            brain_nets_cpu = brain_nets.cpu().numpy()
            vp_counts_cpu = vp_counts.cpu().numpy()
            rel_time_cpu = rel_time.cpu().numpy()
            
            # Save each sample individually
            for i, cache_path in enumerate(cache_paths):
                np.savez_compressed(
                    str(cache_path),
                    brain_nets=brain_nets_cpu[i],
                    valid_patch_counts=vp_counts_cpu[i],
                    rel_time=rel_time_cpu[i],
                )
                saved_count += 1
                
    log.info(f"Done! Saved: {saved_count}, Skipped (already existed): {skipped_count}")

if __name__ == '__main__':
    main()
