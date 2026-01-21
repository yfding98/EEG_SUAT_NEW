#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN Training Script

Trains STGNN_SOZ_Locator model for seizure onset zone localization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

from stgnn_model import STGNN_SOZ_Locator


class STGNNDataset(Dataset):
    """Dataset for STGNN training"""
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        raw_data_root: str,
        n_bands: int = 5,
        time_steps: int = 10000,
        target_sfreq: float = 250.0
    ):
        """
        Args:
            csv_path: Path to CSV file with labels and relate_path
            data_root: Root directory for NPZ files (NHFE features)
            raw_data_root: Root directory for SET files (channel positions)
            n_bands: Number of frequency bands (default: 5)
            time_steps: Number of time steps (default: 10000)
            target_sfreq: Target sampling rate (default: 250.0 Hz)
        """
        self.data_root = Path(data_root)
        self.raw_data_root = Path(raw_data_root)
        self.n_bands = n_bands
        self.time_steps = time_steps
        self.target_sfreq = target_sfreq
        
        # Load CSV
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if 'relate_path' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("CSV must contain 'relate_path' and 'label' columns")
        
        # Load samples
        self.samples = []
        self.channel_names = None
        self.channel_positions = None  # Will store channel positions from first sample
        self._load_samples()
    
    def _parse_soz_channels(self, label_str: str) -> List[str]:
        """Parse SOZ channel labels"""
        if pd.isna(label_str) or not label_str or str(label_str).strip() == '':
            return []
        label_str = str(label_str).strip()
        for delimiter in [',', ';', '|']:
            if delimiter in label_str:
                return [ch.strip() for ch in label_str.split(delimiter) if ch.strip()]
        return [ch.strip() for ch in label_str.split() if ch.strip()]
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """
        Check if a position is valid (not NaN or None)
        
        Args:
            pos: Position array (x, y, z)
        
        Returns:
            True if position is valid, False otherwise
        """
        if pos is None:
            return False
        if isinstance(pos, np.ndarray):
            return not np.any(np.isnan(pos)) and not np.any(np.isinf(pos))
        return False
    
    def _extract_channel_positions(self, montage, ch_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract channel positions from montage and estimate positions for Sph-L and Sph-R
        
        Args:
            montage: MNE montage object
            ch_names: List of channel names
        
        Returns:
            Dictionary mapping channel name to 3D position (x, y, z)
        """
        positions = {}
        
        # Get positions from montage
        if montage is not None:
            try:
                montage_positions = montage.get_positions()
                if montage_positions and 'ch_pos' in montage_positions:
                    positions.update(montage_positions['ch_pos'])
            except:
                pass
        
        # Estimate positions for Sph-L and Sph-R if missing or invalid (NaN)
        # These are temporal bone electrodes, located near T7/T8 but more lateral and inferior
        sph_l_estimated = False
        sph_r_estimated = False
        
        # Check Sph-L: not in positions OR position is NaN
        if 'Sph-L' in ch_names:
            if 'Sph-L' not in positions or not self._is_valid_position(positions.get('Sph-L')):
                # Remove invalid position if exists
                if 'Sph-L' in positions:
                    del positions['Sph-L']
                # Estimate Sph-L position based on T7 or other left temporal electrodes
                reference_chs = ['T7', 'T3', 'FT7', 'TP7']  # Left temporal electrodes
                sph_l_pos = self._estimate_sph_position(positions, reference_chs, side='left')
                if sph_l_pos is not None:
                    positions['Sph-L'] = sph_l_pos
                    sph_l_estimated = True
        
        # Check Sph-R: not in positions OR position is NaN
        if 'Sph-R' in ch_names:
            if 'Sph-R' not in positions or not self._is_valid_position(positions.get('Sph-R')):
                # Remove invalid position if exists
                if 'Sph-R' in positions:
                    del positions['Sph-R']
                # Estimate Sph-R position based on T8 or other right temporal electrodes
                reference_chs = ['T8', 'T4', 'FT8', 'TP8']  # Right temporal electrodes
                sph_r_pos = self._estimate_sph_position(positions, reference_chs, side='right')
                if sph_r_pos is not None:
                    positions['Sph-R'] = sph_r_pos
                    sph_r_estimated = True
        
        if sph_l_estimated or sph_r_estimated:
            print(f"  Estimated positions: Sph-L={sph_l_estimated}, Sph-R={sph_r_estimated}")
        
        return positions
    
    def _estimate_sph_position(
        self,
        positions: Dict[str, np.ndarray],
        reference_chs: List[str],
        side: str = 'left'
    ) -> Optional[np.ndarray]:
        """
        Estimate Sph-L or Sph-R position based on reference temporal electrodes
        
        Args:
            positions: Dictionary of existing channel positions
            reference_chs: List of reference channel names to use for estimation
            side: 'left' for Sph-L, 'right' for Sph-R
        
        Returns:
            3D position array (x, y, z) or None if cannot estimate
        """
        # Find first available reference channel
        ref_pos = None
        ref_ch = None
        
        for ch in reference_chs:
            if ch in positions:
                ref_pos = positions[ch]
                ref_ch = ch
                break
        
        if ref_pos is None:
            return None
        
        # Sph electrodes are located:
        # - More lateral (further from midline): x coordinate more extreme
        # - More inferior (lower on head): z coordinate more negative
        # - Similar y coordinate (anterior-posterior)
        
        # Adjustment factors (in normalized head coordinates)
        # Lateral shift: 15-20% more lateral
        # Inferior shift: 10-15% lower
        lateral_shift = 0.18 if side == 'left' else -0.18  # Left is negative x, right is positive x
        inferior_shift = -0.12  # Lower on head
        
        # Get reference position
        x, y, z = ref_pos[0], ref_pos[1], ref_pos[2]
        
        # Apply shifts
        # For lateral: move away from midline
        if side == 'left':
            # Left side: x is negative, make it more negative
            x_new = x * (1 + abs(lateral_shift)) if x < 0 else x - abs(lateral_shift)
        else:
            # Right side: x is positive, make it more positive
            x_new = x * (1 + abs(lateral_shift)) if x > 0 else x + abs(lateral_shift)
        
        # For inferior: move down (more negative z)
        z_new = z + inferior_shift
        
        # Keep y similar (slight posterior shift)
        y_new = y - 0.02  # Slightly more posterior
        
        estimated_pos = np.array([x_new, y_new, z_new])
        
        return estimated_pos
    
    def _load_samples(self):
        """Load all samples"""
        print(f"Loading {len(self.df)} samples...")
        
        for idx, row in self.df.iterrows():
            relate_path = row['relate_path']
            label_str = row.get('label', '')
            
            # Load NPZ (NHFE features)
            # Handle both string and Path objects
            if isinstance(relate_path, str):
                relate_path = Path(relate_path)
            npz_path = self.data_root / relate_path
            if not npz_path.exists():
                print(f"Warning: NPZ not found: {npz_path}")
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'NHFE' not in data:
                    print(f"Warning: No NHFE in {npz_path}")
                    continue
                
                nhfe_all = data['NHFE']  # (n_channels, n_bands, n_timepoints)
                ch_names = data.get('ch_names', [])
                if hasattr(ch_names, 'tolist'):
                    ch_names = ch_names.tolist()
                
                # Ensure 5 bands
                if nhfe_all.shape[1] != self.n_bands:
                    if nhfe_all.shape[1] > self.n_bands:
                        nhfe_all = nhfe_all[:, :self.n_bands, :]
                    else:
                        print(f"Warning: Only {nhfe_all.shape[1]} bands in {npz_path}")
                        continue
                
                # Resample to target time steps
                if nhfe_all.shape[2] != self.time_steps:
                    from scipy import signal
                    nhfe_resampled = np.zeros((nhfe_all.shape[0], self.n_bands, self.time_steps))
                    for ch in range(nhfe_all.shape[0]):
                        for band in range(self.n_bands):
                            nhfe_resampled[ch, band, :] = signal.resample(
                                nhfe_all[ch, band, :], self.time_steps
                            )
                    nhfe_all = nhfe_resampled
                
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
            
            # Load SET file for channel positions
            set_pattern = "*_filtered_3_45_postICA_eye.set"
            relate_path_obj = Path(relate_path)
            
            # Try multiple strategies to find SET file
            set_files = []
            
            # Strategy 1: Same directory as NPZ
            set_dir = self.raw_data_root / relate_path_obj.parent
            set_files = list(set_dir.glob(set_pattern))
            
            # Strategy 2: Search in raw_data_root recursively
            if not set_files:
                set_files = list(self.raw_data_root.rglob(set_pattern))
                # Filter by patient ID if possible
                if len(set_files) > 1:
                    patient_id = relate_path_obj.stem.replace('_BEI', '').replace('_filtered', '')
                    filtered = [f for f in set_files if patient_id in f.stem]
                    if filtered:
                        set_files = filtered
            
            # Strategy 3: Use first SET file found (fallback)
            if not set_files:
                # Try to find any SET file in the directory structure
                set_files = list(self.raw_data_root.rglob("*.set"))
            
            channel_positions = None
            if not set_files:
                print(f"Warning: SET file not found for {relate_path}")
                # Use channel names from NPZ
                if len(ch_names) == 0:
                    ch_names = [f'Ch{i+1}' for i in range(nhfe_all.shape[0])]
            else:
                try:
                    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=False, verbose='ERROR')
                    montage = raw.get_montage()
                    ch_names = raw.ch_names
                    
                    # Extract channel positions from montage
                    if montage is not None:
                        channel_positions = self._extract_channel_positions(montage, ch_names)
                except Exception as e:
                    print(f"Warning: Error loading SET file {set_files[0]}: {e}")
                    if len(ch_names) == 0:
                        ch_names = [f'Ch{i+1}' for i in range(nhfe_all.shape[0])]
            
            # Store channel positions (first sample only, for reference)
            if self.channel_positions is None and channel_positions is not None:
                self.channel_positions = channel_positions
            
            # Set standard channel order (first sample)
            if self.channel_names is None:
                self.channel_names = ch_names.copy()
            else:
                # Reorder to match standard
                if len(ch_names) != len(self.channel_names):
                    print(f"Warning: Channel count mismatch for {relate_path}")
                    continue
                reorder_idx = [self.channel_names.index(ch) if ch in self.channel_names else None 
                              for ch in ch_names]
                if None in reorder_idx:
                    print(f"Warning: Channel mismatch for {relate_path}")
                    continue
                nhfe_all = nhfe_all[reorder_idx, :, :]
            
            # Parse SOZ channels (multi-label support)
            soz_channels = self._parse_soz_channels(label_str)
            if not soz_channels:
                print(f"Warning: No SOZ channels for {relate_path}")
                continue
            
            # Create multi-label vector: (n_channels,) with 1.0 for SOZ channels, 0.0 for others
            soz_indices = [self.channel_names.index(ch) for ch in soz_channels if ch in self.channel_names]
            if not soz_indices:
                print(f"Warning: SOZ channels not found in channel list for {relate_path}")
                continue
            
            # Multi-hot encoding: binary vector indicating which channels are SOZ
            multi_label = np.zeros(len(self.channel_names), dtype=np.float32)
            for idx in soz_indices:
                multi_label[idx] = 1.0
            
            self.samples.append({
                'nhfe': nhfe_all.astype(np.float32),  # (n_channels, n_bands, time_steps)
                'label': multi_label,  # Multi-label: (n_channels,) binary vector
                'soz_indices': soz_indices,  # List of SOZ channel indices for evaluation
                'patient_id': str(relate_path)
            })
        
        print(f"Loaded {len(self.samples)} valid samples")
        print(f"Channels: {len(self.channel_names)} - {self.channel_names}")
        
        # Print channel positions info
        if self.channel_positions:
            print(f"Channel positions loaded: {len(self.channel_positions)} channels")
            # Check if Sph-L and Sph-R positions are available
            if 'Sph-L' in self.channel_positions:
                print(f"  Sph-L position: {self.channel_positions['Sph-L']}")
            if 'Sph-R' in self.channel_positions:
                print(f"  Sph-R position: {self.channel_positions['Sph-R']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Convert to (n_channels, n_bands, time_steps) -> (n_bands, n_channels, time_steps) -> (n_channels, n_bands, time_steps)
        nhfe = torch.from_numpy(sample['nhfe'])  # (n_channels, n_bands, time_steps)
        
        # Apply log transformation: log(x+1) to compress dynamic range
        # NHFE values can range from 0 to 1000+, log transform is crucial
        nhfe = torch.log1p(nhfe)  # log(x+1) handles zeros safely
        
        return {
            'nhfe': nhfe,
            'label': torch.from_numpy(sample['label']),  # Multi-label: (n_channels,) binary vector
            'soz_indices': sample['soz_indices'],  # List of SOZ indices for evaluation
            'patient_id': sample['patient_id']
        }


def train_epoch(model, dataloader, criterion, optimizer, device, use_augmentation=True):
    """
    Train for one epoch with data augmentation (multi-label)
    
    Args:
        model: Model to train
        dataloader: Data loader
        criterion: Loss function (BCEWithLogitsLoss for multi-label)
        optimizer: Optimizer
        device: Device
        use_augmentation: If True, apply data augmentation (noise injection and channel masking)
    """
    model.train()
    total_loss = 0.0
    total = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        nhfe = batch['nhfe'].to(device)  # (B, N_Ch, N_Bands, Time_Steps)
        labels = batch['label'].to(device)  # (B, N_Ch) - multi-label binary vector
        
        # Data Augmentation (only during training)
        if use_augmentation:
            # 1. Noise Injection: Add Gaussian noise
            noise = torch.randn_like(nhfe) * 0.05
            nhfe = nhfe + noise
            
            # 2. Channel Masking: Randomly mask 1-2 non-seizing channels
            # Get SOZ channels from labels
            batch_size, n_channels = nhfe.shape[0], nhfe.shape[1]
            for b in range(batch_size):
                soz_channels = torch.where(labels[b] > 0.5)[0].cpu().tolist()
                # Get non-seizing channels
                non_seizing = [i for i in range(n_channels) if i not in soz_channels]
                if len(non_seizing) > 0:
                    # Randomly mask 1-2 channels
                    n_mask = torch.randint(1, min(3, len(non_seizing) + 1), (1,)).item()
                    mask_channels = torch.tensor(
                        np.random.choice(non_seizing, size=n_mask, replace=False),
                        device=device
                    )
                    # Set masked channels to zero
                    nhfe[b, mask_channels, :, :] = 0.0
        
        optimizer.zero_grad()
        logits = model(nhfe)  # (B, N_Ch)
        loss = criterion(logits, labels)  # Multi-label BCE loss
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
    
    # Calculate approximate accuracy (using IOU)
    # This is just for monitoring, not used for optimization
    avg_iou = 0.0
    with torch.no_grad():
        for batch in dataloader:
            nhfe = batch['nhfe'].to(device)
            labels = batch['label'].to(device)
            logits = model(nhfe)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
            
            # Calculate IOU for this batch
            for b in range(labels.size(0)):
                true_indices = torch.where(labels[b] > 0.5)[0].cpu().tolist()
                pred_indices = torch.where(pred_mask[b] > 0.5)[0].cpu().tolist()
                if true_indices or pred_indices:
                    intersection = len(set(true_indices) & set(pred_indices))
                    union = len(set(true_indices) | set(pred_indices))
                    if union > 0:
                        iou = intersection / union
                        avg_iou += iou
    
    avg_iou = avg_iou / total if total > 0 else 0.0
    
    return total_loss / len(dataloader), avg_iou


def validate(
    model, 
    dataloader, 
    criterion, 
    device, 
    return_topk=False, 
    print_details=False, 
    channel_names=None,
    top_k: int = 5,
    return_f1: bool = False
):
    """
    Validate model with Top-K ranking metrics (Multi-Label Ranking)
    
    Args:
        model: Model to validate
        dataloader: Data loader
        criterion: Loss function (BCEWithLogitsLoss)
        device: Device
        return_topk: If True, return hit rates and recall@K
        print_details: If True, print patient info and predictions
        channel_names: List of channel names for printing
        top_k: Top-K value for ranking (default: 5)
    
    Returns:
        If return_topk=False: (loss,)
        If return_topk=True: (loss, hit_rate_3, hit_rate_5, recall_3, recall_5)
    """
    model.eval()
    total_loss = 0.0
    
    # For Top-K metrics
    hits_at_3 = 0  # Count of samples where Top-3 contains at least one true SOZ
    hits_at_5 = 0  # Count of samples where Top-5 contains at least one true SOZ
    recall_at_3_sum = 0.0  # Sum of recall@3 (found true channels / total true channels)
    recall_at_5_sum = 0.0  # Sum of recall@5
    
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validating')):
            nhfe = batch['nhfe'].to(device)
            labels = batch['label'].to(device)  # (B, N_Ch) - multi-label
            soz_indices_list = batch.get('soz_indices', [])
            patient_ids = batch.get('patient_id', [f'Sample_{batch_idx}'] * labels.size(0))
            
            logits = model(nhfe)  # (B, N_Ch)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get probabilities for display (not used for ranking)
            probs = torch.sigmoid(logits)  # (B, N_Ch)
            
            # Calculate Top-K metrics
            if return_topk:
                for i in range(labels.size(0)):
                    # Get true SOZ indices
                    true_soz = soz_indices_list[i] if isinstance(soz_indices_list, list) and i < len(soz_indices_list) else torch.where(labels[i] > 0.5)[0].cpu().tolist()
                    true_soz_set = set(true_soz)
                    
                    if not true_soz_set:
                        continue  # Skip if no true SOZ channels
                    
                    # Top-3: select top 3 channels by logits
                    _, top3_indices = logits[i].topk(k=3, dim=0)
                    pred_top3 = set(top3_indices.cpu().tolist())
                    
                    # Top-5: select top 5 channels by logits
                    _, top5_indices = logits[i].topk(k=5, dim=0)
                    pred_top5 = set(top5_indices.cpu().tolist())
                    
                    # Hit Rate: at least one true SOZ in Top-K
                    if true_soz_set & pred_top3:  # Intersection is non-empty
                        hits_at_3 += 1
                    
                    if true_soz_set & pred_top5:  # Intersection is non-empty
                        hits_at_5 += 1
                    
                    # Recall@K: how many true channels found in Top-K
                    found_at_3 = len(true_soz_set & pred_top3)
                    found_at_5 = len(true_soz_set & pred_top5)
                    recall_at_3_sum += found_at_3 / len(true_soz_set) if len(true_soz_set) > 0 else 0.0
                    recall_at_5_sum += found_at_5 / len(true_soz_set) if len(true_soz_set) > 0 else 0.0
            
            # Print details for each sample
            if print_details:
                for i in range(labels.size(0)):
                    # Get true SOZ channels
                    true_soz = soz_indices_list[i] if isinstance(soz_indices_list, list) and i < len(soz_indices_list) else torch.where(labels[i] > 0.5)[0].cpu().tolist()
                    true_soz_set = set(true_soz)
                    
                    # Get Top-K predictions
                    _, top3_indices = logits[i].topk(k=3, dim=0)
                    top3_list = top3_indices.cpu().tolist()
                    top3_probs = probs[i][top3_indices].cpu().tolist()
                    
                    _, top5_indices = logits[i].topk(k=5, dim=0)
                    top5_list = top5_indices.cpu().tolist()
                    top5_probs = probs[i][top5_indices].cpu().tolist()
                    
                    # Get channel names
                    if channel_names:
                        true_ch_names = [channel_names[idx] if idx < len(channel_names) else f'Ch{idx}' for idx in true_soz]
                    else:
                        true_ch_names = [f'Ch{idx}' for idx in true_soz]
                    
                    # Check hits
                    hit_at_3 = bool(true_soz_set & set(top3_list))
                    hit_at_5 = bool(true_soz_set & set(top5_list))
                    
                    # Calculate recall@K
                    found_at_3 = len(true_soz_set & set(top3_list))
                    found_at_5 = len(true_soz_set & set(top5_list))
                    recall_3 = found_at_3 / len(true_soz_set) if len(true_soz_set) > 0 else 0.0
                    recall_5 = found_at_5 / len(true_soz_set) if len(true_soz_set) > 0 else 0.0
                    
                    patient_id = patient_ids[i] if isinstance(patient_ids, list) else patient_ids
                    print(f"\n  Patient: {patient_id}")
                    print(f"    Target: [{', '.join(true_ch_names) if true_ch_names else 'None'}]")
                    
                    # Format Top-3 predictions
                    top3_ch_names = []
                    for idx, prob in zip(top3_list, top3_probs):
                        ch_name = channel_names[idx] if channel_names and idx < len(channel_names) else f'Ch{idx}'
                        is_true = " [GT]" if idx in true_soz_set else ""
                        top3_ch_names.append(f"{ch_name} ({prob:.3f}){is_true}")
                    print(f"    Top-3 Preds: [{', '.join(top3_ch_names)}]")
                    print(f"    -> Hit@3: {'YES' if hit_at_3 else 'NO'}, Recall@3: {recall_3:.3f} ({found_at_3}/{len(true_soz_set)})")
                    
                    # Format Top-5 predictions
                    top5_ch_names = []
                    for idx, prob in zip(top5_list, top5_probs):
                        ch_name = channel_names[idx] if channel_names and idx < len(channel_names) else f'Ch{idx}'
                        is_true = " [GT]" if idx in true_soz_set else ""
                        top5_ch_names.append(f"{ch_name} ({prob:.3f}){is_true}")
                    print(f"    Top-5 Preds: [{', '.join(top5_ch_names)}]")
                    print(f"    -> Hit@5: {'YES' if hit_at_5 else 'NO'}, Recall@5: {recall_5:.3f} ({found_at_5}/{len(true_soz_set)})")
            
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate F1_score if requested
    f1_macro = None
    f1_micro = None
    if return_f1 or return_topk:
        # Collect all predictions and labels for F1 calculation
        all_pred_labels = []
        all_true_labels = []
        
        # Re-run through dataloader to collect predictions for F1 calculation
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                nhfe = batch['nhfe'].to(device)
                labels = batch['label'].to(device)
                logits = model(nhfe)
                
                # Apply sigmoid and threshold for binary predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_pred_labels.append(preds.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())
        
        if len(all_pred_labels) > 0:
            from sklearn.metrics import f1_score
            all_preds = np.concatenate(all_pred_labels, axis=0)
            all_labels = np.concatenate(all_true_labels, axis=0)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    if return_topk:
        hit_rate_3 = hits_at_3 / total if total > 0 else 0.0
        hit_rate_5 = hits_at_5 / total if total > 0 else 0.0
        recall_3 = recall_at_3_sum / total if total > 0 else 0.0
        recall_5 = recall_at_5_sum / total if total > 0 else 0.0
        if return_f1:
            return avg_loss, hit_rate_3, hit_rate_5, recall_3, recall_5, f1_macro, f1_micro
        else:
            return avg_loss, hit_rate_3, hit_rate_5, recall_3, recall_5
    else:
        if return_f1:
            return avg_loss, f1_macro, f1_micro
        else:
            return (avg_loss,)


def _calculate_iou(true_indices: List[int], pred_indices: List[int]) -> float:
    """Calculate Intersection over Union for channel sets"""
    if not true_indices or not pred_indices:
        return 0.0
    
    true_set = set(true_indices)
    pred_set = set(pred_indices)
    
    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def train_stgnn(
    csv_path: str,
    data_root: str,
    raw_data_root: str,
    config_path: Optional[str] = None,
    output_dir: str = "checkpoints_stgnn",
    debug_overfit: bool = False,
    use_loocv: bool = True
):
    """
    Main training function
    
    Args:
        csv_path: Path to CSV file with labels
        data_root: Root directory for NPZ files
        raw_data_root: Root directory for SET files
        config_path: Path to config YAML file
        output_dir: Output directory
        debug_overfit: If True, use single sample repeated 10 times for debugging
        use_loocv: If True, use Leave-One-Out Cross-Validation (recommended for small datasets)
    """
    print("=" * 80)
    print("STGNN Training")
    print("=" * 80)
    
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Training parameters
    batch_size = config.get('training', {}).get('batch_size', 4)
    epochs = config.get('training', {}).get('epochs', 50)
    lr = config.get('training', {}).get('learning_rate', 0.001)  # Increased to 1e-3
    device = torch.device(config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Note: weight_decay is now hardcoded to 1e-2 in optimizer for stronger regularization
    
    # Model parameters
    n_channels = 21
    n_bands = 5
    time_steps = 10000
    
    # Create dataset
    print("\n[1/4] Loading dataset...")
    dataset = STGNNDataset(
        csv_path=csv_path,
        data_root=data_root,
        raw_data_root=raw_data_root,
        n_bands=n_bands,
        time_steps=time_steps
    )
    
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    # DEBUG_OVERFIT mode: Use single sample repeated 10 times
    if debug_overfit:
        print("\n" + "=" * 80)
        print("DEBUG_OVERFIT MODE: Using single sample repeated 10 times")
        print("=" * 80)
        
        if total_samples == 0:
            raise ValueError("No samples loaded! Cannot run debug mode.")
        
        # Create a dataset with single sample repeated
        from torch.utils.data import Dataset, ConcatDataset
        
        class RepeatDataset(Dataset):
            def __init__(self, base_dataset, repeat=10):
                self.base_dataset = base_dataset
                self.repeat = repeat
                self.base_idx = 0  # Use first sample
            
            def __len__(self):
                return self.repeat
            
            def __getitem__(self, idx):
                return self.base_dataset[self.base_idx]
        
        debug_dataset = RepeatDataset(dataset, repeat=10)
        
        # Custom collate function for debug mode
        def debug_collate_fn(batch):
            nhfe = torch.stack([item['nhfe'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            soz_indices = [item['soz_indices'] for item in batch]
            patient_ids = [item['patient_id'] for item in batch]
            return {
                'nhfe': nhfe,
                'label': labels,
                'soz_indices': soz_indices,
                'patient_id': patient_ids
            }
        
        train_loader = DataLoader(debug_dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=debug_collate_fn)
        val_loader = train_loader  # Same for validation
        
        # Create model with reduced capacity and stronger regularization
        print("\n[2/4] Creating model...")
        model = STGNN_SOZ_Locator(
            n_channels=n_channels,
            n_bands=n_bands,
            time_steps=time_steps,
            temporal_hidden_dim=32,  # Reduced from default 128
            graph_hidden_dim=32,  # Reduced from default 128
            dropout=0.6  # Increased from default 0.2
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer with stronger weight decay
        # Use BCEWithLogitsLoss for multi-label classification
        pos_weight = torch.tensor([10.0] * n_channels, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)  # Increased to 1e-2
        
        # Training loop for debug
        print("\n[3/4] Training (DEBUG_OVERFIT mode)...")
        print("Expected: Loss should drop to ~0.00 within 50 epochs if model works correctly")
        
        for epoch in range(epochs):
            train_loss, train_iou = train_epoch(
                model, train_loader, criterion, optimizer, device,
                use_augmentation=True  # Enable data augmentation
            )
            val_loss, val_hit_rate_3, val_hit_rate_5, val_recall_3, val_recall_5, val_f1_macro, val_f1_micro = validate(
                model, val_loader, criterion, device,
                return_topk=True,
                return_f1=True,
                channel_names=dataset.channel_names
            )
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, IOU={train_iou:.4f}, Hit@5={val_hit_rate_5:.4f}, Recall@5={val_recall_5:.4f}, F1_score(macro)={val_f1_macro:.4f}, F1_score(micro)={val_f1_micro:.4f}")
            
            if train_loss < 0.01:
                print(f"✓ Model can overfit! Loss dropped to {train_loss:.4f}")
                break
        
        if train_loss >= 0.01:
            print(f"⚠️  WARNING: Model failed to overfit. Loss={train_loss:.4f}")
            print("This indicates a potential bug in model implementation or data loading.")
        
        return
    
    # LOOCV mode (recommended for small datasets)
    if use_loocv and total_samples > 1:
        print("\n" + "=" * 80)
        print("Using Leave-One-Out Cross-Validation (LOOCV)")
        print("=" * 80)
        
        from torch.utils.data import Subset
        
        accuracies = []
        all_indices = list(range(total_samples))
        
        for fold in range(total_samples):
            print(f"\n{'='*80}")
            print(f"LOOCV Fold {fold+1}/{total_samples}")
            print(f"{'='*80}")
            
            # Test index is the current fold
            test_idx = [fold]
            train_idx = [i for i in all_indices if i != fold]
            
            # Create train/test splits
            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)
            
            # Custom collate function to handle soz_indices (list of lists)
            def collate_fn(batch):
                nhfe = torch.stack([item['nhfe'] for item in batch])
                labels = torch.stack([item['label'] for item in batch])
                soz_indices = [item['soz_indices'] for item in batch]
                patient_ids = [item['patient_id'] for item in batch]
                return {
                    'nhfe': nhfe,
                    'label': labels,
                    'soz_indices': soz_indices,
                    'patient_id': patient_ids
                }
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
            
            print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
            # Re-initialize model for each fold (CRITICAL!)
            # Use reduced capacity and stronger regularization to prevent overfitting
            model = STGNN_SOZ_Locator(
                n_channels=n_channels,
                n_bands=n_bands,
                time_steps=time_steps,
                temporal_hidden_dim=32,  # Reduced from default 128
                graph_hidden_dim=32,  # Reduced from default 128
                dropout=0.6  # Increased from default 0.2
            ).to(device)
            
            # Loss and optimizer with stronger weight decay
            # Use BCEWithLogitsLoss for multi-label classification
            # Calculate pos_weight to handle class imbalance
            # pos_weight = (num_negative / num_positive) for each class
            # Approximate: if on average 2-3 channels are active out of 21, pos_weight ≈ 21/2.5 ≈ 8.4
            pos_weight = torch.tensor([10.0] * n_channels, device=device)  # Roughly 21/2.1
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)  # Increased to 1e-2
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(epochs):
                train_loss, train_iou = train_epoch(
                    model, train_loader, criterion, optimizer, device, 
                    use_augmentation=True  # Enable data augmentation
                )
                
                # Early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(train_loss)
                
                if epoch % 10 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, IOU={train_iou:.4f}")
                
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Evaluate on test sample with Top-K ranking metrics
            print(f"\n  Evaluating test sample (Fold {fold+1}/{total_samples}):")
            test_loss, hit_rate_3, hit_rate_5, recall_3, recall_5, test_f1_macro, test_f1_micro = validate(
                model, test_loader, criterion, device, 
                return_topk=True,
                return_f1=True,
                print_details=True,  # Print patient info and predictions
                channel_names=dataset.channel_names,  # Pass channel names for printing
                top_k=5
            )
            accuracies.append({
                'hit_rate_3': hit_rate_3,
                'hit_rate_5': hit_rate_5,
                'recall_3': recall_3,
                'recall_5': recall_5,
                'f1_score_macro': test_f1_macro,
                'f1_score_micro': test_f1_micro
            })
            print(f"\n  Fold {fold+1} Test Results:")
            print(f"    Hit Rate @ 3: {hit_rate_3:.4f}, Hit Rate @ 5: {hit_rate_5:.4f}")
            print(f"    Recall @ 3: {recall_3:.4f}, Recall @ 5: {recall_5:.4f}")
            print(f"    F1_score (macro): {test_f1_macro:.4f}, F1_score (micro): {test_f1_micro:.4f}")
        
        # Print LOOCV results
        print("\n" + "=" * 80)
        print("LOOCV Results (Multi-Label Ranking)")
        print("=" * 80)
        
        # Extract metrics
        hit_rate_3_scores = [acc['hit_rate_3'] for acc in accuracies]
        hit_rate_5_scores = [acc['hit_rate_5'] for acc in accuracies]
        recall_3_scores = [acc['recall_3'] for acc in accuracies]
        recall_5_scores = [acc['recall_5'] for acc in accuracies]
        f1_macro_scores = [acc.get('f1_score_macro', 0.0) for acc in accuracies]
        f1_micro_scores = [acc.get('f1_score_micro', 0.0) for acc in accuracies]
        
        print(f"Hit Rate @ 3 (At least one true SOZ in Top-3):")
        print(f"  Individual: {[f'{x:.4f}' for x in hit_rate_3_scores]}")
        print(f"  Mean: {np.mean(hit_rate_3_scores):.4f} ± {np.std(hit_rate_3_scores):.4f}")
        print(f"  Best: {np.max(hit_rate_3_scores):.4f}, Worst: {np.min(hit_rate_3_scores):.4f}")
        
        print(f"\nHit Rate @ 5 (At least one true SOZ in Top-5):")
        print(f"  Individual: {[f'{x:.4f}' for x in hit_rate_5_scores]}")
        print(f"  Mean: {np.mean(hit_rate_5_scores):.4f} ± {np.std(hit_rate_5_scores):.4f}")
        print(f"  Best: {np.max(hit_rate_5_scores):.4f}, Worst: {np.min(hit_rate_5_scores):.4f}")
        
        print(f"\nRecall @ 3 (Fraction of true SOZ channels found in Top-3):")
        print(f"  Individual: {[f'{x:.4f}' for x in recall_3_scores]}")
        print(f"  Mean: {np.mean(recall_3_scores):.4f} ± {np.std(recall_3_scores):.4f}")
        print(f"  Best: {np.max(recall_3_scores):.4f}, Worst: {np.min(recall_3_scores):.4f}")
        
        print(f"\nRecall @ 5 (Fraction of true SOZ channels found in Top-5):")
        print(f"  Individual: {[f'{x:.4f}' for x in recall_5_scores]}")
        print(f"  Mean: {np.mean(recall_5_scores):.4f} ± {np.std(recall_5_scores):.4f}")
        print(f"  Best: {np.max(recall_5_scores):.4f}, Worst: {np.min(recall_5_scores):.4f}")
        
        print(f"\nF1_score (macro):")
        print(f"  Individual: {[f'{x:.4f}' for x in f1_macro_scores]}")
        print(f"  Mean: {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}")
        print(f"  Best: {np.max(f1_macro_scores):.4f}, Worst: {np.min(f1_macro_scores):.4f}")
        
        print(f"\nF1_score (micro):")
        print(f"  Individual: {[f'{x:.4f}' for x in f1_micro_scores]}")
        print(f"  Mean: {np.mean(f1_micro_scores):.4f} ± {np.std(f1_micro_scores):.4f}")
        print(f"  Best: {np.max(f1_micro_scores):.4f}, Worst: {np.min(f1_micro_scores):.4f}")
        
        # Save results
        results = {
            'method': 'LOOCV',
            'total_samples': total_samples,
            'individual_results': accuracies,
            'hit_rate_3': {
                'individual': hit_rate_3_scores,
                'mean': float(np.mean(hit_rate_3_scores)),
                'std': float(np.std(hit_rate_3_scores)),
                'best': float(np.max(hit_rate_3_scores)),
                'worst': float(np.min(hit_rate_3_scores))
            },
            'hit_rate_5': {
                'individual': hit_rate_5_scores,
                'mean': float(np.mean(hit_rate_5_scores)),
                'std': float(np.std(hit_rate_5_scores)),
                'best': float(np.max(hit_rate_5_scores)),
                'worst': float(np.min(hit_rate_5_scores))
            },
            'recall_3': {
                'individual': recall_3_scores,
                'mean': float(np.mean(recall_3_scores)),
                'std': float(np.std(recall_3_scores)),
                'best': float(np.max(recall_3_scores)),
                'worst': float(np.min(recall_3_scores))
            },
            'recall_5': {
                'individual': recall_5_scores,
                'mean': float(np.mean(recall_5_scores)),
                'std': float(np.std(recall_5_scores)),
                'best': float(np.max(recall_5_scores)),
                'worst': float(np.min(recall_5_scores))
            },
            'f1_score_macro': {
                'individual': f1_macro_scores,
                'mean': float(np.mean(f1_macro_scores)),
                'std': float(np.std(f1_macro_scores)),
                'best': float(np.max(f1_macro_scores)),
                'worst': float(np.min(f1_macro_scores))
            },
            'f1_score_micro': {
                'individual': f1_micro_scores,
                'mean': float(np.mean(f1_micro_scores)),
                'std': float(np.std(f1_micro_scores)),
                'best': float(np.max(f1_micro_scores)),
                'worst': float(np.min(f1_micro_scores))
            }
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'loocv_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path / 'loocv_results.json'}")
        return
    
    # Fallback: Traditional train/val/test split (not recommended for small datasets)
    print("\n" + "=" * 80)
    print("Using traditional train/val/test split")
    print("WARNING: Not recommended for datasets with < 50 samples!")
    print("=" * 80)
    
    from torch.utils.data import random_split
    train_ratio = config.get('training', {}).get('train_ratio', 0.7)
    val_ratio = config.get('training', {}).get('val_ratio', 0.15)
    
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Custom collate function to handle soz_indices (list of lists)
    def collate_fn(batch):
        nhfe = torch.stack([item['nhfe'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        soz_indices = [item['soz_indices'] for item in batch]
        patient_ids = [item['patient_id'] for item in batch]
        return {
            'nhfe': nhfe,
            'label': labels,
            'soz_indices': soz_indices,
            'patient_id': patient_ids
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model with reduced capacity and stronger regularization
    print("\n[2/4] Creating model...")
    model = STGNN_SOZ_Locator(
        n_channels=n_channels,
        n_bands=n_bands,
        time_steps=time_steps,
        temporal_hidden_dim=32,  # Reduced from default 128
        graph_hidden_dim=32,  # Reduced from default 128
        dropout=0.6  # Increased from default 0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with stronger weight decay
    # Use BCEWithLogitsLoss for multi-label classification
    # Calculate pos_weight to handle class imbalance
    # pos_weight = (num_negative / num_positive) for each class
    # Approximate: if on average 2-3 channels are active out of 21, pos_weight ≈ 21/2.5 ≈ 8.4
    pos_weight = torch.tensor([10.0] * n_channels, device=device)  # Roughly 21/2.1
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)  # Increased to 1e-2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n[3/4] Training...")
    best_val_f1 = 0.0
    history = {
        'train_loss': [], 
        'train_iou': [], 
        'val_loss': [], 
        'val_hit_rate_3': [], 
        'val_hit_rate_5': [], 
        'val_recall_3': [], 
        'val_recall_5': [],
        'val_f1_score_macro': [],
        'val_f1_score_micro': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_augmentation=True  # Enable data augmentation
        )
        val_loss, val_hit_rate_3, val_hit_rate_5, val_recall_3, val_recall_5, val_f1_macro, val_f1_micro = validate(
            model, val_loader, criterion, device,
            return_topk=True,
            return_f1=True,
            channel_names=dataset.channel_names
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_hit_rate_3'].append(val_hit_rate_3)
        history['val_hit_rate_5'].append(val_hit_rate_5)
        history['val_recall_3'].append(val_recall_3)
        history['val_recall_5'].append(val_recall_5)
        history['val_f1_score_macro'].append(val_f1_macro)
        history['val_f1_score_micro'].append(val_f1_micro)
        
        print(f"Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Hit@3: {val_hit_rate_3:.4f}, Hit@5: {val_hit_rate_5:.4f}, Recall@3: {val_recall_3:.4f}, Recall@5: {val_recall_5:.4f}")
        print(f"Val F1_score (macro): {val_f1_macro:.4f}, Val F1_score (micro): {val_f1_micro:.4f}")
        
        # Save best model (based on Hit Rate @ 5)
        if val_hit_rate_5 > best_val_f1:  # Reuse variable name for compatibility
            best_val_f1 = val_hit_rate_5
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_hit_rate_5': val_hit_rate_5,
                'val_hit_rate_3': val_hit_rate_3,
                'val_recall_3': val_recall_3,
                'val_recall_5': val_recall_5,
                'history': history
            }, output_path / 'best_model.pth')
            print(f"Saved best model (val_hit_rate_5: {val_hit_rate_5:.4f})")
    
    # Test
    print("\n[4/4] Testing...")
    test_loss, test_hit_rate_3, test_hit_rate_5, test_recall_3, test_recall_5, test_f1_macro, test_f1_micro = validate(
        model, test_loader, criterion, device,
        return_topk=True,
        return_f1=True,
        channel_names=dataset.channel_names
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Hit@3: {test_hit_rate_3:.4f}, Hit@5: {test_hit_rate_5:.4f}, Recall@3: {test_recall_3:.4f}, Recall@5: {test_recall_5:.4f}")
    print(f"Test F1_score (macro): {test_f1_macro:.4f}, Test F1_score (micro): {test_f1_micro:.4f}")
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_hit_rate_3': test_hit_rate_3,
        'test_hit_rate_5': test_hit_rate_5,
        'test_recall_3': test_recall_3,
        'test_recall_5': test_recall_5,
        'test_f1_score_macro': test_f1_macro,
        'test_f1_score_micro': test_f1_micro,
        'best_val_hit_rate_5': best_val_f1,
        'history': history
    }
    
    output_path = Path(output_dir)
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend([
            # '--csv_path', r'E:/output/segment_results/nhfe_channel_label.csv',
            '--csv_path', r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\results\window_detection\nhfe_channel_label_quant70.0_0.6_50_0.05.csv',
            # '--data_root', 'E:/output/segment_results',
            '--data_root', 'E:/output/multi_segment_nhfe',
            '--raw_data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
        ])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with labels')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for NPZ files')
    parser.add_argument('--raw_data_root', type=str, required=True, help='Root directory for SET files')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='checkpoints_stgnn', help='Output directory')
    parser.add_argument('--debug_overfit', action='store_true', help='Debug mode: overfit on single sample')
    parser.add_argument('--no_loocv', action='store_true', help='Disable LOOCV, use train/val/test split')
    
    args = parser.parse_args()
    
    train_stgnn(
        csv_path=args.csv_path,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        config_path=args.config,
        output_dir=args.output_dir,
        debug_overfit=args.debug_overfit,
        use_loocv=not args.no_loocv
    )

