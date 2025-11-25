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

from STGNN import STGNN_SOZ_Locator


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
            # 去npz_path下找 *filtered_3_45_postICA_eye_BEI.npz的文件并加载
            npz_pattern = "*filtered_3_45_postICA_eye_BEI.npz"
            npz_files = list(npz_path.glob(npz_pattern))
            bei_npz_path = npz_files[0] if len(npz_files) > 0 else None
            if bei_npz_path is None or not bei_npz_path.exists():
                print(f"  ⚠️  BEI.npz file not found: {bei_npz_path}")
                continue
            try:
                data = np.load(bei_npz_path, allow_pickle=True)
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
            
            # Parse SOZ channels
            soz_channels = self._parse_soz_channels(label_str)
            if not soz_channels:
                print(f"Warning: No SOZ channels for {relate_path}")
                continue
            
            # Create label (index of first SOZ channel)
            soz_indices = [self.channel_names.index(ch) for ch in soz_channels if ch in self.channel_names]
            if not soz_indices:
                print(f"Warning: SOZ channels not found in channel list for {relate_path}")
                continue
            
            # Use first SOZ channel as label (chronological order)
            label = min(soz_indices)
            
            self.samples.append({
                'nhfe': nhfe_all.astype(np.float32),  # (n_channels, n_bands, time_steps)
                'label': label,
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
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        nhfe = batch['nhfe'].to(device)  # (B, N_Ch, N_Bands, Time_Steps)
        labels = batch['label'].to(device)  # (B,)
        
        optimizer.zero_grad()
        logits = model(nhfe)  # (B, N_Ch)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            nhfe = batch['nhfe'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(nhfe)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


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
    weight_decay = config.get('training', {}).get('weight_decay', 1e-4)  # L2 regularization
    
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
        train_loader = DataLoader(debug_dataset, batch_size=10, shuffle=False, num_workers=0)
        val_loader = train_loader  # Same for validation
        
        # Create model
        print("\n[2/4] Creating model...")
        model = STGNN_SOZ_Locator(
            n_channels=n_channels,
            n_bands=n_bands,
            time_steps=time_steps
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer with weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop for debug
        print("\n[3/4] Training (DEBUG_OVERFIT mode)...")
        print("Expected: Loss should drop to ~0.00 within 50 epochs if model works correctly")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            
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
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            
            print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
            # Re-initialize model for each fold (CRITICAL!)
            model = STGNN_SOZ_Locator(
                n_channels=n_channels,
                n_bands=n_bands,
                time_steps=time_steps
            ).to(device)
            
            # Loss and optimizer with weight decay
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # Early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(train_loss)
                
                if epoch % 10 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Evaluate on test sample
            test_loss, test_acc = validate(model, test_loader, criterion, device)
            accuracies.append(test_acc)
            print(f"Fold {fold+1} Test Acc: {test_acc:.4f}")
        
        # Print LOOCV results
        print("\n" + "=" * 80)
        print("LOOCV Results")
        print("=" * 80)
        print(f"Individual accuracies: {accuracies}")
        print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Best: {np.max(accuracies):.4f}, Worst: {np.min(accuracies):.4f}")
        
        # Save results
        results = {
            'method': 'LOOCV',
            'total_samples': total_samples,
            'individual_accuracies': accuracies,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'best_accuracy': float(np.max(accuracies)),
            'worst_accuracy': float(np.min(accuracies))
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print("\n[2/4] Creating model...")
    model = STGNN_SOZ_Locator(
        n_channels=n_channels,
        n_bands=n_bands,
        time_steps=time_steps
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n[3/4] Training...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'history': history
            }, output_path / 'best_model.pth')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    # Test
    print("\n[4/4] Testing...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
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
            '--csv_path', r'E:/output/segment_results/nhfe_channel_label.csv',
            '--data_root', 'E:/output/segment_results',
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

