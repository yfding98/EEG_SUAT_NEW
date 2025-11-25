#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHFE Data Loader

Loads NHFE time-series data from NPZ files or CSV files.
Supports per-patient data loading with metadata.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class PatientData:
    """Container for a single patient's NHFE data."""
    
    def __init__(
        self,
        patient_id: str,
        nhfe_data: np.ndarray,
        channel_names: List[str],
        sampling_rate: float = 250.0,
        window_size: float = 0.25,
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            patient_id: Unique patient identifier
            nhfe_data: NHFE time-series array of shape (n_channels, n_timepoints)
            channel_names: List of channel names
            sampling_rate: Sampling rate in Hz
            window_size: Time window size in seconds (default 250ms = 0.25s)
            metadata: Additional metadata dictionary
        """
        self.patient_id = patient_id
        self.nhfe_data = nhfe_data
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.metadata = metadata or {}
        
        # Validate dimensions
        if nhfe_data.ndim != 2:
            raise ValueError(f"NHFE data must be 2D (n_channels, n_timepoints), got {nhfe_data.ndim}D")
        
        if len(channel_names) != nhfe_data.shape[0]:
            raise ValueError(
                f"Channel names count ({len(channel_names)}) doesn't match "
                f"data channels ({nhfe_data.shape[0]})"
            )
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.nhfe_data.shape[0]
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return self.nhfe_data.shape[1]
    
    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.n_timepoints * self.window_size
    
    def get_channel_data(self, channel_name: str) -> np.ndarray:
        """Get NHFE time-series for a specific channel."""
        if channel_name not in self.channel_names:
            raise ValueError(f"Channel '{channel_name}' not found")
        idx = self.channel_names.index(channel_name)
        return self.nhfe_data[idx, :]
    
    def get_time_array(self) -> np.ndarray:
        """Get time array in seconds."""
        return np.arange(self.n_timepoints) * self.window_size


class NHFEDataLoader:
    """Loader for NHFE time-series data from various sources."""
    
    def __init__(
        self,
        data_root: Union[str, Path],
        target_band: str = 'gamma',
        window_size: float = 0.25,
        sampling_rate: float = 250.0
    ):
        """
        Args:
            data_root: Root directory containing NHFE data
            target_band: Target frequency band name (default: 'gamma')
            window_size: Time window size in seconds (default: 0.25s = 250ms)
            sampling_rate: Sampling rate in Hz (default: 250 Hz)
        """
        self.data_root = Path(data_root)
        self.target_band = target_band
        self.window_size = window_size
        self.sampling_rate = sampling_rate
    
    def load_from_npz(
        self,
        npz_path: Union[str, Path],
        patient_id: Optional[str] = None
    ) -> PatientData:
        """
        Load NHFE data from NPZ file (BEI.npz format).
        
        Args:
            npz_path: Path to BEI.npz file
            patient_id: Patient ID (if None, extracted from path)
        
        Returns:
            PatientData object
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        # Extract patient ID from path if not provided
        if patient_id is None:
            patient_id = self._extract_patient_id(npz_path)
        
        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract NHFE data (n_channels, n_bands, n_samples)
        if 'NHFE' not in data:
            raise ValueError(f"NPZ file does not contain 'NHFE' key: {npz_path}")
        
        nhfe_all_bands = data['NHFE']
        band_names = data.get('band_names', np.array([]))
        if hasattr(band_names, 'tolist'):
            band_names = band_names.tolist()
        
        ch_names = data.get('ch_names', np.array([]))
        if hasattr(ch_names, 'tolist'):
            ch_names = ch_names.tolist()
        
        # Get sampling rate and window size from metadata if available
        sfreq = float(
            data.get('sfreq', [self.sampling_rate])[0]
            if hasattr(data.get('sfreq', self.sampling_rate), '__len__')
            else data.get('sfreq', self.sampling_rate)
        )
        window_size = float(
            data.get('window_size', [self.window_size])[0]
            if hasattr(data.get('window_size', self.window_size), '__len__')
            else data.get('window_size', self.window_size)
        )
        
        # Select target band
        if len(band_names) > 0 and self.target_band in band_names:
            band_idx = band_names.index(self.target_band)
        elif nhfe_all_bands.shape[1] > 0:
            # Use first band if target not found
            band_idx = 0
            if len(band_names) > 0:
                print(f"Warning: Target band '{self.target_band}' not found, using '{band_names[0]}'")
        else:
            raise ValueError("No frequency bands found in NPZ file")
        
        # Extract NHFE for target band: (n_channels, n_timepoints)
        nhfe_sequence = nhfe_all_bands[:, band_idx, :]
        
        # Create channel names if missing
        if len(ch_names) == 0:
            ch_names = [f'Ch{i+1}' for i in range(nhfe_sequence.shape[0])]
        
        # Build metadata
        metadata = {
            'band_names': band_names,
            'target_band': band_names[band_idx] if len(band_names) > 0 else 'unknown',
            'band_idx': band_idx,
            'npz_path': str(npz_path),
            'time_points': data.get('times', None)
        }
        
        return PatientData(
            patient_id=patient_id,
            nhfe_data=nhfe_sequence,
            channel_names=ch_names,
            sampling_rate=sfreq,
            window_size=window_size,
            metadata=metadata
        )
    
    def load_from_csv(
        self,
        csv_path: Union[str, Path],
        patient_id_col: str = 'patient_id',
        channel_col: str = 'channel',
        time_col: str = 'time',
        nhfe_col: str = 'nhfe'
    ) -> PatientData:
        """
        Load NHFE data from CSV file.
        
        Expected CSV format:
        - patient_id: Patient identifier
        - channel: Channel name
        - time: Time point (seconds or index)
        - nhfe: NHFE value
        
        Args:
            csv_path: Path to CSV file
            patient_id_col: Column name for patient ID
            channel_col: Column name for channel names
            time_col: Column name for time points
            nhfe_col: Column name for NHFE values
        
        Returns:
            PatientData object
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = [patient_id_col, channel_col, time_col, nhfe_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get unique patient IDs
        patient_ids = df[patient_id_col].unique()
        if len(patient_ids) > 1:
            raise ValueError(
                f"CSV contains multiple patients ({len(patient_ids)}). "
                "Load one patient at a time."
            )
        
        patient_id = str(patient_ids[0])
        
        # Get unique channels and time points
        channels = sorted(df[channel_col].unique())
        time_points = sorted(df[time_col].unique())
        
        # Create 2D array: (n_channels, n_timepoints)
        nhfe_data = np.zeros((len(channels), len(time_points)))
        
        for i, ch in enumerate(channels):
            ch_data = df[df[channel_col] == ch].sort_values(time_col)
            nhfe_data[i, :] = ch_data[nhfe_col].values
        
        metadata = {
            'csv_path': str(csv_path),
            'time_points': time_points
        }
        
        return PatientData(
            patient_id=patient_id,
            nhfe_data=nhfe_data,
            channel_names=channels,
            sampling_rate=self.sampling_rate,
            window_size=self.window_size,
            metadata=metadata
        )
    
    def load_all_patients(
        self,
        patient_list: Optional[List[str]] = None,
        pattern: str = "**/*BEI.npz"
    ) -> Dict[str, PatientData]:
        """
        Load NHFE data for all patients.
        
        Args:
            patient_list: Optional list of patient IDs to load
            pattern: Glob pattern to find NPZ files (default: "**/*BEI.npz")
        
        Returns:
            Dictionary mapping patient_id -> PatientData
        """
        npz_files = list(self.data_root.glob(pattern))
        
        if len(npz_files) == 0:
            raise FileNotFoundError(
                f"No NPZ files found matching pattern '{pattern}' in {self.data_root}"
            )
        
        patients = {}
        for npz_path in npz_files:
            try:
                patient_id = self._extract_patient_id(npz_path)
                
                # Filter by patient_list if provided
                if patient_list is not None and patient_id not in patient_list:
                    continue
                
                patient_data = self.load_from_npz(npz_path, patient_id)
                patients[patient_id] = patient_data
                
            except Exception as e:
                print(f"Warning: Failed to load {npz_path}: {e}")
                continue
        
        return patients
    
    def _extract_patient_id(self, path: Path) -> str:
        """Extract patient ID from file path."""
        # Try to extract from path structure
        parts = path.parts
        for part in reversed(parts):
            if part != path.name and part != path.stem:
                # Common patterns: patient_name, P001, etc.
                if len(part) > 0 and not part.startswith('.'):
                    return part
        
        # Fallback: use stem of filename
        return path.stem.replace('_BEI', '').replace('_filtered', '')


def load_patient_labels(
    labels_path: Union[str, Path],
    patient_id_col: str = 'patient_id',
    channel_col: str = 'channel',
    label_col: str = 'is_onset'
) -> Dict[str, List[str]]:
    """
    Load ground truth seizure onset channel labels.
    
    Args:
        labels_path: Path to CSV file with labels
        patient_id_col: Column name for patient ID
        channel_col: Column name for channel names
        label_col: Column name for binary label (1 = onset, 0 = non-onset)
    
    Returns:
        Dictionary mapping patient_id -> list of onset channel names
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    
    required_cols = [patient_id_col, channel_col, label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter for onset channels (label == 1)
    onset_df = df[df[label_col] == 1]
    
    # Group by patient
    labels = {}
    for patient_id, group in onset_df.groupby(patient_id_col):
        onset_channels = group[channel_col].tolist()
        labels[str(patient_id)] = onset_channels
    
    return labels

