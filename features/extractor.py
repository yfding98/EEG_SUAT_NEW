#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHFE Feature Extractor

Extracts features from normalized NHFE time-series for seizure onset detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import signal, stats
from features.normalization import normalize_nhfe, compute_adaptive_threshold


class NHFEFeatureExtractor:
    """Extract features from NHFE time-series for onset detection."""
    
    def __init__(
        self,
        baseline_duration: float = 10.0,
        window_size: float = 0.25,
        sampling_rate: float = 250.0,
        onset_window_ms: float = 500.0,
        stability_window: float = 2.0
    ):
        """
        Args:
            baseline_duration: Baseline duration in seconds (default: 10s)
            window_size: Time window size in seconds for NHFE calculation (default: 0.25s = 250ms)
            sampling_rate: Sampling rate in Hz (default: 250 Hz)
                Note: NHFE time points = original data time points (one per sample)
            onset_window_ms: Window for multi-label onset grouping in milliseconds (default: 500ms)
            stability_window: Window for stability feature in seconds (default: 2s)
        """
        self.baseline_duration = baseline_duration
        self.window_size = window_size  # Window size for NHFE calculation (250ms)
        self.sampling_rate = sampling_rate  # Original data sampling rate
        self.onset_window_ms = onset_window_ms
        self.stability_window = stability_window
        
        # Compute baseline indices
        # NHFE time points = original sampling points, so each point = 1/sampling_rate seconds
        # baseline_end_idx = baseline_duration * sampling_rate
        self.baseline_end_idx = int(baseline_duration * sampling_rate)
    
    def extract_features(
        self,
        nhfe_data: np.ndarray,
        channel_name: str,
        threshold: Optional[float] = None,
        compute_threshold: bool = True,
        use_raw_nhfe: bool = False
    ) -> Dict[str, float]:
        """
        Extract features for a single channel.
        
        Features:
        1. threshold_crossing_time: Time of first threshold crossing
        2. peak_nhfe_norm: Peak normalized NHFE value (or raw peak if use_raw_nhfe=True)
        3. slope_onset: Slope of NHFE around onset
        4. area_under_curve: Area under curve after onset
        5. stability_duration: How long NHFE stays above threshold
        
        Args:
            nhfe_data: Raw NHFE time-series (1D array)
            channel_name: Channel name (for metadata)
            threshold: Threshold value (if None, computed adaptively)
            compute_threshold: Whether to compute adaptive threshold
            use_raw_nhfe: If True, use raw NHFE values without normalization
        
        Returns:
            Dictionary of feature names -> values
        """
        if use_raw_nhfe:
            # Mode 2: Use raw NHFE values
            nhfe_processed = nhfe_data
            median_base = np.median(nhfe_data[:self.baseline_end_idx])
            mad_base = np.median(np.abs(nhfe_data[:self.baseline_end_idx] - median_base))
            
            # Threshold must be provided for raw mode
            if threshold is None:
                raise ValueError("threshold must be provided when use_raw_nhfe=True")
        else:
            # Mode 1: Normalize NHFE
            nhfe_norm, median_base, mad_base = normalize_nhfe(
                nhfe_data,
                baseline_start=0,
                baseline_end=self.baseline_end_idx
            )
            nhfe_processed = nhfe_norm
            
            # Compute threshold if needed
            if threshold is None and compute_threshold:
                threshold = compute_adaptive_threshold(
                    nhfe_norm,
                    baseline_start=0,
                    baseline_end=self.baseline_end_idx
                )
            elif threshold is None:
                threshold = 3.0  # Default fallback
        
        # Extract features
        features = {}
        
        # 1. Threshold crossing time
        crossing_idx = self._find_threshold_crossing(nhfe_processed, threshold)
        if crossing_idx is not None:
            # Each NHFE time point = 1/sampling_rate seconds
            features['threshold_crossing_time'] = crossing_idx / self.sampling_rate
            features['threshold_crossing_idx'] = float(crossing_idx)
        else:
            features['threshold_crossing_time'] = np.nan
            features['threshold_crossing_idx'] = np.nan
        
        # 2. Peak NHFE (normalized or raw)
        features['peak_nhfe_norm'] = float(np.nanmax(nhfe_processed))
        peak_idx = int(np.nanargmax(nhfe_processed))
        features['peak_time'] = peak_idx / self.sampling_rate
        
        # 3. Slope around onset
        if crossing_idx is not None and crossing_idx < len(nhfe_processed) - 1:
            # Compute slope in a window around onset
            # Use ~1.25s window: 1.25s * sampling_rate samples
            slope_window_samples = int(1.25 * self.sampling_rate)
            slope_window = min(slope_window_samples, len(nhfe_processed) - crossing_idx - 1)
            if slope_window > 1:
                y = nhfe_processed[crossing_idx:crossing_idx + slope_window]
                x = np.arange(len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                features['slope_onset'] = float(slope)
                features['slope_r2'] = float(r_value ** 2)
            else:
                features['slope_onset'] = 0.0
                features['slope_r2'] = 0.0
        else:
            features['slope_onset'] = 0.0
            features['slope_r2'] = 0.0
        
        # 4. Area under curve after onset
        if crossing_idx is not None:
            post_onset = nhfe_processed[crossing_idx:]
            # Only count values above threshold
            above_threshold = post_onset[post_onset > threshold]
            if len(above_threshold) > 0:
                features['area_under_curve'] = float(np.trapz(above_threshold - threshold))
            else:
                features['area_under_curve'] = 0.0
        else:
            features['area_under_curve'] = 0.0
        
        # 5. Stability duration (how long it stays high)
        if crossing_idx is not None:
            stability_duration = self._compute_stability_duration(
                nhfe_processed,
                crossing_idx,
                threshold
            )
            features['stability_duration'] = stability_duration
        else:
            features['stability_duration'] = 0.0
        
        # Additional features
        features['mean_nhfe_norm'] = float(np.nanmean(nhfe_processed))
        features['std_nhfe_norm'] = float(np.nanstd(nhfe_processed))
        features['median_nhfe_norm'] = float(np.nanmedian(nhfe_processed))
        
        # Baseline statistics
        baseline_data = nhfe_processed[:self.baseline_end_idx]
        features['baseline_mean'] = float(np.nanmean(baseline_data))
        features['baseline_std'] = float(np.nanstd(baseline_data))
        
        # Ratio of peak to baseline
        if features['baseline_mean'] > 0:
            features['peak_baseline_ratio'] = features['peak_nhfe_norm'] / features['baseline_mean']
        else:
            features['peak_baseline_ratio'] = np.nan
        
        # Time to peak from crossing
        if crossing_idx is not None and not np.isnan(features['threshold_crossing_idx']):
            features['time_to_peak'] = (peak_idx - crossing_idx) / self.sampling_rate
        else:
            features['time_to_peak'] = np.nan
        
        # Metadata
        features['channel_name'] = channel_name
        features['threshold'] = float(threshold)
        features['median_base'] = float(median_base)
        features['mad_base'] = float(mad_base)
        
        return features
    
    def extract_all_channels(
        self,
        nhfe_data: np.ndarray,
        channel_names: List[str],
        threshold: Optional[float] = None,
        use_raw_nhfe: bool = False
    ) -> Tuple[List[Dict[str, float]], float]:
        """
        Extract features for all channels.
        
        Args:
            nhfe_data: NHFE data array of shape (n_channels, n_timepoints)
            channel_names: List of channel names
            threshold: Optional global threshold (if None, computed adaptively)
            use_raw_nhfe: If True, use raw NHFE values without normalization
        
        Returns:
            Tuple of (list of feature dicts, global threshold used)
        """
        all_features = []
        
        # If threshold not provided, compute from first channel as reference
        if threshold is None and not use_raw_nhfe:
            ref_features = self.extract_features(
                nhfe_data[0, :],
                channel_names[0],
                compute_threshold=True,
                use_raw_nhfe=False
            )
            threshold = ref_features['threshold']
        elif threshold is None and use_raw_nhfe:
            raise ValueError("threshold must be provided when use_raw_nhfe=True")
        
        # Extract features for all channels
        for i, ch_name in enumerate(channel_names):
            features = self.extract_features(
                nhfe_data[i, :],
                ch_name,
                threshold=threshold,
                compute_threshold=False,
                use_raw_nhfe=use_raw_nhfe
            )
            all_features.append(features)
        
        return all_features, threshold
    
    def _find_threshold_crossing(
        self,
        nhfe_norm: np.ndarray,
        threshold: float
    ) -> Optional[int]:
        """Find first index where NHFE crosses threshold."""
        above_threshold = nhfe_norm > threshold
        
        # Only look after baseline period
        above_threshold[:self.baseline_end_idx] = False
        
        if np.any(above_threshold):
            return int(np.argmax(above_threshold))
        return None
    
    def _compute_stability_duration(
        self,
        nhfe_norm: np.ndarray,
        crossing_idx: int,
        threshold: float
    ) -> float:
        """
        Compute how long NHFE stays above threshold after crossing.
        
        Args:
            nhfe_norm: Normalized NHFE values
            crossing_idx: Index where threshold was crossed
            threshold: Threshold value
        
        Returns:
            Duration in seconds that NHFE stays above threshold
        """
        post_onset = nhfe_norm[crossing_idx:]
        above_threshold = post_onset > threshold
        
        # Find first time it drops below threshold
        if np.all(above_threshold):
            # Never drops below threshold
            return (len(post_onset) - 1) / self.sampling_rate
        
        # Find first drop
        drop_idx = np.argmin(above_threshold)
        
        # But allow for brief drops (within stability window)
        # stability_samples = stability_window * sampling_rate
        stability_samples = int(self.stability_window * self.sampling_rate)
        
        # Check if it stays high for at least stability_window
        if drop_idx >= stability_samples:
            return drop_idx / self.sampling_rate
        else:
            # Brief drop, continue checking
            # Find next crossing
            remaining = post_onset[drop_idx:]
            if len(remaining) > 0:
                next_above = remaining > threshold
                if np.any(next_above):
                    # Continues above threshold
                    return (len(post_onset) - 1) / self.sampling_rate
            
            return drop_idx / self.sampling_rate
    
    def group_onset_channels(
        self,
        features_list: List[Dict[str, float]],
        channel_names: List[str]
    ) -> List[List[str]]:
        """
        Group channels that cross threshold within onset_window_ms.
        
        This implements multi-label onset detection: if multiple channels
        rise within 500ms, they are treated as equal onset labels.
        
        Args:
            features_list: List of feature dictionaries
            channel_names: List of channel names
        
        Returns:
            List of channel groups (each group is a list of channel names)
        """
        # Get crossing times
        crossing_times = []
        for i, feat in enumerate(features_list):
            if not np.isnan(feat['threshold_crossing_time']):
                crossing_times.append({
                    'channel': channel_names[i],
                    'time': feat['threshold_crossing_time'],
                    'idx': i
                })
        
        if len(crossing_times) == 0:
            return []
        
        # Sort by crossing time
        crossing_times.sort(key=lambda x: x['time'])
        
        # Group channels within onset_window_ms
        groups = []
        current_group = [crossing_times[0]['channel']]
        current_time = crossing_times[0]['time']
        
        for ct in crossing_times[1:]:
            time_diff = (ct['time'] - current_time) * 1000  # Convert to ms
            
            if time_diff <= self.onset_window_ms:
                # Within window, add to current group
                current_group.append(ct['channel'])
            else:
                # New group
                groups.append(current_group)
                current_group = [ct['channel']]
                current_time = ct['time']
        
        # Add last group
        if len(current_group) > 0:
            groups.append(current_group)
        
        return groups

