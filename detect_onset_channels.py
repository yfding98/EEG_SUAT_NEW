#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to detect seizure onset channels based on NHFE threshold crossing.

For each patient:
1. Normalize NHFE for all channels
2. Compute adaptive threshold for each channel
3. Find threshold crossing time for each channel
4. Identify channels that cross threshold within 500ms of the earliest crossing
5. Print results
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import NHFEDataLoader, load_patient_labels, PatientData
from features.normalization import normalize_nhfe, compute_adaptive_threshold


def find_threshold_crossing(
    nhfe_norm: np.ndarray,
    threshold: float,
    baseline_end_idx: int
) -> Optional[int]:
    """
    Find first index where NHFE crosses threshold (after baseline).
    
    Args:
        nhfe_norm: Normalized NHFE values
        threshold: Threshold value
        baseline_end_idx: End index of baseline period
    
    Returns:
        Index of first crossing, or None if no crossing
    """
    above_threshold = nhfe_norm > threshold
    
    # Only look after baseline period
    above_threshold[:baseline_end_idx] = False
    
    if np.any(above_threshold):
        return int(np.argmax(above_threshold))
    return None


def detect_onset_channels(
    patient_data: PatientData,
    baseline_duration: float = 10.0,
    onset_window_ms: float = 500.0,
    threshold_percentile: float = 99.5,
    fallback_threshold: float = 3.0,
    use_raw_nhfe: bool = False,
    manual_threshold: Optional[float] = None,
    band_idx: Optional[int] = None
) -> Tuple[List[str], Dict[str, float], Dict[str, Optional[int]]]:
    """
    Detect onset channels for a patient based on threshold crossing.
    
    Args:
        patient_data: PatientData object (can contain multi-band data)
        baseline_duration: Baseline duration in seconds
        onset_window_ms: Window for grouping onset channels (milliseconds)
        threshold_percentile: Percentile for adaptive threshold (only used if use_raw_nhfe=False)
        fallback_threshold: Fallback threshold if adaptive fails (only used if use_raw_nhfe=False)
        use_raw_nhfe: If True, use raw NHFE values without normalization
        manual_threshold: Manual threshold value (required if use_raw_nhfe=True)
        band_idx: Band index to use (if None and multi-band, will process all bands)
    
    Returns:
        Tuple of:
        - List of onset channel names
        - Dictionary mapping channel -> crossing time (seconds)
        - Dictionary mapping channel -> crossing index
    """
    sampling_rate = patient_data.sampling_rate
    baseline_end_idx = int(baseline_duration * sampling_rate)
    
    # Check if multi-band data
    is_multi_band = patient_data.nhfe_data.ndim == 3
    
    if is_multi_band and band_idx is None:
        # Process all bands and find earliest onset across all bands
        return detect_onset_channels_multi_band(
            patient_data,
            baseline_duration=baseline_duration,
            onset_window_ms=onset_window_ms,
            threshold_percentile=threshold_percentile,
            fallback_threshold=fallback_threshold,
            use_raw_nhfe=use_raw_nhfe,
            manual_threshold=manual_threshold
        )
    
    # Single band processing (or specific band_idx)
    channel_crossing_times = {}  # channel -> time in seconds
    channel_crossing_indices = {}  # channel -> index
    
    # Process each channel
    for i, ch_name in enumerate(patient_data.channel_names):
        if is_multi_band:
            if band_idx is None:
                band_idx = 0  # Default to first band
            nhfe_channel = patient_data.nhfe_data[i, band_idx, :]
        else:
            nhfe_channel = patient_data.nhfe_data[i, :]
        
        if use_raw_nhfe:
            # Mode 2: Use raw NHFE values with manual threshold
            if manual_threshold is None:
                raise ValueError("manual_threshold must be provided when use_raw_nhfe=True")
            
            nhfe_data = nhfe_channel  # Use raw values
            threshold = manual_threshold
            
        else:
            # Mode 1: Normalize NHFE and use adaptive threshold
            nhfe_norm, median_base, mad_base = normalize_nhfe(
                nhfe_channel,
                baseline_start=0,
                baseline_end=baseline_end_idx
            )
            
            # Compute threshold
            baseline_data = nhfe_norm[:baseline_end_idx]
            threshold = compute_adaptive_threshold(
                baseline_data,
                percentile=threshold_percentile,
                fallback_threshold=fallback_threshold
            )
            
            nhfe_data = nhfe_norm
        
        # Find threshold crossing
        crossing_idx = find_threshold_crossing(
            nhfe_data,
            threshold,
            baseline_end_idx
        )
        
        if crossing_idx is not None:
            crossing_time = crossing_idx / sampling_rate
            channel_crossing_times[ch_name] = crossing_time
            channel_crossing_indices[ch_name] = crossing_idx
        else:
            channel_crossing_times[ch_name] = None
            channel_crossing_indices[ch_name] = None
    
    # Step 1: Find t_active for each channel (first threshold crossing time)
    # channel_crossing_times already contains t_active for each channel
    
    # Step 2: Find t_active_min (minimum t_active among all channels)
    valid_crossings = {
        ch: time for ch, time in channel_crossing_times.items()
        if time is not None
    }
    
    if len(valid_crossings) == 0:
        # No channels crossed threshold
        return [], channel_crossing_times, channel_crossing_indices
    
    t_active_min = min(valid_crossings.values())
    
    # Step 3: Find all channels where t_active - t_active_min <= 500ms
    onset_channels = []
    
    for ch_name, t_active in valid_crossings.items():
        if t_active is not None:
            time_diff_ms = (t_active - t_active_min) * 1000  # Convert to milliseconds
            if time_diff_ms <= onset_window_ms:
                onset_channels.append(ch_name)
    
    # Sort onset channels by crossing time (t_active)
    onset_channels.sort(key=lambda x: (
        channel_crossing_times[x] if channel_crossing_times[x] is not None else float('inf')
    ))
    
    return onset_channels, channel_crossing_times, channel_crossing_indices


def detect_onset_channels_multi_band(
    patient_data: PatientData,
    baseline_duration: float = 10.0,
    onset_window_ms: float = 500.0,
    threshold_percentile: float = 99.5,
    fallback_threshold: float = 3.0,
    use_raw_nhfe: bool = False,
    manual_threshold: Optional[float] = None
) -> Tuple[List[str], Dict[str, float], Dict[str, Optional[int]]]:
    """
    Detect onset channels across all frequency bands.
    
    For each band, finds active channels, then finds the earliest onset time
    across all bands and channels.
    
    Args:
        patient_data: PatientData object with multi-band NHFE data
        baseline_duration: Baseline duration in seconds
        onset_window_ms: Window for grouping onset channels (milliseconds)
        threshold_percentile: Percentile for adaptive threshold
        fallback_threshold: Fallback threshold if adaptive fails
        use_raw_nhfe: If True, use raw NHFE values without normalization
        manual_threshold: Manual threshold value (required if use_raw_nhfe=True)
    
    Returns:
        Tuple of:
        - List of onset channel names (across all bands)
        - Dictionary mapping channel -> crossing time (seconds)
        - Dictionary mapping channel -> crossing index
    """
    sampling_rate = patient_data.sampling_rate
    baseline_end_idx = int(baseline_duration * sampling_rate)
    n_bands = patient_data.n_bands
    band_names = patient_data.band_names
    
    # Collect crossing times for all channels across all bands
    # Structure: {channel_name: {band_name: (time, index)}}
    all_crossings = {}  # channel -> list of (band_name, time, index)
    
    # Process each band
    for band_idx in range(n_bands):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f'band_{band_idx}'
        
        # Process each channel in this band
        for i, ch_name in enumerate(patient_data.channel_names):
            nhfe_channel = patient_data.nhfe_data[i, band_idx, :]
            
            if use_raw_nhfe:
                if manual_threshold is None:
                    raise ValueError("manual_threshold must be provided when use_raw_nhfe=True")
                nhfe_data = nhfe_channel
                threshold = manual_threshold
            else:
                nhfe_norm, _, _ = normalize_nhfe(
                    nhfe_channel,
                    baseline_start=0,
                    baseline_end=baseline_end_idx
                )
                baseline_data = nhfe_norm[:baseline_end_idx]
                threshold = compute_adaptive_threshold(
                    baseline_data,
                    percentile=threshold_percentile,
                    fallback_threshold=fallback_threshold
                )
                nhfe_data = nhfe_norm
            
            # Find threshold crossing
            crossing_idx = find_threshold_crossing(
                nhfe_data,
                threshold,
                baseline_end_idx
            )
            
            if crossing_idx is not None:
                crossing_time = crossing_idx / sampling_rate
                if ch_name not in all_crossings:
                    all_crossings[ch_name] = []
                all_crossings[ch_name].append((band_name, crossing_time, crossing_idx))
    
    # Find earliest crossing time across all channels and bands
    if len(all_crossings) == 0:
        return [], {}, {}
    
    # Get the earliest time for each channel (across all bands)
    channel_earliest_times = {}
    channel_earliest_indices = {}
    channel_earliest_bands = {}
    
    for ch_name, crossings in all_crossings.items():
        # Find earliest crossing for this channel across all bands
        earliest = min(crossings, key=lambda x: x[1])  # x[1] is time
        band_name, time, idx = earliest
        channel_earliest_times[ch_name] = time
        channel_earliest_indices[ch_name] = idx
        channel_earliest_bands[ch_name] = band_name
    
    # Find t_active_min (minimum across all channels)
    t_active_min = min(channel_earliest_times.values())
    
    # Find all channels where t_active - t_active_min <= onset_window_ms
    onset_channels = []
    channel_crossing_times = {}
    channel_crossing_indices = {}
    
    for ch_name, t_active in channel_earliest_times.items():
        time_diff_ms = (t_active - t_active_min) * 1000
        if time_diff_ms <= onset_window_ms:
            onset_channels.append(ch_name)
            channel_crossing_times[ch_name] = t_active
            channel_crossing_indices[ch_name] = channel_earliest_indices[ch_name]
    
    # Sort onset channels by crossing time
    onset_channels.sort(key=lambda x: channel_crossing_times[x])
    
    return onset_channels, channel_crossing_times, channel_crossing_indices


def main():
    """Main function."""
    print("=" * 80)
    print("NHFE Threshold-based Onset Channel Detection")
    print("=" * 80)
    
    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n[1/3] Loading data...")
    # Load data
    data_loader = NHFEDataLoader(
        data_root=config['data']['data_root'],
        target_band=config['features']['target_band'],
        window_size=config['features']['window_size']
    )
    
    patients = data_loader.load_all_patients(
        pattern=config['data']['pattern']
    )
    print(f"Loaded {len(patients)} patients")
    
    # Load labels for comparison
    labels = load_patient_labels(config['data']['labels_path'])
    print(f"Loaded labels for {len(labels)} patients")
    
    # Get configuration
    baseline_duration = config['features'].get('baseline_duration', 10.0)
    onset_window_ms = config['features'].get('onset_window_ms', 200.0)
    
    # Detection mode configuration
    detection_config = config.get('detection', {})
    use_raw_nhfe = detection_config.get('use_raw_nhfe', False)
    manual_threshold = detection_config.get('manual_threshold', None)
    threshold_percentile = detection_config.get('threshold_percentile', 99.5)
    fallback_threshold = detection_config.get('fallback_threshold', 3.0)
    
    print("\n[2/3] Detecting onset channels...")
    print(f"Baseline duration: {baseline_duration}s")
    print(f"Onset window: {onset_window_ms}ms")
    
    if use_raw_nhfe:
        print(f"Mode: Raw NHFE with manual threshold")
        if manual_threshold is None:
            print("Error: manual_threshold must be set when use_raw_nhfe=True")
            return
        print(f"Manual threshold: {manual_threshold}")
    else:
        print(f"Mode: Normalized NHFE with adaptive threshold")
        print(f"Threshold percentile: {threshold_percentile}")
        print(f"Fallback threshold: {fallback_threshold}")
    
    print("=" * 80)
    
    # Process each patient
    results = []
    
    for patient_id, patient_data in patients.items():
        print(f"\nPatient: {patient_id}")
        print(f"  Channels: {len(patient_data.channel_names)}")
        print(f"  Time points: {patient_data.n_timepoints}")
        print(f"  Duration: {patient_data.duration_seconds:.2f}s")
        print(f"  Sampling rate: {patient_data.sampling_rate}Hz")
        
        # Detect onset channels
        onset_channels, crossing_times, crossing_indices = detect_onset_channels(
            patient_data,
            baseline_duration=baseline_duration,
            onset_window_ms=onset_window_ms,
            threshold_percentile=threshold_percentile,
            fallback_threshold=fallback_threshold,
            use_raw_nhfe=use_raw_nhfe,
            manual_threshold=manual_threshold
        )
        
        # Get ground truth labels
        gt_channels = labels.get(patient_id, [])
        
        # Print results
        if len(onset_channels) > 0:
            print(f"  ✓ Detected onset channels ({len(onset_channels)}): {', '.join(onset_channels)}")
            
            # Print crossing times
            print(f"  Crossing times:")
            for ch in onset_channels:
                time = crossing_times[ch]
                idx = crossing_indices[ch]
                print(f"    {ch}: {time:.3f}s (index {idx})")
            
            # Compare with ground truth
            if len(gt_channels) > 0:
                detected_set = set(onset_channels)
                gt_set = set(gt_channels)
                
                correct = detected_set & gt_set
                missed = gt_set - detected_set
                false_positives = detected_set - gt_set
                
                print(f"  Ground truth: {', '.join(gt_channels)}")
                if len(correct) > 0:
                    print(f"  ✓ Correct: {', '.join(correct)}")
                if len(missed) > 0:
                    print(f"  ✗ Missed: {', '.join(missed)}")
                if len(false_positives) > 0:
                    print(f"  ✗ False positives: {', '.join(false_positives)}")
            else:
                print(f"  (No ground truth labels available)")
        else:
            print(f"  ✗ No onset channels detected")
            if len(gt_channels) > 0:
                print(f"  Ground truth: {', '.join(gt_channels)}")
        if not onset_channels:
            print("  (No onset channels detected)")
        # Store results
        results.append({
            'patient_id': patient_id,
            'detected_channels': ', '.join(onset_channels) if onset_channels else 'None',
            'num_detected': len(onset_channels),
            'gt_channels': ', '.join(gt_channels) if gt_channels else 'None',
            'num_gt': len(gt_channels),
            'correct': len(set(onset_channels) & set(gt_channels)) if gt_channels else 0,
            'missed': len(set(gt_channels) - set(onset_channels)) if gt_channels else 0,
            'false_positives': len(set(onset_channels) - set(gt_channels)) if gt_channels else 0
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("[3/3] Summary")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    print(f"\nTotal patients: {len(df)}")
    print(f"Patients with detected channels: {len(df[df['num_detected'] > 0])}")
    print(f"Patients with ground truth: {len(df[df['num_gt'] > 0])}")
    
    if len(df[df['num_gt'] > 0]) > 0:
        print(f"\nDetection Performance:")
        print(f"  Total correct: {df['correct'].sum()}")
        print(f"  Total missed: {df['missed'].sum()}")
        print(f"  Total false positives: {df['false_positives'].sum()}")
        
        # Calculate metrics
        total_gt = df['num_gt'].sum()
        total_detected = df['num_detected'].sum()
        total_correct = df['correct'].sum()
        
        if total_gt > 0:
            recall = total_correct / total_gt
            print(f"  Recall: {recall:.4f} ({total_correct}/{total_gt})")
        
        if total_detected > 0:
            precision = total_correct / total_detected
            print(f"  Precision: {precision:.4f} ({total_correct}/{total_detected})")
        
        if total_correct > 0:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  F1-score: {f1:.4f}")
    
    # Save results to CSV
    output_path = Path("onset_detection_results.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")
    
    # Print detailed table
    print("\n" + "=" * 80)
    print("Detailed Results Table")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print("\nDone!")


if __name__ == "__main__":
    main()

