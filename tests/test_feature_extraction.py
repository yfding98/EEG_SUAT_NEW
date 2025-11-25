#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for feature extraction module.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.normalization import log_transform, normalize_nhfe, compute_adaptive_threshold
from features.extractor import NHFEFeatureExtractor


def test_log_transform():
    """Test log-transform function."""
    # Test with positive values
    nhfe = np.array([1.0, 10.0, 100.0, 1000.0])
    nhfe_log = log_transform(nhfe)
    
    assert nhfe_log.shape == nhfe.shape
    assert np.all(nhfe_log >= 0)
    assert nhfe_log[0] < nhfe_log[-1]  # Monotonic
    
    # Test with zeros
    nhfe_zero = np.array([0.0, 1.0, 2.0])
    nhfe_log_zero = log_transform(nhfe_zero)
    assert nhfe_log_zero[0] == 0.0


def test_normalize_nhfe():
    """Test NHFE normalization."""
    # Create synthetic NHFE data with baseline and onset
    n_samples = 100
    baseline_end = 40
    
    # Baseline: low values
    baseline = np.random.lognormal(mean=0, sigma=0.5, size=baseline_end)
    
    # Onset: high values
    onset = np.random.lognormal(mean=3, sigma=0.5, size=n_samples - baseline_end)
    
    nhfe = np.concatenate([baseline, onset])
    
    # Normalize
    nhfe_norm, median_base, mad_base = normalize_nhfe(
        nhfe,
        baseline_start=0,
        baseline_end=baseline_end
    )
    
    assert nhfe_norm.shape == nhfe.shape
    assert not np.any(np.isnan(nhfe_norm))
    assert not np.any(np.isinf(nhfe_norm))
    
    # Baseline should be centered around 0
    baseline_norm = nhfe_norm[:baseline_end]
    assert np.abs(np.median(baseline_norm)) < 1.0
    
    # Onset should be positive (above baseline)
    onset_norm = nhfe_norm[baseline_end:]
    assert np.mean(onset_norm) > np.mean(baseline_norm)


def test_adaptive_threshold():
    """Test adaptive threshold computation."""
    # Create synthetic data
    n_samples = 100
    baseline_end = 40
    
    baseline = np.random.normal(0, 1, baseline_end)
    onset = np.random.normal(5, 1, n_samples - baseline_end)
    nhfe_norm = np.concatenate([baseline, onset])
    
    threshold = compute_adaptive_threshold(
        nhfe_norm,
        baseline_start=0,
        baseline_end=baseline_end
    )
    
    assert threshold > 0
    assert threshold > np.percentile(baseline, 50)  # Should be above median


def test_feature_extraction():
    """Test feature extraction for a single channel."""
    # Create synthetic NHFE data
    n_samples = 120
    baseline_end = 40
    onset_start = 50
    
    # Baseline: low values
    baseline = np.random.lognormal(mean=0, sigma=0.3, size=baseline_end)
    
    # Transition
    transition = np.linspace(
        np.random.lognormal(0, 0.3),
        np.random.lognormal(2, 0.3),
        onset_start - baseline_end
    )
    
    # Onset: high values
    onset = np.random.lognormal(mean=2.5, sigma=0.3, size=n_samples - onset_start)
    
    nhfe = np.concatenate([baseline, transition, onset])
    
    # Extract features
    extractor = NHFEFeatureExtractor(
        baseline_duration=10.0,
        window_size=0.25
    )
    
    features = extractor.extract_features(
        nhfe,
        channel_name="Ch1"
    )
    
    # Check required features exist
    required_features = [
        'threshold_crossing_time',
        'peak_nhfe_norm',
        'slope_onset',
        'area_under_curve',
        'stability_duration',
        'mean_nhfe_norm',
        'std_nhfe_norm'
    ]
    
    for feat_name in required_features:
        assert feat_name in features
    
    # Check feature values are reasonable
    assert features['peak_nhfe_norm'] > 0
    assert features['mean_nhfe_norm'] is not None
    
    # If threshold was crossed, crossing time should be valid
    if not np.isnan(features['threshold_crossing_time']):
        assert features['threshold_crossing_time'] >= 0
        assert features['threshold_crossing_time'] < n_samples * 0.25


def test_multi_channel_extraction():
    """Test feature extraction for multiple channels."""
    n_channels = 5
    n_samples = 120
    
    # Create synthetic multi-channel data
    nhfe_data = np.random.lognormal(
        mean=1.0,
        sigma=0.5,
        size=(n_channels, n_samples)
    )
    
    channel_names = [f"Ch{i+1}" for i in range(n_channels)]
    
    extractor = NHFEFeatureExtractor(
        baseline_duration=10.0,
        window_size=0.25
    )
    
    features_list, threshold = extractor.extract_all_channels(
        nhfe_data,
        channel_names
    )
    
    assert len(features_list) == n_channels
    assert threshold > 0
    
    for i, feat in enumerate(features_list):
        assert feat['channel_name'] == channel_names[i]
        assert 'peak_nhfe_norm' in feat


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

