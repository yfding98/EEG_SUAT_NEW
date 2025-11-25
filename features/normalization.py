#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHFE Normalization Module

Implements log-transform and robust baseline normalization using median and MAD.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


def log_transform(nhfe: np.ndarray) -> np.ndarray:
    """
    Apply log1p transform to NHFE values.
    
    Args:
        nhfe: NHFE values (can be any shape)
    
    Returns:
        Log-transformed values: log1p(nhfe)
    """
    return np.log1p(np.maximum(nhfe, 0))  # Ensure non-negative


def compute_baseline_stats(
    nhfe_log: np.ndarray,
    baseline_mask: Optional[np.ndarray] = None,
    baseline_start: Optional[int] = None,
    baseline_end: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute robust baseline statistics (median and MAD).
    
    Args:
        nhfe_log: Log-transformed NHFE values (1D array)
        baseline_mask: Boolean mask for baseline period (if None, uses baseline_start/end)
        baseline_start: Start index of baseline period
        baseline_end: End index of baseline period (exclusive)
    
    Returns:
        Tuple of (median, MAD) for baseline period
    """
    if baseline_mask is not None:
        baseline_data = nhfe_log[baseline_mask]
    elif baseline_start is not None and baseline_end is not None:
        baseline_data = nhfe_log[baseline_start:baseline_end]
    else:
        # Use first 10 seconds as baseline (default)
        baseline_end = min(40, len(nhfe_log))  # 10s at 4 Hz (250ms windows)
        baseline_data = nhfe_log[:baseline_end]
    
    if len(baseline_data) == 0:
        raise ValueError("Baseline period is empty")
    
    median_base = np.median(baseline_data)
    mad_base = np.median(np.abs(baseline_data - median_base))
    
    return median_base, mad_base


def normalize_nhfe(
    nhfe: np.ndarray,
    baseline_mask: Optional[np.ndarray] = None,
    baseline_start: Optional[int] = None,
    baseline_end: Optional[int] = None,
    median_base: Optional[float] = None,
    mad_base: Optional[float] = None,
    eps: float = 1e-6
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize NHFE using log-transform and robust baseline normalization.
    
    Steps:
    1. Log-transform: log1p(NHFE)
    2. Compute baseline stats (if not provided)
    3. Normalize: (NHFE_log - median_base) / (MAD_base + eps)
    
    Args:
        nhfe: Raw NHFE values (1D array)
        baseline_mask: Boolean mask for baseline period
        baseline_start: Start index of baseline period
        baseline_end: End index of baseline period
        median_base: Pre-computed baseline median (if None, computed from data)
        mad_base: Pre-computed baseline MAD (if None, computed from data)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Tuple of (normalized_nhfe, median_base, mad_base)
    """
    # Step 1: Log-transform
    nhfe_log = log_transform(nhfe)
    
    # Step 2: Compute baseline stats if not provided
    if median_base is None or mad_base is None:
        median_base, mad_base = compute_baseline_stats(
            nhfe_log,
            baseline_mask=baseline_mask,
            baseline_start=baseline_start,
            baseline_end=baseline_end
        )
    
    # Step 3: Normalize
    nhfe_norm = (nhfe_log - median_base) / (mad_base + eps)
    
    return nhfe_norm, median_base, mad_base


def compute_adaptive_threshold(
    nhfe_norm: np.ndarray,
    baseline_mask: Optional[np.ndarray] = None,
    baseline_start: Optional[int] = None,
    baseline_end: Optional[int] = None,
    percentile: float = 99.5,
    fallback_threshold: float = 3.0
) -> float:
    """
    Compute adaptive threshold from baseline period.
    
    Uses percentile of baseline distribution, with fallback to fixed threshold.
    
    Args:
        nhfe_norm: Normalized NHFE values (1D array)
        baseline_mask: Boolean mask for baseline period
        baseline_start: Start index of baseline period
        baseline_end: End index of baseline period
        percentile: Percentile to use for threshold (default: 99.5)
        fallback_threshold: Fallback threshold in MAD units (default: 3.0)
    
    Returns:
        Threshold value
    """
    if baseline_mask is not None:
        baseline_data = nhfe_norm[baseline_mask]
    elif baseline_start is not None and baseline_end is not None:
        baseline_data = nhfe_norm[baseline_start:baseline_end]
    else:
        baseline_end = min(40, len(nhfe_norm))
        baseline_data = nhfe_norm[:baseline_end]
    
    if len(baseline_data) == 0:
        return fallback_threshold
    
    # Compute percentile threshold
    th_percentile = np.percentile(baseline_data, percentile)
    
    # Use the maximum of percentile and fallback
    threshold = max(th_percentile, fallback_threshold)
    
    return threshold

