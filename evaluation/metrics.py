#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for seizure onset detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix
)
from scipy.stats import spearmanr


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities (optional, for AUC)
    
    Returns:
        Dictionary of metric names -> values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    
    # AUC if probabilities provided
    if y_proba is not None:
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Get positive class probabilities
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['auc'] = np.nan
    
    return metrics


def compute_onset_time_error(
    predicted_times: np.ndarray,
    true_times: np.ndarray,
    threshold_crossing_times: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute onset time prediction error.
    
    Args:
        predicted_times: Predicted onset times (for positive predictions)
        true_times: True onset times (for positive labels)
        threshold_crossing_times: Actual threshold crossing times from features
    
    Returns:
        Dictionary with time error metrics
    """
    metrics = {}
    
    if len(predicted_times) == 0 or len(true_times) == 0:
        metrics['mean_absolute_error'] = np.nan
        metrics['mean_squared_error'] = np.nan
        metrics['median_absolute_error'] = np.nan
        return metrics
    
    # Align predictions with true labels
    # This is simplified - in practice, you'd need to match channels
    errors = np.abs(predicted_times - true_times)
    
    metrics['mean_absolute_error'] = float(np.mean(errors))
    metrics['mean_squared_error'] = float(np.mean(errors ** 2))
    metrics['median_absolute_error'] = float(np.median(errors))
    metrics['std_error'] = float(np.std(errors))
    
    return metrics


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute ranking metrics (Top-K accuracy, Spearman correlation).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        channel_names: Optional channel names for per-patient ranking
    
    Returns:
        Dictionary with ranking metrics
    """
    metrics = {}
    
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    
    # Top-K accuracy
    # Sort by probability
    sorted_indices = np.argsort(y_proba)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Top-1 accuracy
    metrics['top1_accuracy'] = float(sorted_labels[0] == 1) if len(sorted_labels) > 0 else 0.0
    
    # Top-3 accuracy (at least one positive in top 3)
    if len(sorted_labels) >= 3:
        metrics['top3_accuracy'] = float(np.any(sorted_labels[:3] == 1))
    else:
        metrics['top3_accuracy'] = float(np.any(sorted_labels == 1))
    
    # Top-5 accuracy
    if len(sorted_labels) >= 5:
        metrics['top5_accuracy'] = float(np.any(sorted_labels[:5] == 1))
    else:
        metrics['top5_accuracy'] = float(np.any(sorted_labels == 1))
    
    # Spearman correlation between probabilities and labels
    try:
        spearman_corr, p_value = spearmanr(y_proba, y_true)
        metrics['spearman_correlation'] = float(spearman_corr)
        metrics['spearman_pvalue'] = float(p_value)
    except:
        metrics['spearman_correlation'] = np.nan
        metrics['spearman_pvalue'] = np.nan
    
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    channel_names: Optional[List[str]] = None,
    predicted_times: Optional[np.ndarray] = None,
    true_times: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities
        channel_names: Optional channel names
        predicted_times: Optional predicted onset times
        true_times: Optional true onset times
    
    Returns:
        Dictionary with all metrics
    """
    all_metrics = {}
    
    # Classification metrics
    class_metrics = evaluate_predictions(y_true, y_pred, y_proba)
    all_metrics.update(class_metrics)
    
    # Ranking metrics
    rank_metrics = compute_ranking_metrics(y_true, y_proba, channel_names)
    all_metrics.update(rank_metrics)
    
    # Time error metrics (if provided)
    if predicted_times is not None and true_times is not None:
        time_metrics = compute_onset_time_error(predicted_times, true_times)
        all_metrics.update(time_metrics)
    
    return all_metrics


def evaluate_per_patient(
    predictions: Dict[str, Dict[str, float]],
    labels: Dict[str, List[str]],
    channel_names: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions per patient.
    
    Args:
        predictions: Dict mapping patient_id -> dict of channel -> probability
        labels: Dict mapping patient_id -> list of onset channel names
        channel_names: Dict mapping patient_id -> list of channel names
    
    Returns:
        Dictionary mapping patient_id -> metrics
    """
    per_patient_metrics = {}
    
    for patient_id in predictions.keys():
        if patient_id not in labels or patient_id not in channel_names:
            continue
        
        onset_channels = set(labels[patient_id])
        all_channels = channel_names[patient_id]
        pred_dict = predictions[patient_id]
        
        # Build arrays
        y_true = np.array([1 if ch in onset_channels else 0 for ch in all_channels])
        y_proba = np.array([pred_dict.get(ch, 0.0) for ch in all_channels])
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_proba, all_channels)
        per_patient_metrics[patient_id] = metrics
    
    return per_patient_metrics

