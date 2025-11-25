#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold optimization for imbalanced classification.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score, precision_recall_curve


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'f1_macro', or 'youden')
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]  # Get positive class probabilities
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == 'f1':
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
    
    elif metric == 'youden':
        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        # For binary: J = recall + (TN/(TN+FP)) - 1 = recall + precision - 1
        youden_scores = recall + precision - 1
        best_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = youden_scores[best_idx]
    
    else:
        # Default: use F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
    
    return float(optimal_threshold), float(best_score)



