"""Evaluation module for seizure onset detection."""

from .metrics import (
    evaluate_predictions,
    compute_onset_time_error,
    compute_ranking_metrics,
    compute_all_metrics
)
from .threshold_optimization import find_optimal_threshold

__all__ = [
    'evaluate_predictions',
    'compute_onset_time_error',
    'compute_ranking_metrics',
    'compute_all_metrics',
    'find_optimal_threshold'
]

