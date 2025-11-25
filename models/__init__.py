"""Model definitions for seizure onset detection."""

from .xgboost_model import XGBoostOnsetDetector
from .lightgbm_model import LightGBMOnsetDetector
from .temporal_cnn import TemporalCNNOnsetDetector

__all__ = [
    'XGBoostOnsetDetector',
    'LightGBMOnsetDetector',
    'TemporalCNNOnsetDetector'
]

