"""Training module for seizure onset detection."""

from .trainer import OnsetDetectorTrainer
from .split import patient_wise_split

__all__ = ['OnsetDetectorTrainer', 'patient_wise_split']

