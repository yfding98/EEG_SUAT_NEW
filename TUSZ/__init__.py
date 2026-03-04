# TUSZ Processing Pipeline for SOZ Localization
"""
TUSZ (TUH EEG Seizure Corpus) processing module for SOZ (Seizure Onset Zone) localization.

This package provides tools to:
1. Parse TUSZ annotation files
2. Load and preprocess EDF data
3. Generate training manifests with derived SOZ labels
4. Provide PyTorch Dataset for training
"""

from .config import TUSZ_CONFIG, BIPOLAR_CHANNELS, BIPOLAR_TO_REGION
