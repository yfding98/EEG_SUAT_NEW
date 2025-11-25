#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient-wise data splitting utilities.

Ensures that all samples from the same patient stay in the same split.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.model_selection import train_test_split


def patient_wise_split(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patients into train/val/test sets.
    
    Args:
        patient_ids: List of unique patient IDs
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get unique patients
    unique_patients = list(set(patient_ids))
    n_patients = len(unique_patients)
    
    if n_patients < 3:
        raise ValueError(
            f"Need at least 3 patients for train/val/test split, got {n_patients}"
        )
    
    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=(1 - train_ratio),
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    return train_patients, val_patients, test_patients


def split_features_by_patients(
    features_list: List[Dict],
    patient_ids: List[str],
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[str], List[str], List[str]]:
    """
    Split feature dictionaries by patient IDs.
    
    Args:
        features_list: List of feature dictionaries
        patient_ids: List of patient IDs corresponding to features_list
        train_patients: List of training patient IDs
        val_patients: List of validation patient IDs
        test_patients: List of test patient IDs
    
    Returns:
        Tuple of (train_features, val_features, test_features,
                  train_labels, val_labels, test_labels)
    """
    train_features = []
    val_features = []
    test_features = []
    train_labels = []
    val_labels = []
    test_labels = []
    
    for i, (feat, pid) in enumerate(zip(features_list, patient_ids)):
        if pid in train_patients:
            train_features.append(feat)
            train_labels.append(feat.get('label', 0))
        elif pid in val_patients:
            val_features.append(feat)
            val_labels.append(feat.get('label', 0))
        elif pid in test_patients:
            test_features.append(feat)
            test_labels.append(feat.get('label', 0))
    
    return (
        train_features, val_features, test_features,
        train_labels, val_labels, test_labels
    )

