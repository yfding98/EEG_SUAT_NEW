#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for seizure onset detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json

from data.loader import PatientData
from features.extractor import NHFEFeatureExtractor
from features.normalization import normalize_nhfe
from models.xgboost_model import XGBoostOnsetDetector
from models.lightgbm_model import LightGBMOnsetDetector
from models.temporal_cnn import TemporalCNNWrapper
from training.split import patient_wise_split


class OnsetDetectorTrainer:
    """Trainer for seizure onset detection models."""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        feature_extractor: Optional[NHFEFeatureExtractor] = None,
        use_raw_nhfe: bool = False,
        manual_threshold: Optional[float] = None,
        **model_kwargs
    ):
        """
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', or 'temporal_cnn')
            feature_extractor: Optional pre-configured feature extractor
            use_raw_nhfe: If True, use raw NHFE values without normalization
            manual_threshold: Manual threshold value (required if use_raw_nhfe=True)
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type.lower()
        # Initialize feature extractor with default parameters if not provided
        # Note: sampling_rate will be updated from PatientData when extracting features
        self.feature_extractor = feature_extractor or NHFEFeatureExtractor(sampling_rate=250.0)
        self.use_raw_nhfe = use_raw_nhfe
        self.manual_threshold = manual_threshold
        
        if use_raw_nhfe and manual_threshold is None:
            raise ValueError("manual_threshold must be provided when use_raw_nhfe=True")
        
        # Initialize model with appropriate parameters
        if self.model_type == 'xgboost':
            # XGBoost/LightGBM parameters
            tree_params = {
                'n_estimators': model_kwargs.get('n_estimators', 200),
                'max_depth': model_kwargs.get('max_depth', 6),
                'learning_rate': model_kwargs.get('learning_rate', 0.1),
                'subsample': model_kwargs.get('subsample', 0.8),
                'colsample_bytree': model_kwargs.get('colsample_bytree', 0.8),
                'reg_alpha': model_kwargs.get('reg_alpha', 0.1),
                'reg_lambda': model_kwargs.get('reg_lambda', 1.0),
                'random_state': model_kwargs.get('random_state', 42)
            }
            self.model = XGBoostOnsetDetector(**tree_params)
        elif self.model_type == 'lightgbm':
            # LightGBM parameters
            tree_params = {
                'n_estimators': model_kwargs.get('n_estimators', 200),
                'max_depth': model_kwargs.get('max_depth', 6),
                'learning_rate': model_kwargs.get('learning_rate', 0.1),
                'subsample': model_kwargs.get('subsample', 0.8),
                'colsample_bytree': model_kwargs.get('colsample_bytree', 0.8),
                'reg_alpha': model_kwargs.get('reg_alpha', 0.1),
                'reg_lambda': model_kwargs.get('reg_lambda', 1.0),
                'num_leaves': model_kwargs.get('num_leaves', 31),
                'random_state': model_kwargs.get('random_state', 42)
            }
            self.model = LightGBMOnsetDetector(**tree_params)
        elif self.model_type == 'temporal_cnn':
            # Temporal CNN parameters
            cnn_params = {
                'input_length': model_kwargs.get('input_length', 120),
                'n_filters': model_kwargs.get('n_filters', [32, 64, 128]),
                'kernel_sizes': model_kwargs.get('kernel_sizes', [5, 5, 5]),
                'dropout': model_kwargs.get('dropout', 0.3),
                'learning_rate': model_kwargs.get('learning_rate', 0.001),
                'device': model_kwargs.get('device', 'cpu')
            }
            self.model = TemporalCNNWrapper(**cnn_params)
            # Store batch_size and epochs for later use
            self.model.batch_size = model_kwargs.get('batch_size', 32)
            self.model.epochs = model_kwargs.get('epochs', 50)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
    
    def prepare_features(
        self,
        patients: Dict[str, PatientData],
        labels: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extract features from patient data.
        
        Args:
            patients: Dictionary mapping patient_id -> PatientData
            labels: Optional dictionary mapping patient_id -> list of onset channel names
        
        Returns:
            Tuple of (feature_matrix, labels_array, feature_names, patient_ids)
        """
        all_features = []
        all_labels = []
        all_patient_ids = []
        
        for patient_id, patient_data in patients.items():
            # Update feature extractor sampling rate from patient data
            self.feature_extractor.sampling_rate = patient_data.sampling_rate
            # Recompute baseline_end_idx with correct sampling rate
            self.feature_extractor.baseline_end_idx = int(
                self.feature_extractor.baseline_duration * patient_data.sampling_rate
            )
            
            # Get ground truth labels if provided
            onset_channels = labels.get(patient_id, []) if labels else []
            
            # Handle multi-band data: extract features for all bands
            nhfe_data = patient_data.nhfe_data
            is_multi_band = nhfe_data.ndim == 3
            
            if is_multi_band:
                # Multi-band: extract features for each band and concatenate
                n_bands = nhfe_data.shape[1]
                band_names = patient_data.band_names if hasattr(patient_data, 'band_names') else [f'band_{i}' for i in range(n_bands)]
                
                # Extract features for each band
                all_band_features = []
                for band_idx in range(n_bands):
                    band_name = band_names[band_idx] if band_idx < len(band_names) else f'band_{band_idx}'
                    band_nhfe = nhfe_data[:, band_idx, :]
                    
                    features_list, threshold = self.feature_extractor.extract_all_channels(
                        band_nhfe,
                        patient_data.channel_names,
                        threshold=self.manual_threshold if self.use_raw_nhfe else None,
                        use_raw_nhfe=self.use_raw_nhfe
                    )
                    
                    # Add band suffix to feature names
                    for feat in features_list:
                        # Create new dict with band suffix
                        band_feat = {}
                        for key, value in feat.items():
                            if key in ['channel_name', 'label', 'patient_id']:
                                band_feat[key] = value
                            else:
                                # Add band suffix to feature names
                                band_feat[f"{key}_{band_name}"] = value
                        all_band_features.append(band_feat)
                
                # Merge features from all bands for each channel
                # Group by channel name
                channel_features_dict = {}
                for band_feat in all_band_features:
                    ch_name = band_feat['channel_name']
                    if ch_name not in channel_features_dict:
                        channel_features_dict[ch_name] = {'channel_name': ch_name}
                    # Merge features (excluding metadata keys)
                    for key, value in band_feat.items():
                        if key not in ['channel_name', 'label', 'patient_id']:
                            channel_features_dict[ch_name][key] = value
                
                # Convert to list
                features_list = list(channel_features_dict.values())
            else:
                # Single band: extract features normally
                features_list, threshold = self.feature_extractor.extract_all_channels(
                    nhfe_data,
                    patient_data.channel_names,
                    threshold=self.manual_threshold if self.use_raw_nhfe else None,
                    use_raw_nhfe=self.use_raw_nhfe
                )
            
            # Add labels to features
            for feat in features_list:
                channel_name = feat['channel_name']
                is_onset = 1 if channel_name in onset_channels else 0
                feat['label'] = is_onset
                feat['patient_id'] = patient_id
                
                all_features.append(feat)
                all_labels.append(is_onset)
                all_patient_ids.append(patient_id)
        
        # Convert to feature matrix
        # Exclude non-feature keys
        exclude_keys = {'channel_name', 'label', 'patient_id', 'threshold'}
        
        # Get feature names from first feature dict
        if len(all_features) == 0:
            raise ValueError("No features extracted")
        
        # Get all possible feature names (excluding metadata keys)
        all_feature_keys = set()
        for feat in all_features:
            for key in feat.keys():
                if key not in exclude_keys:
                    all_feature_keys.add(key)
        
        feature_names = sorted(list(all_feature_keys))
        
        # Build feature matrix
        feature_matrix = np.array([
            [feat[k] for k in feature_names]
            for feat in all_features
        ])
        
        # Replace NaN with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        labels_array = np.array(all_labels)
        
        return feature_matrix, labels_array, feature_names, all_patient_ids
    
    def train(
        self,
        patients: Dict[str, PatientData],
        labels: Dict[str, List[str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the model with patient-wise splitting.
        
        Args:
            patients: Dictionary mapping patient_id -> PatientData
            labels: Dictionary mapping patient_id -> list of onset channel names
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed
        
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data based on model type
        if self.model_type == 'temporal_cnn':
            # For CNN, prepare sequences
            print("Preparing sequences for CNN...")
            sequence_length = getattr(self.model, 'input_length', None)
            
            # If sequence_length is None or 0, use full sequence length
            if sequence_length is None or sequence_length <= 0:
                # Use full sequence length (find max length from all patients)
                max_length = max(
                    patient_data.n_timepoints 
                    for patient_data in patients.values()
                )
                sequence_length = max_length
                print(f"Using full sequence length: {sequence_length} (auto-detected)")
            else:
                print(f"Using fixed sequence length: {sequence_length}")
            
            X, y, patient_ids = self.prepare_sequences(patients, labels, sequence_length=sequence_length)
            feature_names = None  # CNN doesn't use feature names
            print(f"Prepared sequences: {X.shape}")
        else:
            # For tree-based models, extract features
            print("Extracting features...")
            X, y, feature_names, patient_ids = self.prepare_features(patients, labels)
            print(f"Extracted features: {X.shape}")
        
        print(f"Positive samples: {np.sum(y)}, Negative samples: {np.sum(1-y)}")
        
        # Patient-wise split
        print("Splitting data by patients...")
        unique_patients = list(set(patient_ids))
        train_patients, val_patients, test_patients = patient_wise_split(
            unique_patients,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )
        
        print(f"Train patients: {len(train_patients)}")
        print(f"Val patients: {len(val_patients)}")
        print(f"Test patients: {len(test_patients)}")
        
        # Split data by patients
        train_mask = np.array([pid in train_patients for pid in patient_ids])
        val_mask = np.array([pid in val_patients for pid in patient_ids])
        test_mask = np.array([pid in test_patients for pid in patient_ids])
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train model
        print(f"\nTraining {self.model_type} model...")
        
        if self.model_type == 'temporal_cnn':
            # CNN training - get batch_size and epochs from model config if available
            batch_size = getattr(self.model, 'batch_size', 32)
            epochs = getattr(self.model, 'epochs', 50)
            validation_data = (X_val, y_val) if len(X_val) > 0 else None
            self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=True
            )
        else:
            # Tree-based models
            validation_data = (X_val, y_val) if len(X_val) > 0 else None
            self.model.fit(
                X_train, y_train,
                feature_names=feature_names,
                validation_data=validation_data
            )
        
        self.is_trained = True
        
        # Evaluate
        from evaluation.metrics import evaluate_predictions
        from evaluation.threshold_optimization import find_optimal_threshold
        
        print("\nEvaluating on validation set...")
        y_val_proba = self.model.predict_proba(X_val)
        
        # Find optimal threshold on validation set
        optimal_threshold, best_f1 = find_optimal_threshold(y_val, y_val_proba, metric='f1')
        print(f"Optimal threshold on validation set: {optimal_threshold:.4f} (F1={best_f1:.4f})")
        
        # Predict with optimal threshold
        y_val_pred = self.model.predict(X_val, threshold=optimal_threshold)
        
        # Print class distribution
        print(f"Validation set: {np.sum(y_val)} positive, {len(y_val) - np.sum(y_val)} negative")
        print(f"Validation predictions: {np.sum(y_val_pred)} positive, {len(y_val_pred) - np.sum(y_val_pred)} negative")
        if y_val_proba.ndim > 1:
            print(f"Validation probability range: [{y_val_proba[:, 1].min():.4f}, {y_val_proba[:, 1].max():.4f}], mean={y_val_proba[:, 1].mean():.4f}")
        
        val_metrics = evaluate_predictions(
            y_val, y_val_pred, y_val_proba
        )
        val_metrics['optimal_threshold'] = optimal_threshold
        
        print("\nEvaluating on test set...")
        y_test_proba = self.model.predict_proba(X_test)
        
        # Use optimal threshold from validation set
        y_test_pred = self.model.predict(X_test, threshold=optimal_threshold)
        
        # Print class distribution
        print(f"Test set: {np.sum(y_test)} positive, {len(y_test) - np.sum(y_test)} negative")
        print(f"Test predictions: {np.sum(y_test_pred)} positive, {len(y_test_pred) - np.sum(y_test_pred)} negative")
        if y_test_proba.ndim > 1:
            print(f"Test probability range: [{y_test_proba[:, 1].min():.4f}, {y_test_proba[:, 1].max():.4f}], mean={y_test_proba[:, 1].mean():.4f}")
        
        test_metrics = evaluate_predictions(
            y_test, y_test_pred, y_test_proba
        )
        test_metrics['optimal_threshold'] = optimal_threshold
        
        results = {
            'model_type': self.model_type,
            'feature_names': feature_names,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'test_patients': test_patients,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'n_features': len(feature_names) if feature_names is not None else X.shape[1] if X.ndim > 1 else 1,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
        
        return results
    
    def prepare_sequences(
        self,
        patients: Dict[str, PatientData],
        labels: Dict[str, List[str]],
        sequence_length: int = 120
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequence data for temporal CNN.
        
        Args:
            patients: Dictionary mapping patient_id -> PatientData
            labels: Dictionary mapping patient_id -> list of onset channel names
            sequence_length: Length of sequences to extract
        
        Returns:
            Tuple of (sequences, labels, patient_ids)
        """
        all_sequences = []
        all_labels = []
        all_patient_ids = []
        
        for patient_id, patient_data in patients.items():
            onset_channels = labels.get(patient_id, [])
            
            # Update feature extractor sampling rate from patient data
            self.feature_extractor.sampling_rate = patient_data.sampling_rate
            # Recompute baseline_end_idx with correct sampling rate
            self.feature_extractor.baseline_end_idx = int(
                self.feature_extractor.baseline_duration * patient_data.sampling_rate
            )
            
            # Normalize NHFE for all channels (or use raw if configured)
            # Handle multi-band data: use all bands as channels
            is_multi_band = patient_data.nhfe_data.ndim == 3
            
            for i, ch_name in enumerate(patient_data.channel_names):
                if is_multi_band:
                    # Multi-band: stack all bands as channels
                    # Shape: (n_bands, sequence_length)
                    n_bands = patient_data.nhfe_data.shape[1]
                    band_sequences = []
                    
                    for band_idx in range(n_bands):
                        nhfe_channel = patient_data.nhfe_data[i, band_idx, :]
                        
                        if self.use_raw_nhfe:
                            # Use raw NHFE values
                            nhfe_norm = nhfe_channel
                        else:
                            # Normalize
                            nhfe_norm, _, _ = normalize_nhfe(
                                nhfe_channel,
                                baseline_start=0,
                                baseline_end=self.feature_extractor.baseline_end_idx
                            )
                        
                        # Extract sequence (pad or truncate to sequence_length)
                        if len(nhfe_norm) >= sequence_length:
                            sequence = nhfe_norm[:sequence_length]
                        else:
                            sequence = np.pad(
                                nhfe_norm,
                                (0, sequence_length - len(nhfe_norm)),
                                mode='edge'
                            )
                        band_sequences.append(sequence)
                    
                    # Stack all bands: (n_bands, sequence_length)
                    sequence = np.stack(band_sequences, axis=0)
                else:
                    # Single band
                    nhfe_channel = patient_data.nhfe_data[i, :]
                    
                    if self.use_raw_nhfe:
                        # Use raw NHFE values
                        nhfe_norm = nhfe_channel
                    else:
                        # Normalize
                        nhfe_norm, _, _ = normalize_nhfe(
                            nhfe_channel,
                            baseline_start=0,
                            baseline_end=self.feature_extractor.baseline_end_idx
                        )
                    
                    # Extract sequence (pad or truncate to sequence_length)
                    if len(nhfe_norm) >= sequence_length:
                        sequence = nhfe_norm[:sequence_length]
                    else:
                        sequence = np.pad(
                            nhfe_norm,
                            (0, sequence_length - len(nhfe_norm)),
                            mode='edge'
                        )
                    # Add channel dimension: (1, sequence_length)
                    sequence = sequence[np.newaxis, :]
                
                is_onset = 1 if ch_name in onset_channels else 0
                
                all_sequences.append(sequence)
                all_labels.append(is_onset)
                all_patient_ids.append(patient_id)
        
        sequences = np.array(all_sequences)
        labels = np.array(all_labels)
        
        return sequences, labels, all_patient_ids
    
    def save(self, output_dir: Union[str, Path]) -> None:
        """Save trained model and metadata."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / f"{self.model_type}_model.pkl"
        if hasattr(self.model, 'save'):
            self.model.save(str(model_path))
        else:
            raise ValueError(f"Model {self.model_type} does not support save()")
        
        # Save feature extractor
        import joblib
        extractor_path = output_dir / "feature_extractor.pkl"
        joblib.dump(self.feature_extractor, extractor_path)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: Union[str, Path], model_type: str) -> 'OnsetDetectorTrainer':
        """Load trained model."""
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / f"{model_type}_model.pkl"
        
        if model_type == 'xgboost':
            model = XGBoostOnsetDetector.load(str(model_path))
        elif model_type == 'lightgbm':
            model = LightGBMOnsetDetector.load(str(model_path))
        elif model_type == 'temporal_cnn':
            model = TemporalCNNWrapper.load(str(model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load feature extractor
        import joblib
        extractor_path = model_dir / "feature_extractor.pkl"
        feature_extractor = joblib.load(extractor_path)
        
        instance = cls(model_type=model_type, feature_extractor=feature_extractor)
        instance.model = model
        instance.is_trained = True
        
        return instance

