#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for EEG Seizure Onset Detection Pipeline

Usage:
    python main.py train --config config.yaml
    python main.py inference --config config.yaml --model_dir checkpoints --patient_data path/to/data
"""

import argparse
import yaml
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import NHFEDataLoader, load_patient_labels
from training.trainer import OnsetDetectorTrainer
from evaluation.metrics import compute_all_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_pipeline(config: dict) -> None:
    """Run training pipeline."""
    print("=" * 80)
    print("EEG Seizure Onset Detection - Training Pipeline")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading data...")
    data_loader = NHFEDataLoader(
        data_root=config['data']['data_root'],
        target_band=config['features']['target_band'],
        window_size=config['features']['window_size']
    )
    
    patients = data_loader.load_all_patients(
        pattern=config['data']['pattern']
    )
    print(f"Loaded {len(patients)} patients")
    if len(patients) > 0:
        print(f"Sample patient IDs from data: {list(patients.keys())[:5]}")
    
    # Load labels
    labels = load_patient_labels(config['data']['labels_path'])
    print(f"Loaded labels for {len(labels)} patients")
    if len(labels) > 0:
        print(f"Sample patient IDs from labels: {list(labels.keys())[:5]}")
    
    # Filter patients that have both data and labels
    valid_patients = {
        pid: data for pid, data in patients.items()
        if pid in labels
    }
    print(f"Valid patients (with data and labels): {len(valid_patients)}")
    
    if len(valid_patients) == 0:
        # Show which patient IDs don't match
        data_pids = set(patients.keys())
        label_pids = set(labels.keys())
        missing_in_labels = data_pids - label_pids
        missing_in_data = label_pids - data_pids
        
        if missing_in_labels:
            print(f"\nWarning: {len(missing_in_labels)} patient IDs in data but not in labels:")
            print(f"  {list(missing_in_labels)[:10]}")
        if missing_in_data:
            print(f"\nWarning: {len(missing_in_data)} patient IDs in labels but not in data:")
            print(f"  {list(missing_in_data)[:10]}")
        
        raise ValueError(
            "No patients with both data and labels found!\n"
            "Please check that patient IDs match between NPZ files and CSV labels."
        )
    
    # Initialize trainer
    print("\n[2/4] Initializing model...")
    model_config = config['model']
    model_type = model_config['type']
    
    # Prepare model parameters based on model type
    if model_type in ['xgboost', 'lightgbm']:
        # Tree-based model parameters
        model_kwargs = {
            'n_estimators': model_config.get('n_estimators', 200),
            'max_depth': model_config.get('max_depth', 6),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'subsample': model_config.get('subsample', 0.8),
            'colsample_bytree': model_config.get('colsample_bytree', 0.8),
            'reg_alpha': model_config.get('reg_alpha', 0.1),
            'reg_lambda': model_config.get('reg_lambda', 1.0),
            'random_state': model_config.get('random_state', 42)
        }
        if model_type == 'lightgbm':
            model_kwargs['num_leaves'] = model_config.get('num_leaves', 31)
    elif model_type == 'temporal_cnn':
        # CNN model parameters
        # input_length can be None (use full sequence) or an integer (fixed length)
        input_length = model_config.get('input_length')
        if input_length is None or input_length == 'null':
            input_length = None  # Will be auto-detected during training
        elif isinstance(input_length, str) and input_length.lower() == 'null':
            input_length = None
        
        model_kwargs = {
            'input_length': input_length,  # None means use full sequence length
            'n_filters': model_config.get('n_filters', [32, 64, 128]),
            'kernel_sizes': model_config.get('kernel_sizes', [5, 5, 5]),
            'dropout': model_config.get('dropout', 0.3),
            'learning_rate': model_config.get('learning_rate', 0.001),
            'device': model_config.get('device', 'cpu')
        }
    else:
        model_kwargs = {}
    
    # Get feature extraction configuration
    features_config = config.get('features', {})
    detection_config = config.get('detection', {})
    use_raw_nhfe = detection_config.get('use_raw_nhfe', False)
    manual_threshold = detection_config.get('manual_threshold', None)
    
    if use_raw_nhfe and manual_threshold is None:
        raise ValueError("manual_threshold must be set in config when use_raw_nhfe=True")
    
    trainer = OnsetDetectorTrainer(
        model_type=model_type,
        use_raw_nhfe=use_raw_nhfe,
        manual_threshold=manual_threshold,
        **model_kwargs
    )
    
    # Train
    print("\n[3/4] Training model...")
    training_config = config['training']
    results = trainer.train(
        patients=valid_patients,
        labels=labels,
        train_ratio=training_config['train_ratio'],
        val_ratio=training_config['val_ratio'],
        test_ratio=training_config['test_ratio'],
        random_state=training_config['random_state']
    )
    
    # Save model
    print("\n[4/4] Saving model...")
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir)
    
    # Save results
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "training_results.json"
    with open(results_path, 'w', encoding="utf-8") as f:
        # Convert numpy types to native Python types
        results_serializable = json.loads(
            json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
        )
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Results saved to: {results_path}")
    print("\nValidation Metrics:")
    for key, value in results['val_metrics'].items():
        print(f"  {key}: {value:.4f}")
    print("\nTest Metrics:")
    for key, value in results['test_metrics'].items():
        print(f"  {key}: {value:.4f}")


def inference_pipeline(
    config: dict,
    model_dir: str,
    patient_data_path: Optional[str] = None,
    patient_id: Optional[str] = None
) -> None:
    """Run inference pipeline."""
    print("=" * 80)
    print("EEG Seizure Onset Detection - Inference Pipeline")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    model_config = config['model']
    trainer = OnsetDetectorTrainer.load(
        model_dir=model_dir,
        model_type=model_config['type']
    )
    print(f"Loaded {model_config['type']} model from {model_dir}")
    
    # Load patient data
    print("\n[2/3] Loading patient data...")
    data_loader = NHFEDataLoader(
        data_root=config['data']['data_root'],
        target_band=config['features']['target_band'],
        window_size=config['features']['window_size']
    )
    
    if patient_data_path:
        # Load specific patient from file
        patient_data = data_loader.load_from_npz(patient_data_path, patient_id)
        patients = {patient_data.patient_id: patient_data}
    elif patient_id:
        # Load specific patient by ID
        all_patients = data_loader.load_all_patients(
            pattern=config['data']['pattern']
        )
        if patient_id not in all_patients:
            raise ValueError(f"Patient {patient_id} not found")
        patients = {patient_id: all_patients[patient_id]}
    else:
        # Load all patients
        patients = data_loader.load_all_patients(
            pattern=config['data']['pattern']
        )
    
    print(f"Loaded {len(patients)} patient(s)")
    
    # Run inference
    print("\n[3/3] Running inference...")
    inference_config = config['inference']
    threshold = inference_config.get('threshold', 0.5)
    top_k = inference_config.get('top_k', 5)
    
    results = {}
    
    for patient_id, patient_data in patients.items():
        # Extract features
        X, _, feature_names, _ = trainer.prepare_features(
            {patient_id: patient_data},
            labels=None
        )
        
        # Predict
        y_proba = trainer.model.predict_proba(X)
        y_pred = trainer.model.predict(X, threshold=threshold)
        
        # Get probabilities for each channel
        channel_probs = {}
        for i, ch_name in enumerate(patient_data.channel_names):
            if y_proba.ndim > 1:
                prob = y_proba[i, 1]
            else:
                prob = y_proba[i]
            channel_probs[ch_name] = float(prob)
        
        # Rank channels
        sorted_channels = sorted(
            channel_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get predicted onset channels
        predicted_onset = [
            ch for ch, prob in channel_probs.items()
            if prob >= threshold
        ]
        
        # Get top-K channels
        top_k_channels = [ch for ch, _ in sorted_channels[:top_k]]
        
        results[patient_id] = {
            'predicted_onset_channels': predicted_onset,
            'top_k_channels': top_k_channels,
            'channel_probabilities': channel_probs,
            'channel_ranking': [ch for ch, _ in sorted_channels],
            'n_predicted_onset': len(predicted_onset)
        }
        
        print(f"\nPatient: {patient_id}")
        print(f"  Predicted onset channels: {predicted_onset}")
        print(f"  Top-{top_k} channels: {top_k_channels}")
    
    # Save results
    output_dir = Path(config['output']['inference_output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    results_path = output_dir / "inference_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV (per-patient summary)
    summary_rows = []
    for patient_id, result in results.items():
        summary_rows.append({
            'patient_id': patient_id,
            'n_predicted_onset': result['n_predicted_onset'],
            'predicted_onset_channels': ', '.join(result['predicted_onset_channels']),
            'top_1_channel': result['top_k_channels'][0] if len(result['top_k_channels']) > 0 else '',
            'top_1_probability': result['channel_probabilities'].get(
                result['top_k_channels'][0], 0.0
            ) if len(result['top_k_channels']) > 0 else 0.0
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "inference_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nInference complete!")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="EEG Seizure Onset Detection Pipeline"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'inference'],
        help='Pipeline mode: train or inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Model directory (required for inference)'
    )
    parser.add_argument(
        '--patient_data',
        type=str,
        help='Path to patient data file (optional for inference)'
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        help='Patient ID (optional for inference)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Run pipeline
    if args.mode == 'train':
        train_pipeline(config)
    elif args.mode == 'inference':
        if args.model_dir is None:
            raise ValueError("--model_dir is required for inference")
        inference_pipeline(
            config,
            args.model_dir,
            args.patient_data,
            args.patient_id
        )


if __name__ == '__main__':
    import  sys
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                'train',
                '--config', 'config.yaml',
            ]
        )
    main()

