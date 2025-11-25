#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone inference script for new patient data.

Usage:
    python inference.py --model_dir checkpoints --patient_data path/to/BEI.npz --patient_id P001
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

from data.loader import NHFEDataLoader
from training.trainer import OnsetDetectorTrainer
from features.extractor import NHFEFeatureExtractor


def run_inference(
    model_dir: str,
    patient_data_path: str,
    patient_id: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: float = 0.5,
    top_k: int = 5
) -> Dict:
    """
    Run inference on a single patient.
    
    Args:
        model_dir: Directory containing trained model
        patient_data_path: Path to patient NHFE data (NPZ file)
        patient_id: Optional patient ID (extracted from path if not provided)
        config_path: Optional path to config file (for feature extraction settings)
        threshold: Probability threshold for binary prediction
        top_k: Number of top channels to return
    
    Returns:
        Dictionary with inference results
    """
    # Load config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_type = config['model']['type']
        target_band = config['features']['target_band']
        window_size = config['features']['window_size']
    else:
        # Default values
        model_type = 'xgboost'  # Will be determined from saved model
        target_band = 'gamma'
        window_size = 0.25
    
    # Load model
    print(f"Loading model from {model_dir}...")
    trainer = OnsetDetectorTrainer.load(
        model_dir=model_dir,
        model_type=model_type
    )
    
    # Load patient data
    print(f"Loading patient data from {patient_data_path}...")
    data_loader = NHFEDataLoader(
        data_root=Path(patient_data_path).parent,
        target_band=target_band,
        window_size=window_size
    )
    
    patient_data = data_loader.load_from_npz(patient_data_path, patient_id)
    print(f"Patient ID: {patient_data.patient_id}")
    print(f"Channels: {patient_data.n_channels}")
    print(f"Time points: {patient_data.n_timepoints}")
    print(f"Duration: {patient_data.duration_seconds:.2f} seconds")
    
    # Extract features
    print("Extracting features...")
    X, _, feature_names, _ = trainer.prepare_features(
        {patient_data.patient_id: patient_data},
        labels=None
    )
    
    # Predict
    print("Running predictions...")
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
    
    # Build results
    results = {
        'patient_id': patient_data.patient_id,
        'predicted_onset_channels': predicted_onset,
        'top_k_channels': top_k_channels,
        'channel_probabilities': channel_probs,
        'channel_ranking': [ch for ch, _ in sorted_channels],
        'n_predicted_onset': len(predicted_onset),
        'threshold': threshold
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print(f"\nPatient ID: {patient_data.patient_id}")
    print(f"\nPredicted Onset Channels ({len(predicted_onset)}):")
    for ch in predicted_onset:
        print(f"  - {ch}: {channel_probs[ch]:.4f}")
    
    print(f"\nTop-{top_k} Channels:")
    for i, (ch, prob) in enumerate(sorted_channels[:top_k], 1):
        print(f"  {i}. {ch}: {prob:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on new patient data"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--patient_data',
        type=str,
        required=True,
        help='Path to patient NHFE data (NPZ file)'
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        help='Patient ID (extracted from path if not provided)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for binary prediction (default: 0.5)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top channels to return (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        model_dir=args.model_dir,
        patient_data_path=args.patient_data,
        patient_id=args.patient_id,
        config_path=args.config,
        threshold=args.threshold,
        top_k=args.top_k
    )
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

