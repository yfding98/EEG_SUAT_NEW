# EEG Seizure Onset Channel Detection Pipeline

Production-grade Python pipeline for detecting seizure onset channels from NHFE (Normalized High Frequency Energy) time-series data.

## Overview

This pipeline implements a data-driven approach to predict seizure-onset channels based purely on NHFE values. The system:

1. **Loads** per-patient NHFE time-series data
2. **Processes** data with log-transform and robust baseline normalization
3. **Extracts** temporal and statistical features
4. **Trains** machine learning models (XGBoost, LightGBM, or Temporal CNN)
5. **Predicts** seizure onset channels for new patients
6. **Evaluates** performance with comprehensive metrics

## Key Features

- **Robust Normalization**: Log-transform + median/MAD baseline normalization
- **Adaptive Thresholding**: Percentile-based thresholds with fallback
- **Patient-wise Splitting**: Ensures no data leakage between train/val/test
- **Multi-label Support**: Groups channels that rise within 500ms as equal onset labels
- **Multiple Models**: XGBoost, LightGBM, and Temporal CNN support
- **Comprehensive Evaluation**: F1, Precision/Recall, Top-K accuracy, Spearman correlation

## Installation

### Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
chatGPT/
├── data/              # Data loading modules
│   ├── __init__.py
│   └── loader.py      # NHFE data loader from NPZ/CSV
├── features/          # Feature extraction
│   ├── __init__.py
│   ├── normalization.py  # Log-transform and baseline normalization
│   └── extractor.py      # Feature extraction (threshold crossing, peaks, slopes, etc.)
├── models/            # Model definitions
│   ├── __init__.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── temporal_cnn.py
├── training/          # Training utilities
│   ├── __init__.py
│   ├── split.py       # Patient-wise data splitting
│   └── trainer.py     # Model training orchestration
├── evaluation/        # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py     # F1, Precision/Recall, Top-K, Spearman, etc.
├── tests/             # Unit tests
├── config.yaml        # Configuration file
├── main.py            # Main entry point
├── inference.py       # Standalone inference script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Configuration

Edit `config.yaml` to configure:

- **Data paths**: Root directory and labels file
- **Feature extraction**: Baseline duration, window size, target band
- **Model parameters**: Model type and hyperparameters
- **Training**: Train/val/test ratios
- **Output paths**: Where to save models and results

### Example `config.yaml`

```yaml
data:
  data_root: "path/to/nhfe/data"
  labels_path: "path/to/labels.csv"
  pattern: "**/*BEI.npz"

features:
  baseline_duration: 10.0
  window_size: 0.25
  target_band: "gamma"

model:
  type: "xgboost"
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1

training:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Usage

### 1. Training

Train a model on your dataset:

```bash
python main.py train --config config.yaml
```

This will:
- Load NHFE data from NPZ files
- Load ground truth labels from CSV
- Extract features for all channels
- Split data patient-wise (no leakage)
- Train the selected model
- Evaluate on validation and test sets
- Save model and results

**Expected Labels CSV Format:**
```csv
patient_id,channel,is_onset
P001,Ch1,1
P001,Ch2,0
P001,Ch3,1
...
```

### 2. Inference

Run inference on new patient data:

```bash
# Using main.py
python main.py inference --config config.yaml --model_dir checkpoints

# Or using standalone inference script
python inference.py \
    --model_dir checkpoints \
    --patient_data path/to/patient_BEI.npz \
    --patient_id P001 \
    --threshold 0.5 \
    --top_k 5
```

**Output:**
- Predicted onset channels
- Channel ranking by probability
- Top-K channels
- Per-channel probabilities

### 3. Adding New Patients

To add new patients:

1. **Add NHFE data**: Place NPZ files (BEI.npz format) in your data directory
2. **Add labels** (if available): Add rows to labels CSV:
   ```csv
   patient_id,channel,is_onset
   P023,Ch1,0
   P023,Ch2,1
   ...
   ```
3. **Run inference**: Use inference pipeline to get predictions

## Data Format

### Input: NHFE Time-Series

The pipeline expects NHFE data in NPZ format with the following structure:

```python
{
    'NHFE': array of shape (n_channels, n_bands, n_timepoints),
    'ch_names': list of channel names,
    'band_names': list of frequency band names,
    'sfreq': sampling rate (Hz),
    'window_size': time window size (seconds)
}
```

### Processing Pipeline

1. **Log-transform**: `NHFE_log = log1p(NHFE)`
2. **Baseline normalization**:
   - `median_base = median(NHFE_log[baseline])`
   - `MAD_base = median(abs(NHFE_log[baseline] - median_base))`
   - `NHFE_norm = (NHFE_log - median_base) / (MAD_base + 1e-6)`
3. **Adaptive threshold**: `TH = percentile(NHFE_norm[baseline], 99.5)`
4. **Feature extraction**:
   - Threshold crossing time
   - Peak NHFE_norm
   - Slope around onset
   - Area under curve after onset
   - Stability duration

## Model Selection

The pipeline supports three model types:

### 1. XGBoost (Recommended)
- Fast training and inference
- Good performance on tabular features
- Interpretable feature importance

### 2. LightGBM
- Similar to XGBoost, often faster
- Good for large datasets

### 3. Temporal CNN
- Processes raw NHFE sequences
- Captures temporal patterns directly
- Requires sequence data preparation

**Recommendation**: Start with XGBoost for best balance of performance and speed.

## Evaluation Metrics

The pipeline computes:

- **Classification Metrics**:
  - F1 score
  - Precision
  - Recall
  - AUC-ROC
  - Accuracy

- **Ranking Metrics**:
  - Top-1 accuracy
  - Top-3 accuracy
  - Top-5 accuracy
  - Spearman correlation

- **Time Metrics** (if onset times available):
  - Mean absolute error
  - Mean squared error
  - Median absolute error

## Output Format

### Training Output

```
checkpoints/
├── xgboost_model.pkl
├── feature_extractor.pkl

results/
└── training_results.json
```

### Inference Output

```json
{
  "patient_id": "P001",
  "predicted_onset_channels": ["Ch2", "Ch5"],
  "top_k_channels": ["Ch2", "Ch5", "Ch8", "Ch12", "Ch15"],
  "channel_probabilities": {
    "Ch1": 0.23,
    "Ch2": 0.87,
    ...
  },
  "channel_ranking": ["Ch2", "Ch5", ...],
  "n_predicted_onset": 2
}
```

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Example test included: `tests/test_feature_extraction.py`

## Troubleshooting

### Common Issues

1. **No patients found**: Check `data_root` and `pattern` in config.yaml
2. **Missing labels**: Ensure labels CSV has correct format and patient IDs match
3. **Memory errors**: Reduce batch size (for CNN) or number of estimators (for tree models)
4. **Poor performance**: 
   - Try different model types
   - Adjust feature extraction parameters
   - Check data quality and label accuracy

### Data Quality Checks

- Verify NHFE values are non-negative
- Check baseline period is clean (no artifacts)
- Ensure sufficient baseline duration (≥10 seconds recommended)
- Validate channel names match between data and labels

## Performance Tips

1. **Feature Engineering**: The default features work well, but you can extend `NHFEFeatureExtractor` for domain-specific features
2. **Hyperparameter Tuning**: Adjust model parameters in `config.yaml` based on validation performance
3. **Ensemble**: Train multiple models and combine predictions
4. **Threshold Tuning**: Adjust prediction threshold based on precision/recall trade-off

## Citation

If you use this pipeline, please cite:

```
EEG Seizure Onset Detection Pipeline
Based on NHFE (Normalized High Frequency Energy) analysis
```

## License

[Specify your license here]

## Contact

[Your contact information]

