# Spatio-Temporal Graph Neural Network (ST-GNN) for SOZ Localization

## Overview

This module implements a **Spatio-Temporal Graph Neural Network (ST-GNN)** for Epilepsy Seizure Onset Zone (SOZ) localization. The model identifies the **single channel that seizes FIRST** among all channels using Normalized High-Frequency Energy (NHFE) features.

**Key Focus**: The architecture is specifically designed to identify the **chronological order** of seizure onset, not just which channel has the most energy. This is crucial for understanding the propagation pattern of epileptic activity.

## Task Description

- **Input**: NHFE features with shape `(Batch_Size, N_Channels, N_Bands, Time_Steps)`
  - `N_Channels`: 21 (10-20 system)
  - `N_Bands`: 5 (Delta, Theta, Alpha, Beta, Gamma)
  - `Time_Steps`: 10000 (40 seconds @ 250Hz)
  
- **Output**: Logits with shape `(Batch_Size, N_Channels)` for multi-class classification
- **Problem Type**: Multi-class Classification (Output size = Number of Channels)

## Architecture

The model consists of four main modules with **chronological order focus**:

### Module A: Causal Temporal Feature Extractor
- **Purpose**: Downsamples time dimension from 10000 to ~64 using **Causal Dilated Convolutions (TCN blocks)**
- **Location**: `temporal_extractor.py`, `causal_tcn.py`
- **Key Features**:
  - **Causal convolutions**: Preserves temporal causality - no information leakage from future
  - **Dilated convolutions**: Captures long-range temporal dependencies
  - **Positional Encoding**: Fixed sinusoidal encoding added before temporal extraction to provide exact timing information
  - Batch normalization and ReLU activation
  - MaxPool for downsampling
  - Designed to capture "step changes" while preserving chronological order

### Module B: Adaptive Graph Structure Learning
- **Purpose**: Learns functional connectivity between channels automatically
- **Location**: `graph_learning.py`
- **Key Features**:
  - Learnable adjacency matrix (parameter-based)
  - Feature-based dynamic learning (from temporal features)
  - Combined approach (parameter + feature-based)
  - Can ignore non-seizing channels by assigning low edge weights

### Module C: Graph Convolution
- **Purpose**: Exchanges information between channels using graph neural networks
- **Location**: `graph_conv.py`
- **Key Features**:
  - Supports both GCN (Graph Convolutional Network) and GAT (Graph Attention Network)
  - Multiple layers with batch normalization
  - Helps compare energy rise of one channel against its neighbors

### Module D: Classification Head with Temporal Attention
- **Purpose**: Maps graph features to channel logits using **Temporal Attention**
- **Location**: `classifier_head.py`, `temporal_attention.py`
- **Key Features**:
  - **Temporal Attention**: Learns attention weights for each time step
  - Focuses on the **rising edge** of the seizure onset (not just average)
  - Formula: `Weighted_Sum = Sum(Features_t * Attention_Score_t)`
  - Fully Connected layers with batch normalization
  - Outputs one logit per channel
  - No softmax applied (use `CrossEntropyLoss` in training)

## Usage

### Basic Usage

```python
import torch
from STGNN import STGNN_SOZ_Locator

# Create model
model = STGNN_SOZ_Locator(
    n_channels=21,
    n_bands=5,
    time_steps=10000,
    temporal_hidden_dim=128,
    temporal_reduced_steps=64,
    graph_learning_method='combined',  # 'parameter', 'feature', or 'combined'
    graph_hidden_dim=128,
    graph_conv_type='gcn',  # 'gcn' or 'gat'
    graph_n_layers=2,
    dropout=0.2,
    use_batch_norm=True
)

# Forward pass
# Input: (Batch_Size, 21, 5, 10000)
x = torch.randn(4, 21, 5, 10000)
logits = model(x)  # Output: (4, 21)

# Get adjacency matrix (optional)
logits, adjacency = model(x, return_adjacency=True)
```

### Training Example

```python
import torch
import torch.nn as nn
from STGNN import STGNN_SOZ_Locator

# Create model
model = STGNN_SOZ_Locator()

# Loss function (multi-class classification)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    nhfe_features = batch['nhfe_features']  # (B, 21, 5, 10000)
    target_channel = batch['soz_channel']   # (B,) - index of SOZ channel
    
    # Forward pass
    logits = model(nhfe_features)  # (B, 21)
    
    # Compute loss
    loss = criterion(logits, target_channel)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Model Configuration Options

### Graph Learning Methods

1. **'parameter'**: Uses a learnable parameter matrix (static but trainable)
2. **'feature'**: Computes adjacency from temporal features dynamically
3. **'combined'**: Combines both parameter-based and feature-based (recommended)

### Graph Convolution Types

1. **'gcn'**: Graph Convolutional Network (standard GCN)
2. **'gat'**: Graph Attention Network (with attention mechanism)

## Testing

Run the test script to verify model shapes and functionality:

```bash
python STGNN/test_stgnn.py
```

The test script verifies:
- Input/output shape consistency
- Different batch sizes
- Gradient flow
- Different model configurations

## File Structure

```
STGNN/
├── __init__.py              # Package initialization
├── temporal_extractor.py    # Module A: Temporal feature extraction (wrapper)
├── causal_tcn.py            # Causal TCN blocks implementation
├── positional_encoding.py   # Sinusoidal positional encoding
├── graph_learning.py        # Module B: Adaptive graph learning
├── graph_conv.py            # Module C: Graph convolution layers
├── classifier_head.py       # Module D: Classification head (wrapper)
├── temporal_attention.py     # Temporal attention pooling
├── stgnn_model.py           # Main model class
├── test_stgnn.py            # Test script
└── README.md                # This file
```

## Model Parameters

The default model has approximately **487,078 trainable parameters** (increased due to causal TCN blocks and temporal attention).

## Refined Forward Pass Flow

1. **Input**: `(Batch_Size, 21, 5, 10000)`
2. **Positional Encoding**: Add sinusoidal temporal position information
3. **Causal TCN**: Extract high-level temporal features → `(Batch_Size, 21, 128, 64)`
4. **Graph Learning**: Learn adjacency matrix from temporal features
5. **Graph Convolution**: Exchange info between channels (preserving temporal dim) → `(Batch_Size, 21, 128, 64)`
6. **Temporal Attention**: Aggregate time dimension, focusing on rising edge → `(Batch_Size, 21, 128)`
7. **Classifier**: Project to logits → `(Batch_Size, 21)`

## Key Design Decisions

1. **Causal Convolutions**: Preserves temporal causality - output at time t only depends on inputs ≤ t
2. **Positional Encoding**: Provides exact timing information to compare spike timing across channels
3. **Temporal Attention**: Focuses on the rising edge of seizure onset rather than averaging all time steps
4. **Temporal Downsampling**: Time dimension reduced from 10000 to 64 using causal convolutions
5. **Adaptive Graph Learning**: Learns connectivity patterns rather than using fixed spatial adjacency
6. **Per-Channel Processing**: Each channel is processed independently in the classification head
7. **No Softmax in Model**: Softmax is applied in the loss function (CrossEntropyLoss)

## Why These Changes Matter

- **Causal Convolutions**: Ensure the model analyzes signal history strictly over time, preserving the causality of the onset event. No information leakage from future time steps.

- **Positional Encoding**: Crucial for the model to compare the exact timing of spikes across different channels. Without it, the model might identify the channel with highest energy, not necessarily the first one to spike.

- **Temporal Attention**: The "onset" happens at a specific brief moment. Average pooling dilutes this signal. Attention allows the model to focus specifically on the *rising edge* of the NHFE, which is when the seizure actually begins.

## Notes on NHFE Features

The input NHFE (Normalized High-Frequency Energy) features represent energy ratios. A seizure onset is characterized by a sharp, exponential rise in high-frequency bands relative to the baseline. The Temporal Extractor is designed to catch these "step changes" in the time-series.

## References

- Graph Convolutional Networks (GCN): Kipf & Welling, 2017
- Graph Attention Networks (GAT): Veličković et al., 2018
- Spatio-Temporal GNNs for EEG: Various recent works on brain network analysis

