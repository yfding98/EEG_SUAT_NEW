#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Feature Extractor Module

Module A: Causal Temporal Convolutional Network (TCN) to downsample time dimension
and extract high-level temporal features from NHFE time-series while preserving causality.

Input: (Batch_Size, N_Channels, N_Bands, Time_Steps)
Output: (Batch_Size, N_Channels, Hidden_Dim, Reduced_Time_Steps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import causal TCN
from causal_tcn import CausalTemporalExtractor


class TemporalFeatureExtractor(nn.Module):
    """
    Temporal Feature Extractor using Causal TCN
    
    Downsamples time dimension from 10000 to a latent size (e.g., 64 or 128)
    using causal dilated convolutions. Preserves temporal causality - no information
    leakage from future time steps.
    
    Architecture:
    - Causal dilated convolutions (TCN blocks)
    - Batch normalization and ReLU activation
    - MaxPool for downsampling
    - Designed to capture "step changes" in high-frequency energy while preserving order
    """
    
    def __init__(
        self,
        n_bands: int = 5,
        hidden_dim: int = 128,
        reduced_time_steps: int = 64,
        dropout: float = 0.2,
        num_blocks: int = 4
    ):
        """
        Args:
            n_bands: Number of frequency bands (Delta, Theta, Alpha, Beta, Gamma) = 5
            hidden_dim: Hidden dimension for temporal features (default: 128)
            reduced_time_steps: Target reduced time steps (default: 64)
            dropout: Dropout rate (default: 0.2)
            num_blocks: Number of TCN blocks (default: 4)
        """
        super(TemporalFeatureExtractor, self).__init__()
        
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.reduced_time_steps = reduced_time_steps
        
        # Use CausalTemporalExtractor
        self.causal_tcn = CausalTemporalExtractor(
            n_bands=n_bands,
            hidden_dim=hidden_dim,
            reduced_time_steps=reduced_time_steps,
            dropout=dropout,
            num_blocks=num_blocks
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch_Size, N_Channels, N_Bands, Time_Steps)
               Example: (B, 21, 5, 10000)
        
        Returns:
            Output tensor of shape (Batch_Size, N_Channels, Hidden_Dim, Reduced_Time_Steps)
            Example: (B, 21, 128, 64)
        """
        return self.causal_tcn(x)

