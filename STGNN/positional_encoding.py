#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positional Encoding Module

Sinusoidal positional encoding for temporal sequences.
This is crucial for the model to compare the exact timing of spikes across different channels.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (like in Transformers)
    
    Adds fixed sinusoidal encodings to input features to provide temporal position information.
    This allows the model to understand the chronological order of events.
    
    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension (must match input feature dimension)
            max_len: Maximum sequence length (default: 10000)
            dropout: Dropout rate (default: 0.1)
        """
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Compute div_term: 10000^(2i/d_model) for i in range(0, d_model//2)
        # Handle both even and odd d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (ceil(d_model/2),)
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))
        # Apply cos to odd indices (only if d_model > 1)
        if d_model > 1:
            # For odd d_model, we have one less cos term
            cos_indices = torch.arange(1, d_model, 2)
            if len(cos_indices) > 0:
                pe[:, 1::2] = torch.cos(position * div_term[:len(cos_indices)].unsqueeze(0))
        
        # Register as buffer (not a parameter, but part of model state)
        # Shape: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch_Size, ..., Time_Steps, Features)
               or (Batch_Size, Features, Time_Steps) for Conv1d format
        
        Returns:
            Output tensor with positional encoding added
        """
        # x shape can be:
        # - (B, N_Ch, N_Bands, Time_Steps) - before temporal extraction
        # - (B, N_Ch, Features, Time_Steps) - after temporal extraction
        
        if len(x.shape) == 4:
            # (B, N_Ch, Features, Time_Steps)
            batch_size, n_channels, features, time_steps = x.shape
            
            # Reshape to (B * N_Ch, Features, Time_Steps)
            x_reshaped = x.view(batch_size * n_channels, features, time_steps)
            
            # Add positional encoding
            # pe: (1, max_len, d_model) -> (1, time_steps, features)
            # Transpose x_reshaped: (B*N_Ch, Features, Time_Steps) -> (B*N_Ch, Time_Steps, Features)
            x_reshaped = x_reshaped.transpose(1, 2)  # (B*N_Ch, Time_Steps, Features)
            
            # Add positional encoding: (B*N_Ch, Time_Steps, Features) + (1, Time_Steps, Features)
            x_reshaped = x_reshaped + self.pe[:, :time_steps, :features]
            
            # Transpose back: (B*N_Ch, Time_Steps, Features) -> (B*N_Ch, Features, Time_Steps)
            x_reshaped = x_reshaped.transpose(1, 2)
            
            # Reshape back: (B * N_Ch, Features, Time_Steps) -> (B, N_Ch, Features, Time_Steps)
            x = x_reshaped.view(batch_size, n_channels, features, time_steps)
            
        elif len(x.shape) == 3:
            # (B, Features, Time_Steps) - already processed per channel
            batch_size, features, time_steps = x.shape
            
            # Transpose: (B, Features, Time_Steps) -> (B, Time_Steps, Features)
            x = x.transpose(1, 2)
            
            # Add positional encoding: (B, Time_Steps, Features) + (1, Time_Steps, Features)
            x = x + self.pe[:, :time_steps, :features]
            
            # Transpose back: (B, Time_Steps, Features) -> (B, Features, Time_Steps)
            x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return self.dropout(x)

