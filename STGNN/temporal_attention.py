#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Attention Module

Module D: Temporal Attention Pooling to focus on the rising edge of NHFE.
Instead of averaging all time steps, learns attention weights for each time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalAttention(nn.Module):
    """
    Temporal Attention Pooling
    
    Learns attention weights for each time step to focus on the most relevant
    temporal moments (e.g., the rising edge of the seizure onset).
    
    Formula: Weighted_Sum = Sum(Features_t * Attention_Score_t)
    where Attention_Score_t = softmax(MLP(Features_t))
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden dimension for attention MLP (None = use in_features)
            dropout: Dropout rate (default: 0.1)
            temperature: Temperature for softmax (default: 1.0)
        """
        super(TemporalAttention, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_features
        self.temperature = temperature
        
        # Attention MLP: maps features to attention scores
        # Input: (..., Features) -> Output: (..., 1) - attention score
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch_Size, N_Channels, Features, Time_Steps)
               or (Batch_Size, Features, Time_Steps)
        
        Returns:
            Output tensor of shape (Batch_Size, N_Channels, Features)
            or (Batch_Size, Features)
        """
        if len(x.shape) == 4:
            # (B, N_Ch, Features, Time_Steps)
            batch_size, n_channels, features, time_steps = x.shape
            
            # Reshape: (B, N_Ch, Features, Time_Steps) -> (B * N_Ch, Features, Time_Steps)
            x_reshaped = x.contiguous().view(batch_size * n_channels, features, time_steps)
            
            # Transpose: (B * N_Ch, Features, Time_Steps) -> (B * N_Ch, Time_Steps, Features)
            x_reshaped = x_reshaped.transpose(1, 2)  # (B * N_Ch, Time_Steps, Features)
            
            # Compute attention scores for each time step
            # (B * N_Ch, Time_Steps, Features) -> (B * N_Ch, Time_Steps, 1)
            attention_scores = self.attention_mlp(x_reshaped)  # (B * N_Ch, Time_Steps, 1)
            
            # Apply temperature and softmax
            attention_weights = F.softmax(attention_scores / self.temperature, dim=1)
            # (B * N_Ch, Time_Steps, 1)
            
            # Weighted sum: Sum(Features_t * Attention_Weight_t)
            # (B * N_Ch, Time_Steps, Features) * (B * N_Ch, Time_Steps, 1) 
            # -> (B * N_Ch, Time_Steps, Features) -> sum over time -> (B * N_Ch, Features)
            weighted_features = (x_reshaped * attention_weights).sum(dim=1)
            # (B * N_Ch, Features)
            
            # Reshape back: (B * N_Ch, Features) -> (B, N_Ch, Features)
            output = weighted_features.contiguous().view(batch_size, n_channels, features)
            
        elif len(x.shape) == 3:
            # (B, Features, Time_Steps)
            batch_size, features, time_steps = x.shape
            
            # Transpose: (B, Features, Time_Steps) -> (B, Time_Steps, Features)
            x = x.transpose(1, 2)  # (B, Time_Steps, Features)
            
            # Compute attention scores
            # (B, Time_Steps, Features) -> (B, Time_Steps, 1)
            attention_scores = self.attention_mlp(x)  # (B, Time_Steps, 1)
            
            # Apply temperature and softmax
            attention_weights = F.softmax(attention_scores / self.temperature, dim=1)
            # (B, Time_Steps, 1)
            
            # Weighted sum
            # (B, Time_Steps, Features) * (B, Time_Steps, 1) -> (B, Features)
            output = (x * attention_weights).sum(dim=1)  # (B, Features)
            
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return output

