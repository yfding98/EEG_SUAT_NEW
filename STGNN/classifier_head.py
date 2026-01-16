#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Head Module

Module D: Classification head that maps graph features to channel logits.

Uses Temporal Attention to focus on the rising edge of the seizure onset,
then applies Fully Connected layers to output logits for each channel (N_Channels = 21).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import temporal attention
from temporal_attention import TemporalAttention


class ClassificationHead(nn.Module):
    """
    Classification Head for SOZ Localization with Temporal Attention
    
    Uses Temporal Attention to focus on the rising edge of the seizure onset,
    then maps graph features to channel logits for multi-class classification.
    Output size = Number of Channels (21).
    """
    
    def __init__(
        self,
        in_features: int,
        n_channels: int = 21,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        attention_hidden_dim: Optional[int] = None,
        attention_temperature: float = 1.0
    ):
        """
        Args:
            in_features: Input feature dimension from graph convolution
            n_channels: Number of channels (output classes) (default: 21)
            hidden_dim: Hidden dimension for FC layers (None = use in_features)
            dropout: Dropout rate (default: 0.3)
            attention_hidden_dim: Hidden dimension for attention MLP (None = use in_features)
            attention_temperature: Temperature for attention softmax (default: 1.0)
        """
        super(ClassificationHead, self).__init__()
        
        self.in_features = in_features
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_features
        
        # Temporal Attention: aggregates temporal dimension with learned weights
        # Input: (B, N_Ch, Features, Time_Steps) -> Output: (B, N_Ch, Features)
        self.temporal_attention = TemporalAttention(
            in_features=in_features,
            hidden_dim=attention_hidden_dim,
            dropout=dropout,
            temperature=attention_temperature
        )
        
        # Fully Connected layers
        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Final output layer: maps to 1 (single logit per channel)
        # Each channel gets one logit indicating its SOZ score
        self.fc_out = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            graph_features: Graph features of shape (Batch_Size, N_Channels, In_Features, Time_Steps)
                           The temporal dimension should be preserved from graph convolution
        
        Returns:
            Logits of shape (Batch_Size, N_Channels)
            Note: No softmax applied (use CrossEntropyLoss later)
        """
        batch_size = graph_features.shape[0]
        
        # Handle different input shapes
        if len(graph_features.shape) == 3:
            # (B, N_Ch, Features) - already aggregated, skip attention
            node_features = graph_features
        elif len(graph_features.shape) == 4:
            # (B, N_Ch, Features, Time_Steps) - apply temporal attention
            # This focuses on the rising edge of the seizure onset
            node_features = self.temporal_attention(graph_features)  # (B, N_Ch, Features)
        else:
            raise ValueError(f"Unexpected input shape: {graph_features.shape}")
        
        # For SOZ localization, we want per-channel predictions
        # Process each channel independently to get one logit per channel
        
        # Reshape: (B, N_Ch, Features) -> (B * N_Ch, Features)
        x = node_features.view(batch_size * self.n_channels, self.in_features)
        
        # FC Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # FC Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer: one logit per channel
        x = self.fc_out(x)  # (B * N_Ch, 1)
        
        # Reshape: (B * N_Ch, 1) -> (B, N_Ch)
        logits = x.view(batch_size, self.n_channels)
        
        return logits

