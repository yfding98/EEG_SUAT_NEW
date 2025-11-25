#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Graph Structure Learning Module

Module B: Learnable Adjacency Matrix that automatically learns functional
connectivity/dependencies between channels based on training data.

This allows the model to ignore non-seizing channels by assigning low edge weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptiveGraphLearning(nn.Module):
    """
    Adaptive Graph Structure Learning
    
    Learns a dynamic adjacency matrix A that represents functional connectivity
    between EEG channels. The adjacency matrix is learned from the temporal features
    and can adapt to ignore non-seizing channels.
    
    Methods:
    1. Learnable parameter matrix (static but trainable)
    2. Feature-based dynamic learning (from temporal features)
    3. Combined approach (parameter + feature-based)
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        hidden_dim: int = 128,
        method: str = 'combined',
        temperature: float = 0.1,
        k_nearest: Optional[int] = None
    ):
        """
        Args:
            n_channels: Number of channels (default: 21)
            hidden_dim: Hidden dimension from temporal features (default: 128)
            method: Learning method - 'parameter', 'feature', or 'combined' (default: 'combined')
            temperature: Temperature for softmax in feature-based learning (default: 0.1)
            k_nearest: K-nearest neighbors for sparsification (None = fully connected)
        """
        super(AdaptiveGraphLearning, self).__init__()
        
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.method = method
        self.temperature = temperature
        self.k_nearest = k_nearest
        
        if method in ['parameter', 'combined']:
            # Learnable adjacency matrix (symmetric)
            # Initialize with small random values
            self.adjacency_param = nn.Parameter(
                torch.randn(n_channels, n_channels) * 0.1
            )
            # Make it symmetric by averaging with transpose
            self.adjacency_param.data = (
                self.adjacency_param.data + self.adjacency_param.data.t()
            ) / 2
        
        if method in ['feature', 'combined']:
            # MLP to compute edge weights from temporal features
            # Input: concatenated features from two channels
            # Output: edge weight (scalar)
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Normalization layer
        self.layer_norm = nn.LayerNorm(n_channels)
        
    def forward(
        self,
        temporal_features: torch.Tensor,
        return_adjacency: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to compute adjacency matrix
        
        Args:
            temporal_features: Tensor of shape (Batch_Size, N_Channels, Hidden_Dim, Reduced_Time_Steps)
            return_adjacency: If True, also return the adjacency matrix
        
        Returns:
            Adjacency matrix of shape (Batch_Size, N_Channels, N_Channels)
            If return_adjacency=False, returns the normalized adjacency matrix
        """
        batch_size = temporal_features.shape[0]
        
        # Aggregate temporal dimension: average pooling over time
        # (B, N_Ch, Hidden_Dim, Reduced_Time_Steps) -> (B, N_Ch, Hidden_Dim)
        node_features = temporal_features.mean(dim=-1)  # Global average pooling
        
        if self.method == 'parameter':
            # Use learnable parameter matrix
            # (N_Ch, N_Ch) -> (B, N_Ch, N_Ch)
            adjacency = self.adjacency_param.unsqueeze(0).expand(batch_size, -1, -1)
            
        elif self.method == 'feature':
            # Compute adjacency from features
            adjacency = self._compute_feature_based_adjacency(node_features)
            
        elif self.method == 'combined':
            # Combine parameter-based and feature-based
            param_adj = self.adjacency_param.unsqueeze(0).expand(batch_size, -1, -1)
            feature_adj = self._compute_feature_based_adjacency(node_features)
            # Weighted combination
            adjacency = 0.5 * param_adj + 0.5 * feature_adj
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Make symmetric (average with transpose)
        adjacency = (adjacency + adjacency.transpose(-2, -1)) / 2
        
        # Apply ReLU to ensure non-negative weights
        adjacency = F.relu(adjacency)
        
        # K-nearest neighbors sparsification (optional)
        if self.k_nearest is not None and self.k_nearest < self.n_channels:
            adjacency = self._k_nearest_sparsify(adjacency)
        
        # Normalize adjacency matrix (row normalization)
        # Add self-loops for stability
        identity = torch.eye(self.n_channels, device=adjacency.device).unsqueeze(0)
        adjacency = adjacency + identity
        
        # Row normalization: D^(-1) * A
        degree = adjacency.sum(dim=-1, keepdim=True) + 1e-8
        adjacency = adjacency / degree
        
        if return_adjacency:
            return adjacency, node_features
        return adjacency
    
    def _compute_feature_based_adjacency(
        self,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adjacency matrix from node features (vectorized)
        
        Args:
            node_features: (Batch_Size, N_Channels, Hidden_Dim)
        
        Returns:
            Adjacency matrix: (Batch_Size, N_Channels, N_Channels)
        """
        batch_size, n_channels, hidden_dim = node_features.shape
        
        # Vectorized computation: expand features for all pairs
        # Expand node_features: (B, N_Ch, Hidden_Dim) -> (B, N_Ch, 1, Hidden_Dim) and (B, 1, N_Ch, Hidden_Dim)
        h_i = node_features.unsqueeze(2)  # (B, N_Ch, 1, Hidden_Dim)
        h_j = node_features.unsqueeze(1)  # (B, 1, N_Ch, Hidden_Dim)
        
        # Broadcast and concatenate: (B, N_Ch, N_Ch, 2*Hidden_Dim)
        pair_features = torch.cat([h_i.expand(-1, -1, n_channels, -1), 
                                   h_j.expand(-1, n_channels, -1, -1)], dim=-1)
        
        # Reshape for MLP: (B, N_Ch, N_Ch, 2*Hidden_Dim) -> (B*N_Ch*N_Ch, 2*Hidden_Dim)
        pair_features_flat = pair_features.view(batch_size * n_channels * n_channels, hidden_dim * 2)
        
        # Compute edge weights: (B*N_Ch*N_Ch, 1)
        edge_weights_flat = self.edge_mlp(pair_features_flat)
        
        # Reshape back: (B*N_Ch*N_Ch, 1) -> (B, N_Ch, N_Ch)
        adjacency = edge_weights_flat.view(batch_size, n_channels, n_channels)
        
        return adjacency
    
    def _k_nearest_sparsify(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Keep only K nearest neighbors for each node
        
        Args:
            adjacency: (Batch_Size, N_Channels, N_Channels)
        
        Returns:
            Sparsified adjacency: (Batch_Size, N_Channels, N_Channels)
        """
        batch_size, n_channels, _ = adjacency.shape
        
        # For each node, keep top-k connections
        sparsified = torch.zeros_like(adjacency)
        
        for b in range(batch_size):
            for i in range(n_channels):
                # Get top-k neighbors (excluding self)
                values, indices = torch.topk(
                    adjacency[b, i, :],
                    k=min(self.k_nearest + 1, n_channels),  # +1 to account for self
                    dim=0
                )
                # Set top-k values
                sparsified[b, i, indices] = adjacency[b, i, indices]
        
        return sparsified

