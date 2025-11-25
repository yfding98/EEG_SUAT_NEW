#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN_SOZ_Locator Model

Main model class integrating all modules:
- Module A: Temporal Feature Extractor
- Module B: Adaptive Graph Structure Learning
- Module C: Graph Convolution
- Module D: Classification Head

Input: (Batch_Size, N_Channels, N_Bands, Time_Steps)
Output: (Batch_Size, N_Channels) - Logits for Softmax
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .temporal_extractor import TemporalFeatureExtractor
from .graph_learning import AdaptiveGraphLearning
from .graph_conv import GraphConvolutionBlock
from .classifier_head import ClassificationHead
from .positional_encoding import PositionalEncoding


class STGNN_SOZ_Locator(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for SOZ Localization
    
    Identifies the single channel that seizes first among all channels.
    This is a multi-class classification problem where output size = Number of Channels.
    
    Architecture:
    1. Temporal Feature Extractor: Downsamples time dimension and extracts temporal features
    2. Adaptive Graph Learning: Learns functional connectivity between channels
    3. Graph Convolution: Exchanges information between channels
    4. Classification Head: Maps to channel logits
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_bands: int = 5,
        time_steps: int = 10000,
        temporal_hidden_dim: int = 128,
        temporal_reduced_steps: int = 64,
        graph_learning_method: str = 'combined',
        graph_hidden_dim: int = 128,
        graph_conv_type: str = 'gcn',
        graph_n_layers: int = 2,
        classifier_hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Args:
            n_channels: Number of channels (default: 21)
            n_bands: Number of frequency bands (default: 5)
            time_steps: Number of time steps (default: 10000)
            temporal_hidden_dim: Hidden dimension for temporal features (default: 128)
            temporal_reduced_steps: Reduced time steps after temporal extraction (default: 64)
            graph_learning_method: 'parameter', 'feature', or 'combined' (default: 'combined')
            graph_hidden_dim: Hidden dimension for graph features (default: 128)
            graph_conv_type: 'gcn' or 'gat' (default: 'gcn')
            graph_n_layers: Number of graph convolution layers (default: 2)
            classifier_hidden_dim: Hidden dimension for classifier (None = use graph_hidden_dim)
            dropout: Dropout rate (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(STGNN_SOZ_Locator, self).__init__()
        
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.time_steps = time_steps
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_reduced_steps = temporal_reduced_steps
        
        # Positional Encoding: Add temporal position information
        # Applied before temporal extraction to preserve chronological order
        self.positional_encoding = PositionalEncoding(
            d_model=n_bands,  # Encode the band dimension
            max_len=time_steps,
            dropout=dropout * 0.5  # Lower dropout for positional encoding
        )
        
        # Module A: Causal Temporal Feature Extractor
        self.temporal_extractor = TemporalFeatureExtractor(
            n_bands=n_bands,
            hidden_dim=temporal_hidden_dim,
            reduced_time_steps=temporal_reduced_steps,
            dropout=dropout
        )
        
        # Module B: Adaptive Graph Structure Learning
        self.graph_learning = AdaptiveGraphLearning(
            n_channels=n_channels,
            hidden_dim=temporal_hidden_dim,
            method=graph_learning_method,
            temperature=0.1,
            k_nearest=None  # Fully connected
        )
        
        # Module C: Graph Convolution
        self.graph_conv = GraphConvolutionBlock(
            in_features=temporal_hidden_dim,
            hidden_features=graph_hidden_dim,
            out_features=graph_hidden_dim,
            n_layers=graph_n_layers,
            conv_type=graph_conv_type,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # Module D: Classification Head with Temporal Attention
        classifier_hidden = classifier_hidden_dim if classifier_hidden_dim is not None else graph_hidden_dim
        self.classifier = ClassificationHead(
            in_features=graph_hidden_dim,
            n_channels=n_channels,
            hidden_dim=classifier_hidden,
            dropout=dropout
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_adjacency: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with refined chronological order focus
        
        Flow:
        1. Input: (B, 21, 5, 10000)
        2. Add Positional Encoding
        3. Causal TCN: Extract high-level temporal features -> (B, 21, 128, 64)
        4. GNN: Exchange info between channels -> (B, 21, 128, 64)
        5. Temporal Attention: Aggregate time dimension -> (B, 21, 128)
        6. Classifier: Project to logits -> (B, 21)
        
        Args:
            x: Input tensor of shape (Batch_Size, N_Channels, N_Bands, Time_Steps)
               Example: (B, 21, 5, 10000)
            return_adjacency: If True, also return the learned adjacency matrix
        
        Returns:
            Logits of shape (Batch_Size, N_Channels)
            If return_adjacency=True, returns (logits, adjacency_matrix)
        """
        # Step 1: Add Positional Encoding
        # (B, 21, 5, 10000) -> (B, 21, 5, 10000) with positional info
        x = self.positional_encoding(x)
        
        # Step 2: Module A: Causal Temporal Feature Extractor
        # (B, 21, 5, 10000) -> (B, 21, 128, 64)
        # Uses causal convolutions to preserve temporal order
        temporal_features = self.temporal_extractor(x)
        
        # Step 3: Module B: Adaptive Graph Learning
        # Compute adjacency matrix from temporal features
        # (B, 21, 128, 64) -> adjacency: (B, 21, 21), node_features: (B, 21, 128)
        # Note: graph_learning aggregates temporal dimension internally
        adjacency, node_features = self.graph_learning(
            temporal_features,
            return_adjacency=True
        )
        
        # Step 4: Module C: Graph Convolution
        # Apply graph convolution while preserving temporal dimension
        # We need to process temporal features with graph convolution
        # Reshape temporal_features for graph conv: (B, 21, 128, 64)
        batch_size, n_channels, features, time_steps = temporal_features.shape
        
        # Process each time step through graph convolution
        # Reshape: (B, 21, 128, 64) -> (B, 64, 21, 128) -> (B*64, 21, 128)
        temporal_features_reshaped = temporal_features.permute(0, 3, 1, 2)  # (B, 64, 21, 128)
        temporal_features_reshaped = temporal_features_reshaped.contiguous().view(
            batch_size * time_steps, n_channels, features
        )  # (B*64, 21, 128)
        
        # Expand adjacency for all time steps
        adjacency_expanded = adjacency.unsqueeze(1).expand(
            -1, time_steps, -1, -1
        )  # (B, 64, 21, 21)
        adjacency_expanded = adjacency_expanded.contiguous().view(
            batch_size * time_steps, n_channels, n_channels
        )  # (B*64, 21, 21)
        
        # Graph convolution: (B*64, 21, 128) -> (B*64, 21, 128)
        graph_features_reshaped = self.graph_conv(temporal_features_reshaped, adjacency_expanded)
        
        # Reshape back: (B*64, 21, 128) -> (B, 64, 21, 128) -> (B, 21, 128, 64)
        graph_features = graph_features_reshaped.view(
            batch_size, time_steps, n_channels, graph_features_reshaped.shape[-1]
        )  # (B, 64, 21, 128)
        graph_features = graph_features.permute(0, 2, 3, 1)  # (B, 21, 128, 64)
        
        # Step 5: Module D: Classification Head with Temporal Attention
        # Temporal Attention aggregates time dimension, focusing on rising edge
        # (B, 21, 128, 64) -> (B, 21)
        logits = self.classifier(graph_features)
        
        if return_adjacency:
            return logits, adjacency
        return logits
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model configuration
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'n_channels': self.n_channels,
            'n_bands': self.n_bands,
            'time_steps': self.time_steps,
            'temporal_hidden_dim': self.temporal_hidden_dim,
            'temporal_reduced_steps': self.temporal_reduced_steps,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

