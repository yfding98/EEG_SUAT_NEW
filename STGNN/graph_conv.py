#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Convolution Module

Module C: Graph Convolutional layers (GCN/GAT) applied on the spatial dimension
(nodes=21 channels) to exchange information between channels.

This helps the model compare the energy rise of one channel against its neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer (GCN)
    
    Implements the standard GCN operation:
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    
    where:
    - A is the adjacency matrix
    - D is the degree matrix
    - H^(l) is the node features at layer l
    - W^(l) is the learnable weight matrix
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias (default: True)
        """
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features of shape (Batch_Size, N_Channels, In_Features)
            adjacency: Normalized adjacency matrix of shape (Batch_Size, N_Channels, N_Channels)
        
        Returns:
            Output features of shape (Batch_Size, N_Channels, Out_Features)
        """
        # Linear transformation: H * W
        # (B, N_Ch, In_Feat) @ (In_Feat, Out_Feat) -> (B, N_Ch, Out_Feat)
        support = torch.matmul(node_features, self.weight)
        
        # Graph convolution: A * (H * W)
        # (B, N_Ch, N_Ch) @ (B, N_Ch, Out_Feat) -> (B, N_Ch, Out_Feat)
        output = torch.bmm(adjacency, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    
    Implements attention mechanism for graph convolution:
    h_i^(l+1) = σ(Σ_{j∈N(i)} α_{ij} W h_j^(l))
    
    where α_{ij} is the attention coefficient computed as:
    α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout rate (default: 0.2)
            alpha: Negative slope for LeakyReLU (default: 0.2)
            concat: Whether to concatenate multi-head outputs (default: True)
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features of shape (Batch_Size, N_Channels, In_Features)
            adjacency: Adjacency matrix of shape (Batch_Size, N_Channels, N_Channels)
        
        Returns:
            Output features of shape (Batch_Size, N_Channels, Out_Features)
        """
        batch_size, n_channels, in_features = node_features.shape
        
        # Linear transformation: Wh
        # (B, N_Ch, In_Feat) @ (In_Feat, Out_Feat) -> (B, N_Ch, Out_Feat)
        h = torch.matmul(node_features, self.W)
        
        # Compute attention coefficients
        # For each pair (i, j), compute a^T [Wh_i || Wh_j]
        # Expand h for pairwise computation
        h_i = h.unsqueeze(2)  # (B, N_Ch, 1, Out_Feat)
        h_j = h.unsqueeze(1)  # (B, 1, N_Ch, Out_Feat)
        
        # Concatenate: [Wh_i || Wh_j]
        h_concat = torch.cat([h_i.expand(-1, -1, n_channels, -1), h_j.expand(-1, n_channels, -1, -1)], dim=-1)
        # (B, N_Ch, N_Ch, 2*Out_Feat)
        
        # Compute attention: a^T [Wh_i || Wh_j]
        e = torch.matmul(h_concat, self.a).squeeze(-1)  # (B, N_Ch, N_Ch)
        e = self.leaky_relu(e)
        
        # Apply mask (only attend to connected nodes)
        attention_mask = adjacency > 0
        e = e.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)  # (B, N_Ch, N_Ch)
        attention = self.dropout_layer(attention)
        
        # Apply attention: Σ α_{ij} * Wh_j
        # (B, N_Ch, N_Ch) @ (B, N_Ch, Out_Feat) -> (B, N_Ch, Out_Feat)
        h_prime = torch.bmm(attention, h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphConvolutionBlock(nn.Module):
    """
    Graph Convolution Block with multiple layers
    
    Can use either GCN or GAT layers
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int = 2,
        conv_type: str = 'gcn',
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            n_layers: Number of graph convolution layers (default: 2)
            conv_type: 'gcn' or 'gat' (default: 'gcn')
            dropout: Dropout rate (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(GraphConvolutionBlock, self).__init__()
        
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.use_batch_norm = use_batch_norm
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        # First layer
        if conv_type == 'gcn':
            self.layers.append(GraphConvolution(in_features, hidden_features))
        elif conv_type == 'gat':
            self.layers.append(GraphAttentionLayer(in_features, hidden_features, dropout=dropout))
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for _ in range(n_layers - 2):
            if conv_type == 'gcn':
                self.layers.append(GraphConvolution(hidden_features, hidden_features))
            elif conv_type == 'gat':
                self.layers.append(GraphAttentionLayer(hidden_features, hidden_features, dropout=dropout))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_features))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Last layer
        if n_layers > 1:
            if conv_type == 'gcn':
                self.layers.append(GraphConvolution(hidden_features, out_features))
            elif conv_type == 'gat':
                self.layers.append(GraphAttentionLayer(hidden_features, out_features, dropout=dropout, concat=False))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_features))
            self.dropout_layers.append(nn.Dropout(dropout))
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features of shape (Batch_Size, N_Channels, In_Features)
            adjacency: Adjacency matrix of shape (Batch_Size, N_Channels, N_Channels)
        
        Returns:
            Output features of shape (Batch_Size, N_Channels, Out_Features)
        """
        x = node_features
        
        for i, layer in enumerate(self.layers):
            # Graph convolution
            x = layer(x, adjacency)
            
            # Batch normalization (applied per channel)
            if self.use_batch_norm:
                batch_size, n_channels, n_features = x.shape
                x = x.view(-1, n_features)  # (B*N_Ch, Features)
                x = self.batch_norms[i](x)
                x = x.view(batch_size, n_channels, n_features)  # (B, N_Ch, Features)
            
            # Activation
            if i < len(self.layers) - 1:  # Not last layer
                x = F.relu(x)
            
            # Dropout
            x = self.dropout_layers[i](x)
        
        return x


