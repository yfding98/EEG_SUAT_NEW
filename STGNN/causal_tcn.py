#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Temporal Convolutional Network (TCN) Blocks

Causal dilated convolutions that preserve temporal causality.
No information leakage from future time steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution
    
    Ensures that output at time t only depends on inputs at time <= t.
    Uses left padding: padding = (kernel_size - 1) * dilation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            dilation: Dilation rate (default: 1)
            stride: Stride (default: 1)
            groups: Groups for grouped convolution (default: 1)
            bias: Whether to use bias (default: True)
        """
        super(CausalConv1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        # Causal padding: (kernel_size - 1) * dilation
        # This ensures output[t] only depends on input[<=t]
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch_Size, In_Channels, Time_Steps)
        
        Returns:
            Output tensor of shape (Batch_Size, Out_Channels, Time_Steps)
            (Same time steps due to causal padding)
        """
        # Apply causal convolution
        x = self.conv(x)
        
        # Crop the right side to remove extra padding
        # The padding adds (kernel_size - 1) * dilation on the right
        if self.stride == 1:
            # Crop to original length
            crop_size = (self.kernel_size - 1) * self.dilation
            if crop_size > 0:
                x = x[:, :, :-crop_size]
        
        return x


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block
    
    A residual block with causal dilated convolutions.
    Architecture:
    - Causal Conv1d with dilation
    - Batch Normalization
    - ReLU activation
    - Dropout
    - Residual connection (if input/output dimensions match)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size (default: 3)
            dilation: Dilation rate (default: 1)
            dropout: Dropout rate (default: 0.2)
            use_residual: Whether to use residual connection (default: True)
        """
        super(TCNBlock, self).__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Causal convolution
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution (optional, for deeper blocks)
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 conv for residual connection if dimensions don't match
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch_Size, In_Channels, Time_Steps)
        
        Returns:
            Output tensor of shape (Batch_Size, Out_Channels, Time_Steps)
        """
        residual = x
        
        # First causal conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second causal conv + BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.use_residual:
            out = out + residual
        elif self.residual_conv is not None:
            residual = self.residual_conv(residual)
            out = out + residual
        
        out = F.relu(out)
        
        return out


class CausalTemporalExtractor(nn.Module):
    """
    Causal Temporal Feature Extractor using TCN blocks
    
    Downsamples time dimension using causal dilated convolutions.
    Preserves temporal causality - no information leakage from future.
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
            n_bands: Number of frequency bands (default: 5)
            hidden_dim: Hidden dimension (default: 128)
            reduced_time_steps: Target reduced time steps (default: 64)
            dropout: Dropout rate (default: 0.2)
            num_blocks: Number of TCN blocks (default: 4)
        """
        super(CausalTemporalExtractor, self).__init__()
        
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.reduced_time_steps = reduced_time_steps
        self.dropout = dropout
        
        # First TCN block: n_bands -> hidden_dim//2
        self.tcn1 = TCNBlock(
            in_channels=n_bands,
            out_channels=hidden_dim // 2,
            kernel_size=7,
            dilation=1,
            dropout=dropout
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Second TCN block: hidden_dim//2 -> hidden_dim
        self.tcn2 = TCNBlock(
            in_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            kernel_size=5,
            dilation=2,
            dropout=dropout
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Third TCN block: hidden_dim -> hidden_dim
        self.tcn3 = TCNBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=4,
            dropout=dropout
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Additional TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_blocks - 3):
            dilation = 2 ** (i + 3)  # 8, 16, 32, ...
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Adaptive pooling to ensure exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(reduced_time_steps)
        
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
        batch_size, n_channels, n_bands, time_steps = x.shape
        
        # Reshape: (B, N_Channels, N_Bands, Time_Steps) -> (B * N_Channels, N_Bands, Time_Steps)
        # Process each channel independently
        x = x.view(batch_size * n_channels, n_bands, time_steps)
        
        # TCN Block 1: (B*N_Ch, N_Bands, 10000) -> (B*N_Ch, Hidden_Dim//2, 10000)
        x = self.tcn1(x)
        x = self.pool1(x)  # -> (B*N_Ch, Hidden_Dim//2, 5000)
        
        # TCN Block 2: (B*N_Ch, Hidden_Dim//2, 5000) -> (B*N_Ch, Hidden_Dim, 5000)
        x = self.tcn2(x)
        x = self.pool2(x)  # -> (B*N_Ch, Hidden_Dim, 2500)
        
        # TCN Block 3: (B*N_Ch, Hidden_Dim, 2500) -> (B*N_Ch, Hidden_Dim, 2500)
        x = self.tcn3(x)
        x = self.pool3(x)  # -> (B*N_Ch, Hidden_Dim, 1250)
        
        # Additional TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Adaptive pooling to exact size
        # -> (B*N_Ch, Hidden_Dim, Reduced_Time_Steps)
        x = self.adaptive_pool(x)
        
        # Reshape back: (B * N_Channels, Hidden_Dim, Reduced_Time_Steps) 
        # -> (B, N_Channels, Hidden_Dim, Reduced_Time_Steps)
        x = x.view(batch_size, n_channels, self.hidden_dim, self.reduced_time_steps)
        
        return x


