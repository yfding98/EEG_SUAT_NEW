#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal CNN Model for Seizure Onset Detection

1D CNN that processes normalized NHFE time-series directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class TemporalCNNOnsetDetector(nn.Module):
    """1D CNN for temporal NHFE sequence classification."""
    
    def __init__(
        self,
        input_length: int = 120,
        n_channels: int = 1,
        n_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 5, 5],
        dropout: float = 0.3,
        n_classes: int = 2
    ):
        """
        Args:
            input_length: Length of input time-series
            n_channels: Number of input channels (1 for single band, n_bands for multi-band)
            n_filters: Number of filters in each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            dropout: Dropout rate
            n_classes: Number of output classes (2 for binary)
        """
        super().__init__()
        
        self.input_length = input_length
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        
        # Build convolutional layers
        conv_layers = []
        in_channels = n_channels
        
        for i, (n_filt, k_size) in enumerate(zip(n_filters, kernel_sizes)):
            conv_layers.append(
                nn.Conv1d(in_channels, n_filt, kernel_size=k_size, padding=k_size//2)
            )
            conv_layers.append(nn.BatchNorm1d(n_filt))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = n_filt
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Compute output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, input_length)
            dummy_output = self.conv_layers(dummy_input)
            conv_output_size = dummy_output.numel()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
        
        Returns:
            Logits of shape (batch, n_classes)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = F.softmax(logits, dim=1)
            return proba.cpu().numpy()
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(x)
        return (proba[:, 1] >= threshold).astype(int)


class TemporalCNNWrapper:
    """Wrapper for TemporalCNNOnsetDetector with training utilities."""
    
    def __init__(
        self,
        input_length: Optional[int] = None,
        n_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 5, 5],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Args:
            input_length: Length of input time-series. 
                         If None, will be determined from data during training.
                         Recommended: None (auto-detect) or 5000-10000 for fixed length.
            n_filters: Number of filters in each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu' or 'cuda')
        """
        self.input_length = input_length  # Can be None for auto-detection
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model only if input_length is provided
        # Otherwise, will be initialized in fit() method
        if input_length is not None:
            self.model = TemporalCNNOnsetDetector(
                input_length=input_length,
                n_channels=1,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                dropout=dropout
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = None  # Will be initialized in fit() method
            self.optimizer = None  # Will be initialized in fit() method
            self.criterion = None  # Will be initialized in fit() method
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 50,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True
    ) -> 'TemporalCNNWrapper':
        """
        Train the model.
        
        Args:
            X: Input sequences of shape (n_samples, seq_len)
            y: Binary labels of shape (n_samples,)
            batch_size: Batch size
            epochs: Number of epochs
            validation_data: Optional (X_val, y_val)
            verbose: Whether to print training progress
        
        Returns:
            Self
        """
        # Initialize model if input_length was None (auto-detect from data)
        if self.model is None:
            # Determine input_length from data
            if X.ndim == 2:
                detected_length = X.shape[1]
            else:
                detected_length = X.shape[2]
            
            self.input_length = detected_length
            if verbose:
                print(f"Auto-detected input_length: {detected_length}")
            
            self.model = TemporalCNNOnsetDetector(
                input_length=self.input_length,
                n_channels=1,
                n_filters=self.n_filters,
                kernel_sizes=self.kernel_sizes,
                dropout=self.dropout
            ).to(self.device)
            
            # Initialize optimizer and criterion
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            self.criterion = nn.CrossEntropyLoss()
        
        # Reshape X for CNN
        # Expected input shape: (n_samples, n_channels, seq_len)
        if X.ndim == 2:
            # Single band: add channel dimension
            X = X[:, np.newaxis, :]
        elif X.ndim == 3:
            # Multi-band: X is already (n_samples, n_bands, seq_len)
            # This is correct for CNN input
            pass
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}, expected 2D or 3D")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Validation dataloader
        val_dataloader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim == 2:
                X_val = X_val[:, np.newaxis, :]
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        logits = self.model(batch_X)
                        loss = self.criterion(logits, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_dataloader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.model.predict_proba(X_tensor)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_length': self.input_length,
            'device': str(self.device)
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu') -> 'TemporalCNNWrapper':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Reconstruct model architecture (assuming default)
        instance = cls(
            input_length=checkpoint['input_length'],
            device=device
        )
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return instance

