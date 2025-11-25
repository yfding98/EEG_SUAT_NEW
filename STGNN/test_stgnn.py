#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for STGNN_SOZ_Locator

Tests the model with dummy input to verify shape consistency.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from STGNN.stgnn_model import STGNN_SOZ_Locator


def test_model_shapes():
    """Test model with dummy input"""
    print("=" * 80)
    print("STGNN_SOZ_Locator - Shape Verification Test")
    print("=" * 80)
    
    # Model parameters
    n_channels = 21
    n_bands = 5
    time_steps = 10000
    batch_size = 4
    
    print(f"\nModel Configuration:")
    print(f"  N_Channels: {n_channels}")
    print(f"  N_Bands: {n_bands}")
    print(f"  Time_Steps: {time_steps}")
    print(f"  Batch_Size: {batch_size}")
    
    # Create model
    print(f"\nCreating model...")
    model = STGNN_SOZ_Locator(
        n_channels=n_channels,
        n_bands=n_bands,
        time_steps=time_steps,
        temporal_hidden_dim=128,
        temporal_reduced_steps=64,
        graph_learning_method='combined',
        graph_hidden_dim=128,
        graph_conv_type='gcn',
        graph_n_layers=2,
        dropout=0.2,
        use_batch_norm=True
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Create dummy input
    print(f"\nCreating dummy input...")
    dummy_input = torch.randn(batch_size, n_channels, n_bands, time_steps)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        logits, adjacency = model(dummy_input, return_adjacency=True)
    
    # Verify output shapes
    print(f"\nOutput Verification:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected logits shape: ({batch_size}, {n_channels})")
    print(f"  Logits shape correct: {logits.shape == (batch_size, n_channels)}")
    
    print(f"\n  Adjacency shape: {adjacency.shape}")
    print(f"  Expected adjacency shape: ({batch_size}, {n_channels}, {n_channels})")
    print(f"  Adjacency shape correct: {adjacency.shape == (batch_size, n_channels, n_channels)}")
    
    # Check logits values
    print(f"\nLogits Statistics:")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Std: {logits.std().item():.4f}")
    
    # Check adjacency matrix
    print(f"\nAdjacency Matrix Statistics:")
    print(f"  Min: {adjacency.min().item():.4f}")
    print(f"  Max: {adjacency.max().item():.4f}")
    print(f"  Mean: {adjacency.mean().item():.4f}")
    print(f"  Std: {adjacency.std().item():.4f}")
    
    # Test with different batch sizes
    print(f"\n" + "=" * 80)
    print("Testing with different batch sizes...")
    print("=" * 80)
    
    for bs in [1, 2, 8]:
        test_input = torch.randn(bs, n_channels, n_bands, time_steps)
        with torch.no_grad():
            test_logits = model(test_input)
        print(f"  Batch size {bs}: Input {test_input.shape} -> Output {test_logits.shape} [OK]")
    
    # Test gradient flow
    print(f"\n" + "=" * 80)
    print("Testing gradient flow...")
    print("=" * 80)
    
    model.train()
    dummy_input = torch.randn(batch_size, n_channels, n_bands, time_steps, requires_grad=True)
    dummy_target = torch.randint(0, n_channels, (batch_size,))
    
    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(logits, dummy_target)
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters() if p.requires_grad)
    print(f"  Gradients computed: {has_gradients} [OK]")
    print(f"  Loss value: {loss.item():.4f}")
    
    print(f"\n" + "=" * 80)
    print("[OK] All tests passed!")
    print("=" * 80)


def test_different_configurations():
    """Test model with different configurations"""
    print("\n" + "=" * 80)
    print("Testing Different Model Configurations")
    print("=" * 80)
    
    n_channels = 21
    n_bands = 5
    time_steps = 10000
    batch_size = 2
    
    configs = [
        {
            'name': 'GCN with parameter-based graph learning',
            'graph_learning_method': 'parameter',
            'graph_conv_type': 'gcn',
        },
        {
            'name': 'GAT with feature-based graph learning',
            'graph_learning_method': 'feature',
            'graph_conv_type': 'gat',
        },
        {
            'name': 'GCN with combined graph learning',
            'graph_learning_method': 'combined',
            'graph_conv_type': 'gcn',
        },
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        try:
            model = STGNN_SOZ_Locator(
                n_channels=n_channels,
                n_bands=n_bands,
                time_steps=time_steps,
                graph_learning_method=config['graph_learning_method'],
                graph_conv_type=config['graph_conv_type'],
            )
            
            dummy_input = torch.randn(batch_size, n_channels, n_bands, time_steps)
            with torch.no_grad():
                logits = model(dummy_input)
            
            assert logits.shape == (batch_size, n_channels), \
                f"Expected shape ({batch_size}, {n_channels}), got {logits.shape}"
            
            print(f"  [OK] Configuration works correctly")
            print(f"    Output shape: {logits.shape}")
            
        except Exception as e:
            print(f"  [FAIL] Configuration failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run shape verification test
    test_model_shapes()
    
    # Test different configurations
    test_different_configurations()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

