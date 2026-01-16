#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN Model Visualization Tool

This script generates visualizations of the STGNN model structure.
It attempts to use torchviz to generate a computation graph image.
"""

import sys
import os
from pathlib import Path
import torch
from torchviz import make_dot
import logging

# Add path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from model_wrapper import create_model
except ImportError:
    # If running from root directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from DeepSOZ.train.model_wrapper import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_model():
    output_dir = Path(__file__).parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Instantiating STGNN model...")
    # Create model with default config
    model = create_model('channel')
    model.eval()
    
    # Create dummy input: (Batch, Windows, Channels, TimePoints)
    # Default: 30 windows, 19 channels, 200 points
    dummy_input = torch.randn(1, 30, 19, 200)
    
    logger.info("Generating computation graph...")
    try:
        # Forward pass to trace the graph
        y = model(dummy_input)
        
        # Generate dot object
        # params=dict(model.named_parameters()) include parameters in graph
        dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        
        # Save
        output_path = output_dir / 'stgnn_architecture'
        dot.format = 'png'
        dot.render(str(output_path))
        
        logger.info(f"Visualization saved to: {output_path}.png")
        print(f"\nSuccessfully generated model graph at: {output_path}.png")
        
    except Exception as e:
        logger.error(f"Failed to generate visualization with torchviz: {e}")
        logger.info("Make sure 'torchviz' and 'graphviz' are installed.")
        logger.info("pip install torchviz")
        logger.info("And install Graphviz from https://graphviz.org/download/")

    # Also print text structure
    print("\nModel Structure (Text):")
    print("="*80)
    print(model)
    print("="*80)
    
    with open(output_dir / 'model_structure.txt', 'w') as f:
        f.write(str(model))
    logger.info(f"Text structure saved to: {output_dir / 'model_structure.txt'}")

if __name__ == '__main__':
    visualize_model()
