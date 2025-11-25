#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for STGNN training
"""

from train_stgnn import train_stgnn

if __name__ == '__main__':
    # Default paths - modify as needed
    train_stgnn(
        csv_path="E:/output/segment_results/nhfe_channel_label.csv",
        data_root="E:/output/segment_results",
        raw_data_root="E:/DataSet/EEG/EEG dataset_SUAT_processed",
        config_path=None,  # Optional: path to config.yaml
        output_dir="checkpoints_stgnn"
    )


