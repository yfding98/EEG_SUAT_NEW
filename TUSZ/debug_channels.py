#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check which montage types have FZ, CZ, PZ channels"""

import sys
sys.path.insert(0, '.')
from pathlib import Path

from config import normalize_channel_name
from data_loader import read_edf, detect_montage_type

data_root = Path('F:/dataset/TUSZ/v2.0.3/edf/train')

# Sample some files from different montages
montage_samples = {}

for patient_dir in list(data_root.iterdir())[:5]:
    if not patient_dir.is_dir():
        continue
    for session_dir in patient_dir.iterdir():
        if not session_dir.is_dir():
            continue
        for montage_dir in session_dir.iterdir():
            if not montage_dir.is_dir():
                continue
            
            montage = detect_montage_type(str(montage_dir))
            if montage not in montage_samples:
                # Get first EDF file
                edf_files = list(montage_dir.glob('*.edf'))
                if edf_files:
                    montage_samples[montage] = edf_files[0]

print("Checking FZ/CZ/PZ availability by montage type:")
print("=" * 70)

for montage, edf_path in sorted(montage_samples.items()):
    try:
        data, fs, ch_names = read_edf(str(edf_path))
        
        # Build normalized channel map
        ch_map = set()
        for name in ch_names:
            normalized = normalize_channel_name(name)
            ch_map.add(normalized)
        
        has_fz = 'FZ' in ch_map
        has_cz = 'CZ' in ch_map
        has_pz = 'PZ' in ch_map
        
        status = "[OK]" if (has_fz and has_cz and has_pz) else "[MISSING]"
        print(f"{status} {montage:15s} - FZ:{has_fz}, CZ:{has_cz}, PZ:{has_pz}")
        
        if not (has_fz and has_cz and has_pz):
            # Show what channels are available
            eeg_channels = [n for n in ch_map if len(n) <= 4]
            print(f"   Available: {sorted(eeg_channels)}")
    except Exception as e:
        print(f"  {montage}: Error - {e}")
