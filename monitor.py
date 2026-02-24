#!/usr/bin/env python3
"""
Monitor training progress in real-time
"""

import os
import time
from pathlib import Path

def monitor_training():
    """Display training status"""
    
    print("=" * 60)
    print("S4 GALAXY CLASSIFICATION - TRAINING MONITOR")
    print("=" * 60)
    print()
    
    # Check if data has been downloaded
    data_dir = Path("./data")
    if data_dir.exists():
        print("✓ Dataset downloaded")
        files = list(data_dir.rglob("*"))
        print(f"  Total files in data folder: {len(files)}")
    else:
        print("✗ Dataset not yet downloaded")
    
    print()
    
    # Check for model checkpoint
    if Path("galaxy_s4_model.pth").exists():
        size = os.path.getsize("galaxy_s4_model.pth") / 1024 / 1024
        print(f"✓ Model checkpoint saved: galaxy_s4_model.pth ({size:.1f} MB)")
    else:
        print("⏳ Model checkpoint not yet created (training in progress)")
    
    print()
    
    # Check for model parameters file
    if Path("s4_model_params.h").exists():
        size = os.path.getsize("s4_model_params.h") / 1024
        print(f"✓ Model parameters exported: s4_model_params.h ({size:.1f} KB)")
    else:
        print("⏳ Model parameters not yet exported")
    
    print()
    
    # Check for CSV samples
    if Path("galaxy_samples.csv").exists():
        size = os.path.getsize("galaxy_samples.csv") / 1024
        print(f"✓ Galaxy samples exported: galaxy_samples.csv ({size:.1f} KB)")
    else:
        print("⏳ Galaxy samples not yet exported")
    
    print()
    print("=" * 60)
    print("Training is running in the background...")
    print("The model will be saved when training completes.")
    print("=" * 60)

if __name__ == "__main__":
    monitor_training()
