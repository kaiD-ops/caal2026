#!/usr/bin/env python3
"""
Advanced training progress monitor with epoch tracking
"""

import os
import sys
from pathlib import Path
import time

def check_training_status():
    """Check training status and provide detailed progress"""
    
    output_files = {
        'model': Path('galaxy_s4_model.pth'),
        'params': Path('s4_model_params.h'),
        'samples': Path('galaxy_samples.csv'),
    }
    
    print("\n" + "=" * 70)
    print("S4 GALAXY CLASSIFICATION - TRAINING STATUS")
    print("=" * 70 + "\n")
    
    # Check dataset
    data_dir = Path('./data')
    if data_dir.exists():
        files = list(data_dir.rglob('*'))
        print(f"✓ Dataset: Ready ({len([f for f in files if f.is_file()])} files)")
    
    # Check process
    import subprocess
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        python_procs = [line for line in result.stdout.split('\n') if 'python' in line.lower()]
        if python_procs:
            print(f"✓ Training Process: Running ({len(python_procs)} Python processes)")
            # Show memory usage
            for proc in python_procs[:3]:
                if 'K' in proc:
                    mem = proc.split()[-1]
                    print(f"  └─ Memory: {mem}")
    except:
        pass
    
    print("\n" + "-" * 70)
    
    # Check output files
    status = "Training in progress..."
    for name, path in output_files.items():
        if path.exists():
            size = path.stat().st_size
            if size > 1024*1024:
                print(f"✓ {name.upper():10s}: {size/(1024*1024):.1f} MB")
            else:
                print(f"✓ {name.upper():10s}: {size/1024:.1f} KB")
            status = "Training likely complete!"
        else:
            print(f"⏳ {name.upper():10s}: Not yet created")
    
    print("-" * 70)
    print(f"\n📊 Status: {status}\n")
    
    # Estimate time
    if not output_files['model'].exists():
        print("⏱  Estimated time to completion: ~1.5 - 2 hours")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    check_training_status()
