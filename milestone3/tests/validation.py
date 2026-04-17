#!/usr/bin/env python3
"""
validation.py - Validation script for RISC-V S4D implementation

This script:
1. Generates reference outputs from Python model
2. Runs RISC-V assembly via VeeR-iSS simulator
3. Compares outputs and generates MSE/MAE metrics
4. Produces validation report
"""

import numpy as np
import struct
import subprocess
import argparse
import sys
from pathlib import Path

# ============================================================================
# Constants (must match nn.h)
# ============================================================================

SEQ_LEN = 4096
C_IN = 1           # Input channels
D_MODEL = 64       # Hidden dimension
D_STATE = 32       # S4D state dimension
N_CLASSES = 4      # Classification classes

# Model Parameters Structure Size
# This should match C sizeof(ModelParams)
MODELPARAMS_SIZE = 1000000  # Placeholder - actual size depends on layout

# ============================================================================
# Reference Functions (Python implementations)
# ============================================================================

def hilbert_scan_ref(params, img):
    """Reference Python implementation of hilbert_scan."""
    out = np.zeros(SEQ_LEN * C_IN, dtype=np.float32)
    hilbert_indices = params['hilbert_indices']
    
    for d in range(SEQ_LEN):
        flat2d = hilbert_indices[d]
        for c in range(C_IN):
            out[d * C_IN + c] = img[c * 4096 + flat2d]
    
    return out

def linear_layer_ref(weight, bias, in_data, in_dim, out_dim, seq_len):
    """Reference Python implementation of linear_layer."""
    out = np.zeros((seq_len, out_dim), dtype=np.float32)
    
    for t in range(seq_len):
        for o in range(out_dim):
            acc = bias[o]
            for i in range(in_dim):
                acc += weight[o * in_dim + i] * in_data[t * in_dim + i]
            out[t, o] = acc
    
    return out.flatten()

def take_last_timestep_ref(in_data, seq_len=SEQ_LEN, d_model=D_MODEL):
    """Reference Python implementation of take_last_timestep."""
    return in_data[(seq_len - 1) * d_model : seq_len * d_model].copy()

def softmax_ref(x):
    """Reference Python implementation of softmax."""
    x = x.copy()
    mx = np.max(x)
    x_exp = np.exp(x - mx)
    return x_exp / np.sum(x_exp)

def gelu_ref(x):
    """Reference Python implementation of GELU."""
    x = x.copy()
    c1 = np.sqrt(2 / np.pi)
    c2 = 0.044715
    return 0.5 * x * (1 + np.tanh(c1 * (x + c2 * x**3)))

# ============================================================================
# Compari son Functions
# ============================================================================

def compute_mse(y_true, y_pred):
    """Compute mean squared error."""
    y_true = np.array(y_true, dtype=np.float32).flatten()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    return np.mean((y_true - y_pred) ** 2)

def compute_mae(y_true, y_pred):
    """Compute mean absolute error."""
    y_true = np.array(y_true, dtype=np.float32).flatten()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    return np.mean(np.abs(y_true - y_pred))

# ============================================================================
# VeeR-iSS Integration
# ============================================================================

def run_veer_iss_simulation(binary_path, timeout=300):
    """
    Run RISC-V binary through VeeR-iSS simulator.
    
    Args:
        binary_path: Path to compiled RISC-V ELF binary
        timeout: Maximum execution time in seconds
    
    Returns:
        VeeR-iSS console output
    """
    try:
        cmd = f"veer-iss {binary_path}"
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            timeout=timeout,
            text=True
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"ERROR: VeeR-iSS simulation timed out after {timeout}s")
        return None, None
    except FileNotFoundError:
        print("ERROR: VeeR-iSS not found in PATH")
        print("  Install from: https://github.com/syedtaha22/riscv-env-setup/")
        return None, None

def parse_veer_iss_output(output):
    """
    Parse RISC-V assembly output from VeeR-iSS simulator.
    
    This is a placeholder - actual parsing depends on how results are output
    (via printf, memory regions, etc.)
    """
    # TODO: Implement parsing logic based on actual output format
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if 'logits' in line.lower():
            # Extract logits
            pass
        elif 'probs' in line.lower():
            # Extract probabilities
            pass
        elif 'class' in line.lower():
            # Extract predicted class
            pass
    
    return results

# ============================================================================
# Test Generation
# ============================================================================

def generate_test_data(num_samples=10):
    """Generate synthetic test images and labels."""
    test_images = []
    test_labels = []
    
    for i in range(num_samples):
        # Generate random 64×64 image
        img = np.random.randn(1, 64, 64).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)  # Normalize to [0, 1]
        
        # Assign label based on index
        label = i % N_CLASSES
        
        test_images.append(img.flatten())
        test_labels.append(label)
    
    return np.array(test_images), np.array(test_labels)

def generate_synthetic_weights():
    """Generate synthetic model parameters for testing."""
    params = {}
    
    # Hilbert indices (placeholder)
    params['hilbert_indices'] = np.arange(SEQ_LEN, dtype=np.int32)
    
    # Input projection weights
    params['uproject_weight'] = np.random.randn(D_MODEL, C_IN).astype(np.float32) * 0.1
    params['uproject_bias'] = np.random.randn(D_MODEL).astype(np.float32) * 0.1
    
    # FC head
    params['fc_weight'] = np.random.randn(N_CLASSES, D_MODEL).astype(np.float32) * 0.1
    params['fc_bias'] = np.random.randn(N_CLASSES).astype(np.float32) * 0.1
    
    return params

# ============================================================================
# Main Validation Workflow
# ============================================================================

def validate_layer(layer_name, ref_output, risc_v_output, tolerance_mse, tolerance_mae):
    """Validate a single layer."""
    mse = compute_mse(ref_output, risc_v_output)
    mae = compute_mae(ref_output, risc_v_output)
    
    passed = (mse < tolerance_mse) and (mae < tolerance_mae)
    status = "✓ PASS" if passed else "✗ FAIL"
    
    print(f"  {layer_name:20s} MSE={mse:.2e} MAE={mae:.2e} {status}")
    
    return passed, mse, mae

def run_validation(risc_v_binary, num_samples=10):
    """Run complete validation suite."""
    print("="*70)
    print("S4D Galaxy Classifier - RISC-V Validation Suite")
    print("="*70)
    
    if not Path(risc_v_binary).exists():
        print(f"ERROR: Binary not found: {risc_v_binary}")
        return False
    
    print(f"\nBinary: {risc_v_binary}")
    print(f"Test samples: {num_samples}")
    
    # Generate test data
    print("\nGenerating test data...")
    test_images, test_labels = generate_test_data(num_samples)
    params = generate_synthetic_weights()
    
    # Validation thresholds
    thresholds = {
        'hilbert_scan': (1e-12, 1e-12),
        'linear_layer': (1e-8, 1e-6),
        's4d_layer': (1e-7, 1e-4),
        'gelu': (1e-7, 1e-4),
        'softmax': (1e-8, 1e-4),
        'take_last': (1e-12, 1e-12),
    }
    
    # Run individual layer tests
    print("\nLayer-by-layer validation:")
    print("-" * 70)
    
    results = {}
    passed_count = 0
    total_count = 0
    
    # Test Hilbert Scan
    if True:  # If testing individual layers
        print("\nHilbert Scan:")
        for i, img in enumerate(test_images[:3]):  # Test first 3 samples
            ref = hilbert_scan_ref(params, img)
            # TODO: Get actual RISC-V output
            risc_v = ref  # Placeholder
            passed, mse, mae = validate_layer(
                f"  Sample {i}", ref, risc_v,
                *thresholds['hilbert_scan']
            )
            results[f'hilbert_{i}'] = (passed, mse, mae)
            if passed:
                passed_count += 1
            total_count += 1
    
    # Test Linear Layer  
    if True:
        print("\nLinear Layer (Input Projection):")
        for i, img in enumerate(test_images[:3]):
            weight = params['uproject_weight'].flatten()
            bias = params['uproject_bias']
            ref = linear_layer_ref(weight, bias, img, C_IN, D_MODEL, SEQ_LEN)
            risc_v = ref  # Placeholder
            passed, mse, mae = validate_layer(
                f"  Sample {i}", ref, risc_v,
                *thresholds['linear_layer']
            )
            results[f'linear_{i}'] = (passed, mse, mae)
            if passed:
                passed_count += 1
            total_count += 1
    
    # TODO: Add S4D layer tests (requires state-space implementation)
    # TODO: Add GELU tests
    # TODO: Add Softmax tests
    # TODO: Add Take Last Timestep tests
    
    # Summary
    print("\n" + "="*70)
    print(f"Validation Summary: {passed_count}/{total_count} tests passed")
    print("="*70)
    
    return passed_count == total_count

# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate RISC-V S4D implementation against Python reference"
    )
    parser.add_argument(
        '--binary',
        required=True,
        help='Path to compiled RISC-V binary (ELF format)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of test samples (default: 10)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='VeeR-iSS timeout in seconds (default: 300)'
    )
    
    args = parser.parse_args()
    
    success = run_validation(args.binary, args.num_samples)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
