#!/usr/bin/env python3
"""
Member 1 Validation Suite - Generate Reference Outputs for Assembly Layers

This script generates test data and reference outputs for each RISC-V assembly layer
implemented by Member 1. These reference outputs will be compared against assembly
implementations running on VeeR-iSS.

Layers validated:
  1. hilbert_scan: Pure integer indexing - must have MSE < 1e-12 (exact match)
  2. linear_layer: Matrix multiplication - must have MSE < 1e-6
  3. take_last_timestep: Extraction - should have MSE < 1e-12 (exact match)
  4. softmax_inplace: Normalization - must have MSE < 1e-6
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hilbert import HilbertScan
from model.s4d import S4D
from model.tlts import TakeLastTimestep
from model.gclassifier import GalaxyClassifierS4D

# Constants matching C implementation
SEQ_LEN = 4096
D_MODEL = 64
D_STATE = 64
C_IN = 1
N_CLASSES = 4
IMAGE_SIZE = 64


def generate_hilbert_indices():
    """Generate Hilbert curve indices for 64x64 grid."""
    hilbert_scan = HilbertScan()
    return hilbert_scan.indices.cpu().numpy()


def generate_test_image(channels=1, height=64, width=64, seed=42):
    """Generate random test image."""
    np.random.seed(seed)
    return np.random.randn(1, channels, height, width).astype(np.float32)


def validate_hilbert_scan():
    """Validate hilbert_scan layer.
    
    Expected: Pure integer indexing with no floating point error.
    MSE threshold: < 1e-12 (should be exact match)
    """
    print("\n" + "="*70)
    print("LAYER 1: HILBERT SCAN VALIDATION")
    print("="*70)
    
    # Generate test image
    test_img = generate_test_image(channels=1)
    
    # PyTorch reference
    hilbert_scan = HilbertScan()
    with torch.no_grad():
        output_torch = hilbert_scan(torch.from_numpy(test_img))
    output_ref = output_torch.cpu().numpy()  # Shape: (1, 4096, 1)
    
    print(f"✓ Input shape: {test_img.shape}")
    print(f"✓ Output shape: {output_ref.shape}")
    print(f"✓ Number of unique values in output: {len(np.unique(output_ref.flatten()))}")
    
    # Save reference output
    reference_file = "validation_data/hilbert_output_ref.npy"
    Path("validation_data").mkdir(exist_ok=True)
    np.save(reference_file, output_ref.astype(np.float32))
    
    # Save test image
    test_file = "validation_data/hilbert_input.npy"
    np.save(test_file, test_img.astype(np.float32))
    
    # Save hilbert indices for C code
    indices = generate_hilbert_indices()
    indices_file = "validation_data/hilbert_indices.bin"
    indices.astype(np.int32).tofile(indices_file)
    
    print(f"✓ Reference output saved to {reference_file}")
    print(f"✓ Test input saved to {test_file}")
    print(f"✓ Hilbert indices saved to {indices_file}")
    print(f"✓ Expected MSE when comparing assembly output: < 1e-12")
    
    return output_ref, test_img, indices


def validate_linear_layer():
    """Validate linear_layer (input projection).
    
    Expected: Matrix multiplication with float32 precision.
    MSE threshold: < 1e-6
    """
    print("\n" + "="*70)
    print("LAYER 2: LINEAR LAYER VALIDATION")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    input_data = np.random.randn(SEQ_LEN, C_IN).astype(np.float32)
    weight = np.random.randn(D_MODEL, C_IN).astype(np.float32)
    bias = np.random.randn(D_MODEL).astype(np.float32)
    
    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    weight_torch = torch.from_numpy(weight)
    bias_torch = torch.from_numpy(bias)
    
    with torch.no_grad():
        output_torch = torch.nn.functional.linear(input_torch, weight_torch, bias_torch)
    
    output_ref = output_torch.numpy()  # Shape: (4096, 64)
    
    print(f"✓ Input shape: {input_data.shape}")
    print(f"✓ Weight shape: {weight.shape}")
    print(f"✓ Bias shape: {bias.shape}")
    print(f"✓ Output shape: {output_ref.shape}")
    print(f"✓ Output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
    
    # Save test data
    Path("validation_data").mkdir(exist_ok=True)
    np.save("validation_data/linear_input.npy", input_data)
    np.save("validation_data/linear_weight.npy", weight)
    np.save("validation_data/linear_bias.npy", bias)
    np.save("validation_data/linear_output_ref.npy", output_ref)
    
    print(f"✓ Test data saved to validation_data/linear_*.npy")
    print(f"✓ Expected MSE when comparing assembly output: < 1e-6")
    
    return output_ref, input_data, weight, bias


def validate_take_last_timestep():
    """Validate take_last_timestep layer.
    
    Expected: Simple extraction of last row - must be exact match.
    MSE threshold: < 1e-12
    """
    print("\n" + "="*70)
    print("LAYER 3: TAKE LAST TIMESTEP VALIDATION")
    print("="*70)
    
    # Generate test data: (4096, 64) sequence
    np.random.seed(42)
    input_seq = np.random.randn(SEQ_LEN, D_MODEL).astype(np.float32)
    
    # Reference: just take last row
    output_ref = input_seq[-1, :]  # Shape: (64,)
    
    print(f"✓ Input shape: {input_seq.shape}")
    print(f"✓ Output shape: {output_ref.shape}")
    print(f"✓ Output value range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
    
    # Save test data
    Path("validation_data").mkdir(exist_ok=True)
    np.save("validation_data/tlts_input.npy", input_seq)
    np.save("validation_data/tlts_output_ref.npy", output_ref)
    
    print(f"✓ Test data saved to validation_data/tlts_*.npy")
    print(f"✓ Expected MSE when comparing assembly output: < 1e-12 (exact match)")
    
    return output_ref, input_seq


def validate_softmax():
    """Validate softmax_inplace layer.
    
    Expected: Numerically stable softmax normalization.
    MSE threshold: < 1e-6
    Output constraint: probabilities must sum to 1.0 ± 1e-6
    """
    print("\n" + "="*70)
    print("LAYER 4: SOFTMAX VALIDATION")
    print("="*70)
    
    # Generate test data: (4,) logit vector
    np.random.seed(42)
    logits = np.random.randn(N_CLASSES).astype(np.float32)
    
    # Reference: numerically stable softmax
    logits_torch = torch.from_numpy(logits)
    with torch.no_grad():
        # PyTorch softmax
        output_torch = torch.nn.functional.softmax(logits_torch, dim=0)
    
    output_ref = output_torch.numpy()  # Shape: (4,)
    
    print(f"✓ Input logits: {logits}")
    print(f"✓ Output probabilities: {output_ref}")
    print(f"✓ Sum of probabilities: {output_ref.sum():.10f}")
    print(f"✓ Min probability: {output_ref.min():.10f}")
    print(f"✓ Max probability: {output_ref.max():.10f}")
    
    # Verify sum constraint
    prob_sum = output_ref.sum()
    if abs(prob_sum - 1.0) < 1e-5:
        print(f"✓ Sum constraint satisfied: {prob_sum:.10f} ≈ 1.0")
    else:
        print(f"⚠ WARNING: Sum constraint violated: {prob_sum:.10f}")
    
    # Save test data
    Path("validation_data").mkdir(exist_ok=True)
    np.save("validation_data/softmax_input.npy", logits)
    np.save("validation_data/softmax_output_ref.npy", output_ref)
    
    print(f"✓ Test data saved to validation_data/softmax_*.npy")
    print(f"✓ Expected MSE when comparing assembly output: < 1e-6")
    
    return output_ref, logits


def main():
    """Run all validation tests and generate reference outputs."""
    print("\n" + "="*70)
    print("MEMBER 1 VALIDATION SUITE")
    print("Generate Reference Outputs for Assembly Layers")
    print("="*70)
    
    try:
        # Validate each layer in sequence
        hilbert_out, hilbert_in, hilbert_idx = validate_hilbert_scan()
        linear_out, linear_in, linear_w, linear_b = validate_linear_layer()
        tlts_out, tlts_in = validate_take_last_timestep()
        softmax_out, softmax_in = validate_softmax()
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print("✓ All reference outputs generated successfully")
        print("\nNext steps:")
        print("1. Copy assembly files to VeeR-iSS environment")
        print("2. Assemble and link with C runtime")
        print("3. Run test harnesses and capture outputs")
        print("4. Compare assembly outputs with reference outputs in validation_data/")
        print("5. Record MSE and pass/fail status in validation_m1.txt")
        
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
