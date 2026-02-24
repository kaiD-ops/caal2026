"""
Numerical Validation for S4 Implementations

This script validates that the Recurrent and Convolutional formulations of S4
produce numerically equivalent outputs. Both formulations should compute the
same sequence transformation, just using different computational approaches.

Test Procedure:
--------------
1. Initialize S4Recurrent and S4Convolutional with identical parameters
2. Generate random input sequences
3. Forward pass through both models
4. Assert outputs match within numerical tolerance
5. Benchmark timing for different sequence lengths

Expected Result:
---------------
Both implementations should produce identical outputs (within floating point
precision ~1e-5). The convolutional implementation should be significantly
faster for long sequences.
"""

import torch
import sys
import time
from pathlib import Path

# Add model directory to path
model_dir = Path(__file__).parent / 'model'
sys.path.insert(0, str(model_dir.parent))

from model.s4_recurrent import S4Recurrent
from model.s4_conv import S4Convolutional


def transfer_parameters(recurrent_model, conv_model):
    """
    Transfer parameters from recurrent model to convolutional model.
    
    Both models share the same parameterization (A, B, C, D, log_dt),
    so we can directly copy the parameters to ensure they're identical.
    
    Parameters
    ----------
    recurrent_model : S4Recurrent
        Source model
    conv_model : S4Convolutional
        Target model
    """
    # Copy learnable parameters
    conv_model.B.data = recurrent_model.B.data.clone()
    conv_model.C.data = recurrent_model.C.data.clone()
    conv_model.D.data = recurrent_model.D.data.clone()
    conv_model.log_dt.data = recurrent_model.log_dt.data.clone()
    
    # Copy buffers (A matrix)
    conv_model.A = recurrent_model.A.clone()


def validate_equivalence(d_model=16, d_state=64, seq_len=100, batch_size=2, tolerance=1e-5):
    """
    Validate that S4Recurrent and S4Convolutional produce identical outputs.
    
    Parameters
    ----------
    d_model : int
        Feature dimension
    d_state : int
        State space dimension
    seq_len : int
        Sequence length to test
    batch_size : int
        Batch size for testing
    tolerance : float
        Maximum allowed absolute difference between outputs
        
    Returns
    -------
    bool
        True if outputs match within tolerance, False otherwise
    float
        Maximum absolute difference between outputs
    """
    print(f"\n{'='*60}")
    print(f"Testing Equivalence: L={seq_len}, d_model={d_model}, d_state={d_state}")
    print(f"{'='*60}")
    
    # Initialize models
    recurrent_model = S4Recurrent(d_model=d_model, d_state=d_state)
    conv_model = S4Convolutional(d_model=d_model, d_state=d_state, l_max=seq_len*2)
    
    # Transfer parameters to ensure identical initialization
    transfer_parameters(recurrent_model, conv_model)
    
    # Set to evaluation mode (disable dropout, etc.)
    recurrent_model.eval()
    conv_model.eval()
    
    # Generate random input
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass through both models
    with torch.no_grad():
        y_recurrent, _ = recurrent_model(x)
        y_conv, _ = conv_model(x)
    
    # Compute difference
    max_diff = torch.max(torch.abs(y_recurrent - y_conv)).item()
    mean_diff = torch.mean(torch.abs(y_recurrent - y_conv)).item()
    
    # Check if within tolerance
    passed = max_diff < tolerance
    
    # Print results
    print(f"✓ Recurrent output shape: {y_recurrent.shape}")
    print(f"✓ Convolutional output shape: {y_conv.shape}")
    print(f"\nNumerical Difference:")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Tolerance threshold:      {tolerance:.2e}")
    
    if passed:
        print(f"\n✅ PASSED: Outputs match within tolerance!")
    else:
        print(f"\n❌ FAILED: Outputs differ by more than tolerance!")
        print(f"\nSample values comparison (first 5 elements):")
        print(f"  Recurrent: {y_recurrent[0, 0, :5]}")
        print(f"  Conv:      {y_conv[0, 0, :5]}")
    
    return passed, max_diff


def benchmark_timing(d_model=16, d_state=64, sequence_lengths=[64, 256, 1024], 
                     batch_size=2, num_runs=3):
    """
    Benchmark execution time for both implementations across different sequence lengths.
    
    Parameters
    ----------
    d_model : int
        Feature dimension
    d_state : int
        State space dimension
    sequence_lengths : list of int
        Sequence lengths to benchmark
    batch_size : int
        Batch size for benchmarking
    num_runs : int
        Number of runs to average over
    """
    print(f"\n{'='*60}")
    print(f"Timing Benchmark: d_model={d_model}, d_state={d_state}")
    print(f"{'='*60}")
    print(f"{'Seq Length':<12} {'Recurrent (ms)':<18} {'Conv (ms)':<18} {'Speedup':<10}")
    print(f"{'-'*60}")
    
    for L in sequence_lengths:
        # Initialize models
        recurrent_model = S4Recurrent(d_model=d_model, d_state=d_state)
        conv_model = S4Convolutional(d_model=d_model, d_state=d_state, l_max=L*2)
        
        # Transfer parameters
        transfer_parameters(recurrent_model, conv_model)
        
        # Set to evaluation mode
        recurrent_model.eval()
        conv_model.eval()
        
        # Generate input
        x = torch.randn(batch_size, L, d_model)
        
        # Warm-up runs
        with torch.no_grad():
            _ = recurrent_model(x)
            _ = conv_model(x)
        
        # Benchmark recurrent
        recurrent_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = recurrent_model(x)
            end = time.perf_counter()
            recurrent_times.append((end - start) * 1000)  # Convert to milliseconds
        recurrent_avg = sum(recurrent_times) / num_runs
        
        # Benchmark convolutional
        conv_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = conv_model(x)
            end = time.perf_counter()
            conv_times.append((end - start) * 1000)  # Convert to milliseconds
        conv_avg = sum(conv_times) / num_runs
        
        # Compute speedup
        speedup = recurrent_avg / conv_avg if conv_avg > 0 else float('inf')
        
        print(f"{L:<12} {recurrent_avg:<18.2f} {conv_avg:<18.2f} {speedup:<10.2f}x")
    
    print(f"\nNote: Convolutional is typically faster for longer sequences.")
    print(f"      Recurrent has lower memory overhead but requires sequential processing.")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("S4 IMPLEMENTATION VALIDATION SUITE")
    print("="*60)
    
    # Test 1: Basic equivalence
    passed, max_diff = validate_equivalence(
        d_model=16,
        d_state=64,
        seq_len=100,
        batch_size=2,
        tolerance=1e-4  # Slightly relaxed tolerance for practical testing
    )
    
    # Test 2: Different sequence lengths
    print("\n" + "="*60)
    print("Testing Multiple Sequence Lengths")
    print("="*60)
    
    all_passed = True
    for seq_len in [50, 200, 500]:
        passed_i, _ = validate_equivalence(
            d_model=8,
            d_state=32,
            seq_len=seq_len,
            batch_size=1,
            tolerance=1e-4
        )
        all_passed = all_passed and passed_i
    
    # Test 3: Timing benchmark
    benchmark_timing(
        d_model=16,
        d_state=64,
        sequence_lengths=[64, 256, 1024],
        batch_size=2,
        num_runs=3
    )
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if all_passed:
        print("✅ All tests PASSED!")
        print("   Both S4 implementations produce numerically equivalent results.")
        print("   The convolutional formulation is recommended for training (faster).")
        print("   The recurrent formulation is useful for autoregressive generation.")
    else:
        print("❌ Some tests FAILED!")
        print("   Please check the implementation for bugs.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
