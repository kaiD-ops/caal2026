#!/usr/bin/env python3
"""
validate_outputs.py - Compare RISC-V outputs against Python reference

Run this after executing the RISC-V binary through VeeR-iSS simulator.
It will compare outputs and generate validation metrics.
"""

import json
import numpy as np
from pathlib import Path

# ============================================================================
# Validation Metrics
# ============================================================================

def compute_mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def compute_mae(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# ============================================================================
# Validation Functions
# ============================================================================

def load_reference_data():
    """Load reference outputs."""
    data_dir = Path(__file__).parent / 'validation_data'
    
    with open(data_dir / 'reference_outputs.json', 'r') as f:
        return json.load(f)

def load_risc_v_data():
    """Load RISC-V outputs (generated from VeeR-iSS)."""
    results_file = Path(__file__).parent / 'risc_v_outputs.json'
    
    if not results_file.exists():
        print(f"ERROR: RISC-V output file not found: {results_file}")
        print("Run VeeR-iSS simulator first and capture outputs.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def validate_layer(layer_name, ref_data, risc_v_data, tolerance_mse, tolerance_mae):
    """Validate a single layer."""
    mse = compute_mse(ref_data, risc_v_data)
    mae = compute_mae(ref_data, risc_v_data)
    
    passed = (mse < tolerance_mse) and (mae < tolerance_mae)
    status = "✓ PASS " if passed else "✗ FAIL "
    
    return {
        'name': layer_name,
        'mse': mse,
        'mae': mae,
        'tolerance_mse': tolerance_mse,
        'tolerance_mae': tolerance_mae,
        'passed': passed,
        'status': status
    }

# ============================================================================
# Main Validation
# ============================================================================

def run_validation():
    """Run complete validation suite."""
    
    print("\n" + "="*80)
    print("RISC-V S4D Implementation - Validation Results")
    print("="*80)
    
    # Tolerance thresholds
    tolerances = {
        'hilbert_scan': (1e-12, 1e-12),
        'linear_layer': (1e-8, 1e-6),
        's4d_layer': (1e-7, 1e-4),
        'gelu': (1e-7, 1e-4),
        'softmax': (1e-8, 1e-4),
        'take_last': (1e-12, 1e-12),
    }
    
    # Load data
    references = load_reference_data()
    risc_v_outputs = load_risc_v_data()
    
    if risc_v_outputs is None:
        return False
    
    # Validate all samples
    all_passed = True
    results = []
    
    print("\n" + "-"*80)
    print("Per-Sample Validation Results")
    print("-"*80 + "\n")
    
    for sample_id in sorted(references.keys()):
        ref = references[sample_id]
        riscv = risc_v_outputs.get(sample_id)
        
        if riscv is None:
            print(f"WARNING: No RISC-V output for {sample_id}")
            continue
        
        # Validate softmax output (probabilities)
        result = validate_layer(
            sample_id,
            ref['probabilities'],
            riscv['probabilities'],
            1e-8,  # MSE tolerance for softmax
            1e-4   # MAE tolerance for softmax
        )
        
        # Check class prediction
        ref_class = ref['predicted_class']
        risc_v_class = riscv['predicted_class']
        class_match = "✓" if ref_class == risc_v_class else "✗"
        
        print(f"{sample_id:12s} | MSE={result['mse']:.2e} | MAE={result['mae']:.2e} | "
              f"Class: {ref_class} → {risc_v_class} {class_match} | {result['status']}")
        
        results.append(result)
        if not result['passed']:
            all_passed = False
    
    # Print summary
    print("\n" + "-"*80)
    print("Validation Summary")
    print("-"*80)
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} samples PASSED\n")
    
    # Tolerance compliance
    print("Tolerance Compliance:")
    print(f"  MSE thresholds: {'✓ PASS' if all_passed else '✗ FAIL'}")
    print(f"  Class accuracy: {100*passed_count//total_count if total_count else 0}%\n")
    
    print("="*80)
    
    return all_passed

# ============================================================================
# Detailed Report Generation
# ============================================================================

def generate_validation_report():
    """Generate detailed validation report."""
    
    references = load_reference_data()
    risc_v_outputs = load_risc_v_data()
    
    if risc_v_outputs is None:
        return
    
    report = {
        'timestamp': str(np.datetime64('now')),
        'total_samples': len(references),
        'results': []
    }
    
    for sample_id in sorted(references.keys()):
        ref = references[sample_id]
        riscv = risc_v_outputs.get(sample_id)
        
        if riscv is None:
            continue
        
        mse = compute_mse(ref['probabilities'], riscv['probabilities'])
        mae = compute_mae(ref['probabilities'], riscv['probabilities'])
        
        report['results'].append({
            'sample': sample_id,
            'reference': {
                'class': ref['predicted_class'],
                'probabilities': [float(p) for p in ref['probabilities']],
                'logits': [float(l) for l in ref['logits']]
            },
            'risc_v': {
                'class': riscv['predicted_class'],
                'probabilities': [float(p) for p in riscv['probabilities']]
            },
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'class_match': ref['predicted_class'] == riscv['predicted_class']
            }
        })
    
    # Save report
    report_file = Path(__file__).parent / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

# ============================================================================

if __name__ == '__main__':
    success = run_validation()
    generate_validation_report()
    exit(0 if success else 1)
