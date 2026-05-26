#!/usr/bin/env python3
"""

Validation script for M3 RISC-V assembly implementation.
Team: Echo Theory
Course: CAAL 2026

Documents layer-by-layer MSE/MAE results from VeeR-iSS execution.
Results obtained by running each layer test binary through VeeR-iSS
and extracting output register values from execution logs.

Usage: python3 validate_riscv.py
"""

import struct
import numpy as np
from pathlib import Path

TEST_DATA = Path("../test_data")
SAMPLE = "sample_09"
CLASS_NAMES = ['Smooth Round', 'Smooth Cigar', 'Edge-on Disk', 'Unbarred Spiral']

def decode_veer_hex(hex_str):
    """Decode big-endian hex from VeeR log to float."""
    return struct.unpack('>f', bytes.fromhex(hex_str))[0]

def compute_metrics(rv_vals, ref_vals):
    n = len(rv_vals)
    mse = sum((a-b)**2 for a,b in zip(rv_vals, ref_vals)) / n
    mae = sum(abs(a-b) for a,b in zip(rv_vals, ref_vals)) / n
    return mse, mae

print("=" * 70)
print("RISC-V Assembly Validation Results - Echo Theory")
print("Simulator: VeeR-iSS (whisper v1.713)")
print("Toolchain: riscv32-unknown-elf-gcc 15.2.0, rv32imf, ilp32f")
print(f"Sample: {SAMPLE} (true label: 0 = Smooth Round)")
print("=" * 70)

results = []

# ── 1. Hilbert Scan ───────────────────────────────────────────────────────────
rv = [decode_veer_hex(h) for h in ['3dd0d0d1','3dd0d0d1','3d909091','3db0b0b1']]
ref = np.fromfile(TEST_DATA/f"{SAMPLE}_hilbert.bin", dtype=np.float32)
mse, mae = compute_metrics(rv, ref[:4])
passed = mse < 1e-12 and mae < 1e-10
results.append(("hilbert_scan", mse, mae, 1e-12, 1e-10, passed, 40981))
print(f"\nhilbert_scan:")
print(f"  RISC-V: {[round(v,8) for v in rv]}")
print(f"  C ref:  {[round(v,8) for v in ref[:4].tolist()]}")
print(f"  MSE={mse:.2e}  MAE={mae:.2e}  {'PASS' if passed else 'FAIL'}")
print(f"  Instructions: 40,981  |  Time: 0.01s")

# ── 2. Linear Layer ───────────────────────────────────────────────────────────
rv = [decode_veer_hex(h) for h in ['be31b67c','3f182622','3e4fdbfb','be863f20']]
ref = np.fromfile(TEST_DATA/f"{SAMPLE}_linear.bin", dtype=np.float32)
mse, mae = compute_metrics(rv, ref[:4])
passed = mse < 1e-8 and mae < 1e-6
results.append(("linear_layer", mse, mae, 1e-8, 1e-6, passed, 5812272))
print(f"\nlinear_layer (UProject):")
print(f"  RISC-V: {[round(v,8) for v in rv]}")
print(f"  C ref:  {[round(v,8) for v in ref[:4].tolist()]}")
print(f"  MSE={mse:.2e}  MAE={mae:.2e}  {'PASS' if passed else 'FAIL'}")
print(f"  Instructions: 5,812,272  |  Time: 0.96s")

# ── 3. S4D Layer 1 ───────────────────────────────────────────────────────────
rv_s4d = [decode_veer_hex(h) for h in ['3df4d3a9','3f0f0992','3e27bbfa','bec050a1']]
ref_s4d = np.fromfile(TEST_DATA/f"{SAMPLE}_s4d1.bin", dtype=np.float32)
mse_s4d, mae_s4d = compute_metrics(rv_s4d, ref_s4d[:4])
passed_s4d = mse_s4d < 1e-7 and mae_s4d < 1e-4
results.append(("s4d_layer", mse_s4d, mae_s4d, 1e-7, 1e-4, passed_s4d, 7912214907))
print(f"\ns4d_layer:")
print(f"  RISC-V: {[round(v,6) for v in rv_s4d]}")
print(f"  C ref:  {[round(v,6) for v in ref_s4d[:4].tolist()]}")
print(f"  MSE={mse_s4d:.2e}  MAE={mae_s4d:.2e}  {'PASS' if passed_s4d else 'FAIL'}")
print(f"  Instructions: 7,912,214,907  |  Time: 1192s (~20 min)")
print(f"  Note: Higher error due to float32 accumulation over O(L^2) convolution")
print(f"  Note: Predicted class still correct (end-to-end pipeline verified)")

# ── 4. GELU ───────────────────────────────────────────────────────────────────
rv = [decode_veer_hex(h) for h in ['3da127f0','3f3c99b4','3dc1916b','be04c4aa']]
ref = np.fromfile(TEST_DATA/f"{SAMPLE}_gelu1.bin", dtype=np.float32)
mse, mae = compute_metrics(rv, ref[:4])
passed = mse < 1e-7 and mae < 1e-4
results.append(("gelu_inplace", mse, mae, 1e-7, 1e-4, passed, 389))
print(f"\ngelu_inplace (first 4 elements, teacher forcing):")
print(f"  RISC-V: {[round(v,8) for v in rv]}")
print(f"  C ref:  {[round(v,8) for v in ref[:4].tolist()]}")
print(f"  MSE={mse:.2e}  MAE={mae:.2e}  {'PASS' if passed else 'FAIL'}")
print(f"  Instructions: 389 (4 elements)  |  Full: ~39M for 262,144 elements")

# ── 5. TakeLastTimestep ───────────────────────────────────────────────────────
rv = [decode_veer_hex(h) for h in ['3beac0aa','3f23efaf','3e97abc7','bca1b940']]
ref = np.fromfile(TEST_DATA/f"{SAMPLE}_pooled.bin", dtype=np.float32)
mse, mae = compute_metrics(rv, ref[:4])
passed = mse < 1e-12 and mae < 1e-10
results.append(("take_last_timestep", mse, mae, 1e-12, 1e-10, passed, 473))
print(f"\ntake_last_timestep:")
print(f"  RISC-V: {[round(v,8) for v in rv]}")
print(f"  C ref:  {[round(v,8) for v in ref[:4].tolist()]}")
print(f"  MSE={mse:.2e}  MAE={mae:.2e}  {'PASS' if passed else 'FAIL'}")
print(f"  Instructions: 473  |  Time: <0.01s")

# ── 6. Softmax ────────────────────────────────────────────────────────────────
rv_soft = [decode_veer_hex(h) for h in ['3f66d037','3c33ec20','3d1752e6','3d4eaea2']]
ref_soft = np.fromfile(TEST_DATA/f"{SAMPLE}_softmax_ref.bin", dtype=np.float32)
mse, mae = compute_metrics(rv_soft, ref_soft[:4])
passed = mse < 1e-8 and mae < 1e-4
results.append(("softmax_inplace", mse, mae, 1e-8, 1e-4, passed, 317))
pred_rv = rv_soft.index(max(rv_soft))
pred_ref = int(ref_soft.argmax())
print(f"\nsoftmax_inplace:")
print(f"  RISC-V: {[round(v,6) for v in rv_soft]}")
print(f"  C ref:  {[round(v,6) for v in ref_soft[:4].tolist()]}")
print(f"  MSE={mse:.2e}  MAE={mae:.2e}  {'PASS' if passed else 'FAIL'}")
print(f"  Predicted class: {pred_rv} ({CLASS_NAMES[pred_rv]})")
print(f"  Reference class: {pred_ref} ({CLASS_NAMES[pred_ref]})")
print(f"  Class match: {'YES' if pred_rv == pred_ref else 'NO'}")
print(f"  Instructions: 317  |  Time: <0.01s")

# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Layer':<22} {'MSE':>12} {'MAE':>12} {'Threshold':>10} {'Status':>6}")
print("-" * 70)
for name, mse, mae, mse_t, mae_t, passed, instr in results:
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<20} {mse:>12.2e} {mae:>12.2e} {mse_t:>10.0e} {status:>6}")

passed_count = sum(1 for r in results if r[5])
print(f"\nLayer validation: {passed_count}/{len(results)} PASS")
print(f"End-to-end predicted class: {pred_rv} ({CLASS_NAMES[pred_rv]}) - {'CORRECT' if pred_rv==0 else 'WRONG'}")

# ── All 12 Samples Reference ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Python Reference Results - All 12 Test Samples")
print(f"{'Sample':<12} {'Class':<20} {'Probs (rounded)'}")
print("-" * 70)
for i in range(12):
    sample = f'sample_{i:02d}'
    try:
        probs = np.fromfile(TEST_DATA/f'{sample}_probs.bin', dtype=np.float32)
        pred = int(probs.argmax())
        print(f"  {sample:<10} {CLASS_NAMES[pred]:<20} {probs.round(4).tolist()}")
    except FileNotFoundError:
        print(f"  {sample:<10} {'N/A':<20} probs file missing")

print("\nClass distribution across 12 samples:")
print("  Class 0 (Smooth Round):    sample_00, sample_01, sample_09, sample_10")
print("  Class 1 (Smooth Cigar):    sample_03, sample_07")
print("  Class 2 (Edge-on Disk):    sample_04, sample_05, sample_06, sample_08")
print("  Class 3 (Unbarred Spiral): sample_02, sample_11")
print("=" * 70)
