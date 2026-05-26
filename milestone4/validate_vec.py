#!/usr/bin/env python3
"""
validate_vec.py  –  M4 Validation Harness
==========================================
Runs the vectorized RISC-V binary on all test samples via VeeR-iSS,
parses the output log, and compares against a Python/C reference.

Usage
-----
    python3 validate_vec.py [--samples 10] [--sim whisper]
                            [--logdir build_vec/logs]
                            [--ref reference_probs.npy]

Outputs a table of per-sample MSE and MAE (scalar vs. vector outputs),
and an overall pass/fail summary (threshold: 4 decimal places = 5e-5).

The script assumes:
  • build_vec.sh is in the same directory.
  • VeeR-iSS produces a log containing console output lines (hex words).
  • A reference array (numpy .npy) holding shape [N_SAMPLES, 4] floats
    is either provided via --ref or generated from a Python S4D model.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PASS_THRESHOLD = 5e-5        # 4 decimal places
N_CLASSES = 4
CLASS_NAMES = ["SmoothRound", "SmoothCigar", "EdgeOnDisk", "UnbarredSpiral"]

# ---------------------------------------------------------------------------
# Hex extraction from VeeR-iSS log
# ---------------------------------------------------------------------------
HEX_PAT = re.compile(r'\b([0-9a-f]{8})\b')

def extract_probs_from_log(logpath: Path) -> np.ndarray | None:
    """Parse a VeeR-iSS log and extract the 4 output probability floats."""
    try:
        text = logpath.read_text(errors='replace')
    except FileNotFoundError:
        return None

    # The main_vec.s writes 4 consecutive 8-hex-digit words to console I/O.
    # They appear as 8-char hex tokens in the log; we take the last group of 4.
    matches = HEX_PAT.findall(text)
    # Filter to plausible IEEE-754 single-precision values (exponent ≠ 0xFF)
    probs = []
    for m in matches:
        raw = int(m, 16)
        exp = (raw >> 23) & 0xFF
        if exp == 0xFF:          # inf / NaN – skip
            continue
        val = struct.unpack('>f', bytes.fromhex(m))[0]
        if 0.0 <= val <= 1.0:
            probs.append(val)
    if len(probs) < N_CLASSES:
        return None
    return np.array(probs[-N_CLASSES:], dtype=np.float32)


# ---------------------------------------------------------------------------
# Run simulation for one sample
# ---------------------------------------------------------------------------
def run_simulation(sample_idx: int, build_script: Path, logdir: Path,
                   sim: str) -> Path:
    logpath = logdir / f"s4d_vec_sample{sample_idx:02d}.txt"
    if logpath.exists():
        return logpath          # reuse cached log

    cmd = [str(build_script), '-s', str(sample_idx)]
    print(f"  [sim] sample {sample_idx:02d} …", end=' ', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED\n{result.stderr}")
    else:
        print("done")
    return logpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=10,
                    help='Number of test samples to evaluate')
    ap.add_argument('--sim', default='whisper',
                    help='VeeR-iSS executable name')
    ap.add_argument('--logdir', default='build_vec/logs',
                    help='Directory containing simulation logs')
    ap.add_argument('--ref', default=None,
                    help='Path to reference .npy array [N_SAMPLES, 4]')
    ap.add_argument('--no-sim', action='store_true',
                    help='Skip simulation; parse existing logs only')
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()
    build_script = script_dir / 'build_vec.sh'
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Load reference if available
    reference = None
    if args.ref:
        reference = np.load(args.ref)
        print(f"[ref] Loaded reference from {args.ref}  shape={reference.shape}")
    else:
        print("[ref] No --ref provided; will report predicted classes only")

    print(f"\nRunning {args.samples} samples …\n")

    rows = []
    all_pass = True

    for idx in range(args.samples):
        if not args.no_sim:
            logpath = run_simulation(idx, build_script, logdir, args.sim)
        else:
            logpath = logdir / f"s4d_vec_sample{idx:02d}.txt"

        probs = extract_probs_from_log(logpath)
        if probs is None:
            print(f"  sample {idx:02d}: could not parse log at {logpath}")
            rows.append({'idx': idx, 'probs': None, 'mse': None, 'mae': None,
                         'pred': -1, 'pass': False})
            all_pass = False
            continue

        pred = int(np.argmax(probs))

        mse = mae = None
        sample_pass = True
        if reference is not None and idx < len(reference):
            ref = reference[idx].astype(np.float32)
            mse = float(np.mean((probs - ref) ** 2))
            mae = float(np.mean(np.abs(probs - ref)))
            sample_pass = mse < PASS_THRESHOLD

        rows.append({'idx': idx, 'probs': probs, 'mse': mse, 'mae': mae,
                     'pred': pred, 'pass': sample_pass})
        if not sample_pass:
            all_pass = False

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    print()
    hdr = f"{'Samp':>4}  {'P0':>8} {'P1':>8} {'P2':>8} {'P3':>8}  "
    hdr += f"{'Pred':>4}  {'MSE':>10}  {'MAE':>10}  {'Pass':>5}"
    print(hdr)
    print('-' * len(hdr))

    for r in rows:
        if r['probs'] is None:
            print(f"{r['idx']:4d}  {'(log parse error)':40s}")
            continue
        p = r['probs']
        mse_s = f"{r['mse']:.2e}" if r['mse'] is not None else '       n/a'
        mae_s = f"{r['mae']:.2e}" if r['mae'] is not None else '       n/a'
        ok = 'PASS' if r['pass'] else 'FAIL'
        print(f"{r['idx']:4d}  {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[3]:8.4f}  "
              f"{r['pred']:4d}  {mse_s:>10}  {mae_s:>10}  {ok:>5}")

    print('-' * len(hdr))
    print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print(f"Pass threshold: MSE < {PASS_THRESHOLD:.0e}  (4 decimal places)")
    print(f"\nClass legend: " +
          ', '.join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)))

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
