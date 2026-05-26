#!/usr/bin/env python3
"""
count_instructions.py - Parse whisper --profileinst log and report
dynamic instruction counts, comparing vectorized (M4) vs scalar (M3).

Usage: python3 count_instructions.py build/galaxy_vec_prof.log
"""

import sys
import re
from pathlib import Path

M3_TOTAL = 7_912_214_907   # M3 dynamic instruction count

VECTOR_MNEMONICS = {
    'vsetvli', 'vle32.v', 'vse32.v', 'vluxei32.v',
    'vfmul.vv', 'vfmul.vf', 'vfadd.vv', 'vfadd.vf',
    'vfsub.vv', 'vfdiv.vf', 'vfmacc.vv', 'vfnmsac.vv',
    'vfredusum.vs', 'vfredmax.vs',
    'vfmv.s.f', 'vfmv.f.s', 'vfmv.v.f',
    'vmv.v.v', 'vsll.vi', 'vor.vv',
}

def parse_profile(path):
    counts = {}
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                count = int(parts[0])
                mnem  = parts[1].lower()
                counts[mnem] = counts.get(mnem, 0) + count
                total += count
            except ValueError:
                continue
    return counts, total

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 count_instructions.py <profileinst.log>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    counts, total = parse_profile(path)

    vec_count   = sum(v for k, v in counts.items() if k in VECTOR_MNEMONICS)
    scalar_count = total - vec_count
    speedup = M3_TOTAL / total if total > 0 else 0

    print("=" * 60)
    print("M4 Vectorized  -  Dynamic Instruction Count Report")
    print("=" * 60)
    print(f"  Total instructions : {total:>15,}")
    print(f"  Vector instructions: {vec_count:>15,}  ({100*vec_count/total:.1f}%)")
    print(f"  Scalar instructions: {scalar_count:>15,}  ({100*scalar_count/total:.1f}%)")
    print(f"  M3 total           : {M3_TOTAL:>15,}")
    print(f"  Speedup (instr)    : {speedup:>14.2f}x")
    print()

    # Top 20 most-executed instructions
    print("Top 20 instructions by count:")
    print(f"  {'Count':>14}  {'%Total':>7}  Mnemonic")
    print("  " + "-"*40)
    for mnem, cnt in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        tag = " [vec]" if mnem in VECTOR_MNEMONICS else ""
        print(f"  {cnt:>14,}  {100*cnt/total:>6.2f}%  {mnem}{tag}")
    print("=" * 60)

if __name__ == "__main__":
    main()
