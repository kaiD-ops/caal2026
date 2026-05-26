#!/usr/bin/env python3
"""
count_instructions.py  –  Static & Dynamic Instruction Count for M4
=====================================================================
Analyses the RVV assembly source files (static count) and VeeR-iSS
execution log (dynamic count), then prints a formatted breakdown by
instruction family and per-layer, matching the M4 rubric.

Static families
---------------
  R-type : add sub and or xor sll srl sra mul div rem ...
  I-type : addi lw lh lb jalr srai srli slli andi ori xori ...
  S-type : sw sh sb
  B-type : beq bne blt bge bltu bgeu
  U-type : lui auipc
  J-type : jal (call expands to auipc+jalr but count separately)
  F-type : fadd.s fsub.s fmul.s fdiv.s fmadd.s fmsub.s fneg.s fabs.s
           flw fsw fcvt.* fmv.* fsgnj.* flt.s fle.s feq.s
  V-type : vsetvli vle32.v vse32.v vluxei32.v vlse32.v vfmul.vv vfadd.vv
           vfsub.vv vfmacc.vv vfredusum.vs vfmv.* vmv.* vsll.vi ...

Usage
-----
    python3 count_instructions.py --src .
                                  --log build_vec/logs/s4d_vec_sample09.txt
                                  [--scalar-log build/logs/sample.txt]
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Instruction family classification
# ---------------------------------------------------------------------------
FAMILIES = {
    'R': re.compile(
        r'\b(add|sub|and|or|xor|sll|srl|sra|mul|mulh|div|divu|rem|remu'
        r'|mv|neg|not|seqz|snez|sltz|sgtz|slt|sltu)\b'),
    'I': re.compile(
        r'\b(addi|lw|lh|lb|lhu|lbu|jalr|srai|srli|slli|andi|ori|xori|li'
        r'|la|nop|ret)\b'),
    'S': re.compile(r'\b(sw|sh|sb)\b'),
    'B': re.compile(r'\b(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|bltz|bgez|blez|bgtz|j|bnez)\b'),
    'U': re.compile(r'\b(lui|auipc)\b'),
    'J': re.compile(r'\b(jal|call|tail)\b'),
    'F': re.compile(
        r'\b(fadd\.s|fsub\.s|fmul\.s|fdiv\.s|fmadd\.s|fmsub\.s|fnmadd\.s'
        r'|fnmsub\.s|fsqrt\.s|fneg\.s|fabs\.s|flw|fsw|fcvt\.[a-z.]+|fmv\.[a-z.]+(?:\.[a-z.]+)?'
        r'|fsgnj\.s|fsgnjn\.s|fsgnjx\.s|flt\.s|fle\.s|feq\.s)\b'),
    'V': re.compile(
        r'\b(vsetvli|vle\d+\.v|vse\d+\.v|vluxei\d+\.v|vlse\d+\.v|vlsseg\w+'
        r'|vfmul\.vv|vfadd\.vv|vfsub\.vv|vfmacc\.vv|vfredusum\.vs|vfredosum\.vs'
        r'|vfmax\.vv|vfmin\.vv|vfmv\.[a-z.]+|vmv\.[a-z.]+|vsll\.vi|vsrl\.vi'
        r'|vadd\.\w+|vsub\.\w+|vmul\.\w+|vmerge\.\w+)\b'),
}

# Layer → source file mapping
LAYER_FILES = {
    'Hilbert Scan':         'hilbert_vec.s',
    'Linear Layer':         'linear_vec.s',
    'S4D Layer':            's4d_vec.s',
    'GELU':                 'gelu_vec.s',
    'TakeLastTimestep':     'take_last_vec.s',
    'Softmax':              'softmax_vec.s',
    'Math helpers':         'math_vec.s',
    'Main/pipeline':        'main_vec.s',
}


def classify(mnemonic: str) -> str:
    for fam, pat in FAMILIES.items():
        if pat.match(mnemonic):
            return fam
    return '?'


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------
# Match non-comment, non-directive lines that look like instructions
INSN_LINE = re.compile(r'^\s+([a-z][a-z0-9_.]*)\s', re.MULTILINE)


def static_count(src_dir: Path) -> dict:
    """Count instructions per family per source file."""
    results = {}
    for layer, fname in LAYER_FILES.items():
        path = src_dir / fname
        if not path.exists():
            continue
        text = path.read_text(errors='replace')
        # Strip comments
        text = re.sub(r'#.*', '', text)
        counts = defaultdict(int)
        total = 0
        for m in INSN_LINE.finditer(text):
            mn = m.group(1)
            fam = classify(mn)
            counts[fam] += 1
            total += 1
        results[layer] = dict(counts)
        results[layer]['_total'] = total
    return results


def print_static(results: dict):
    fams = list(FAMILIES.keys()) + ['?']
    print("\n" + "=" * 70)
    print("STATIC INSTRUCTION COUNT (source-level, per module)")
    print("=" * 70)
    hdr = f"{'Layer':<22}" + ''.join(f"{f:>6}" for f in fams) + f"{'Total':>8}"
    print(hdr)
    print('-' * len(hdr))
    grand = defaultdict(int)
    for layer, counts in results.items():
        tot = counts.get('_total', 0)
        row = f"{layer:<22}"
        for f in fams:
            c = counts.get(f, 0)
            grand[f] += c
            row += f"{c:>6}"
        row += f"{tot:>8}"
        print(row)
    print('-' * len(hdr))
    grand_total = sum(grand[f] for f in fams)
    row = f"{'TOTAL':<22}" + ''.join(f"{grand[f]:>6}" for f in fams)
    row += f"{grand_total:>8}"
    print(row)
    # Percentages
    if grand_total:
        prow = f"{'%':<22}"
        for f in fams:
            pct = 100 * grand[f] / grand_total
            prow += f"{pct:>5.1f}%"[:-1] + ' '  # compact
        print(prow)
    print()


# ---------------------------------------------------------------------------
# Dynamic analysis
# ---------------------------------------------------------------------------
# VeeR-iSS log line format (scalar):
#   #N 0 <PC> <hex> r <rd>  <val>  <mnemonic> ...
# VeeR-iSS log line for vector:
#   #N 0 <PC> <hex> v <vd>  <vec_val>  <mnemonic> ...
LOG_LINE = re.compile(
    r'^#\d+\s+\d+\s+[0-9a-f]+\s+[0-9a-f]+\s+[rv]\s+\S+\s+\S+\s+(\S+)')


def dynamic_count(logpath: Path) -> dict:
    """Count instructions per family from VeeR-iSS log."""
    counts = defaultdict(int)
    total = 0
    for line in logpath.open(errors='replace'):
        m = LOG_LINE.match(line)
        if m:
            mn = m.group(1).split('.')[0] if '.' not in m.group(1)[:3] else m.group(1)
            fam = classify(m.group(1))
            counts[fam] += 1
            total += 1
    counts['_total'] = total
    return dict(counts)


def print_dynamic(vec_counts: dict, scalar_counts: dict | None = None):
    fams = list(FAMILIES.keys()) + ['?']
    print("=" * 70)
    print("DYNAMIC INSTRUCTION COUNT (VeeR-iSS execution trace)")
    print("=" * 70)
    vec_total = vec_counts.get('_total', 0)

    if scalar_counts:
        sc_total = scalar_counts.get('_total', 0)
        print(f"\n{'Family':>8}  {'Scalar':>14} {'%':>6}  {'Vector':>14} {'%':>6}  {'Speedup':>8}")
        print('-' * 68)
        for f in fams:
            sc = scalar_counts.get(f, 0)
            vc = vec_counts.get(f, 0)
            sc_pct = 100 * sc / sc_total if sc_total else 0
            vc_pct = 100 * vc / vec_total if vec_total else 0
            spd = f"{sc/vc:.2f}x" if vc else "—"
            print(f"{f:>8}  {sc:>14,} {sc_pct:>5.1f}%  {vc:>14,} {vc_pct:>5.1f}%  {spd:>8}")
        print('-' * 68)
        spd = f"{sc_total/vec_total:.2f}x" if vec_total else "—"
        print(f"{'TOTAL':>8}  {sc_total:>14,} {'100.0%':>6}  {vec_total:>14,} {'100.0%':>6}  {spd:>8}")
    else:
        print(f"\n{'Family':>8}  {'Count':>14}  {'%':>6}")
        print('-' * 36)
        for f in fams:
            c = vec_counts.get(f, 0)
            pct = 100 * c / vec_total if vec_total else 0
            print(f"{f:>8}  {c:>14,}  {pct:>5.1f}%")
        print('-' * 36)
        print(f"{'TOTAL':>8}  {vec_total:>14,}  {'100.0%':>6}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='.',
                    help='Directory containing *_vec.s source files')
    ap.add_argument('--log', default=None,
                    help='VeeR-iSS log for the vector binary')
    ap.add_argument('--scalar-log', default=None,
                    help='VeeR-iSS log for the M3 scalar binary (optional)')
    args = ap.parse_args()

    src_dir = Path(args.src)
    print(f"\nStatic analysis of: {src_dir.resolve()}")
    static = static_count(src_dir)
    print_static(static)

    if args.log:
        logpath = Path(args.log)
        if not logpath.exists():
            print(f"[warn] Log not found: {logpath}")
        else:
            vec_dyn = dynamic_count(logpath)
            scalar_dyn = None
            if args.scalar_log:
                sp = Path(args.scalar_log)
                if sp.exists():
                    scalar_dyn = dynamic_count(sp)
            print_dynamic(vec_dyn, scalar_dyn)
    else:
        print("[info] Pass --log <path> for dynamic count analysis")


if __name__ == '__main__':
    main()
