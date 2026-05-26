# Milestone 4 – RISC-V Vector (RVV) Implementation

## File Overview

| File | Description | Vectorized? |
|---|---|---|
| `main_vec.s` | Entry point; loads weights & sample, calls all layers | Pipeline only |
| `math_vec.s` | Scalar transcendentals: `expf`, `cosf`, `sinf`, `tanhf` | No (scalar, reused from M3) |
| `hilbert_vec.s` | Hilbert scan gather | **Yes** – `vluxei32.v` |
| `linear_vec.s` | Linear (FC) layer — dot product | **Yes** – `vfmul.vv` + `vfredusum.vs` |
| `s4d_vec.s` | S4D recurrence + causal convolution | **Yes** – Phase-2 & Phase-3 |
| `gelu_vec.s` | GELU activation | **Partial** – polynomial vectorized; `tanhf` scalar |
| `take_last_vec.s` | Extract last timestep [4096,64]→[64] | **Yes** – single `vle32/vse32` |
| `softmax_vec.s` | Softmax (N=4) | **No** – N=4 is below breakeven point |
| `build_vec.sh` | Build + simulate script | — |
| `validate_vec.py` | End-to-end validation vs Python/C reference | — |
| `count_instructions.py` | Static + dynamic instruction count analysis | — |

## Build & Run

```bash
# One-shot: compile, link, simulate sample_09
./build_vec.sh

# Run a different sample (0-indexed)
./build_vec.sh -s 3

# Clean build artifacts
./build_vec.sh -c

# Re-run existing binary without recompiling
./build_vec.sh -e
```

## Validation

```bash
# Validate 10 samples (logs must exist or --no-sim not set)
python3 validate_vec.py --samples 10 --ref ../reference_probs.npy

# Parse existing logs only (skip simulation)
python3 validate_vec.py --samples 10 --ref ../reference_probs.npy --no-sim
```

## Instruction Count

```bash
# Static count only
python3 count_instructions.py --src .

# Static + dynamic (with scalar comparison)
python3 count_instructions.py \
    --src . \
    --log build_vec/logs/s4d_vec_sample09.txt \
    --scalar-log ../build/logs/sample.txt
```

## Vectorization Strategy Summary

### Layers Vectorized

**Hilbert Scan** — uses `vluxei32.v` (indexed load / gather). The index array
gives element offsets; we scale to byte offsets with `vsll.vi 2` then do a
single-trip gather + `vse32.v`.  Reduces 4096 scalar iterations to ⌈4096/VLMAX⌉.

**Linear Layer** — the inner dot-product over `in_dim` elements is vectorized
with strip-mined `vfmul.vv` + `vfredusum.vs`.  The outer loops (timestep,
output neuron) remain scalar because each output neuron requires an independent
accumulator.

**S4D Layer — Phase 2 (kernel build)** — the 32-state accumulation
`kval += 2·Re(Ct·cur)` and state update `cur *= step` are both vectorized over
N=32 states using `m1` register groups (32 × e32 = 128 bytes, fits in one
register for VLEN≥128).  The reduction `vfredusum.vs` collapses to a scalar
`kval` per timestep.

**S4D Layer — Phase 3 (causal convolution)** — the j-loop
`acc += K[j] · u[t−j][h]` is vectorized using `vle32.v` (for K) and
`vlse32.v` with stride −256 bytes (for the reversed u column).  A
`vfredusum.vs` collapses each strip into a scalar partial sum.

**GELU** — the polynomial `c1·x³ + x` and the final `0.5·x·(1+tanh)` scaling
are fully vectorized.  The `tanhf` call is scalar (one call per element)
because `math_vec.s` implements it as a scalar routine; vectorizing `tanhf`
would require an RVV-native exp approximation.

**TakeLastTimestep** — trivially vectorized: offset to row 4095, then
strip-mined `vle32.v` / `vse32.v`.

### Layers Retained as Scalar

**Softmax (N=4)** — operates on exactly 4 elements (one per class).  The
overhead of `vsetvli` + reduction exceeds any gain.  Dynamic instruction count
contribution is negligible (<0.001% of total).

**Math helpers (expf / cosf / sinf / tanhf)** — these scalar routines are
called O(N_STATES) = O(96) times per forward pass during Phase-1 precomputation.
Vectorizing them would require a full RVV polynomial approximation library.
Since they account for <0.01% of dynamic instructions, retaining scalar versions
is justified.

## Memory Layout

Intermediate buffers in `.bss` (same as M3):

```
buf_hilbert  :  4096×1×4  =    16 384 B
buf_proj     :  4096×64×4 = 1 048 576 B
buf_s4d1     :  4096×64×4 = 1 048 576 B
buf_s4d2     :  4096×64×4 = 1 048 576 B
buf_pooled   :    64×4    =       256 B
buf_logits   :     4×4    =        16 B
```

S4D scratchpad (per-channel, reset each channel invocation):
`sv_smag, sv_scos, sv_ssin, sv_step_r, sv_step_i, sv_cur_r, sv_cur_i,
sv_Ct_r, sv_Ct_i` — each 32×4 = 128 B.  `sv_kernel` — 4096×4 = 16 384 B.
