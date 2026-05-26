# Milestone 4: Vectorized RISC-V Assembly (RVV)

S4D galaxy classifier with RISC-V V extension vectorization.

## Hardware Target
- VeeR-EL2 simulator (whisper)
- VLEN = 256 bits (32 bytes/vector)
- Float32 (SEW=32): 8 elements per LMUL=1 register
- Architecture: `rv32gcv`

## Files

| File | Description |
|------|-------------|
| `main_vec.s` | Same 9-stage pipeline as M3; calls same function names |
| `math_vec.s` | Scalar transcendentals (expf/cosf/sinf/tanhf — unchanged) |
| `hilbert_vec.s` | `vluxei32.v` gather: 512 batches of 8, ~8x fewer instructions |
| `linear_vec.s` | Strip-mined dot product over in_dim with VL=8 |
| `take_last_vec.s` | LMUL=8 (VL=64): one vle32.v + one vse32.v replaces 64 scalar loads |
| `gelu_vec.s` | Batch-8 vectorized polynomial; tanh remains scalar |
| `softmax_vec.s` | vfredmax.vs for max; vfdiv.vf for normalize |
| `s4d_vec.s` | **Phase 2 vectorized**: LMUL=4 (VL=32) processes all 32 states simultaneously |

## Vectorization Summary

### take_last_vec.s
```asm
vsetvli zero, t0, e32, m8, ta, ma   # VL=64
vle32.v v0, (a0)                     # load 64 floats
vse32.v v0, (a1)                     # store 64 floats
```
M3: 473 instructions → M4: ~10 instructions (~47x reduction)

### hilbert_vec.s
```asm
vle32.v   v1, (a0)           # load 8 indices
vsll.vi   v1, v1, 2          # * 4 (byte offsets)
vluxei32.v v0, (a1), v1     # gather 8 pixels
vse32.v   v0, (a2)           # store 8 pixels
```
M3: ~41K instructions → M4: ~5K instructions (~8x reduction)

### s4d_vec.s Phase 2 (main win)
```asm
# All 32 states processed in one pass per timestep:
vfmul.vv  v24, v0, v16          # Ct_r * cur_r
vfnmsac.vv v24, v4, v20         # -= Ct_i * cur_i  -> Re(Ct*cur)
vfredusum.vs v28, v24, v28      # sum over 32 states
# ... complex multiply for cur advancement
```
Phase 2: 4096 t × 32 n iterations → 4096 t × 1 vector iteration (~32x reduction in phase 2)

## Build (from WSL, inside milestone4/)

```bash
# Generate test data first (if not done):
cd .. && python3 generate_test_data.py && cd milestone4

# Build:
make

# Build + profile:
make test
python3 count_instructions.py build/galaxy_vec_prof.log

# Validate:
python3 validate_vec.py
```

Or use the shell script:
```bash
bash build_vec.sh          # build + run
bash build_vec.sh --test   # build + profile + count instructions
```

## Expected Results

All layers produce numerically identical outputs to M3 (same algorithm, different execution path):
- `hilbert_scan`: MSE < 1e-12
- `linear_layer`: MSE < 1e-8
- `s4d_layer`: MSE < 1e-7
- `gelu_inplace`: MSE < 1e-7
- `take_last_timestep`: MSE < 1e-12
- `softmax_inplace`: MSE < 1e-8

Predicted class: **0 (Smooth Round)** for sample_09.

## Differences from M3

| Aspect | M3 | M4 |
|--------|----|----|
| Architecture | `rv32imf` | `rv32gcv` |
| s4d Phase 2 | Scalar n-loop (32 iterations/t) | LMUL=4 vector (1 iteration/t) |
| hilbert_scan | Scalar gather | `vluxei32.v` indexed gather |
| take_last | 64 scalar loads | 1 `vle32.v` LMUL=8 |
| gelu | Scalar per element | Batch-8 vectorized polynomial |
| softmax | Scalar max + norm | `vfredmax.vs` + `vfdiv.vf` |
| linear | Scalar dot product | Strip-mined VL=8 reduction |
| Phase 3 (conv) | Scalar | Scalar (O(L²) sequential dependency) |
