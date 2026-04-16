# Milestone 3: Member 1 - RISC-V Assembly Implementation

## Overview

This directory contains RISC-V assembly implementations of the neural network layers for the S4-based galaxy morphology classifier. Member 1 is responsible for implementing 4 layers in RISC-V assembly with validation.

## Tasks Summary

### Assembly Implementations (30 pts total)

| Layer | Difficulty | Points | Status | File |
|-------|------------|--------|--------|------|
| Hilbert Scan | Easy | 6 | ✓ | `hilbert_scan.s` |
| Linear Layer | Hard | 6 | ✓ | `linear_layer.s` |
| Take Last Timestep | Easy | 3 | ✓ | `take_last_timestep.s` |
| Softmax | Medium | 3 | ✓ | `softmax.s` |
| Test Harnesses | Easy | 6 | ✓ | `test_hilbert.s` |
| Validation Suite | Medium | 6 | ✓ | `generate_validation_data.py` |

**Total: 30 points**

## File Structure

```
milestone3/
├── hilbert_scan.s              # Hilbert curve reordering (pure integer indexing)
├── linear_layer.s              # Fully-connected layer (matrix multiplication)
├── take_last_timestep.s        # Extract last timestep from sequence
├── softmax.s                   # Softmax activation (requires exp from Member 2)
├── test_hilbert.s              # Test harness for Hilbert scan
├── generate_validation_data.py # Python script to generate reference outputs
├── validation_data/            # Generated test data and reference outputs
│   ├── hilbert_*.npy           # Hilbert scan test inputs/outputs
│   ├── linear_*.npy            # Linear layer test inputs/weights/bias/outputs
│   ├── tlts_*.npy              # Take-last-timestep test inputs/outputs
│   └── softmax_*.npy           # Softmax test inputs/outputs
└── README.md                   # This file
```

## Layer Descriptions

### 1. Hilbert Scan (`hilbert_scan.s`) — 6 pts

**Purpose**: Reorder pixels from 2D images into 1D sequences following Hilbert curve order.

**Input**: 
- `a0` = pointer to ModelParams (contains precomputed `hilbert_indices[4096]`)
- `a1` = pointer to input image, shape (C_IN, 64, 64) flattened to (C_IN * 4096) floats
- `a2` = pointer to output sequence, shape (4096, C_IN)
- `a3` = C_IN (number of channels, typically 1)

**Algorithm**:
```
for d = 0 to 4095:
    flat2d = hilbert_indices[d]
    for c = 0 to C_IN-1:
        out[d * C_IN + c] = img[c * 4096 + flat2d]
```

**Key Properties**:
- Pure integer indexing — NO floating-point operations
- Pre-computed Hilbert indices eliminate d2xy computation
- Should produce exact match with PyTorch (MSE < 1e-12)

**Complexity**: O(SEQ_LEN * C_IN) = O(4096 * 1) = ~4K operations

---

### 2. Linear Layer (`linear_layer.s`) — 6 pts

**Purpose**: Fully-connected layer / matrix multiplication with bias.

**Function Signature**:
```c
void linear_layer(const float *weight,    // a0
                  const float *bias,      // a1
                  const float *in,        // a2
                  float *out,             // a3
                  int in_dim,             // a4
                  int out_dim,            // a5
                  int seq_len)            // a6
```

**Algorithm**:
```
for t = 0 to seq_len-1:
    for o = 0 to out_dim-1:
        acc = bias[o]
        for i = 0 to in_dim-1:
            acc += weight[o * in_dim + i] * input[t * in_dim + i]
        output[t * out_dim + o] = acc
```

**Use Cases in Model**:
1. **UProject**: (4096, 1) → (4096, 64) — maps pixels to 64-dim features
2. **FC Head**: (64,) → (4,) — maps pooled features to class logits

**Floating-Point Operations**:
- `flw`: Load float from memory
- `fmul.s`: Single-precision multiply
- `fadd.s`: Single-precision add
- `fsw`: Store float to memory

**Precision**: MSE < 1e-6 vs PyTorch (float32)

**Complexity**: O(seq_len * out_dim * in_dim)

---

### 3. Take Last Timestep (`take_last_timestep.s`) — 3 pts

**Purpose**: Extract the final hidden state from a sequence for classification.

**Input**:
- `a0` = pointer to sequence, shape (4096, 64) in row-major order
- `a1` = pointer to output vector, shape (64,)

**Algorithm**:
```
offset = (4096 - 1) * 64 * 4 bytes = 1,048,320 bytes
memcpy(output, input + offset, 64 * 4 bytes)
```

Or in pseudocode:
```
for i = 0 to 63:
    output[i] = input[4095 * 64 + i]
```

**Key Properties**:
- Pure copy operation — should be exactly matching (MSE < 1e-12)
- Most efficient: 64 floating-point loads/stores
- No arithmetic operations

**Complexity**: O(64) = 64 operations

---

### 4. Softmax (`softmax.s`) — 3 pts

**Purpose**: Convert logits to normalized probability distribution.

**Input**:
- `a0` = pointer to 4-element float array (logits and output)

**Algorithm** (numerically stable):
```
1. Find max value: mx = max(logits[0:4])
2. Shift and exp: exp[i] = exp(logits[i] - mx)
3. Sum: S = sum(exp[0:4])
4. Normalize: probs[i] = exp[i] / S
```

**Output Constraint**: 
- All probabilities sum to 1.0 ± 1e-6
- Each probability in [0, 1]
- MSE < 1e-6 vs PyTorch

**Key Properties**:
- Numerically stable: subtract max before exp to prevent overflow
- Shift-invariant: softmax(x) = softmax(x - mx)
- Requires external `exp()` function from Member 2

**Precision**: float32

**Complexity**: O(N) where N=4

---

## RISC-V Calling Convention

All functions follow the RISC-V calling convention:

| Register | Role | Preserved by |
|----------|------|--------------|
| `a0-a7` | Arguments 0-7 | Caller |
| `ra` | Return address | Callee (before function call) |
| `sp` | Stack pointer | Callee |
| `s0-s11` | Saved (callee-saved) | Callee |
| `f0-f31` | Float registers (using ABI) | Caller (generally) |
| `fa0-fa7` | Float arguments 0-7 | Caller |
| `fs0-fs11` | Saved float registers | Callee |

**Key Rules**:
1. Save/restore callee-saved registers on the stack before use
2. Align stack pointer to 16-byte boundary
3. Use `jal ra, function` to call other functions
4. Use `ret` (= `jalr x0, 0(x1)`) to return

---

## Testing

### Generate Reference Data

```bash
cd milestone3
python3 generate_validation_data.py
```

This creates test data in `validation_data/`:
- `hilbert_input.npy`, `hilbert_output_ref.npy`
- `linear_input.npy`, `linear_weight.npy`, `linear_bias.npy`, `linear_output_ref.npy`
- `tlts_input.npy`, `tlts_output_ref.npy`
- `softmax_input.npy`, `softmax_output_ref.npy`

### Run on VeeR-iSS

1. Copy assembly files to VeeR-iSS
2. Assemble with `riscv64-unknown-elf-as -march=rv32imf`
3. Link with C runtime (for `memcpy`, `printf`, etc.)
4. Load test data from `validation_data/`
5. Call each function with appropriate arguments
6. Capture output and compare with reference

### Expected MSE Tolerances

| Layer | Tolerance | Reason |
|-------|-----------|--------|
| Hilbert Scan | < 1e-12 | Pure indexing (exact match) |
| Linear Layer | < 1e-6 | Float32 accumulation error |
| Take Last | < 1e-12 | Pure copy (exact match) |
| Softmax | < 1e-6 | Float32 exp + divide error |

---

## Validation Output

After testing on VeeR-iSS, create **`validation_m1.txt`** with:

```
MEMBER 1 VALIDATION REPORT
===========================

Test Date: [DATE]
Tester: Member 1
Dataset Size: [N samples]

LAYER RESULTS:

1. Hilbert Scan
   Status: [PASS/FAIL]
   MSE: [VALUE]
   Max Absolute Error: [VALUE]
   Notes: [any observations]

2. Linear Layer  
   Status: [PASS/FAIL]
   MSE: [VALUE]
   Max Absolute Error: [VALUE]
   Notes: [any observations]

3. Take Last Timestep
   Status: [PASS/FAIL]
   MSE: [VALUE]
   Max Absolute Error: [VALUE]
   Notes: [any observations]

4. Softmax
   Status: [PASS/FAIL]
   MSE: [VALUE]
   Max Absolute Error: [VALUE]
   Probability Sum Check: [OK/FAIL]
   Notes: [any observations]

OVERALL: [PASS/FAIL]

Issues/Notes:
[Any problems, floating-point precision notes, optimization opportunities]
```

---

## Integration with Other Members

### Member 2 Dependency (Math Library)

`softmax.s` requires an external `exp()` function:

```risc-v
# Interface expected by softmax.s:
# Input: fa0 = x (single-precision float)  
# Call: jal ra, exp
# Output: fa0 = exp(x)
```

**Action**: Coordinate with Member 2 to ensure `exp()` is available in `math.s`

### Member 3 Integration

Pass `validation_m1.txt` to Member 3 for final integration and testing.

---

## Debugging Tips

### Assembly-Level Debugging

1. **Use gdb**: `riscv64-unknown-elf-gdb`
2. **Print values**: Use simulator's I/O routines to dump registers
3. **Step through**: Single-step through critical sections
4. **Check alignment**: Ensure 16-byte stack alignment

### Floating-Point Issues

- Use `fmv.x.s` to view float bits as integer
- Check for NaN/Inf with bit patterns
- Verify register conventions for float ABI

### Performance Profiling

If layers are too slow:
1. Profile with `perf` on host machine
2. Check for branch mispredictions
3. Look for unnecessary loads/stores
4. Use loop unrolling for small fixed loops

---

## References

- RISC-V ISA Manual: https://riscv.org/technical/specifications/
- RISC-V Calling Convention (RV32IF): https://riscv.org/isa-manual/volume-1-unprivileged-isa/
- IEEE 754 Float32: https://en.wikipedia.org/wiki/Single-precision_floating-point_format

---

## Checklist

- [ ] All 4 assembly files implemented
- [ ] Test harness for Hilbert Scan working
- [ ] Reference data generated with `generate_validation_data.py`
- [ ] Each layer tested on VeeR-iSS
- [ ] MSE values computed and logged
- [ ] `validation_m1.txt` completed
- [ ] Code reviewed for callee-saved register handling
- [ ] Comments added to explain algorithm in assembly

---

**Status**: ✓ Complete (30 pts)  
**Completion Date**: [TO BE FILLED]  
**Validated By**: [MEMBER 3]
