# RISC-V S4D Implementation Guide

This document provides detailed information about the assembly implementations for each layer of the S4D galaxy classifier.

## Table of Contents

1. [Overview](#overview)
2. [Register Management](#register-management)
3. [Memory Layout](#memory-layout)
4. [Layer Implementations](#layer-implementations)

## Overview

### Architecture Target

- **ISA**: RISC-V RV32IMF (32-bit integer + multiply/divide + floating-point)
- **Calling Convention**: RISC-V Application Binary Interface (ABI)
- **Simulator**: VeeR-iSS (VeeR ISS Simulator)

### Key Design Principles

1. **Correctness First**: Exact numerical match against Python/C reference
2. **Calling Convention**: Strict adherence to RISC-V ABI for interoperability
3. **Modularity**: Each layer is an independent callable routine
4. **Comments**: Inline documentation for non-obvious operations

## Register Management

### Integer Registers (x0-x31)

| Type | Registers | Usage | Preserved |
|------|-----------|-------|-----------|
| Zero | x0 | Always 0 | N/A |
| RA | x1 | Return address | Caller |
| SP | x2 | Stack pointer | Callee |
| GP | x3 | Global pointer | - |
| TP | x4 | Thread pointer | - |
| Temporaries | x5-x7 (t0-t2) | Caller-saved scratch | Caller |
| | x28-x31 (t3-t6) | | |
| Saved | x8-x9 (s0-s1) | Callee-saved | Callee |
| | x18-x27 (s2-s11) | | |
| Arguments | x10-x17 (a0-a7) | Function arguments | Caller |

### Floating-Point Registers (f0-f31)

| Type | Registers | Usage | Preserved |
|------|-----------|-------|-----------|
| Temporaries | f0-f7 (ft0-ft7) | Caller-saved scratch | Caller |
| | f28-f31 (ft8-ft11) | | |
| Saved | f8-f9 (fs0-fs1) | Callee-saved | Callee |
| | f18-f27 (fs2-fs11) | | |
| Arguments | f10-f17 (fa0-fa7) | FP arguments | Caller |

### Our Register Allocation Strategy

**Integer registers (by layer):**
- Loop counters: t0, t1
- Address calculations: t2, t3, t4
- Saved registers: s0, s1

**Floating-point registers (by layer):**
- Accumulators: fs0, fs1, fs2
- Temporaries: ft0, ft1, ft2, ft3
- Arguments/return: fa0, fa1

## Memory Layout

### Data Segment Organization

```
0x80000000 +------------------+
           | Model Parameters |  (~1 MB)
           | (weights, biases)|
           +------------------+
           | Hilbert indices  |
           +------------------+
           | Intermediate     |
0x80100000 | Buffers          |  (~8 MB for temp arrays)
           |                  |
           +------------------+
0x80800000 | Stack (grows ↓)  |  (grows downward)
           +------------------+
```

### Buffer Allocation (in .data section)

1. **Model Parameters** (1 MB): All learned weights and biases
2. **Hilbert Indices** (16 KB): Pre-computed Hilbert curve mapping
3. **Intermediate Buffers**: Layer outputs used by next layer
   - After Hilbert: 16 KB (4096 floats)
   - After U-project: 256 KB (4096×64 floats)
   - After S4D-1: 256 KB (4096×64 floats)
   - etc.

### Memory Access Patterns

Key calculation for array indexing:

```
Physical Address = Base Register + Byte Offset
Byte Offset = (Linear Index) × 4   [for float32]

Examples:
  weight[o*in_dim + i] → offset = (o*in_dim + i) * 4
  data[t*D_MODEL + d] → offset = (t*64 + d) * 4
```

## Layer Implementations

### 1. Hilbert Scan (`hilbert_scan.s`)

**Purpose**: Reorder pixels along a Hilbert space-filling curve.

**Algorithm**:
```
for d = 0 to 4095:
    flat2d = hilbert_indices[d]  // 1D→2D mapping
    for c = 0 to 0:               // C_IN=1
        out[d*1 + c] = img[c*4096 + flat2d]
```

**Key Points**:
- Pure integer operations (no floating-point)
- Triple-nested loop (outer: 4096, middle: 1, inner: read/write)
- MSE tolerance: 10⁻¹² (exact match expected)
- Performance: ~4M dynamic instructions

**Register Allocation**:
- t0: outer loop counter (d)
- t1: inner loop counter (c)
- t2: loaded hilbert index
- t3, t4: address calculations
- ft0: temporary float load/store

**Critical Loop**:
```assembly
slli t3, t0, 2              # byte offset for hilbert_indices[d]
add t3, a0, t3
lw t2, 0(t3)                # load hilbert_indices[d]

slli t4, t0, 2              # byte offset for output
add t4, a2, t4
flw ft0, addr(a1, t2)       # load from input
fsw ft0, 0(t4)              # store to output
```

### 2. Linear Layer (`linear_layer.s`)

**Purpose**: Affine transformation (matrix-vector multiplication with bias).

**Algorithm**:
```
for t = 0 to seq_len-1:        // timestep
    for o = 0 to out_dim-1:    // output neuron
        acc = bias[o]
        for i = 0 to in_dim-1: // input feature
            acc += weight[o*in_dim + i] * in[t*in_dim + i]
        out[t*out_dim + o] = acc
```

**Usage**:
- Input Projection: in_dim=1, out_dim=64, seq_len=4096 (262M dynamic instr)
- FC Head: in_dim=64, out_dim=4, seq_len=1 (~4M dynamic instr)

**Key Points**:
- Innermost loop is tight (typically 1-64 iterations)
- Uses fused multiply-accumulate (fmadd.s) for efficiency
- MSE tolerance: 10⁻⁸

**Register Allocation**:
- t0: outer loop (t)
- t1: middle loop (o)
- t2: inner loop (i)
- t3, t4: address calculations
- fs0: accumulator
- fs1: loaded weight
- fs2: loaded input

**Critical Instruction**:
```assembly
fmadd.s fs0, fs1, fs2, fs0  # fs0 += fs1 * fs2 (fused op)
```

### 3. Take Last Timestep (`take_last_timestep.s`)

**Purpose**: Extract the final timestep from a sequence.

**Algorithm**:
```
offset = (SEQ_LEN - 1) * D_MODEL * 4
memcpy(out, &in[offset], D_MODEL * sizeof(float))
```

**Key Points**:
- Simple loop: 64 iterations (one per D_MODEL)
- D-model=64 floats = 256 bytes
- MSE tolerance: 10⁻¹² (exact match)
- Performance: ~180 instructions

**Implementation**:
```assembly
li t0, 4095                 # SEQ_LEN - 1
li t1, 64                   # D_MODEL
mul t0, t0, t1              # offset in float units
slli t0, t0, 2              # convert to bytes
add t1, a0, t0              # source address

# Copy loop (64 iterations)
flw ft0, 0(t1)
fsw ft0, 0(a1)
addi t1, t1, 4
addi a1, a1, 4
```

### 4. Softmax (`softmax_inplace.s`)

**Purpose**: Convert logits to probabilities.

**Algorithm**:
```
1. Find max:  max_val = max(x[0..n-1])
2. Exp sum:   for i: x[i] = exp(x[i] - max); sum += x[i]
3. Normalize: for i: x[i] /= sum
```

**Key Points**:
- Numerical stability: subtract max before exp
- In-place modification of input array
- Calls expf_fast (from math.s)
- MSE tolerance: 10⁻⁸
- Typical input size: N_CLASSES=4

**Three-Phase Implementation**:

```assembly
# Phase 1: Find maximum
flw fs3, 0(a0)              # fs3 = x[0]
loop:
    flw fs0, ...            # load x[i]
    fle.s t2, fs0, fs3      # if x[i] <= max, skip
    fmv.s fs3, fs0          # fs3 = x[i] (new max)

# Phase 2: Compute exp(x - max) and sum
fli.s fs2, 0.0              # sum = 0
loop:
    flw fs0, ...            # load x[i]
    fsub.s fs0, fs0, fs3    # x[i] - max
    jal ra, expf_fast       # call exp
    fsw fa0, ...            # store result
    fadd.s fs2, fs2, fa0    # sum += exp

# Phase 3: Normalize by sum
loop:
    flw fs0, ...            # load exp(x[i] - max)
    fdiv.s fs0, fs0, fs2    # divide by sum
    fsw fs0, ...            # store result
```

### 5. GELU (`gelu_inplace.s`)

**Purpose**: Apply Gaussian Error Linear Unit activation.

**Formula**:
```
gelu(x) ≈ 0.5 * x * (1 + tanh(c₁ * (x + c₂ * x³)))

where:
  c₁ = √(2/π) ≈ 0.7979
  c₂ = 0.044715
```

**Key Points**:
- Uses tanh approximation (calls tanhf_fast from math.s)
- Applied in-place to activation tensors
- MSE tolerance: 10⁻⁷
- Applied after S4D Layer 1 and Layer 2

**Implementation One-Liner** (per element):
```assembly
# Compute x³
fmul.s x_sq, x, x           # x²
fmul.s x_cube, x_sq, x      # x³

# Compute argument to tanh: c₁ * (x + c₂ * x³)
fmul.s term, c2, x_cube     # c₂ * x³
fadd.s arg, x, term         # x + c₂*x³
fmul.s arg, arg, c1         # c₁ * (x + c₂*x³)

# Call tanh
jal ra, tanhf_fast          # result in fa0

# Compute 0.5 * x * (1 + tanh(...))
fadd.s one_plus, fa0, 1.0   # 1 + tanh(...)
fmul.s prod, x, one_plus    # x * (1 + tanh(...))
fmul.s result, prod, 0.5    # 0.5 * x * (1 + tanh(...))
```

### 6. Math Functions (`math.s`)

**Function Library**:
- `expf_fast(x)`: e^x using Taylor series (6th order)
- `sinf_fast(x)`: sin(x) using Taylor series (for phase/complex)
- `cosf_fast(x)`: cos(x) using Taylor series
- `tanhf_fast(x)`: tanh(x) using Taylor series
- `sqrtf_fast(x)`: √x using hardware fsqrt.s

**Implementation Template for Transcendentals**:

```assembly
.globl expf_fast
expf_fast:
    # fa0 = x
    # Compute: 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
    
    fli.s ft0, 1.0          # constant 1
    fmv.s result, ft0       # result = 1
    
    # x term and higher
    fmul.s x_sq, x, x       # x²
    fmul.s term, x_sq, 0.5  # x²/2
    fadd.s result, result, x
    fadd.s result, result, term
    
    # Continue for more terms as needed
    # ...
    
    fmv.s fa0, result
    ret
```

**Accuracy Notes**:
- Truncating Taylor series limits both range and precision
- Suitable for moderate ranges (typical network activations)
- For extreme values, accuracy may degrade
- Used by softmax, GELU, and S4D computations

### 7. S4D Layer (`s4d_layer.s`)

**Purpose**: Process sequence through selective state-space model.

**Algorithm** (Simplified):
```
1. Initialize state h = 0
2. For each timestep t:
   a. Load input x[t]
   b. Update state: h := A*h + B*x (where B ≈ implicit 1)
   c. Readout: y[t] = Re(C*h) + D*x[t]
3. Return output y[0..seq_len-1]
```

**Current Status**:
⚠️ **PLACEHOLDER IMPLEMENTATION** - Does not implement full state-space dynamics.

Current version computes:
```
y[t] = D[d] * x[t*D_MODEL + d]  // Feedthrough only
```

**Full Implementation Needed**:
```
1. Discrete state update: A_disc = exp(dt * A_continuous)
2. Complex arithmetic: A_real + j*A_imag
3. Matrix-vector product: h = A*h + x
4. Complex readout: y = Re(C*h) + D*x
```

**Register Usage** (for full implementation):
- Stack: h_real[D_STATE] and h_imag[D_STATE] buffers
- Temps: t0-t7 for loop counters and indices
- Floats: fs0-fs3 for state computations

**TODO**:
- [ ] Implement matrix exponential: exp(M) for complex matrices
- [ ] Implement complex number arithmetic
- [ ] Full state-space update loop
- [ ] Validation against Python reference

---

## Debugging Tips

### Printing Values (VeeR-iSS Environment)

```assembly
# Print a single float value
fmv.x.s a0, fa0             # Convert float to integer bits
li a7, 1                    # Syscall for print
ecall

# Print an integer
li a0, 42
li a7, 1
ecall
```

### Memory Inspection

Use VeeR-iSS debugger:
```bash
veer-iss --debug bin/program
(gdb) x/4f $sp-16   # Examine 4 floats on stack
```

### Comparing Assembly vs Python

1. Add intermediate printf statements to C code
2. Run both C and assembly versions
3. Compare outputs at checkpoint
4. Binary-search to find divergence point

---

## Performance Expectations

### Instruction Counts (Approximate)

| Layer | Static | Dynamic (per sample) |
|-------|--------|---------------------|
| Hilbert | 30 | 4.1M |
| Linear (U) | 25 | 262M |
| Linear (FC) | 25 | 4M |
| Softmax | 35 | 512K |
| GELU | 40 | 1.8M |
| S4D | 50+ | 50M+ |

### Optimization Targets (Milestone 4)

- Loop unrolling for small inner loops
- RVV vectorization for batch operations
- Cache-friendly memory access patterns

---

## References

- RISC-V ISA Spec: https://riscv.org/technical/specifications/
- Calling Convention: https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf
- VeeR-iSS: https://github.com/chipsalliance/VeeR-ISS
- Assembly Examples: Examples in [Milestone 2 Report]

---

**Document Version**: 1.0
**Last Updated**: April 17, 2026
