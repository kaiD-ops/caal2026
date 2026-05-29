# Milestone 3 Calling Conventions

## Register Usage

### Integer Registers

| Register | ABI Name | Role in M3 |
|----------|----------|------------|
| x0       | zero     | Hardwired zero |
| x1       | ra       | Return address (caller-saved, must save before call) |
| x2       | sp       | Stack pointer (always 4-byte aligned) |
| x10-x17  | a0-a7    | Function arguments / return values (caller-saved) |
| x5-x7    | t0-t2    | Temporaries (caller-saved) |
| x28-x31  | t3-t6    | Temporaries (caller-saved) |
| x8-x9    | s0-s1    | Callee-saved (function must save/restore if used) |
| x18-x27  | s2-s11   | Callee-saved (function must save/restore if used) |

### Floating-Point Registers

| Register | ABI Name | Role |
|----------|----------|------|
| f0-f7    | ft0-ft7  | Temporaries (caller-saved) |
| f10-f17  | fa0-fa7  | FP arguments / return values (caller-saved) |
| f8-f9    | fs0-fs1  | Callee-saved |
| f18-f27  | fs2-fs11 | Callee-saved |
| f28-f31  | ft8-ft11 | Temporaries (caller-saved) |

## Function Signatures

### math.s

```
expf(fa0: float) -> fa0: float
cosf(fa0: float) -> fa0: float
sinf(fa0: float) -> fa0: float
tanhf(fa0: float) -> fa0: float
```
Clobbers: ft0-ft7, fa1-fa7, t0-t2. Preserves all s-registers.

### hilbert_scan.s

```
hilbert_scan(a0: int32_t* indices, a1: float* img, a2: float* out) -> void
```
Reorders 4096 floats using pre-computed Hilbert curve indices.
Clobbers: t0-t6, fa0.

### linear_layer.s

```
linear_layer(a0: float* W, a1: float* b, a2: float* in, a3: float* out,
             a4: int in_dim, a5: int out_dim, a6: int seq_len) -> void
```
Computes Y = X * W^T + b. W is stored row-major [out_dim, in_dim].
Clobbers: t0-t6, fa0-fa2.

### s4d_layer.s

```
s4d_layer(a0: float* log_dt, a1: float* log_A_real, a2: float* A_imag,
          a3: float* C_mat,  a4: float* D_vec,       a5: float* in,
          a6: float* out) -> void
```
C_mat is interleaved complex: C_mat[h*32+n] = (float Cr, float Ci).
Shape: D_MODEL=64, N_STATES=32.
Uses BSS scratch buffers: s4d_smag, s4d_scos, s4d_ssin, s4d_cur_r,
s4d_cur_i, s4d_Ct_r, s4d_Ct_i, s4d_kernel (16KB).

### gelu.s

```
gelu_inplace(a0: float* x, a1: int n) -> void
```
Applies GELU in-place. x is modified. Calls tanhf internally.

### take_last_timestep.s

```
take_last_timestep(a0: float* in, a1: float* out) -> void
```
Copies 64 floats from in[4095*64 .. 4095*64+63] to out[0..63].

### softmax.s

```
softmax_inplace(a0: float* x, a1: int n) -> void
```
In-place numerically-stable softmax. Calls expf internally.

## Memory Layout

### Weight File: model_weights.bin (84496 bytes total)

| Offset  | Size  | Shape        | Description          |
|---------|-------|--------------|----------------------|
| 0       | 16384 | [4096]       | Hilbert indices (int32) |
| 16384   | 256   | [64,1]       | UProject W           |
| 16640   | 256   | [64]         | UProject b           |
| 16896   | 256   | [64]         | S4D-1 log_dt         |
| 17152   | 8192  | [64,32]      | S4D-1 log_A_real     |
| 25344   | 8192  | [64,32]      | S4D-1 A_imag         |
| 33536   | 16384 | [64,32,2]    | S4D-1 C (interleaved)|
| 49920   | 256   | [64]         | S4D-1 D              |
| 50176   | 256   | [64]         | S4D-2 log_dt         |
| 50432   | 8192  | [64,32]      | S4D-2 log_A_real     |
| 58624   | 8192  | [64,32]      | S4D-2 A_imag         |
| 66816   | 16384 | [64,32,2]    | S4D-2 C (interleaved)|
| 83200   | 256   | [64]         | S4D-2 D              |
| 83456   | 1024  | [4,64]       | FC weight            |
| 84480   | 16    | [4]          | FC bias              |

### Activation Buffer Sizes

| Buffer      | Size (bytes) | Shape       |
|-------------|-------------|-------------|
| buf_hilbert | 16384       | [4096,1]    |
| buf_proj    | 1048576     | [4096,64]   |
| buf_s4d1    | 1048576     | [4096,64]   |
| buf_s4d2    | 1048576     | [4096,64]   |
| buf_pooled  | 256         | [64]        |
| buf_logits  | 16          | [4]         |

## Stack Convention

- Stack grows downward (decrement sp before use)
- Align sp to 4 bytes before any call
- Callee saves ra + all s-registers it modifies
- Callee restores all saved registers before ret

## VeeR-iSS I/O

- Console output: write bytes to 0xd0580004 (ASCII)
- Halt: write 0xff to 0xd0580000
