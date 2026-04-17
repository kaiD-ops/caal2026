# =============================================================================
# layers.s  –  S4D inference layer routines in RISC-V assembly
#
# Layers: hilbert_scan, linear_layer, s4d_layer, gelu_inplace,
#         softmax_inplace, take_last_timestep
#
# Constants (from nn.h):
#   SEQ_LEN  = 4096
#   D_MODEL  = 64
#   D_STATE  = 32
#   N_CLASSES= 4
#   C_IN     = 1
# =============================================================================

.section .text

# =============================================================================
# hilbert_scan
# Reorders image pixels into 1D Hilbert-curve sequence.
#
# C signature: void hilbert_scan(int32_t* hilbert_idx, float* img, float* out)
# Arguments:
#   a0 = hilbert_indices (int32 array, length SEQ_LEN=4096)
#   a1 = img             (float array, length C_IN*64*64=4096)
#   a2 = out             (float array, length SEQ_LEN*C_IN=4096)
#
# Algorithm: for d in [0,4096): out[d] = img[hilbert_indices[d]]
# Pure index lookup – no arithmetic, exact match guaranteed.
# =============================================================================
.global hilbert_scan
hilbert_scan:
    li      t0, 4096                # loop counter = SEQ_LEN
    mv      t1, a0                  # t1 = ptr to hilbert_indices
    mv      t2, a1                  # t2 = ptr to img
    mv      t3, a2                  # t3 = ptr to out
hs_loop:
    beqz    t0, hs_done
    lw      t4, 0(t1)               # t4 = hilbert_indices[d]  (flat2d index)
    slli    t5, t4, 2               # t5 = flat2d * 4  (byte offset into img)
    add     t6, t2, t5              # t6 = &img[flat2d]
    flw     fa0, 0(t6)              # fa0 = img[flat2d]
    fsw     fa0, 0(t3)              # out[d] = fa0
    addi    t1, t1, 4               # advance hilbert_indices ptr
    addi    t3, t3, 4               # advance out ptr
    addi    t0, t0, -1
    j       hs_loop
hs_done:
    ret

# =============================================================================
# linear_layer
# Affine transform: Y = X * W^T + b
# Handles both sequence (seq_len > 1) and vector (seq_len = 1) inputs.
#
# C signature: void linear_layer(float* W, float* b, float* in, float* out,
#                                int in_dim, int out_dim, int seq_len)
# Arguments:
#   a0 = W       (row-major, shape [out_dim, in_dim])
#   a1 = b       (shape [out_dim])
#   a2 = in      (shape [seq_len, in_dim])
#   a3 = out     (shape [seq_len, out_dim])
#   a4 = in_dim
#   a5 = out_dim
#   a6 = seq_len
# =============================================================================
.global linear_layer
linear_layer:
    # Save callee-saved registers
    addi    sp, sp, -28
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)

    mv      s0, a0                  # W
    mv      s1, a1                  # b
    mv      s2, a2                  # in
    mv      s3, a3                  # out
    mv      s4, a4                  # in_dim
    mv      s5, a5                  # out_dim
    mv      s6, a6                  # seq_len

    # outer loop: timestep t in [0, seq_len)
ll_t_loop:
    beqz    s6, ll_done

    # middle loop: output neuron o in [0, out_dim)
    mv      t3, s5                  # o counter
    mv      t4, s1                  # b pointer
    mv      t5, s3                  # out row pointer

ll_o_loop:
    beqz    t3, ll_t_next

    # acc = b[o]
    flw     fa0, 0(t4)              # acc = b[o]

    # inner loop: i in [0, in_dim) — dot product W[o] . x_t
    # W row o starts at s0 + o*in_dim*4
    # Compute pointer: t6 = s0 + (s5-t3) * in_dim * 4
    sub     t6, s5, t3              # t6 = s5 - t3 = o (0-indexed)
    mul     t6, t6, s4              # t6 = o * in_dim
    slli    t6, t6, 2               # t6 = o * in_dim * 4
    add     t6, s0, t6              # t6 = &W[o][0]

    mv      t0, s2                  # in row pointer (x_t)
    mv      t1, s4                  # i counter

ll_i_loop:
    beqz    t1, ll_i_done
    flw     fa1, 0(t6)              # W[o][i]
    flw     fa2, 0(t0)              # x_t[i]
    fmadd.s fa0, fa1, fa2, fa0      # acc += W[o][i] * x_t[i]
    addi    t6, t6, 4
    addi    t0, t0, 4
    addi    t1, t1, -1
    j       ll_i_loop
ll_i_done:
    fsw     fa0, 0(t5)              # out[t][o] = acc
    addi    t4, t4, 4               # next b[o]
    addi    t5, t5, 4               # next out[t][o]
    addi    t3, t3, -1
    j       ll_o_loop

ll_t_next:
    # advance in by in_dim*4
    slli    t0, s4, 2
    add     s2, s2, t0
    # advance out by out_dim*4
    slli    t0, s5, 2
    add     s3, s3, t0
    addi    s6, s6, -1
    j       ll_t_loop

ll_done:
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 28
    ret

# =============================================================================
# s4d_layer
# S4D diagonal structured state space layer using direct convolution.
# K[t] = 2*Re(sum_n C_tilde_n * exp(dtA_n * t))   (direct, no iterative power)
# y[t] = D*u[t] + sum_{j=0}^{t} K[j]*u[t-j]
#
# C signature:
#   void s4d_layer(float* log_dt, float* log_A_real, float* A_imag,
#                  float* C_mat, float* D_vec, float* in, float* out)
# Arguments:
#   a0 = log_dt      [D_MODEL]
#   a1 = log_A_real  [D_MODEL][D_STATE]
#   a2 = A_imag      [D_MODEL][D_STATE]
#   a3 = C_mat       [D_MODEL][D_STATE][2]
#   a4 = D_vec       [D_MODEL]
#   a5 = in          [SEQ_LEN][D_MODEL]
#   a6 = out         [SEQ_LEN][D_MODEL]
#
# Note: uses s4d_kernel buffer for temporary kernel storage
# =============================================================================
.global s4d_layer
s4d_layer:
    addi    sp, sp, -48
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)
    sw      s10, 44(sp)

    mv      s0, a0      # log_dt
    mv      s1, a1      # log_A_real
    mv      s2, a2      # A_imag
    mv      s3, a3      # C_mat
    mv      s4, a4      # D_vec
    mv      s5, a5      # in
    mv      s6, a6      # out

    # outer loop: channel h in [0, D_MODEL=64)
    li      s7, 0                   # h = 0
s4d_h_loop:
    li      t0, 64                  # D_MODEL
    bge     s7, t0, s4d_h_done

    # dt = exp(log_dt[h])
    slli    t0, s7, 2               # h*4
    add     t0, s0, t0
    flw     fa0, 0(t0)              # log_dt[h]
    call    expf                    # fa0 = dt
    fsw     fa0, s4d_dt, t0         # save dt (use temp storage)
    # Actually store in a float register — use fs0
    fmv.s   fs0, fa0                # fs0 = dt

    # Generate kernel K[t] for t in [0, SEQ_LEN=4096)
    # K[t] = sum_{n=0}^{D_STATE-1} 2*Re(C_tilde_n * exp(dtA_n * t))
    # where C_tilde_n = C_n * (exp(dtA_n) - 1) / A_n
    #
    # For each t, compute exp(dtA_n * t) directly:
    #   dtA_n = (-exp(log_A_real[h,n]) + j*A_imag[h,n]) * dt
    #   exp(dtA_n * t) = exp(dtAr*t) * (cos(dtAi*t) + j*sin(dtAi*t))

    # Precompute dtAr[n] and dtAi[n] and C_tilde[n] for all n
    # Store in per-channel temp arrays on stack or in .bss
    # For simplicity: compute kernel[t] by iterating over n for each t

    # Pointer to kernel buffer
    lui     t0, %hi(s4d_kernel)
    addi    s8, t0, %lo(s4d_kernel) # s8 = &s4d_kernel[0]

    # Precompute per-state: dtAr[n], dtAi[n], Ct_r[n], Ct_i[n]
    # Store in s4d_dtAr, s4d_dtAi, s4d_Ct_r, s4d_Ct_i (32 floats each)
    li      s9, 0                   # n = 0
s4d_precomp_loop:
    li      t0, 32                  # D_STATE
    bge     s9, t0, s4d_precomp_done

    # Ar = -exp(log_A_real[h*D_STATE + n])
    li      t0, 32
    mul     t0, s7, t0              # h * D_STATE
    add     t0, t0, s9              # h*D_STATE + n
    slli    t0, t0, 2               # * 4
    add     t0, s1, t0              # &log_A_real[h,n]
    flw     fa0, 0(t0)
    call    expf                    # fa0 = exp(log_A_real[h,n])
    fneg.s  fs1, fa0                # fs1 = Ar = -exp(...)

    # Ai = A_imag[h*D_STATE + n]
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s2, t0
    flw     fs2, 0(t0)              # fs2 = Ai

    # dtAr = Ar * dt
    fmul.s  fs3, fs1, fs0           # fs3 = dtAr = Ar * dt
    # dtAi = Ai * dt
    fmul.s  fs4, fs2, fs0           # fs4 = dtAi = Ai * dt

    # A_bar_r = exp(dtAr) * cos(dtAi)
    fmv.s   fa0, fs3
    call    expf                    # fa0 = exp(dtAr)
    fmv.s   fs5, fa0                # fs5 = mag = exp(dtAr)
    fmv.s   fa0, fs4
    call    cosf                    # fa0 = cos(dtAi)
    fmul.s  fs6, fs5, fa0           # fs6 = A_bar_r = mag*cos
    fmv.s   fa0, fs4
    call    sinf                    # fa0 = sin(dtAi)
    fmul.s  fs7, fs5, fa0           # fs7 = A_bar_i = mag*sin

    # C_tilde = C * (A_bar - 1) / A  (complex arithmetic)
    # Load C[h, n] = C_mat[(h*D_STATE+n)*2] (re), C_mat[(h*D_STATE+n)*2+1] (im)
    li      t0, 32
    mul     t0, s7, t0              # h*D_STATE
    add     t0, t0, s9              # h*D_STATE+n
    li      t1, 2
    mul     t0, t0, t1              # (h*D_STATE+n)*2
    slli    t0, t0, 2               # *4
    add     t0, s3, t0
    flw     fa1, 0(t0)              # Cr = Re(C[h,n])
    flw     fa2, 4(t0)              # Ci = Im(C[h,n])

    # em1r = A_bar_r - 1,  em1i = A_bar_i
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fs6, ft0           # ft1 = em1r = A_bar_r - 1
    fmv.s   ft2, fs7                # ft2 = em1i = A_bar_i

    # num = C * (A_bar-1): complex multiply
    # num_r = Cr*em1r - Ci*em1i
    fmul.s  ft3, fa1, ft1           # Cr*em1r
    fmul.s  ft4, fa2, ft2           # Ci*em1i
    fsub.s  ft3, ft3, ft4           # num_r
    # num_i = Cr*em1i + Ci*em1r
    fmul.s  ft4, fa1, ft2           # Cr*em1i
    fmul.s  ft5, fa2, ft1           # Ci*em1r
    fadd.s  ft4, ft4, ft5           # num_i

    # Amag2 = Ar^2 + Ai^2
    fmul.s  ft5, fs1, fs1           # Ar^2
    fmul.s  ft6, fs2, fs2           # Ai^2
    fadd.s  ft5, ft5, ft6           # Amag2

    # Ct_r = (num_r*Ar + num_i*Ai) / Amag2
    fmul.s  ft6, ft3, fs1           # num_r*Ar
    fmul.s  ft7, ft4, fs2           # num_i*Ai
    fadd.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft5           # Ct_r

    # Ct_i = (num_i*Ar - num_r*Ai) / Amag2
    fmul.s  ft7, ft4, fs1           # num_i*Ar
    fmul.s  fa3, ft3, fs2           # num_r*Ai
    fsub.s  ft7, ft7, fa3
    fdiv.s  ft7, ft7, ft5           # Ct_i

    # Store dtAr, dtAi, Ct_r, Ct_i for this n
    slli    t0, s9, 2               # n*4
    lui     t1, %hi(s4d_dtAr)
    add     t1, t1, t0
    fsw     fs3, %lo(s4d_dtAr)(t1)

    lui     t1, %hi(s4d_dtAi)
    add     t1, t1, t0
    fsw     fs4, %lo(s4d_dtAi)(t1)

    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    fsw     ft6, %lo(s4d_Ct_r)(t1)

    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    fsw     ft7, %lo(s4d_Ct_i)(t1)

    addi    s9, s9, 1
    j       s4d_precomp_loop
s4d_precomp_done:

    # Build kernel K[t] for t in [0, 4096)
    # K[t] = sum_n 2*Re(Ct_n * exp(dtA_n*t))
    #       = sum_n 2*(Ct_r_n*exp(dtAr_n*t)*cos(dtAi_n*t) - Ct_i_n*exp(dtAr_n*t)*sin(dtAi_n*t))
    li      s9, 0                   # t = 0
    mv      s10, s8                 # ptr into kernel
s4d_kern_t:
    li      t0, 4096
    bge     s9, t0, s4d_kern_done

    # kval = 0
    fmv.w.x fa5, zero               # fa5 = kval = 0.0

    li      s10, 0                  # n = 0 (reuse s10 as n here)
s4d_kern_n:
    li      t0, 32
    bge     s10, t0, s4d_kern_n_done

    # Load precomputed dtAr[n], dtAi[n], Ct_r[n], Ct_i[n]
    slli    t0, s10, 2
    lui     t1, %hi(s4d_dtAr)
    add     t1, t1, t0
    flw     fs3, %lo(s4d_dtAr)(t1)  # dtAr_n

    lui     t1, %hi(s4d_dtAi)
    add     t1, t1, t0
    flw     fs4, %lo(s4d_dtAi)(t1)  # dtAi_n

    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    flw     fs5, %lo(s4d_Ct_r)(t1)  # Ct_r_n

    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    flw     fs6, %lo(s4d_Ct_i)(t1)  # Ct_i_n

    # exp(dtAr_n * t)
    fcvt.s.w fa0, s9                # fa0 = float(t)
    fmul.s  fa0, fa0, fs3           # fa0 = dtAr_n * t
    call    expf                    # fa0 = exp(dtAr_n*t)
    fmv.s   fs7, fa0                # fs7 = mag_t

    # cos(dtAi_n * t)
    fcvt.s.w fa0, s9
    fmul.s  fa0, fa0, fs4           # dtAi_n * t
    call    cosf                    # fa0 = cos(dtAi_n*t)
    fmul.s  ft0, fs7, fa0           # ft0 = exp_r = mag_t * cos

    # sin(dtAi_n * t)
    fcvt.s.w fa0, s9
    fmul.s  fa0, fa0, fs4
    call    sinf                    # fa0 = sin(dtAi_n*t)
    fmul.s  ft1, fs7, fa0           # ft1 = exp_i = mag_t * sin

    # 2*Re(Ct_n * exp_t) = 2*(Ct_r*exp_r - Ct_i*exp_i)
    fmul.s  ft2, fs5, ft0           # Ct_r * exp_r
    fmul.s  ft3, fs6, ft1           # Ct_i * exp_i
    fsub.s  ft2, ft2, ft3           # Re(Ct * exp)
    lui     t1, %hi(math_two)
    flw     ft3, %lo(math_two)(t1)
    fmul.s  ft2, ft3, ft2           # 2 * Re(...)
    fadd.s  fa5, fa5, ft2           # kval += ...

    addi    s10, s10, 1
    j       s4d_kern_n

s4d_kern_n_done:
    # Store K[t]
    li      t0, 4096
    sub     t0, t0, s9              # recompute offset since s10 was reused
    # s9 still = t, s8 = base of kernel
    slli    t0, s9, 2               # t * 4
    add     t0, s8, t0
    fsw     fa5, 0(t0)              # kernel[t] = kval

    addi    s9, s9, 1
    j       s4d_kern_t
s4d_kern_done:

    # Causal convolution: y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j]*u[t-j][h]
    # D[h]
    slli    t0, s7, 2
    add     t0, s4, t0
    flw     fs1, 0(t0)              # fs1 = D[h]

    li      s9, 0                   # t = 0
s4d_conv_t:
    li      t0, 4096
    bge     s9, t0, s4d_conv_done

    # acc = D[h] * in[t][h]
    # in[t][h] = in[t*D_MODEL + h] = in[t*64 + h]
    li      t0, 64
    mul     t0, s9, t0              # t*64
    add     t0, t0, s7              # t*64 + h
    slli    t0, t0, 2               # *4
    add     t0, s5, t0              # &in[t][h]
    flw     fa0, 0(t0)              # u[t][h]
    fmul.s  fa5, fs1, fa0           # acc = D[h] * u[t][h]

    # sum_{j=0}^{t} K[j] * u[t-j][h]
    li      s10, 0                  # j = 0
s4d_conv_j:
    bgt     s10, s9, s4d_conv_j_done

    # K[j]
    slli    t0, s10, 2
    add     t0, s8, t0
    flw     fa1, 0(t0)              # K[j]

    # u[t-j][h]
    sub     t0, s9, s10             # t-j
    li      t1, 64
    mul     t0, t0, t1              # (t-j)*64
    add     t0, t0, s7              # (t-j)*64 + h
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa2, 0(t0)              # u[t-j][h]

    fmadd.s fa5, fa1, fa2, fa5      # acc += K[j] * u[t-j][h]

    addi    s10, s10, 1
    j       s4d_conv_j
s4d_conv_j_done:

    # out[t][h] = acc
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s6, t0
    fsw     fa5, 0(t0)

    addi    s9, s9, 1
    j       s4d_conv_t
s4d_conv_done:

    addi    s7, s7, 1               # h++
    j       s4d_h_loop
s4d_h_done:
    lw      ra, 0(sp)
    lw      s0, 4(sp)
    lw      s1, 8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    lw      s4, 20(sp)
    lw      s5, 24(sp)
    lw      s6, 28(sp)
    lw      s7, 32(sp)
    lw      s8, 36(sp)
    lw      s9, 40(sp)
    lw      s10, 44(sp)
    addi    sp, sp, 48
    ret

# =============================================================================
# gelu_inplace
# GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
# Applied element-wise in place.
#
# C signature: void gelu_inplace(float* x, int n)
# Arguments: a0 = x ptr, a1 = n
# =============================================================================
.global gelu_inplace
gelu_inplace:
    addi    sp, sp, -8
    sw      ra, 0(sp)
    sw      s0, 4(sp)

    mv      s0, a0                  # ptr
    mv      s1, a1                  # n (use t-reg — but save s1)
    # s1 not saved above — fix:
    addi    sp, sp, -4
    sw      s1, 0(sp)
    mv      s1, a1

gelu_loop:
    beqz    s1, gelu_done
    flw     fa0, 0(s0)              # x

    # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul.s  ft0, fa0, fa0           # x^2
    fmul.s  ft0, ft0, fa0           # x^3
    lui     t0, %hi(gelu_c1)
    flw     ft1, %lo(gelu_c1)(t0)   # 0.044715
    fmul.s  ft0, ft0, ft1           # 0.044715*x^3
    fadd.s  ft0, fa0, ft0           # x + 0.044715*x^3
    lui     t0, %hi(gelu_c2)
    flw     ft1, %lo(gelu_c2)(t0)   # sqrt(2/pi) = 0.7978845608
    fmul.s  ft0, ft0, ft1           # inner

    # tanh(inner)
    fmv.s   fa0, ft0
    call    tanhf                   # fa0 = tanh(inner)

    # 0.5 * x * (1 + tanh)
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fadd.s  fa0, fa0, ft1           # 1 + tanh
    flw     ft2, 0(s0)              # reload x
    fmul.s  fa0, ft2, fa0           # x * (1+tanh)
    lui     t0, %hi(gelu_half)
    flw     ft1, %lo(gelu_half)(t0)
    fmul.s  fa0, fa0, ft1           # 0.5 * ...
    fsw     fa0, 0(s0)              # store back

    addi    s0, s0, 4
    addi    s1, s1, -1
    j       gelu_loop
gelu_done:
    lw      s1, 0(sp)
    addi    sp, sp, 4
    lw      s0, 4(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 8
    ret

# =============================================================================
# softmax_inplace
# Numerically stable: subtract max before exp, then normalize.
# Output probabilities sum to 1.0.
#
# C signature: void softmax_inplace(float* x, int n)
# Arguments: a0 = x, a1 = n (always N_CLASSES=4 for us)
# =============================================================================
.global softmax_inplace
softmax_inplace:
    addi    sp, sp, -12
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)

    mv      s0, a0
    mv      s1, a1

    # Find max
    flw     fa0, 0(s0)              # max = x[0]
    li      t0, 1
sm_max_loop:
    bge     t0, s1, sm_max_done
    slli    t1, t0, 2
    add     t1, s0, t1
    flw     ft0, 0(t1)
    flt.s   t2, fa0, ft0
    beqz    t2, sm_max_no_update
    fmv.s   fa0, ft0
sm_max_no_update:
    addi    t0, t0, 1
    j       sm_max_loop
sm_max_done:
    fmv.s   fs0, fa0                # fs0 = max

    # exp(x[i] - max) and accumulate sum
    fmv.w.x fa3, zero               # sum = 0
    li      t0, 0
    mv      t1, s0
sm_exp_loop:
    bge     t0, s1, sm_exp_done
    flw     fa0, 0(t1)
    fsub.s  fa0, fa0, fs0           # x[i] - max
    call    expf                    # exp(x[i]-max)
    fsw     fa0, 0(t1)
    fadd.s  fa3, fa3, fa0           # sum += exp(...)
    addi    t1, t1, 4
    addi    t0, t0, 1
    j       sm_exp_loop
sm_exp_done:

    # normalize
    li      t0, 0
    mv      t1, s0
sm_norm_loop:
    bge     t0, s1, sm_norm_done
    flw     fa0, 0(t1)
    fdiv.s  fa0, fa0, fa3
    fsw     fa0, 0(t1)
    addi    t1, t1, 4
    addi    t0, t0, 1
    j       sm_norm_loop
sm_norm_done:

    lw      s0, 4(sp)
    lw      s1, 8(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 12
    ret

# =============================================================================
# take_last_timestep
# Extract last row from (SEQ_LEN, D_MODEL) buffer.
# output[d] = input[(SEQ_LEN-1)*D_MODEL + d]
#
# C signature: void take_last_timestep(float* in, float* out)
# Arguments: a0 = in, a1 = out
# =============================================================================
.global take_last_timestep
take_last_timestep:
    # offset = (4096-1)*64*4 = 4095*256 = 1048320 bytes
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1              # 4095 * 64 = 262080 elements
    slli    t0, t0, 2               # * 4 = 1048320 bytes
    add     a0, a0, t0              # in + offset = &in[4095][0]

    li      t0, 64                  # D_MODEL
tl_loop:
    beqz    t0, tl_done
    flw     ft0, 0(a0)
    fsw     ft0, 0(a1)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    t0, t0, -1
    j       tl_loop
tl_done:
    ret

# =============================================================================
# Layer constants and temporary buffers
# =============================================================================
.section .data
.align 2

gelu_c1:    .float 0.044715
gelu_c2:    .float 0.79788456080
gelu_half:  .float 0.5

.section .bss
.align 2

s4d_dt:     .space 4                # temp: current dt
s4d_dtAr:   .space 128              # D_STATE * 4 = 32*4
s4d_dtAi:   .space 128
s4d_Ct_r:   .space 128
s4d_Ct_i:   .space 128
s4d_kernel: .space 16384            # SEQ_LEN * 4 = 4096*4
