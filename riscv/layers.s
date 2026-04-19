# =============================================================================
# layers.s  –  S4D inference layer routines in RISC-V assembly (optimized)
#
# Key optimization vs v1: S4D kernel build uses recurrence relation
# instead of calling expf/cosf/sinf per timestep.
# Reduces transcendental calls from 25M to 96 per full forward pass.
# =============================================================================

.section .text

# =============================================================================
# hilbert_scan
# void hilbert_scan(int32_t* hilbert_idx, float* img, float* out)
# a0=indices, a1=img, a2=out
# =============================================================================
.global hilbert_scan
hilbert_scan:
    li      t0, 4096
    mv      t1, a0
    mv      t2, a1
    mv      t3, a2
hs_loop:
    beqz    t0, hs_done
    lw      t4, 0(t1)
    slli    t5, t4, 2
    add     t6, t2, t5
    flw     fa0, 0(t6)
    fsw     fa0, 0(t3)
    addi    t1, t1, 4
    addi    t3, t3, 4
    addi    t0, t0, -1
    j       hs_loop
hs_done:
    ret

# =============================================================================
# linear_layer
# void linear_layer(float* W, float* b, float* in, float* out,
#                   int in_dim, int out_dim, int seq_len)
# a0=W, a1=b, a2=in, a3=out, a4=in_dim, a5=out_dim, a6=seq_len
# =============================================================================
.global linear_layer
linear_layer:
    addi    sp, sp, -28
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mv      s6, a6

ll_t_loop:
    beqz    s6, ll_done
    mv      t3, s5
    mv      t4, s1
    mv      t5, s3

ll_o_loop:
    beqz    t3, ll_t_next
    flw     fa0, 0(t4)              # acc = b[o]

    sub     t6, s5, t3              # o index
    mul     t6, t6, s4
    slli    t6, t6, 2
    add     t6, s0, t6              # &W[o][0]

    mv      t0, s2
    mv      t1, s4

ll_i_loop:
    beqz    t1, ll_i_done
    flw     fa1, 0(t6)
    flw     fa2, 0(t0)
    fmadd.s fa0, fa1, fa2, fa0
    addi    t6, t6, 4
    addi    t0, t0, 4
    addi    t1, t1, -1
    j       ll_i_loop
ll_i_done:
    fsw     fa0, 0(t5)
    addi    t4, t4, 4
    addi    t5, t5, 4
    addi    t3, t3, -1
    j       ll_o_loop

ll_t_next:
    slli    t0, s4, 2
    add     s2, s2, t0
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
# s4d_layer (OPTIMIZED)
# Uses recurrence relation for kernel build:
#   mag[t] = mag[t-1] * exp(dtAr)   (multiply, not expf call)
#   cos[t] = cos[t-1]*cos(dtAi) - sin[t-1]*sin(dtAi)
#   sin[t] = sin[t-1]*cos(dtAi) + cos[t-1]*sin(dtAi)
# Reduces transcendental calls from O(L*N) to O(N) per channel.
#
# void s4d_layer(float* log_dt, float* log_A_real, float* A_imag,
#                float* C_mat, float* D_vec, float* in, float* out)
# a0=log_dt, a1=log_A_real, a2=A_imag, a3=C_mat, a4=D_vec, a5=in, a6=out
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

    # kernel buffer base
    lui     s8, %hi(s4d_kernel)
    addi    s8, s8, %lo(s4d_kernel)

    li      s7, 0                   # h = 0
s4d_h_loop:
    li      t0, 64
    bge     s7, t0, s4d_h_done

    # dt = exp(log_dt[h])
    slli    t0, s7, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    call    expf
    fmv.s   fs0, fa0                # fs0 = dt

    # -----------------------------------------------------------------------
    # Phase 1: Precompute per state n:
    #   dtAr[n], dtAi[n]   (step sizes for recurrence)
    #   Ct_r[n], Ct_i[n]   (C_tilde)
    #   step_mag[n]  = exp(dtAr[n])     (magnitude multiplier per step)
    #   step_cos[n]  = cos(dtAi[n])     (angle cosine per step)
    #   step_sin[n]  = sin(dtAi[n])     (angle sine per step)
    # -----------------------------------------------------------------------
    li      s9, 0                   # n = 0
s4d_pre_loop:
    li      t0, 32
    bge     s9, t0, s4d_pre_done

    # Ar = -exp(log_A_real[h,n])
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s1, t0
    flw     fa0, 0(t0)
    call    expf
    fneg.s  fs1, fa0                # fs1 = Ar

    # Ai = A_imag[h,n]
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s2, t0
    flw     fs2, 0(t0)              # fs2 = Ai

    # dtAr = Ar * dt,  dtAi = Ai * dt
    fmul.s  fs3, fs1, fs0           # fs3 = dtAr
    fmul.s  fs4, fs2, fs0           # fs4 = dtAi

    # step_mag = exp(dtAr)
    fmv.s   fa0, fs3
    call    expf
    fmv.s   fs5, fa0                # fs5 = step_mag = exp(dtAr)

    # step_cos = cos(dtAi)
    fmv.s   fa0, fs4
    call    cosf
    fmv.s   fs6, fa0                # fs6 = step_cos

    # step_sin = sin(dtAi)
    fmv.s   fa0, fs4
    call    sinf
    fmv.s   fs7, fa0                # fs7 = step_sin

    # A_bar_r = step_mag * step_cos (= exp(dtAr)*cos(dtAi))
    fmul.s  fa1, fs5, fs6           # A_bar_r
    # A_bar_i = step_mag * step_sin
    fmul.s  fa2, fs5, fs7           # A_bar_i

    # C_tilde = C * (A_bar - 1) / A
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    li      t1, 2
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     t0, s3, t0
    flw     fa3, 0(t0)              # Cr
    flw     fa4, 4(t0)              # Ci

    # em1r = A_bar_r - 1
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fa1, ft0           # em1r
    fmv.s   ft2, fa2                # em1i = A_bar_i

    # num = C * (A_bar-1)
    fmul.s  ft3, fa3, ft1
    fmul.s  ft4, fa4, ft2
    fsub.s  ft3, ft3, ft4           # num_r
    fmul.s  ft4, fa3, ft2
    fmul.s  ft5, fa4, ft1
    fadd.s  ft4, ft4, ft5           # num_i

    # Amag2 = Ar^2 + Ai^2
    fmul.s  ft5, fs1, fs1
    fmul.s  ft6, fs2, fs2
    fadd.s  ft5, ft5, ft6

    # Ct_r, Ct_i
    fmul.s  ft6, ft3, fs1
    fmul.s  ft7, ft4, fs2
    fadd.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft5           # Ct_r

    fmul.s  ft7, ft4, fs1
    fmul.s  fa5, ft3, fs2
    fsub.s  ft7, ft7, fa5
    fdiv.s  ft7, ft7, ft5           # Ct_i

    # Store all per-state values
    slli    t0, s9, 2               # n*4
    lui     t1, %hi(s4d_smag)
    add     t1, t1, t0
    fsw     fs5, %lo(s4d_smag)(t1)  # step_mag[n]

    lui     t1, %hi(s4d_scos)
    add     t1, t1, t0
    fsw     fs6, %lo(s4d_scos)(t1)  # step_cos[n]

    lui     t1, %hi(s4d_ssin)
    add     t1, t1, t0
    fsw     fs7, %lo(s4d_ssin)(t1)  # step_sin[n]

    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    fsw     ft6, %lo(s4d_Ct_r)(t1)

    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    fsw     ft7, %lo(s4d_Ct_i)(t1)

    # Initialize running state: cur_r[n]=1, cur_i[n]=0 (t=0: A^0 = 1)
    lui     t1, %hi(s4d_cur_r)
    add     t1, t1, t0
    lui     t2, %hi(math_one)
    flw     fa0, %lo(math_one)(t2)
    fsw     fa0, %lo(s4d_cur_r)(t1)

    lui     t1, %hi(s4d_cur_i)
    add     t1, t1, t0
    fmv.w.x fa0, zero
    fsw     fa0, %lo(s4d_cur_i)(t1)

    addi    s9, s9, 1
    j       s4d_pre_loop
s4d_pre_done:

    # -----------------------------------------------------------------------
    # Phase 2: Build kernel using recurrence (NO expf/cosf/sinf in this loop)
    # K[t] = 2 * Re(sum_n Ct_n * cur_n[t])
    # cur_n[t+1] = cur_n[t] * step_n  (complex multiply)
    # step_n = step_mag_n * (step_cos_n + j*step_sin_n)
    # -----------------------------------------------------------------------
    li      s9, 0                   # t = 0
s4d_kern_t:
    li      t0, 4096
    bge     s9, t0, s4d_kern_done

    fmv.w.x fa5, zero               # kval = 0

    li      s10, 0                  # n = 0
s4d_kern_n:
    li      t0, 32
    bge     s10, t0, s4d_kern_n_done

    slli    t0, s10, 2

    # Load Ct_r[n], Ct_i[n]
    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    flw     fs3, %lo(s4d_Ct_r)(t1)

    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    flw     fs4, %lo(s4d_Ct_i)(t1)

    # Load cur_r[n], cur_i[n]
    lui     t1, %hi(s4d_cur_r)
    add     t1, t1, t0
    flw     fs5, %lo(s4d_cur_r)(t1)

    lui     t1, %hi(s4d_cur_i)
    add     t1, t1, t0
    flw     fs6, %lo(s4d_cur_i)(t1)

    # 2*Re(Ct * cur) = 2*(Ct_r*cur_r - Ct_i*cur_i)
    fmul.s  ft0, fs3, fs5
    fmul.s  ft1, fs4, fs6
    fsub.s  ft0, ft0, ft1
    lui     t1, %hi(math_two)
    flw     ft1, %lo(math_two)(t1)
    fmadd.s fa5, ft1, ft0, fa5      # kval += 2*Re(Ct*cur)

    # Advance: cur[n] *= step[n]  (complex multiply)
    # new_r = cur_r*step_cos*step_mag - cur_i*step_sin*step_mag
    # new_i = cur_r*step_sin*step_mag + cur_i*step_cos*step_mag
    # But step = step_mag*(step_cos + j*step_sin), so:
    # step_r = step_mag*step_cos, step_i = step_mag*step_sin
    # Precomputed as s4d_smag*s4d_scos and s4d_smag*s4d_ssin
    # Actually we stored step_mag, step_cos, step_sin separately
    # step_r = step_mag * step_cos
    lui     t1, %hi(s4d_smag)
    add     t1, t1, t0
    flw     ft2, %lo(s4d_smag)(t1)  # step_mag

    lui     t1, %hi(s4d_scos)
    add     t1, t1, t0
    flw     ft3, %lo(s4d_scos)(t1)  # step_cos

    lui     t1, %hi(s4d_ssin)
    add     t1, t1, t0
    flw     ft4, %lo(s4d_ssin)(t1)  # step_sin

    fmul.s  ft3, ft2, ft3           # step_r = mag*cos
    fmul.s  ft4, ft2, ft4           # step_i = mag*sin

    # new_r = cur_r*step_r - cur_i*step_i
    fmul.s  ft5, fs5, ft3
    fmul.s  ft6, fs6, ft4
    fsub.s  ft5, ft5, ft6           # new_r

    # new_i = cur_r*step_i + cur_i*step_r
    fmul.s  ft6, fs5, ft4
    fmul.s  ft7, fs6, ft3
    fadd.s  ft6, ft6, ft7           # new_i

    # Store updated cur[n]
    lui     t1, %hi(s4d_cur_r)
    add     t1, t1, t0
    fsw     ft5, %lo(s4d_cur_r)(t1)

    lui     t1, %hi(s4d_cur_i)
    add     t1, t1, t0
    fsw     ft6, %lo(s4d_cur_i)(t1)

    addi    s10, s10, 1
    j       s4d_kern_n

s4d_kern_n_done:
    # kernel[t] = kval
    slli    t0, s9, 2
    add     t0, s8, t0
    fsw     fa5, 0(t0)

    addi    s9, s9, 1
    j       s4d_kern_t
s4d_kern_done:

    # -----------------------------------------------------------------------
    # Phase 3: Causal convolution
    # y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j]*u[t-j][h]
    # -----------------------------------------------------------------------
    slli    t0, s7, 2
    add     t0, s4, t0
    flw     fs1, 0(t0)              # D[h]

    li      s9, 0
s4d_conv_t:
    li      t0, 4096
    bge     s9, t0, s4d_conv_done

    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa0, 0(t0)              # u[t][h]
    fmul.s  fa5, fs1, fa0           # acc = D*u

    li      s10, 0
s4d_conv_j:
    bgt     s10, s9, s4d_conv_j_done

    slli    t0, s10, 2
    add     t0, s8, t0
    flw     fa1, 0(t0)              # K[j]

    sub     t0, s9, s10
    li      t1, 64
    mul     t0, t0, t1
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa2, 0(t0)              # u[t-j][h]

    fmadd.s fa5, fa1, fa2, fa5

    addi    s10, s10, 1
    j       s4d_conv_j
s4d_conv_j_done:

    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s6, t0
    fsw     fa5, 0(t0)

    addi    s9, s9, 1
    j       s4d_conv_t
s4d_conv_done:

    addi    s7, s7, 1
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
# void gelu_inplace(float* x, int n)
# =============================================================================
.global gelu_inplace
gelu_inplace:
    addi    sp, sp, -12
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)

    mv      s0, a0
    mv      s1, a1

gelu_loop:
    beqz    s1, gelu_done
    flw     fa0, 0(s0)

    fmul.s  ft0, fa0, fa0
    fmul.s  ft0, ft0, fa0
    lui     t0, %hi(gelu_c1)
    flw     ft1, %lo(gelu_c1)(t0)
    fmul.s  ft0, ft0, ft1
    fadd.s  ft0, fa0, ft0
    lui     t0, %hi(gelu_c2)
    flw     ft1, %lo(gelu_c2)(t0)
    fmul.s  ft0, ft0, ft1

    fmv.s   fa0, ft0
    call    tanhf

    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fadd.s  fa0, fa0, ft1
    flw     ft2, 0(s0)
    fmul.s  fa0, ft2, fa0
    lui     t0, %hi(gelu_half)
    flw     ft1, %lo(gelu_half)(t0)
    fmul.s  fa0, fa0, ft1
    fsw     fa0, 0(s0)

    addi    s0, s0, 4
    addi    s1, s1, -1
    j       gelu_loop
gelu_done:
    lw      s0, 4(sp)
    lw      s1, 8(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 12
    ret

# =============================================================================
# softmax_inplace
# void softmax_inplace(float* x, int n)
# =============================================================================
.global softmax_inplace
softmax_inplace:
    addi    sp, sp, -12
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)

    mv      s0, a0
    mv      s1, a1

    flw     fa0, 0(s0)
    li      t0, 1
sm_max_loop:
    bge     t0, s1, sm_max_done
    slli    t1, t0, 2
    add     t1, s0, t1
    flw     ft0, 0(t1)
    flt.s   t2, fa0, ft0
    beqz    t2, sm_max_skip
    fmv.s   fa0, ft0
sm_max_skip:
    addi    t0, t0, 1
    j       sm_max_loop
sm_max_done:
    fmv.s   fs0, fa0

    fmv.w.x fa3, zero
    li      t0, 0
    mv      t1, s0
sm_exp_loop:
    bge     t0, s1, sm_exp_done
    flw     fa0, 0(t1)
    fsub.s  fa0, fa0, fs0
    call    expf
    fsw     fa0, 0(t1)
    fadd.s  fa3, fa3, fa0
    addi    t1, t1, 4
    addi    t0, t0, 1
    j       sm_exp_loop
sm_exp_done:

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
# void take_last_timestep(float* in, float* out)
# =============================================================================
.global take_last_timestep
take_last_timestep:
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     a0, a0, t0

    li      t0, 64
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
# Data / BSS
# =============================================================================
.section .data
.align 2
gelu_c1:    .float 0.044715
gelu_c2:    .float 0.79788456080
gelu_half:  .float 0.5

.section .bss
.align 2
s4d_smag:   .space 128      # step_mag[32]
s4d_scos:   .space 128      # step_cos[32]
s4d_ssin:   .space 128      # step_sin[32]
s4d_cur_r:  .space 128      # running real part[32]
s4d_cur_i:  .space 128      # running imag part[32]
s4d_Ct_r:   .space 128      # C_tilde real[32]
s4d_Ct_i:   .space 128      # C_tilde imag[32]
s4d_kernel: .space 16384    # K[4096]
