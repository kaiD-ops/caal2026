# =============================================================================
# s4d.s - S4D (Structured State Space Diagonal) layer
#
# Implements causal convolution using the S4D formulation:
#   K[t] = 2*Re(sum_n C_tilde_n * A_bar_n^t)  (recurrence - not repeated exp)
#   y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j]*u[t-j][h]
#
# Optimization: recurrence relation for kernel build
#   cur[t+1] = cur[t] * step  (complex multiply, NO expf/cosf/sinf in loop)
#   Reduces transcendental calls from O(L*N) to O(N) per channel
#
# Signature:
#   void s4d_layer(float* log_dt, float* log_A_real, float* A_imag,
#                  float* C_mat, float* D_vec, float* in, float* out)
#   a0=log_dt, a1=log_A_real, a2=A_imag, a3=C_re, a4=C_im,
#   a5=D_vec, a6=in, a7=out
#
# Validation: MSE < 1e-7, MAE < 1e-4 vs C/Python reference
# Dynamic instruction count: ~7.91 billion (SEQ_LEN=4096, D_MODEL=64)
# =============================================================================

.section .text
.global s4d_layer

s4d_layer:
    addi    sp, sp, -52
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
    sw      s11, 48(sp)

    mv      s0, a0      # log_dt
    mv      s1, a1      # log_A_real
    mv      s2, a2      # A_imag
    mv      s3, a3      # C_re
    mv      s4, a4      # C_im
    mv      s5, a5      # D_vec
    mv      s6, a6      # in
    mv      s11, a7     # out

    lui     s8, %hi(s4d_kernel)
    addi    s8, s8, %lo(s4d_kernel)

    li      s7, 0                   # h = 0
s4d_h:
    li      t0, 64
    bge     s7, t0, s4d_h_done

    # dt = exp(log_dt[h])
    slli    t0, s7, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    call    expf
    fmv.s   fs0, fa0                # fs0 = dt

    # Phase 1: precompute step values and C_tilde for each state n
    li      s9, 0                   # n = 0
s4d_pre:
    li      t0, 32
    bge     s9, t0, s4d_pre_done

    # Ar = -exp(log_A_real[h*32+n])
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s1, t0
    flw     fa0, 0(t0)
    call    expf
    fneg.s  fs1, fa0                # fs1 = Ar

    # Ai = A_imag[h*32+n]
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s2, t0
    flw     fs2, 0(t0)              # fs2 = Ai

    # step_mag=exp(Ar*dt), step_cos=cos(Ai*dt), step_sin=sin(Ai*dt)
    fmul.s  fs3, fs1, fs0           # dtAr
    fmul.s  fs4, fs2, fs0           # dtAi
    fmv.s   fa0, fs3
    call    expf
    fmv.s   fs5, fa0                # step_mag
    fmv.s   fa0, fs4
    call    cosf
    fmv.s   fs6, fa0                # step_cos
    fmv.s   fa0, fs4
    call    sinf
    fmv.s   fs7, fa0                # step_sin

    # A_bar_r = step_mag*step_cos, A_bar_i = step_mag*step_sin
    fmul.s  fa1, fs5, fs6
    fmul.s  fa2, fs5, fs7

    # C_re[h*32+n] and C_im[h*32+n] (separate arrays)
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t1, s3, t0
    flw     fa3, 0(t1)              # Cr
    add     t1, s4, t0
    flw     fa4, 0(t1)              # Ci

    # C_tilde = C*(A_bar-1)/A
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fa1, ft0           # em1r
    fmv.s   ft2, fa2                # em1i
    fmul.s  ft3, fa3, ft1
    fmul.s  ft4, fa4, ft2
    fsub.s  ft3, ft3, ft4           # num_r
    fmul.s  ft4, fa3, ft2
    fmul.s  ft5, fa4, ft1
    fadd.s  ft4, ft4, ft5           # num_i
    fmul.s  ft5, fs1, fs1
    fmul.s  ft6, fs2, fs2
    fadd.s  ft5, ft5, ft6           # Amag2
    fmul.s  ft6, ft3, fs1
    fmul.s  ft7, ft4, fs2
    fadd.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft5           # Ct_r
    fmul.s  ft7, ft4, fs1
    fmul.s  fa5, ft3, fs2
    fsub.s  ft7, ft7, fa5
    fdiv.s  ft7, ft7, ft5           # Ct_i

    # Store precomputed values
    slli    t0, s9, 2
    lui     t1, %hi(s4d_smag)
    add     t1, t1, t0
    fsw     fs5, %lo(s4d_smag)(t1)
    lui     t1, %hi(s4d_scos)
    add     t1, t1, t0
    fsw     fs6, %lo(s4d_scos)(t1)
    lui     t1, %hi(s4d_ssin)
    add     t1, t1, t0
    fsw     fs7, %lo(s4d_ssin)(t1)
    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    fsw     ft6, %lo(s4d_Ct_r)(t1)
    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    fsw     ft7, %lo(s4d_Ct_i)(t1)

    # Init cur_r=1, cur_i=0
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
    j       s4d_pre
s4d_pre_done:

    # Phase 2: build kernel via recurrence (no transcendental calls in loop)
    li      s9, 0                   # t = 0
s4d_kern:
    li      t0, 4096
    bge     s9, t0, s4d_kern_done
    fmv.w.x fa5, zero               # kval = 0
    li      s10, 0                  # n = 0
s4d_kern_n:
    li      t0, 32
    bge     s10, t0, s4d_kern_n_done
    slli    t0, s10, 2
    lui     t1, %hi(s4d_Ct_r)
    add     t1, t1, t0
    flw     fs3, %lo(s4d_Ct_r)(t1)
    lui     t1, %hi(s4d_Ct_i)
    add     t1, t1, t0
    flw     fs4, %lo(s4d_Ct_i)(t1)
    lui     t1, %hi(s4d_cur_r)
    add     t1, t1, t0
    flw     fs5, %lo(s4d_cur_r)(t1)
    lui     t1, %hi(s4d_cur_i)
    add     t1, t1, t0
    flw     fs6, %lo(s4d_cur_i)(t1)
    fmul.s  ft0, fs3, fs5
    fmul.s  ft1, fs4, fs6
    fsub.s  ft0, ft0, ft1
    lui     t1, %hi(math_two)
    flw     ft1, %lo(math_two)(t1)
    fmadd.s fa5, ft1, ft0, fa5      # kval += 2*Re(Ct*cur)
    lui     t1, %hi(s4d_smag)
    add     t1, t1, t0
    flw     ft2, %lo(s4d_smag)(t1)
    lui     t1, %hi(s4d_scos)
    add     t1, t1, t0
    flw     ft3, %lo(s4d_scos)(t1)
    lui     t1, %hi(s4d_ssin)
    add     t1, t1, t0
    flw     ft4, %lo(s4d_ssin)(t1)
    fmul.s  ft3, ft2, ft3           # step_r = mag*cos
    fmul.s  ft4, ft2, ft4           # step_i = mag*sin
    fmul.s  ft5, fs5, ft3
    fmul.s  ft6, fs6, ft4
    fsub.s  ft5, ft5, ft6           # new_r
    fmul.s  ft6, fs5, ft4
    fmul.s  ft7, fs6, ft3
    fadd.s  ft6, ft6, ft7           # new_i
    lui     t1, %hi(s4d_cur_r)
    add     t1, t1, t0
    fsw     ft5, %lo(s4d_cur_r)(t1)
    lui     t1, %hi(s4d_cur_i)
    add     t1, t1, t0
    fsw     ft6, %lo(s4d_cur_i)(t1)
    addi    s10, s10, 1
    j       s4d_kern_n
s4d_kern_n_done:
    slli    t0, s9, 2
    add     t0, s8, t0
    fsw     fa5, 0(t0)
    addi    s9, s9, 1
    j       s4d_kern
s4d_kern_done:

    # Phase 3: causal convolution y[t][h] = D*u[t][h] + sum K[j]*u[t-j][h]
    slli    t0, s7, 2
    add     t0, s5, t0
    flw     fs1, 0(t0)              # D[h]
    li      s9, 0                   # t = 0
s4d_conv:
    li      t0, 4096
    bge     s9, t0, s4d_conv_done
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s6, t0
    flw     fa0, 0(t0)
    fmul.s  fa5, fs1, fa0
    li      s10, 0                  # j = 0
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
    add     t0, s6, t0
    flw     fa2, 0(t0)              # u[t-j][h]
    fmadd.s fa5, fa1, fa2, fa5
    addi    s10, s10, 1
    j       s4d_conv_j
s4d_conv_j_done:
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s11, t0
    fsw     fa5, 0(t0)
    addi    s9, s9, 1
    j       s4d_conv
s4d_conv_done:
    addi    s7, s7, 1
    j       s4d_h
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
    lw      s11, 48(sp)
    addi    sp, sp, 52
    ret

.section .bss
.align 2
s4d_smag:   .space 128
s4d_scos:   .space 128
s4d_ssin:   .space 128
s4d_cur_r:  .space 128
s4d_cur_i:  .space 128
s4d_Ct_r:   .space 128
s4d_Ct_i:   .space 128
s4d_kernel: .space 16384
