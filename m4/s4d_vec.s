# =============================================================================
# s4d_layer.s - S4D (Structured State Space Diagonal) Layer
#
# Implements causal convolution using the S4D formulation:
#   K[t] = 2*Re(sum_n C_tilde_n * A_bar_n^t)
#   y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j]*u[t-j][h]
#
# Optimization: kernel built via recurrence (NO repeated expf/cosf/sinf):
#   cur[t+1] = cur[t] * step   (complex multiply)
#   Reduces transcendental calls from O(L*N) to O(N) per channel.
#   ~7.9 billion dynamic instructions for SEQ_LEN=4096, D_MODEL=64.
#
# Weight layout for C_mat: interleaved complex, shape [D_MODEL, N_STATES, 2]
#   C_mat[h*32+n] = (re, im) as two consecutive float32 words.
#
# void s4d_layer(float* log_dt, float* log_A_real, float* A_imag,
#                float* C_mat,  float* D_vec,       float* in, float* out)
# a0=log_dt, a1=log_A_real, a2=A_imag, a3=C_mat, a4=D_vec, a5=in, a6=out
#
# MSE target: < 1e-7, MAE < 1e-4
# =============================================================================

.section .text
.global s4d_layer

s4d_layer:
    addi    sp, sp, -64
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
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
    mv      s3, a3      # C_mat (interleaved complex)
    mv      s4, a4      # D_vec
    mv      s5, a5      # in
    mv      s6, a6      # out

    lui     s8, %hi(s4d_kernel)
    addi    s8, s8, %lo(s4d_kernel)

    li      s7, 0                   # h = 0

s4d_h_loop:
    li      t0, 64
    bge     s7, t0, s4d_h_done

    # -----------------------------------------------------------------------
    # dt = exp(log_dt[h])
    # -----------------------------------------------------------------------
    slli    t0, s7, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    call    expf
    fmv.s   fs0, fa0                # fs0 = dt

    # -----------------------------------------------------------------------
    # Phase 1: Precompute per-state (n=0..31):
    #   Ar = -exp(log_A_real[h,n])
    #   Ai = A_imag[h,n]
    #   step_mag = exp(Ar*dt), step_cos = cos(Ai*dt), step_sin = sin(Ai*dt)
    #   A_bar = step_mag*(step_cos + j*step_sin)
    #   C_tilde = C*(A_bar-1)/A   (C from interleaved array)
    #   cur_r[n]=1, cur_i[n]=0   (initial running state)
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

    # step values
    fmul.s  fs3, fs1, fs0           # fs3 = dtAr = Ar*dt
    fmul.s  fs4, fs2, fs0           # fs4 = dtAi = Ai*dt

    fmv.s   fa0, fs3
    call    expf
    fmv.s   fs5, fa0                # fs5 = step_mag = exp(dtAr)

    fmv.s   fa0, fs4
    call    cosf
    fmv.s   fs6, fa0                # fs6 = step_cos = cos(dtAi)

    fmv.s   fa0, fs4
    call    sinf
    fmv.s   fs7, fa0                # fs7 = step_sin = sin(dtAi)

    # Normalize (cos,sin) to unit magnitude so |step| = step_mag EXACTLY.
    # The poly cosf/sinf give cos^2+sin^2 = 1 + eps; over 4096 recurrence
    # steps that eps compounds as (1+eps/2)^4096 and corrupts the kernel at
    # large t (correct at t=0, wrong by t=4095). Forcing unit magnitude here
    # makes the step a pure rotation * step_mag, so the kernel decays cleanly.
    fmul.s  ft0, fs6, fs6
    fmul.s  ft1, fs7, fs7
    fadd.s  ft0, ft0, ft1           # cos^2 + sin^2
    fsqrt.s ft0, ft0                # norm
    fdiv.s  fs6, fs6, ft0           # cos / norm
    fdiv.s  fs7, fs7, ft0           # sin / norm

    # A_bar_r = step_mag * step_cos,  A_bar_i = step_mag * step_sin
    # (cos,sin) now unit-magnitude so the recurrence step magnitude stays
    # = step_mag < 1 (kernel decays, no blowup).
    fmul.s  fa1, fs5, fs6           # fa1 = A_bar_r
    fmul.s  fa2, fs5, fs7           # fa2 = A_bar_i

    # Load C[h*32+n] = (Cr, Ci) interleaved
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    li      t1, 2
    mul     t0, t0, t1
    slli    t0, t0, 2               # byte offset = (h*32+n)*2*4
    add     t0, s3, t0
    flw     fa3, 0(t0)              # fa3 = Cr
    flw     fa4, 4(t0)              # fa4 = Ci

    # C_tilde = C*(A_bar-1)/A
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fa1, ft0           # em1r = A_bar_r - 1
    fmv.s   ft2, fa2                # em1i = A_bar_i

    # num = C * (A_bar-1)  (complex multiply)
    fmul.s  ft3, fa3, ft1
    fmul.s  ft4, fa4, ft2
    fsub.s  ft3, ft3, ft4           # num_r = Cr*em1r - Ci*em1i
    fmul.s  ft4, fa3, ft2
    fmul.s  ft5, fa4, ft1
    fadd.s  ft4, ft4, ft5           # num_i = Cr*em1i + Ci*em1r

    # Amag2 = Ar^2 + Ai^2
    fmul.s  ft5, fs1, fs1
    fmul.s  ft6, fs2, fs2
    fadd.s  ft5, ft5, ft6

    # Ct_r = Re(num/A) = (num_r*Ar + num_i*Ai)/Amag2
    fmul.s  ft6, ft3, fs1
    fmul.s  ft7, ft4, fs2
    fadd.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft5           # Ct_r

    # Ct_i = Im(num/A) = (num_i*Ar - num_r*Ai)/Amag2
    fmul.s  ft7, ft4, fs1
    fmul.s  fa5, ft3, fs2
    fsub.s  ft7, ft7, fa5
    fdiv.s  ft7, ft7, ft5           # Ct_i

    # Store precomputed per-state values
    slli    t0, s9, 2               # n*4

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

    # Init cur: cur_r[n]=1.0, cur_i[n]=0.0
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
    # Phase 2: Build kernel K[t] = 2*Re(sum_n Ct_n * cur_n[t])
    # cur_n[t+1] = cur_n[t] * step_n  (complex multiply, no transcendentals)
    # -----------------------------------------------------------------------
    li      s9, 0                   # t = 0

s4d_kern_t:
    li      t0, 4096
    bge     s9, t0, s4d_kern_done

    fmv.w.x fa5, zero               # kval = 0.0

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

    # Re(Ct * cur) = Ct_r*cur_r - Ct_i*cur_i
    fmul.s  ft0, fs3, fs5
    fmul.s  ft1, fs4, fs6
    fsub.s  ft0, ft0, ft1

    # kval += 2 * Re(Ct * cur)
    lui     t1, %hi(math_two)
    flw     ft1, %lo(math_two)(t1)
    fmadd.s fa5, ft1, ft0, fa5

    # Advance cur: cur[t+1] = cur[t] * step
    # step_r = step_mag * step_cos,  step_i = step_mag * step_sin
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

    # new_r = cur_r*step_r - cur_i*step_i
    fmul.s  ft5, fs5, ft3
    fmul.s  ft6, fs6, ft4
    fsub.s  ft5, ft5, ft6

    # new_i = cur_r*step_i + cur_i*step_r
    fmul.s  ft6, fs5, ft4
    fmul.s  ft7, fs6, ft3
    fadd.s  ft6, ft6, ft7

    # Flush-to-zero: if |new_r|+|new_i| < 1e-20, force cur=0.
    # smag<1 so cur decays monotonically; once tiny it stays tiny.
    # Prevents denormal values (< 1.2e-38) that the VeeR FPU turns into NaN,
    # which would then propagate K=NaN into the Phase-3 convolution.
    fabs.s  ft8, ft5
    fabs.s  ft9, ft6
    fadd.s  ft8, ft8, ft9
    lui     t1, %hi(s4d_ftz)
    flw     ft9, %lo(s4d_ftz)(t1)
    flt.s   t1, ft8, ft9            # mag < 1e-20 ?
    beqz    t1, s4d_kern_nostore
    fmv.w.x ft5, zero
    fmv.w.x ft6, zero
s4d_kern_nostore:

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
    fsw     fa5, 0(t0)              # K[t] = kval

    addi    s9, s9, 1
    j       s4d_kern_t

s4d_kern_done:

    # Guard: replace NaN/inf K values with 0.0 to prevent 0*NaN=NaN in Phase 3.
    # Triggered when Ct overflows (Amag2 near-zero) and cur later underflows to 0.
    li      s9, 0
s4d_kfix_loop:
    li      t0, 4096
    bge     s9, t0, s4d_kfix_done
    slli    t0, s9, 2
    add     t0, s8, t0
    flw     fa0, 0(t0)
    feq.s   t1, fa0, fa0              # t1=0 if NaN, t1=1 if finite or inf
    bnez    t1, s4d_kfix_chkinf
    fmv.w.x fa0, zero                 # replace NaN with 0
    fsw     fa0, 0(t0)
    j       s4d_kfix_next
s4d_kfix_chkinf:
    # check for inf: fabs then compare against 3.4e38
    fabs.s  ft0, fa0
    lui     t1, %hi(s4d_finf_limit)
    flw     ft1, %lo(s4d_finf_limit)(t1)
    flt.s   t1, ft1, ft0              # t1=1 if |K| > limit (inf)
    beqz    t1, s4d_kfix_next
    fmv.w.x fa0, zero                 # replace inf with 0
    fsw     fa0, 0(t0)
s4d_kfix_next:
    addi    s9, s9, 1
    j       s4d_kfix_loop
s4d_kfix_done:

    # -----------------------------------------------------------------------
    # Phase 3: Causal convolution  (RVV, UNIT-stride)
    # y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j]*u[t-j][h]
    #
    # Build reversed channel column once:  urev[k] = u[(4095-k)][h].
    # Then u[t-j] = urev[4095-t+j], so the causal dot becomes
    #   acc = sum_{j=0}^{t} K[j] * urev[(4095-t)+j]
    # Both operands now stride FORWARD by 1 -> plain vle32 (no slow vlse32
    # strided gather, which whisper emulates lane-by-lane).
    # -----------------------------------------------------------------------
    slli    t0, s7, 2
    add     t0, s4, t0
    flw     fs1, 0(t0)              # D[h]

    # ---- build urev[0..4095] = u[4095..0][h]  (one pass per channel) -------
    lui     s11, %hi(s4d_urev)
    addi    s11, s11, %lo(s4d_urev) # s11 = &urev[0]     (dst, +4)
    li      t1, 262080              # 4095*64
    add     t1, t1, s7
    slli    t1, t1, 2
    add     t1, s5, t1              # t1 = &u[4095][h]   (src, -256)
    mv      t2, s11
    li      t3, 4096
s4d_urev_build:
    beqz    t3, s4d_urev_done
    flw     ft0, 0(t1)
    fsw     ft0, 0(t2)
    addi    t1, t1, -256
    addi    t2, t2, 4
    addi    t3, t3, -1
    j       s4d_urev_build
s4d_urev_done:

    li      s9, 0                   # t = 0

s4d_conv_t:
    li      t0, 4096
    bge     s9, t0, s4d_conv_done

    # acc = D[h] * u[t][h]
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa0, 0(t0)
    fmul.s  fa5, fs1, fa0

    # vector dot:  K[0..t] . urev[(4095-t)..4095]
    mv      t3, s8                  # &K[0]            (unit +4)
    li      t1, 4095
    sub     t1, t1, s9             # 4095 - t
    slli    t1, t1, 2
    add     t2, s11, t1            # &urev[4095-t]     (unit +4)
    mv      t4, s9
    addi    t4, t4, 1              # count = t + 1

    li      t5, 64
    vsetvli t6, t5, e32, m8, ta, ma
    vmv.v.i v0, 0                  # zero ALL 64 acc lanes

s4d_conv_j:
    beqz    t4, s4d_conv_j_done
    vsetvli t5, t4, e32, m8, tu, ma  # tu: keep tail partials intact
    vle32.v  v8,  (t3)            # K[j .. j+vl-1]
    vle32.v  v16, (t2)            # urev[4095-t+j .. ]
    vfmacc.vv v0, v8, v16
    slli    t6, t5, 2
    add     t3, t3, t6            # K   += vl*4
    add     t2, t2, t6            # urev += vl*4
    sub     t4, t4, t5
    j       s4d_conv_j

s4d_conv_j_done:
    li      t5, 64
    vsetvli t6, t5, e32, m8, ta, ma
    fmv.w.x ft0, zero
    vfmv.s.f v24, ft0
    vfredusum.vs v24, v0, v24
    vfmv.f.s ft1, v24
    fadd.s  fa5, fa5, ft1         # acc += sum of all lanes

    # store y[t][h]
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
    lw      ra,  0(sp)
    lw      s0,  4(sp)
    lw      s1,  8(sp)
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
    addi    sp, sp, 64
    ret

# =============================================================================
# BSS: per-state scratch buffers (N_STATES=32, 4 bytes each = 128 bytes)
.section .data
.align 2
s4d_finf_limit: .float 3.4e38   # largest finite float32
s4d_ftz:        .float 1.0e-20  # flush-to-zero threshold (above denormal range)

# =============================================================================
.section .bss
.align 2
s4d_smag:   .space 128      # step_mag[32]
s4d_scos:   .space 128      # step_cos[32]
s4d_ssin:   .space 128      # step_sin[32]
s4d_cur_r:  .space 128      # running real  part[32]
s4d_cur_i:  .space 128      # running imag  part[32]
s4d_Ct_r:   .space 128      # C_tilde real[32]
s4d_Ct_i:   .space 128      # C_tilde imag[32]
s4d_kernel: .space 16384    # K[4096] = one channel's kernel
s4d_urev:   .space 16384    # reversed channel column u[4095..0][h]
