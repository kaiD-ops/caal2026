# =============================================================================
# s4d_vec.s  –  RVV-vectorized S4D Layer (the primary hotspot)
#
# Mathematical background (identical to M3):
#   K[t]    = 2 * Re( sum_n  C_tilde_n * A_bar_n^t )
#   y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j] * u[t-j][h]
#
# Vectorization strategy
# ──────────────────────
# Phase 1  – Per-channel precompute (scalar, 64 channels × 32 states)
#            Unchanged from M3; the transcendental functions (expf/cosf/sinf)
#            are called O(N) = 96 times total regardless of SEQ_LEN.
#
# Phase 2  – Kernel build (VECTORIZED over N=32 states)
#            For each timestep t:
#              kval  = 2 * Re(Ct * cur)  summed over n=0..31
#              cur  *= step              (complex multiply)
#            Both the dot-product (Ct · cur) and the state update (cur *= step)
#            operate over 32 floats → vectorize with e32 / m1 (fits in one
#            register group for most VLEN≥128 implementations).
#
# Phase 3  – Causal convolution (VECTORIZED over input/kernel strips)
#            Inner j-loop: for each output sample t, multiply K[0..t] by
#            the reversed input strip u[t..0][h] and accumulate.
#            We vectorize the j-loop: load K[j..j+vl-1] and the
#            corresponding reversed u-strip, fmul then freadusum.
#
# Signature (unchanged from M3):
#   void s4d_layer(float* log_dt, float* log_A_real, float* A_imag,
#                  float* C_mat,  float* D_vec, float* in, float* out)
#   a0=log_dt  a1=log_A_real  a2=A_imag  a3=C_mat  a4=D_vec
#   a5=in      a6=out
#
# C_mat layout: interleaved [h][n] real then imag → C[h*32+n].re, C[h*32+n].im
# (matches M3 layers.s where flw fa3,0(t0) / flw fa4,4(t0))
# =============================================================================

.section .text
.global s4d_layer

s4d_layer:
    addi    sp, sp, -52
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

    mv      s0, a0          # log_dt
    mv      s1, a1          # log_A_real
    mv      s2, a2          # A_imag
    mv      s3, a3          # C_mat (interleaved re/im)
    mv      s4, a4          # D_vec
    mv      s5, a5          # in
    mv      s6, a6          # out

    lui     s8, %hi(sv_kernel)
    addi    s8, s8, %lo(sv_kernel)   # s8 = &kernel[0]

    li      s7, 0                    # h = 0

sv_h_loop:
    li      t0, 64
    bge     s7, t0, sv_h_done

    # ── Phase 1: scalar precompute for this channel h ────────────────────────
    # dt = exp(log_dt[h])
    slli    t0, s7, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    call    expf
    fmv.s   fs0, fa0                # fs0 = dt

    li      s9, 0                   # n = 0
sv_pre_loop:
    li      t0, 32
    bge     s9, t0, sv_pre_done

    # Ar = -exp(log_A_real[h*32+n])
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s1, t0
    flw     fa0, 0(t0)
    call    expf
    fneg.s  fs1, fa0                # fs1 = Ar (negative real)

    # Ai = A_imag[h*32+n]
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s2, t0
    flw     fs2, 0(t0)              # fs2 = Ai

    # dtAr = Ar*dt,  dtAi = Ai*dt
    fmul.s  fs3, fs1, fs0
    fmul.s  fs4, fs2, fs0

    # step_mag = exp(dtAr)
    fmv.s   fa0, fs3
    call    expf
    fmv.s   fs5, fa0

    # step_cos = cos(dtAi)
    fmv.s   fa0, fs4
    call    cosf
    fmv.s   fs6, fa0

    # step_sin = sin(dtAi)
    fmv.s   fa0, fs4
    call    sinf
    fmv.s   fs7, fa0

    # A_bar_r = step_mag*step_cos,  A_bar_i = step_mag*step_sin
    fmul.s  fa1, fs5, fs6
    fmul.s  fa2, fs5, fs7

    # C_re, C_im: interleaved layout [h*32+n]*2
    li      t0, 32
    mul     t0, s7, t0
    add     t0, t0, s9
    li      t1, 2
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     t0, s3, t0
    flw     fa3, 0(t0)              # Cr
    flw     fa4, 4(t0)              # Ci

    # C_tilde = C*(A_bar-1)/A  (scalar complex division, same as M3)
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fa1, ft0           # em1r = A_bar_r - 1
    fmv.s   ft2, fa2                # em1i = A_bar_i

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

    # Store into per-state scratchpad arrays
    slli    t0, s9, 2               # n*4
    lui     t1, %hi(sv_smag)
    add     t1, t1, t0
    fsw     fs5, %lo(sv_smag)(t1)

    lui     t1, %hi(sv_scos)
    add     t1, t1, t0
    fsw     fs6, %lo(sv_scos)(t1)

    lui     t1, %hi(sv_ssin)
    add     t1, t1, t0
    fsw     fs7, %lo(sv_ssin)(t1)

    lui     t1, %hi(sv_Ct_r)
    add     t1, t1, t0
    fsw     ft6, %lo(sv_Ct_r)(t1)

    lui     t1, %hi(sv_Ct_i)
    add     t1, t1, t0
    fsw     ft7, %lo(sv_Ct_i)(t1)

    # Precompute step_r = step_mag*step_cos, step_i = step_mag*step_sin
    # and store for vectorized Phase 2
    fmul.s  ft6, fs5, fs6           # step_r
    fmul.s  ft7, fs5, fs7           # step_i
    lui     t1, %hi(sv_step_r)
    add     t1, t1, t0
    fsw     ft6, %lo(sv_step_r)(t1)
    lui     t1, %hi(sv_step_i)
    add     t1, t1, t0
    fsw     ft7, %lo(sv_step_i)(t1)

    # Init cur_r[n]=1, cur_i[n]=0
    lui     t1, %hi(sv_cur_r)
    add     t1, t1, t0
    lui     t2, %hi(math_one)
    flw     fa0, %lo(math_one)(t2)
    fsw     fa0, %lo(sv_cur_r)(t1)

    lui     t1, %hi(sv_cur_i)
    add     t1, t1, t0
    fmv.w.x fa0, zero
    fsw     fa0, %lo(sv_cur_i)(t1)

    addi    s9, s9, 1
    j       sv_pre_loop
sv_pre_done:

    # ── Phase 2: kernel build  (vectorized over N=32 states) ─────────────────
    # Load the 32-element arrays into fixed vector registers once
    # We use m1 groups (32 floats fit in 1 register for VLEN≥128)
    li      t0, 32
    vsetvli zero, t0, e32, m1, ta, ma  # set vl=32 permanently for this phase

    lui     t0, %hi(sv_Ct_r)
    addi    t0, t0, %lo(sv_Ct_r)
    vle32.v  v0, (t0)               # v0 = Ct_r[0..31]

    lui     t0, %hi(sv_Ct_i)
    addi    t0, t0, %lo(sv_Ct_i)
    vle32.v  v1, (t0)               # v1 = Ct_i[0..31]

    lui     t0, %hi(sv_step_r)
    addi    t0, t0, %lo(sv_step_r)
    vle32.v  v2, (t0)               # v2 = step_r[0..31]

    lui     t0, %hi(sv_step_i)
    addi    t0, t0, %lo(sv_step_i)
    vle32.v  v3, (t0)               # v3 = step_i[0..31]

    lui     t0, %hi(sv_cur_r)
    addi    t0, t0, %lo(sv_cur_r)
    vle32.v  v4, (t0)               # v4 = cur_r (=1.0 initially)

    lui     t0, %hi(sv_cur_i)
    addi    t0, t0, %lo(sv_cur_i)
    vle32.v  v5, (t0)               # v5 = cur_i (=0.0 initially)

    # Reduction identity vector: v6 = 0.0
    fmv.w.x  ft0, zero
    vfmv.v.f v6, ft0                # v6 = broadcast 0.0

    li      s9, 0                   # t = 0
sv_kern_loop:
    li      t0, 4096
    bge     s9, t0, sv_kern_done

    # kval = 2 * Re(Ct · cur) = 2 * (Ct_r*cur_r - Ct_i*cur_i)
    # element-wise: tmp = Ct_r*cur_r - Ct_i*cur_i  (v8 = tmp)
    vfmul.vv v8, v0, v4             # v8 = Ct_r * cur_r
    vfmul.vv v9, v1, v5             # v9 = Ct_i * cur_i
    vfsub.vv v8, v8, v9             # v8 = Re(Ct*cur)

    # Sum v8 → scalar using vfredusum; accumulator init = 0
    vfredusum.vs v7, v8, v6         # v7[0] = sum(v8)
    vfmv.f.s ft1, v7                # ft1 = sum(Re(Ct*cur))

    # kval = 2 * ft1
    lui      t0, %hi(math_two)
    flw      ft2, %lo(math_two)(t0)
    fmul.s   fa5, ft2, ft1

    # Store kernel value
    slli    t0, s9, 2
    add     t0, s8, t0
    fsw     fa5, 0(t0)

    # Advance state: cur *= step  (complex multiply, vectorized)
    # new_r = cur_r*step_r - cur_i*step_i
    # new_i = cur_r*step_i + cur_i*step_r
    vfmul.vv v10, v4, v2            # v10 = cur_r * step_r
    vfmul.vv v11, v5, v3            # v11 = cur_i * step_i
    vfsub.vv v12, v10, v11          # v12 = new_r

    vfmul.vv v10, v4, v3            # v10 = cur_r * step_i
    vfmul.vv v11, v5, v2            # v11 = cur_i * step_r
    vfadd.vv v13, v10, v11          # v13 = new_i

    vmv.v.v  v4, v12                # cur_r = new_r
    vmv.v.v  v5, v13                # cur_i = new_i

    addi    s9, s9, 1
    j       sv_kern_loop
sv_kern_done:

    # ── Phase 3: causal convolution  (vectorized over j-strip) ───────────────
    # y[t][h] = D[h]*u[t][h] + sum_{j=0}^{t} K[j] * u[t-j][h]
    #
    # Vectorize the inner j-loop:
    #   for each timestep t:
    #     accumulate strips of K[j..j+vl-1] ⊙ u[t-j..t-j-vl+1][h]
    # NOTE: u[t-j][h] requires stride -1 in time (i.e. stride=-64 floats
    # between consecutive elements), so we use vlse32.v with a negative stride.

    slli    t0, s7, 2
    add     t0, s4, t0
    flw     fs1, 0(t0)              # D[h]

    # Stride for u column accesses: -64*4 = -256 bytes (going backwards in t)
    li      s10, -256               # byte stride = -D_MODEL * sizeof(float)

    li      s9, 0                   # t = 0
sv_conv_t:
    li      t0, 4096
    bge     s9, t0, sv_conv_done

    # acc = D[h] * u[t][h]
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa0, 0(t0)
    fmul.s  fa5, fs1, fa0           # acc = D*u[t][h]

    # Vectorized j-loop
    # K pointer: s8 (base), offset j*4
    # u pointer for u[t-j][h]: start = &u[t][h], stride = -256 bytes
    mv      t1, s9                  # j remaining = t+1 (j goes 0..t)
    addi    t1, t1, 1               # t+1 iterations

    # Pointer to K[0]
    mv      t2, s8                  # K pointer, advances by vl*4

    # Pointer to u[t][h]  (start; stride will go backwards)
    li      t3, 64
    mul     t3, s9, t3
    add     t3, t3, s7
    slli    t3, t3, 2
    add     t3, s5, t3              # t3 = &u[t][h]

    # Zero reduction init
    fmv.w.x  ft0, zero
    vfmv.v.f v6, ft0

sv_conv_j:
    beqz    t1, sv_conv_j_done

    vsetvli t4, t1, e32, m4, ta, ma    # t4 = min(t1, VLMAX)

    # Load K[j..j+vl-1]
    vle32.v  v0, (t2)

    # Load u[t-j..t-j-vl+1][h] with stride -256 bytes
    vlse32.v v4, (t3), s10

    # Multiply and accumulate
    vfmul.vv v8, v0, v4             # v8 = K[j]*u[t-j]
    vfredusum.vs v7, v8, v6         # v7[0] = sum(v8)
    vfmv.f.s ft1, v7
    fadd.s  fa5, fa5, ft1

    # Advance K pointer forward by vl
    slli    t5, t4, 2
    add     t2, t2, t5              # K += vl*4

    # Advance u pointer backward by vl (stride = -256, so t3 -= vl*256)
    # But vlse32.v with stride s10 already consumed vl elements;
    # the *base* pointer for next strip must move by vl*stride bytes
    li      t5, 256
    mul     t5, t4, t5
    sub     t3, t3, t5              # u base moves back vl*256 bytes

    sub     t1, t1, t4
    j       sv_conv_j

sv_conv_j_done:
    # Store y[t][h]
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s7
    slli    t0, t0, 2
    add     t0, s6, t0
    fsw     fa5, 0(t0)

    addi    s9, s9, 1
    j       sv_conv_t
sv_conv_done:

    addi    s7, s7, 1
    j       sv_h_loop

sv_h_done:
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
    addi    sp, sp, 52
    ret

# =============================================================================
# BSS scratchpad for Phase 1 precomputed values (per-channel, N=32 states)
# =============================================================================
.section .bss
.align 4
sv_smag:    .space 128          # step_mag[32]
sv_scos:    .space 128          # step_cos[32]
sv_ssin:    .space 128          # step_sin[32]
sv_step_r:  .space 128          # step_r = mag*cos [32]
sv_step_i:  .space 128          # step_i = mag*sin [32]
sv_cur_r:   .space 128          # running real part [32]
sv_cur_i:   .space 128          # running imag part [32]
sv_Ct_r:    .space 128          # C_tilde real [32]
sv_Ct_i:    .space 128          # C_tilde imag [32]
sv_kernel:  .space 16384        # K[4096]
