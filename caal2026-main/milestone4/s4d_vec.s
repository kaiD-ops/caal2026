.section .text
.global s4d_layer_vectorized

s4d_layer_vectorized:
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
    sw      s10,44(sp)
    fsw     fs0,48(sp)
    fsw     fs1,52(sp)
    fsw     fs2,56(sp)
    fsw     fs3,60(sp)

    mv      s0, a0      # log_dt
    mv      s1, a1      # log_A_real
    mv      s2, a2      # A_imag
    mv      s3, a3      # C_mat (interleaved re/im pairs)
    mv      s4, a4      # D_vec
    mv      s5, a5      # in
    mv      s6, a6      # out

    # Allocate kernel buffer (4096 floats)
    li      a0, 16384
    call    malloc
    mv      s7, a0      # s7 = kernel buffer

    li      s8, 0       # h = 0

s4d_h_loop:
    li      t0, 64
    bge     s8, t0, s4d_h_done

    # dt = exp(log_dt[h])
    slli    t0, s8, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    call    expf
    fmv.s   fs0, fa0

    la      t0, s4d_Ct_r
    la      t1, s4d_Ct_i
    la      t2, s4d_step_r
    la      t3, s4d_step_i
    la      t4, s4d_cur_r
    la      t5, s4d_cur_i
    
    li      s9, 0       # n = 0
s4d_pre_loop:
    li      t0, 32
    bge     s9, t0, s4d_pre_done
    
    # Compute Ar = -exp(log_A_real[h*32+n])
    li      t0, 32
    mul     t0, s8, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s1, t0
    flw     fa0, 0(t0)
    call    expf
    fneg.s  fs1, fa0
    
    # Ai = A_imag[h*32+n]
    li      t0, 32
    mul     t0, s8, t0
    add     t0, t0, s9
    slli    t0, t0, 2
    add     t0, s2, t0
    flw     fs2, 0(t0)
    
    # Compute dtAr, dtAi with NaN guards
    fmul.s  fs3, fs1, fs0    # dtAr
    fmul.s  fs4, fs2, fs0    # dtAi
    
    fmv.s   fa0, fs3
    call    expf
    fmv.s   fs5, fa0         # step_mag
    
    fmv.s   fa0, fs4
    call    cosf
    fmv.s   fs6, fa0         # step_cos
    
    fmv.s   fa0, fs4
    call    sinf
    fmv.s   fs7, fa0         # step_sin
    
    # A_bar = step_mag * (step_cos + j*step_sin)
    fmul.s  fa1, fs5, fs6    # A_bar_r
    fmul.s  fa2, fs5, fs7    # A_bar_i
    
    # Load C_re, C_im from interleaved array
    li      t0, 32
    mul     t0, s8, t0
    add     t0, t0, s9
    slli    t0, t0, 3        # *8 (2 floats per n)
    add     t0, s3, t0
    flw     fa3, 0(t0)       # C_re
    flw     fa4, 4(t0)       # C_im
    
    # Compute C_tilde = C * (A_bar - 1) / A
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  ft1, fa1, ft0    # em1r
    fmv.s   ft2, fa2         # em1i
    
    fmul.s  ft3, fa3, ft1
    fmul.s  ft4, fa4, ft2
    fsub.s  ft3, ft3, ft4    # num_r
    
    fmul.s  ft4, fa3, ft2
    fmul.s  ft5, fa4, ft1
    fadd.s  ft4, ft4, ft5    # num_i
    
    fmul.s  ft5, fs1, fs1
    fmul.s  ft6, fs2, fs2
    fadd.s  ft5, ft5, ft6    # denom = |A|^2
    
    fmv.w.x ft0, zero
    feq.s   t0, ft5, ft0
    bnez    t0, s4d_ct_zero
    
    fmul.s  ft6, ft3, fs1
    fmul.s  ft7, ft4, fs2
    fadd.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft5    # Ct_r
    
    fmul.s  ft7, ft4, fs1
    fmul.s  fa5, ft3, fs2
    fsub.s  ft7, ft7, fa5
    fdiv.s  ft7, ft7, ft5    # Ct_i
    j       s4d_ct_done
    
s4d_ct_zero:
    fmv.w.x ft6, zero
    fmv.w.x ft7, zero
    
s4d_ct_done:
    # Store precomputed values
    slli    t0, s9, 2
    la      t1, s4d_Ct_r
    add     t1, t1, t0
    fsw     ft6, 0(t1)
    
    la      t1, s4d_Ct_i
    add     t1, t1, t0
    fsw     ft7, 0(t1)
    
    fmul.s  ft0, fs5, fs6    # step_r
    fmul.s  ft1, fs5, fs7    # step_i
    
    la      t1, s4d_step_r
    add     t1, t1, t0
    fsw     ft0, 0(t1)
    
    la      t1, s4d_step_i
    add     t1, t1, t0
    fsw     ft1, 0(t1)
    
    # Initialize cur_r = 1, cur_i = 0
    la      t1, s4d_cur_r
    add     t1, t1, t0
    lui     t2, %hi(math_one)
    flw     fa0, %lo(math_one)(t2)
    fsw     fa0, 0(t1)
    
    la      t1, s4d_cur_i
    add     t1, t1, t0
    fmv.w.x fa0, zero
    fsw     fa0, 0(t1)
    
    addi    s9, s9, 1
    j       s4d_pre_loop

s4d_pre_done:

    vsetvli t0, x0, e32, m4, ta, ma   # LMUL=4 for 32 elements (8 regs)
    
    # Load all Ct_r, Ct_i, step_r, step_i, cur_r, cur_i into vector registers
    la      t0, s4d_Ct_r
    vle32.v v0, (t0)          # v0 = Ct_r[0..31]
    la      t0, s4d_Ct_i
    vle32.v v1, (t0)          # v1 = Ct_i[0..31]
    la      t0, s4d_step_r
    vle32.v v2, (t0)          # v2 = step_r[0..31]
    la      t0, s4d_step_i
    vle32.v v3, (t0)          # v3 = step_i[0..31]
    la      t0, s4d_cur_r
    vle32.v v4, (t0)          # v4 = cur_r[0..31]
    la      t0, s4d_cur_i
    vle32.v v5, (t0)          # v5 = cur_i[0..31]
    
    # Constant 2.0 for scaling
    lui     t0, %hi(math_two)
    flw     ft0, %lo(math_two)(t0)
    vfmv.v.f v8, ft0          # v8 = [2.0, 2.0, ...]
    
    li      s9, 0             # t = 0

s4d_kern_t_vec:
    li      t0, 4096
    bge     s9, t0, s4d_kern_done_vec
    
    # Compute kval = 2 * sum_n(Ct_r*cur_r - Ct_i*cur_i) over all 32 states
    vfmul.vv v10, v0, v4      # v10 = Ct_r * cur_r
    vfmul.vv v11, v1, v5      # v11 = Ct_i * cur_i
    vfsub.vv v12, v10, v11    # v12 = Ct_r*cur_r - Ct_i*cur_i
    vfmul.vv v12, v12, v8     # v12 = 2 * (product)
    
    # Horizontal sum across vector register group (32 elements)
    vfredusum.vs v13, v12, v9  # v13[0] = sum of all 32 values
    vmv.x.s t1, v13            # t1 = kval (scalar)
    
    # Store kernel value
    slli    t0, s9, 2
    add     t0, s7, t0
    fsw     ft0, 0(t0)         # K[t] = kval
    
    # Update cur[n] = cur[n] * step[n] for all n in parallel
    # Complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    vfmul.vv v10, v4, v2      # v10 = cur_r * step_r
    vfmul.vv v11, v5, v3      # v11 = cur_i * step_i
    vfsub.vv v12, v10, v11    # new_cur_r = cur_r*step_r - cur_i*step_i
    
    vfmul.vv v10, v4, v3      # v10 = cur_r * step_i
    vfmul.vv v11, v5, v2      # v11 = cur_i * step_r
    vfadd.vv v13, v10, v11    # new_cur_i = cur_r*step_i + cur_i*step_r
    
    vmv.v.v v4, v12           # Update cur_r
    vmv.v.v v5, v13           # Update cur_i
    
    addi    s9, s9, 1
    j       s4d_kern_t_vec

s4d_kern_done_vec:

    li      s9, 0             # t = 0
    flw     fs1, 0(s4)        # D (assuming single D for all channels for simplicity)

s4d_conv_t_vec:
    li      t0, 4096
    bge     s9, t0, s4d_conv_done_vec
    
    # Load u[t][h]
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s8
    slli    t0, t0, 2
    add     t0, s5, t0
    flw     fa0, 0(t0)
    fmul.s  fa5, fs1, fa0     # acc = D * u[t][h]
    
    # Vectorized j-loop using strided loads
    # Set up for j = 0..t
    li      t1, 0             # j = 0
    mv      t2, s7            # K_ptr = &K[0]
    mv      t3, t0            # u_ptr = &u[t][h]
    
    # Configure for strided load: stride = -256 bytes (-64 floats)
    li      t4, -256
    vsetvli t5, x0, e32, m1, ta, ma
    
s4d_conv_j_vec:
    bgt     t1, s9, s4d_conv_j_done_vec
    
    # Load K[j] (contiguous)
    vle32.v v0, (t2)
    
    # Load u[t-j][h] with stride -256
    vlse32.v v1, (t3), t4
    
    # Multiply-accumulate: acc += K[j] * u[t-j][h]
    vfmacc.vv v2, v0, v1
    
    addi    t1, t1, 1
    addi    t2, t2, 4         # K_ptr += 4
    # u_ptr automatically advances by stride
    j       s4d_conv_j_vec
    
s4d_conv_j_done_vec:
    # Extract scalar result (assuming single channel for now)
    vmv.x.s t0, v2
    fmv.w.x fa5, t0
    
    # Store y[t][h]
    li      t0, 64
    mul     t0, s9, t0
    add     t0, t0, s8
    slli    t0, t0, 2
    add     t0, s6, t0
    fsw     fa5, 0(t0)
    
    addi    s9, s9, 1
    j       s4d_conv_t_vec

s4d_conv_done_vec:
    addi    s8, s8, 1
    j       s4d_h_loop

s4d_h_done:
    # Free kernel buffer
    mv      a0, s7
    call    free
    
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
    lw      s10,44(sp)
    flw     fs0,48(sp)
    flw     fs1,52(sp)
    flw     fs2,56(sp)
    flw     fs3,60(sp)
    addi    sp, sp, 64
    ret

.section .rodata
.align 2
math_one:  .float 1.0
math_two:  .float 2.0

.section .bss
.align 4
s4d_Ct_r:   .space 128
s4d_Ct_i:   .space 128
s4d_step_r: .space 128
s4d_step_i: .space 128
s4d_cur_r:  .space 128
s4d_cur_i:  .space 128