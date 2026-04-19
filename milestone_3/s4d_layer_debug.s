# ============================================================
# s4d_layer_debug.s - Simplified version for debugging
# seq_len=2, d_model=2 only
# ============================================================
.section .text
.global s4d_layer
.global _start

_start:
    la   sp, _stack_start
    
    # Test with small parameters (seq_len=2, d_model=2)
    la   a0, test_input
    la   a1, test_A_real
    la   a2, test_A_imag
    la   a3, test_B
    la   a4, test_C_real
    la   a5, test_C_imag
    la   a6, test_output
    li   a7, 2
    call s4d_layer
    
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

s4d_layer:
    addi  sp, sp, -80
    sw    ra, 76(sp)
    sw    s0, 72(sp)   # input ptr
    sw    s1, 68(sp)   # A_real ptr
    sw    s2, 64(sp)   # A_imag ptr
    sw    s3, 60(sp)   # B ptr
    sw    s4, 56(sp)   # C_real ptr
    sw    s5, 52(sp)   # C_imag ptr
    sw    s6, 48(sp)   # output ptr
    sw    s7, 44(sp)   # seq_len
    sw    s8, 40(sp)   # t (timestep)
    sw    s9, 36(sp)   # d (dimension)
    
    # Save FP temporaries
    fsw   ft0, 32(sp)
    fsw   ft1, 28(sp)
    fsw   ft2, 24(sp)
    fsw   ft3, 20(sp)
    fsw   ft4, 16(sp)
    fsw   ft5, 12(sp)
    fsw   ft6, 8(sp)
    fsw   ft7, 4(sp)

    mv    s0,  a0
    mv    s1,  a1
    mv    s2,  a2
    mv    s3,  a3
    mv    s4,  a4
    mv    s5,  a5
    mv    s6,  a6
    mv    s7,  a7

    # State buffers (local, small for debug)
    addi  sp, sp, -32   # allocate h_real and h_imag (8 floats total for d_model=2)
    mv    s10, sp       # h_real pointer
    addi  s11, s10, 16  # h_imag pointer
    
    # Zero out state
    sw    zero, 0(s10)
    sw    zero, 4(s10)
    sw    zero, 0(s11)
    sw    zero, 4(s11)

    li    s8, 0           # t = 0
t_loop:
    bge   s8, s7, t_done

    li    s9, 0           # d = 0
d_loop:
    li    t0, 2           # d_model=2 for debug
    bge   s9, t0, d_done

    # ── load A[d] ──
    slli  t1, s9, 2
    add   t2, s1, t1
    flw   ft0, 0(t2)      # A_real[d]
    add   t2, s2, t1
    flw   ft1, 0(t2)      # A_imag[d]

    # ── Compute A_disc = exp(A) ──
    # exp(a+ib) = exp(a) * (cos(b) + i*sin(b))
    fmv.s fa0, ft0
    call  exp_f
    fmv.s ft4, fa0        # exp(A_real)
    
    fmv.s fa0, ft1
    call  cos_f
    fmv.s ft5, fa0        # cos(A_imag)
    
    fmv.s fa0, ft1
    call  sin_f
    fmv.s ft6, fa0        # sin(A_imag)
    
    # A_disc_real = exp(a)*cos(b)
    fmul.s ft7, ft4, ft5
    # A_disc_imag = exp(a)*sin(b)
    fmul.s ft8, ft4, ft6

    # ── load h[d] ──
    slli  t1, s9, 2
    add   t2, s10, t1
    flw   ft0, 0(t2)      # h_real[d]
    add   t2, s11, t1
    flw   ft1, 0(t2)      # h_imag[d]

    # ── h_new = A_disc * h ──
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    fmv.s ft2, fa0        # A*h real
    fmv.s ft3, fa1        # A*h imag

    # ── load u = input[t][d] ──
    li    t3, 2           # d_model=2
    mul   t4, s8, t3
    add   t4, t4, s9
    slli  t4, t4, 2
    add   t4, s0, t4
    flw   ft4, 0(t4)      # u

    # ── load B[d] ──
    add   t2, s3, t1
    flw   ft5, 0(t2)      # B[d]

    # ── h_new = A*h + B*u ──
    fmul.s ft6, ft4, ft5  # B*u
    fadd.s ft2, ft2, ft6  # h_new_real
    # h_new_imag unchanged

    # ── store updated h ──
    add   t2, s10, t1
    fsw   ft2, 0(t2)
    add   t2, s11, t1
    fsw   ft3, 0(t2)

    # ── y[t][d] = real(C[d] * h_new) ──
    add   t2, s4, t1
    flw   ft4, 0(t2)      # C_real[d]
    add   t2, s5, t1
    flw   ft5, 0(t2)      # C_imag[d]
    
    fmv.s fa0, ft4
    fmv.s fa1, ft5
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    # store output[t][d]
    li    t3, 2
    mul   t4, s8, t3
    add   t4, t4, s9
    slli  t4, t4, 2
    add   t4, s6, t4
    fsw   fa0, 0(t4)

    addi  s9, s9, 1
    j     d_loop
d_done:
    addi  s8, s8, 1
    j     t_loop
t_done:
    # Restore and return
    flw   ft0, 32(sp)
    flw   ft1, 28(sp)
    flw   ft2, 24(sp)
    flw   ft3, 20(sp)
    flw   ft4, 16(sp)
    flw   ft5, 12(sp)
    flw   ft6, 8(sp)
    flw   ft7, 4(sp)
    
    addi  sp, sp, 32     # free state buffers
    lw    ra, 76(sp)
    lw    s0, 72(sp)
    lw    s1, 68(sp)
    lw    s2, 64(sp)
    lw    s3, 60(sp)
    lw    s4, 56(sp)
    lw    s5, 52(sp)
    lw    s6, 48(sp)
    lw    s7, 44(sp)
    lw    s8, 40(sp)
    lw    s9, 36(sp)
    addi  sp, sp, 80
    ret

# ─── Test data ───────────────────────────────────────────
.section .data
.align 4
test_input:
    .float 1.0, 2.0    # t=0
    .float 3.0, 4.0    # t=1
test_A_real:
    .float -0.1, -0.2
test_A_imag:
    .float 0.5, 0.6
test_B:
    .float 1.0, 1.0
test_C_real:
    .float 1.0, 1.0
test_C_imag:
    .float 0.0, 0.0
test_output:
    .space 32

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
