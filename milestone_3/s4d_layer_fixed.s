# ============================================================
# s4d_layer_fixed.s - Fixed stack allocation
# ============================================================
.section .text
.global s4d_layer
.global _start

_start:
    la   sp, _stack_start
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
    # Save registers
    addi  sp, sp, -96
    sw    ra, 92(sp)
    sw    s0, 88(sp)
    sw    s1, 84(sp)
    sw    s2, 80(sp)
    sw    s3, 76(sp)
    sw    s4, 72(sp)
    sw    s5, 68(sp)
    sw    s6, 64(sp)
    sw    s7, 60(sp)
    sw    s8, 56(sp)
    sw    s9, 52(sp)
    sw    s10,48(sp)
    sw    s11,44(sp)
    
    fsw   ft0, 40(sp)
    fsw   ft1, 36(sp)
    fsw   ft2, 32(sp)
    fsw   ft3, 28(sp)
    fsw   ft4, 24(sp)
    fsw   ft5, 20(sp)
    fsw   ft6, 16(sp)
    fsw   ft7, 12(sp)

    mv    s0,  a0
    mv    s1,  a1
    mv    s2,  a2
    mv    s3,  a3
    mv    s4,  a4
    mv    s5,  a5
    mv    s6,  a6
    mv    s7,  a7

    # Use static buffers instead of stack allocation
    la    s10, s4d_h_real
    la    s11, s4d_h_imag
    
    # Zero out state (64 floats each)
    li    t0, 0
zero_loop:
    li    t1, 64
    bge   t0, t1, zero_done
    slli  t2, t0, 2
    add   t3, s10, t2
    sw    zero, 0(t3)
    add   t3, s11, t2
    sw    zero, 0(t3)
    addi  t0, t0, 1
    j     zero_loop
zero_done:

    li    s8, 0           # t = 0
t_loop:
    bge   s8, s7, t_done

    li    s9, 0           # d = 0
d_loop:
    li    t0, 64
    bge   s9, t0, d_done

    # Load A[d]
    slli  t1, s9, 2
    add   t2, s1, t1
    flw   ft0, 0(t2)      # A_real[d]
    add   t2, s2, t1
    flw   ft1, 0(t2)      # A_imag[d]

    # Compute A_disc = exp(A)
    fmv.s fa0, ft0
    call  exp_f
    fmv.s ft4, fa0        # exp(A_real)
    
    fmv.s fa0, ft1
    call  cos_f
    fmv.s ft5, fa0        # cos(A_imag)
    
    fmv.s fa0, ft1
    call  sin_f
    fmv.s ft6, fa0        # sin(A_imag)
    
    fmul.s ft7, ft4, ft5  # A_disc_real
    fmul.s ft8, ft4, ft6  # A_disc_imag

    # Load h[d]
    add   t2, s10, t1
    flw   ft0, 0(t2)      # h_real[d]
    add   t2, s11, t1
    flw   ft1, 0(t2)      # h_imag[d]

    # h_new = A_disc * h
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    fmv.s ft2, fa0        # A*h real
    fmv.s ft3, fa1        # A*h imag

    # Load u = input[t][d]
    li    t3, 64
    mul   t4, s8, t3
    add   t4, t4, s9
    slli  t4, t4, 2
    add   t4, s0, t4
    flw   ft4, 0(t4)      # u

    # Load B[d]
    add   t2, s3, t1
    flw   ft5, 0(t2)      # B[d]

    # h_new = A*h + B*u
    fmul.s ft6, ft4, ft5
    fadd.s ft2, ft2, ft6
    # h_new_imag unchanged

    # Store updated h
    add   t2, s10, t1
    fsw   ft2, 0(t2)
    add   t2, s11, t1
    fsw   ft3, 0(t2)

    # y[t][d] = real(C[d] * h_new)
    add   t2, s4, t1
    flw   ft4, 0(t2)      # C_real[d]
    add   t2, s5, t1
    flw   ft5, 0(t2)      # C_imag[d]
    
    fmv.s fa0, ft4
    fmv.s fa1, ft5
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    # Store output[t][d]
    li    t3, 64
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
    flw   ft0, 40(sp)
    flw   ft1, 36(sp)
    flw   ft2, 32(sp)
    flw   ft3, 28(sp)
    flw   ft4, 24(sp)
    flw   ft5, 20(sp)
    flw   ft6, 16(sp)
    flw   ft7, 12(sp)
    
    lw    ra, 92(sp)
    lw    s0, 88(sp)
    lw    s1, 84(sp)
    lw    s2, 80(sp)
    lw    s3, 76(sp)
    lw    s4, 72(sp)
    lw    s5, 68(sp)
    lw    s6, 64(sp)
    lw    s7, 60(sp)
    lw    s8, 56(sp)
    lw    s9, 52(sp)
    lw    s10,48(sp)
    lw    s11,44(sp)
    addi  sp, sp, 96
    ret

# ─── Static state buffers ───────────────────────────────
.section .bss
.align 4
s4d_h_real:
    .space 256
s4d_h_imag:
    .space 256

# ─── Test data ───────────────────────────────────────────
.section .data
.align 4
test_input:
    .rept 128
        .float 1.0
    .endr
test_A_real:
    .rept 64
        .float -0.1
    .endr
test_A_imag:
    .rept 64
        .float 0.5
    .endr
test_B:
    .rept 64
        .float 1.0
    .endr
test_C_real:
    .rept 64
        .float 1.0
    .endr
test_C_imag:
    .rept 64
        .float 0.0
    .endr
test_output:
    .space 512

_stack_start:
    .space 8192
_stack_end:
