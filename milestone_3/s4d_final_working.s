# ============================================================
# s4d_final_working.s - Production S4D layer (matches working test)
# ============================================================
.section .text
.global _start
.global s4d_layer

_start:
    la   sp, _stack_start
    
    # Test with d_model=64, seq_len=2
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

# ============================================================
# S4D Layer - Function version
# ============================================================
s4d_layer:
    # Save registers
    addi  sp, sp, -80
    sw    ra, 76(sp)
    sw    s0, 72(sp)
    sw    s1, 68(sp)
    sw    s2, 64(sp)
    sw    s3, 60(sp)
    sw    s4, 56(sp)
    sw    s5, 52(sp)
    sw    s6, 48(sp)
    sw    s7, 44(sp)
    sw    s8, 40(sp)
    sw    s9, 36(sp)
    sw    s10,32(sp)
    sw    s11,28(sp)
    
    fsw   ft0, 24(sp)
    fsw   ft1, 20(sp)
    fsw   ft2, 16(sp)
    fsw   ft3, 12(sp)
    fsw   ft4, 8(sp)
    fsw   ft5, 4(sp)

    mv    s0,  a0          # input
    mv    s1,  a1          # A_real
    mv    s2,  a2          # A_imag
    mv    s3,  a3          # B
    mv    s4,  a4          # C_real
    mv    s5,  a5          # C_imag
    mv    s6,  a6          # output
    mv    s7,  a7          # seq_len

    # Use static buffers (must be in .bss)
    la    s10, h_real
    la    s11, h_imag
    
    # Zero out state
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

    li    s8, 0            # t
t_loop:
    bge   s8, s7, t_done

    li    s9, 0            # d
d_loop:
    li    t0, 64
    bge   s9, t0, d_done

    slli  t1, s9, 2
    
    # Load A[d]
    add   t2, s1, t1
    flw   ft0, 0(t2)
    add   t2, s2, t1
    flw   ft1, 0(t2)

    # Compute A_disc = exp(A)
    fmv.s fa0, ft0
    call  exp_f
    fmv.s ft4, fa0
    
    fmv.s fa0, ft1
    call  cos_f
    fmv.s ft5, fa0
    
    fmv.s fa0, ft1
    call  sin_f
    fmv.s ft6, fa0
    
    fmul.s ft7, ft4, ft5
    fmul.s ft8, ft4, ft6

    # Load h[d]
    add   t2, s10, t1
    flw   ft0, 0(t2)
    add   t2, s11, t1
    flw   ft1, 0(t2)

    # h = A_disc * h
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    fmv.s ft2, fa0
    fmv.s ft3, fa1

    # Load u = input[t][d]
    li    t3, 64
    mul   t4, s8, t3
    add   t4, t4, s9
    slli  t4, t4, 2
    add   t4, s0, t4
    flw   ft4, 0(t4)

    # Load B[d]
    add   t2, s3, t1
    flw   ft5, 0(t2)

    # h = A*h + B*u
    fmul.s ft6, ft4, ft5
    fadd.s ft2, ft2, ft6

    # Store h
    add   t2, s10, t1
    fsw   ft2, 0(t2)
    add   t2, s11, t1
    fsw   ft3, 0(t2)

    # Load C[d]
    add   t2, s4, t1
    flw   ft4, 0(t2)
    add   t2, s5, t1
    flw   ft5, 0(t2)
    
    # y = real(C * h)
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
    flw   ft0, 24(sp)
    flw   ft1, 20(sp)
    flw   ft2, 16(sp)
    flw   ft3, 12(sp)
    flw   ft4, 8(sp)
    flw   ft5, 4(sp)
    
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
    lw    s10,32(sp)
    lw    s11,28(sp)
    addi  sp, sp, 80
    ret

# ============================================================
# Math Functions (must be before data section)
# ============================================================
exp_f:
    li    t0, 0x40000000
    fmv.w.x ft0, t0
    fmin.s fa0, fa0, ft0
    li    t0, 0xc0000000
    fmv.w.x ft0, t0
    fmax.s fa0, fa0, ft0
    li    t0, 0x3f800000
    fmv.w.x ft0, t0
    fadd.s ft0, ft0, fa0
    fmul.s ft1, fa0, fa0
    li    t0, 0x3f000000
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2
    fadd.s ft0, ft0, ft1
    fmul.s ft1, ft1, fa0
    li    t0, 0x3e2aaaab
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2
    fadd.s fa0, ft0, ft1
    ret

sin_f:
    fmv.s ft0, fa0
    fmul.s ft1, fa0, fa0
    fmul.s ft2, ft1, fa0
    li    t0, 0x3e2aaaab
    fmv.w.x ft3, t0
    fmul.s ft2, ft2, ft3
    fsub.s fa0, ft0, ft2
    ret

cos_f:
    li    t0, 0x3f800000
    fmv.w.x ft0, t0
    fmul.s ft1, fa0, fa0
    li    t0, 0x3f000000
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2
    fsub.s fa0, ft0, ft1
    ret

complex_mul:
    fmul.s ft0, fa0, fa2
    fmul.s ft1, fa1, fa3
    fsub.s ft2, ft0, ft1
    fmul.s ft0, fa0, fa3
    fmul.s ft1, fa1, fa2
    fadd.s ft3, ft0, ft1
    fmv.s fa0, ft2
    fmv.s fa1, ft3
    ret

# ============================================================
# Test Data
# ============================================================
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

# ============================================================
# BSS Section
# ============================================================
.section .bss
.align 4
h_real:
    .space 256
h_imag:
    .space 256
_stack_start:
    .space 8192
_stack_end:
