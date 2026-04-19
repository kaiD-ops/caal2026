# ============================================================
# Test s4d_layer with state buffer memory operations
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Initialize state buffers
    la    s0, h_real
    la    s1, h_imag
    sw    zero, 0(s0)
    sw    zero, 4(s0)
    sw    zero, 8(s0)
    sw    zero, 12(s0)
    sw    zero, 0(s1)
    sw    zero, 4(s1)
    sw    zero, 8(s1)
    sw    zero, 12(s1)
    
    li    s2, 0           # d counter
    li    s3, 4           # d_model=4
    
test_loop:
    bge   s2, s3, done
    
    # Load A_real[d], A_imag[d]
    la    t0, test_A_real
    slli  t1, s2, 2
    add   t2, t0, t1
    flw   ft0, 0(t2)
    
    la    t0, test_A_imag
    add   t2, t0, t1
    flw   ft1, 0(t2)
    
    # Compute A_disc
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
    
    # Load h from state buffer
    add   t2, s0, t1
    flw   ft0, 0(t2)      # h_real[d]
    add   t2, s1, t1
    flw   ft1, 0(t2)      # h_imag[d]
    
    # complex_mul A_disc * h
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    fmv.s ft2, fa0
    fmv.s ft3, fa1
    
    # Load u = 1.0, B = 1.0
    li    t0, 0x3f800000
    fmv.w.x ft4, t0
    fmv.w.x ft5, t0
    
    fmul.s ft6, ft4, ft5
    fadd.s ft2, ft2, ft6
    
    # Store h back to state buffer
    add   t2, s0, t1
    fsw   ft2, 0(t2)
    add   t2, s1, t1
    fsw   ft3, 0(t2)
    
    # Load C
    la    t0, test_C_real
    add   t2, t0, t1
    flw   ft4, 0(t2)
    la    t0, test_C_imag
    add   t2, t0, t1
    flw   ft5, 0(t2)
    
    # complex_mul C * h_new
    fmv.s fa0, ft4
    fmv.s fa1, ft5
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    addi  s2, s2, 1
    j     test_loop
done:
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

# Math functions (same as before)
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

.section .data
.align 4
test_A_real: .float -0.1, -0.2, -0.3, -0.4
test_A_imag: .float 0.5, 0.6, 0.7, 0.8
test_C_real: .float 1.0, 1.0, 1.0, 1.0
test_C_imag: .float 0.0, 0.0, 0.0, 0.0

.section .bss
.align 4
h_real: .space 16
h_imag: .space 16
_stack_start: .space 8192
_stack_end:
