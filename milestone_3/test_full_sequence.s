# ============================================================
# Test the exact s4d_layer inner loop sequence
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    li    s0, 0           # d_model counter
    li    s1, 4           # d_model=4 for test
    
test_loop:
    bge   s0, s1, done
    
    # Load test A_real and A_imag
    la    t0, test_A_real
    slli  t1, s0, 2
    add   t2, t0, t1
    flw   ft0, 0(t2)      # A_real
    
    la    t0, test_A_imag
    add   t2, t0, t1
    flw   ft1, 0(t2)      # A_imag
    
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
    
    # Load h (zero for test)
    fmv.w.x ft0, zero
    fmv.w.x ft1, zero
    
    # complex_mul
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    
    # Load u and B
    li    t0, 0x3f800000
    fmv.w.x ft4, t0       # u = 1.0
    fmv.w.x ft5, t0       # B = 1.0
    
    fmul.s ft6, ft4, ft5
    fadd.s ft2, fa0, ft6
    
    # Load C
    fmv.w.x ft4, t0       # C_real = 1.0
    fmv.w.x ft5, zero     # C_imag = 0.0
    
    fmv.s fa0, ft4
    fmv.s fa1, ft5
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    addi  s0, s0, 1
    j     test_loop
done:
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

# Math functions
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

.section .bss
_stack_start: .space 8192
_stack_end:
