# ============================================================
# Minimal S4D test - single timestep, single dimension
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Test data
    la   t0, test_A_real
    flw  ft0, 0(t0)      # A_real = -0.1
    la   t0, test_A_imag
    flw  ft1, 0(t0)      # A_imag = 0.5
    
    # Test exp_f
    fmv.s fa0, ft0
    call  exp_f
    fmv.s ft4, fa0        # exp(A_real)
    
    # Test cos_f
    fmv.s fa0, ft1
    call  cos_f
    fmv.s ft5, fa0        # cos(A_imag)
    
    # Test sin_f
    fmv.s fa0, ft1
    call  sin_f
    fmv.s ft6, fa0        # sin(A_imag)
    
    # Compute A_disc
    fmul.s ft7, ft4, ft5  # A_disc_real
    fmul.s ft8, ft4, ft6  # A_disc_imag
    
    # Load h (initially 0)
    li    t0, 0
    fmv.w.x ft0, t0       # h_real = 0
    fmv.w.x ft1, t0       # h_imag = 0
    
    # Test complex_mul: A_disc * h (should be 0)
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft7
    fmv.s fa3, ft8
    call  complex_mul
    
    # Load u = 1.0
    li    t0, 0x3f800000
    fmv.w.x ft4, t0
    
    # Load B = 1.0
    fmv.w.x ft5, t0
    
    # B*u
    fmul.s ft6, ft4, ft5
    
    # h_new = A*h + B*u (since A*h=0, h_new = B*u)
    fadd.s ft2, fa0, ft6   # h_new_real
    fmv.s ft3, fa1         # h_new_imag
    
    # Load C = 1.0 + 0i
    fmv.w.x ft4, t0        # C_real = 1.0
    fmv.w.x ft5, t0        # C_imag = 0.0
    
    # y = real(C * h_new)
    fmv.s fa0, ft4
    fmv.s fa1, ft5
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    # Store result
    la   t0, test_output
    fsw  fa0, 0(t0)
    
    # Signal success
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

# Use the same math functions
.global exp_f
.global sin_f
.global cos_f
.global complex_mul

# Include math_simple.s functions
.section .text
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
test_A_real:    .float -0.1
test_A_imag:    .float 0.5
test_output:    .word 0

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
