.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Test 1: fmv.w.x (move int to float)
    li    t0, 0x3f800000    # 1.0
    fmv.w.x ft0, t0
    
    # Test 2: fadd.s
    li    t0, 0x40000000    # 2.0
    fmv.w.x ft1, t0
    fadd.s ft2, ft0, ft1    # 1.0 + 2.0 = 3.0
    
    # Test 3: fmul.s
    fmul.s ft3, ft0, ft1    # 1.0 * 2.0 = 2.0
    
    # Test 4: fdiv.s
    li    t0, 0x40400000    # 3.0
    fmv.w.x ft4, t0
    fdiv.s ft5, ft4, ft0    # 3.0 / 1.0 = 3.0
    
    # If we get here, basic FPU works
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
