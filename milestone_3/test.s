# ============================================================
# test_fpu.s - Check if FPU is supported
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Try to use floating-point instruction
    li    t0, 0x3f800000    # 1.0 in IEEE754
    fmv.w.x ft0, t0         # Move to FP register
    
    # If we get here, FPU might work
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    
    j     .

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
