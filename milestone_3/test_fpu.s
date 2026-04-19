# ============================================================
# test_fpu.s - Check if floating-point instructions work
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Try a floating-point operation
    fmv.w.x ft0, zero        # Try to move zero to FP register
    
    # If we reach here, FPU might work
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
