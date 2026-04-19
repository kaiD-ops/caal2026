# ============================================================
# Test loops without function calls
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Simple nested loops
    li    s0, 0           # outer loop counter
outer:
    li    t0, 2
    bge   s0, t0, outer_done
    
    li    s1, 0           # inner loop counter
inner:
    li    t0, 4
    bge   s1, t0, inner_done
    
    # Do some simple FP operations
    li    t0, 0x3f800000
    fmv.w.x ft0, t0
    fmv.w.x ft1, t0
    fadd.s ft2, ft0, ft1
    
    addi  s1, s1, 1
    j     inner
inner_done:
    addi  s0, s0, 1
    j     outer
outer_done:
    
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

.section .bss
_stack_start: .space 8192
_stack_end:
