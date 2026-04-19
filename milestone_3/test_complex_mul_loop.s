# ============================================================
# Test complex_mul in a loop
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Test data
    li    t0, 0x3f800000    # 1.0
    fmv.w.x ft0, t0
    fmv.w.x ft1, t0
    fmv.w.x ft2, t0
    fmv.w.x ft3, t0
    
    li    s0, 0             # loop counter
loop:
    li    t0, 10
    bge   s0, t0, done
    
    # Call complex_mul
    fmv.s fa0, ft0
    fmv.s fa1, ft1
    fmv.s fa2, ft2
    fmv.s fa3, ft3
    call  complex_mul
    
    addi  s0, s0, 1
    j     loop
done:
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

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

.section .bss
_stack_start: .space 8192
_stack_end:
