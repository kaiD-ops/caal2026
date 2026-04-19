# ============================================================
# Test sin_f and cos_f in a loop
# ============================================================
.section .text
.global _start

_start:
    la   sp, _stack_start
    
    li    t0, 0x3f800000    # 1.0
    fmv.w.x fa0, t0
    
    li    s0, 0
loop:
    li    t0, 10
    bge   s0, t0, done
    
    call  sin_f
    call  cos_f
    
    addi  s0, s0, 1
    j     loop
done:
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

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

.section .bss
_stack_start: .space 8192
_stack_end:
