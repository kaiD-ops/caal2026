.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Test exp_f(0) should be 1.0
    fmv.w.x fa0, zero
    call exp_f
    
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)
    j     .

.section .bss
_stack_start:
    .space 8192
_stack_end:
