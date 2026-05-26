.section .text
.global _start
_start:
    lui     sp, %hi(stack_top)
    addi    sp, sp, %lo(stack_top)
    li      t0, 10000000
loop:
    addi    t0, t0, -1
    bnez    t0, loop
_finish:
    lui     x3, 0xd0580
    addi    x5, x0, 0xff
    sb      x5, 0(x3)
    beq     x0, x0, _finish
.rept 100
    nop
.endr
.section .bss
.align 2
.space 8192
stack_top:
