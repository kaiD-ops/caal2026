#define STDOUT 0xd0580000
.section .text
.global _start
_start:
    lui     sp, %hi(stack_top)
    addi    sp, sp, %lo(stack_top)

    # Test expf(1.0) -- expected 2.71828
    lui     t0, %hi(test_val_1)
    flw     fa0, %lo(test_val_1)(t0)
    call    expf
    fsw     fa0, 0(sp)    # store result

    # Test expf(0.0) -- expected 1.0
    fmv.w.x fa0, zero
    call    expf
    fsw     fa0, -4(sp)

    # Test cosf(0.0) -- expected 1.0
    fmv.w.x fa0, zero
    call    cosf
    fsw     fa0, -8(sp)

    # Test sinf(0.0) -- expected 0.0
    fmv.w.x fa0, zero
    call    sinf
    fsw     fa0, -12(sp)

_finish:
    lui     x3, 0xd0580
    addi    x5, x0, 0xff
    sb      x5, 0(x3)
    beq     x0, x0, _finish
.rept 100
    nop
.endr
.section .data
.align 2
test_val_1: .float 1.0
.section .bss
.align 2
.space 8192
stack_top:
