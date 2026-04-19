.section .text
.global _start
_start:
    lui     sp, %hi(gl_stack)
    addi    sp, sp, %lo(gl_stack)

    # Copy just first 4 elements from s4d1 output
    lui     a0, %hi(ref_s4d1)
    addi    a0, a0, %lo(ref_s4d1)
    lui     a1, %hi(gl_result)
    addi    a1, a1, %lo(gl_result)
    lw      t0, 0(a0)
    sw      t0, 0(a1)
    lw      t0, 4(a0)
    sw      t0, 4(a1)
    lw      t0, 8(a0)
    sw      t0, 8(a1)
    lw      t0, 12(a0)
    sw      t0, 12(a1)

    lui     a0, %hi(gl_result)
    addi    a0, a0, %lo(gl_result)
    li      a1, 4
    call    gelu_inplace

    lui     t0, %hi(gl_result)
    addi    t0, t0, %lo(gl_result)
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

_finish:
    lui     t0, 0xd0580
    li      t1, 0xff
    sb      t1, 0(t0)
    beq     zero, zero, _finish
.rept 100
    nop
.endr

.section .data
.align 2
ref_s4d1:
.incbin "../test_data/sample_09_s4d1.bin"
gl_result: .space 16

.section .bss
.align 2
.space 8192
gl_stack:
