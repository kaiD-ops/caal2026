.section .text
.global _start
_start:
    lui     sp, %hi(gl_stack)
    addi    sp, sp, %lo(gl_stack)
    # Copy s4d1 output to work buffer
    lui     a0, %hi(ref_s4d1)
    addi    a0, a0, %lo(ref_s4d1)
    lui     a1, %hi(gl_result)
    addi    a1, a1, %lo(gl_result)
    li      t0, 262144
gl_copy:
    beqz    t0, gl_copy_done
    lw      t1, 0(a0)
    sw      t1, 0(a1)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    t0, t0, -1
    j       gl_copy
gl_copy_done:
    lui     a0, %hi(gl_result)
    addi    a0, a0, %lo(gl_result)
    li      a1, 262144
    call    gelu_inplace
    # Load first 4 results
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
.section .bss
.align 2
gl_result: .space 1048576
.space 8192
gl_stack:
