.section .text
.global _start
_start:
    lui     sp, %hi(sm_stack)
    addi    sp, sp, %lo(sm_stack)
    # Copy logits to result buffer (softmax is in-place)
    lui     a0, %hi(ref_logits)
    addi    a0, a0, %lo(ref_logits)
    lui     a1, %hi(sm_result)
    addi    a1, a1, %lo(sm_result)
    li      t0, 4
sm_copy:
    beqz    t0, sm_copy_done
    lw      t1, 0(a0)
    sw      t1, 0(a1)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    t0, t0, -1
    j       sm_copy
sm_copy_done:
    lui     a0, %hi(sm_result)
    addi    a0, a0, %lo(sm_result)
    li      a1, 4
    call    softmax_inplace
    # Load results into registers for log inspection
    lui     t0, %hi(sm_result)
    addi    t0, t0, %lo(sm_result)
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
ref_logits:
.incbin "../test_data/sample_09_logits.bin"
sm_result: .space 16
.section .bss
.align 2
.space 8192
sm_stack:
