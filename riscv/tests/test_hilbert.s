.section .text
.global _start
_start:
    lui     sp, %hi(hl_stack)
    addi    sp, sp, %lo(hl_stack)
    lui     a0, %hi(ref_indices)
    addi    a0, a0, %lo(ref_indices)
    lui     a1, %hi(ref_img)
    addi    a1, a1, %lo(ref_img)
    lui     a2, %hi(hl_result)
    addi    a2, a2, %lo(hl_result)
    call    hilbert_scan
    lui     t0, %hi(hl_result)
    addi    t0, t0, %lo(hl_result)
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
ref_indices:
.incbin "../model_weights.bin", 0, 16384
ref_img:
.incbin "../test_data/sample_09_input.bin"
hl_result: .space 16384
.section .bss
.align 2
.space 8192
hl_stack:
