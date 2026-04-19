.section .text
.global _start
_start:
    lui     sp, %hi(tl_stack)
    addi    sp, sp, %lo(tl_stack)
    lui     a0, %hi(ref_gelu2)
    addi    a0, a0, %lo(ref_gelu2)
    lui     a1, %hi(tl_result)
    addi    a1, a1, %lo(tl_result)
    call    take_last_timestep
    # Load results into float registers so they appear in log
    lui     t0, %hi(tl_result)
    addi    t0, t0, %lo(tl_result)
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
ref_gelu2:
.incbin "../test_data/sample_09_gelu2.bin"
tl_result: .space 256
.section .bss
.align 2
.space 8192
tl_stack:
