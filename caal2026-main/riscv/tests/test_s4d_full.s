.section .text
.global _start
_start:
    lui     sp, %hi(s4_stack)
    addi    sp, sp, %lo(s4_stack)
    lui     a0, %hi(ref_log_dt)
    addi    a0, a0, %lo(ref_log_dt)
    lui     a1, %hi(ref_log_ar)
    addi    a1, a1, %lo(ref_log_ar)
    lui     a2, %hi(ref_aim)
    addi    a2, a2, %lo(ref_aim)
    lui     a3, %hi(ref_c)
    addi    a3, a3, %lo(ref_c)
    lui     a4, %hi(ref_d)
    addi    a4, a4, %lo(ref_d)
    lui     a5, %hi(ref_linear)
    addi    a5, a5, %lo(ref_linear)
    lui     a6, %hi(s4_result)
    addi    a6, a6, %lo(s4_result)
    call    s4d_layer
    lui     t0, %hi(s4_result)
    addi    t0, t0, %lo(s4_result)
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
ref_log_dt: .incbin "../model_weights.bin", 16896, 256
ref_log_ar: .incbin "../model_weights.bin", 17152, 8192
ref_aim:    .incbin "../model_weights.bin", 25344, 8192
ref_c:      .incbin "../model_weights.bin", 33536, 16384
ref_d:      .incbin "../model_weights.bin", 49920, 256
ref_linear: .incbin "../test_data/sample_09_linear.bin"
.section .bss
.align 2
s4_result:  .space 1048576
.space 8192
s4_stack:
