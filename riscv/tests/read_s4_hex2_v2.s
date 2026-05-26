.section .text
.global _start

_start:
    lui     sp, %hi(rs_stack)
    addi    sp, sp, %lo(rs_stack)

    # Run s4d_layer
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

    # Setup MMIO and result pointers
    lui     s1, 0xd0580
    addi    s1, s1, 4          # s1 = 0xd0580004 consoleio
    lui     s0, %hi(s4_result)
    addi    s0, s0, %lo(s4_result)

    # Write 4 floats as hex - INLINED (no function call to avoid ra issues)
    # Float 0
    lw      s2, 0(s0)
    li      s3, 28
hex0:
    bltz    s3, hex0_done
    srl     t0, s2, s3
    andi    t0, t0, 0xF
    li      t1, 10
    blt     t0, t1, hex0_digit
    addi    t0, t0, 87
    j       hex0_store
hex0_digit:
    addi    t0, t0, 48
hex0_store:
    sb      t0, 0(s1)
    addi    s3, s3, -4
    j       hex0
hex0_done:
    li      t0, 32
    sb      t0, 0(s1)

    # Float 1
    lw      s2, 4(s0)
    li      s3, 28
hex1:
    bltz    s3, hex1_done
    srl     t0, s2, s3
    andi    t0, t0, 0xF
    li      t1, 10
    blt     t0, t1, hex1_digit
    addi    t0, t0, 87
    j       hex1_store
hex1_digit:
    addi    t0, t0, 48
hex1_store:
    sb      t0, 0(s1)
    addi    s3, s3, -4
    j       hex1
hex1_done:
    li      t0, 32
    sb      t0, 0(s1)

    # Float 2
    lw      s2, 8(s0)
    li      s3, 28
hex2:
    bltz    s3, hex2_done
    srl     t0, s2, s3
    andi    t0, t0, 0xF
    li      t1, 10
    blt     t0, t1, hex2_digit
    addi    t0, t0, 87
    j       hex2_store
hex2_digit:
    addi    t0, t0, 48
hex2_store:
    sb      t0, 0(s1)
    addi    s3, s3, -4
    j       hex2
hex2_done:
    li      t0, 32
    sb      t0, 0(s1)

    # Float 3
    lw      s2, 12(s0)
    li      s3, 28
hex3:
    bltz    s3, hex3_done
    srl     t0, s2, s3
    andi    t0, t0, 0xF
    li      t1, 10
    blt     t0, t1, hex3_digit
    addi    t0, t0, 87
    j       hex3_store
hex3_digit:
    addi    t0, t0, 48
hex3_store:
    sb      t0, 0(s1)
    addi    s3, s3, -4
    j       hex3
hex3_done:
    li      t0, 10          # newline
    sb      t0, 0(s1)

    # Load into float regs for log
    flw     fa0, 0(s0)
    flw     fa1, 4(s0)
    flw     fa2, 8(s0)
    flw     fa3, 12(s0)

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
rs_stack:
