# read_s4_hex.s
# Runs s4d_layer then writes first 4 floats as ASCII hex to MMIO
# ASCII bytes (0x30-0x66) never trigger --tohost (0xff)
# --consoleoutfile captures the 32 ASCII hex chars + newline

.section .text
.global _start

# write_hex_word: writes t0 as 8 hex chars to MMIO at s1
write_hex_word:
    li      t2, 28          # start from bit 28 (most significant nibble)
whw_loop:
    bltz    t2, whw_done
    srl     t3, t0, t2      # shift right
    andi    t3, t3, 0xF     # get nibble
    li      t4, 10
    blt     t3, t4, whw_digit
    addi    t3, t3, 87      # 'a' - 10 = 87
    j       whw_store
whw_digit:
    addi    t3, t3, 48      # '0' = 48
whw_store:
    sb      t3, 0(s1)
    addi    t2, t2, -4
    j       whw_loop
whw_done:
    # write space separator
    li      t3, 32
    sb      t3, 0(s1)
    ret

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

    # Write first 4 floats as hex to MMIO
    lui     s1, 0xd0580     # MMIO address
    lui     s0, %hi(s4_result)
    addi    s0, s0, %lo(s4_result)

    lw      t0, 0(s0)
    call    write_hex_word
    lw      t0, 4(s0)
    call    write_hex_word
    lw      t0, 8(s0)
    call    write_hex_word
    lw      t0, 12(s0)
    call    write_hex_word

    # newline
    li      t3, 10
    sb      t3, 0(s1)

    # Also load into float regs for log inspection
    flw     fa0, 0(s0)
    flw     fa1, 4(s0)
    flw     fa2, 8(s0)
    flw     fa3, 12(s0)

_finish:
    li      t0, 0xff
    sb      t0, 0(s1)
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
