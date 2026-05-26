.section .text
.global _start
_start:
    lui     sp, %hi(ln_stack)
    addi    sp, sp, %lo(ln_stack)
    # linear_layer(W, b, in, out, in_dim=1, out_dim=64, seq_len=4096)
    lui     a0, %hi(ref_w)
    addi    a0, a0, %lo(ref_w)
    lui     a1, %hi(ref_b)
    addi    a1, a1, %lo(ref_b)
    lui     a2, %hi(ref_hilbert)
    addi    a2, a2, %lo(ref_hilbert)
    lui     a3, %hi(ln_result)
    addi    a3, a3, %lo(ln_result)
    li      a4, 1
    li      a5, 64
    li      a6, 4096
    call    linear_layer
    lui     t0, %hi(ln_result)
    addi    t0, t0, %lo(ln_result)
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
ref_w:
.incbin "../model_weights.bin", 16384, 256
ref_b:
.incbin "../model_weights.bin", 16640, 256
ref_hilbert:
.incbin "../test_data/sample_09_hilbert.bin"
ln_result: .space 16
.section .bss
.align 2
.space 8192
ln_stack:
