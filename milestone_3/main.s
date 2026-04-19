# ============================================================
# main.s — Full S4D galaxy classifier forward pass
# Uses VeeR-iSS entry point _start
# ============================================================
.extern hilbert_scan
.extern linear_layer
.extern working_s4d_test
.extern gelu_vec
.extern take_last_timestep
.extern softmax

.section .text
.global _start

_start:
    # ── set up stack (ensure 16-byte alignment) ──
    la    sp, stack_top
    addi  sp, sp, -16
    sw    ra, 0(sp)

    # ── LAYER 1: Hilbert Scan (C,64,64) -> (4096,C) ──
    la    a0, input_image
    la    a1, hilbert_out
    li    a2, 1               # C = 1 channel
    call  hilbert_scan

    # ── LAYER 2: Linear / Input Projection (4096,1) -> (4096,64) ──
    la    a0, hilbert_out
    la    a1, linear_w
    la    a2, linear_b
    la    a3, linear_out
    li    a4, 1               # in_dim = C
    li    a5, 64              # out_dim
    li    a6, 4096            # batch = seq_len
    call  linear_layer

    # ── LAYER 3: S4D Layer 1 (4096,64) -> (4096,64) ──
    la    a0, linear_out
    la    a1, s4d1_A_real
    la    a2, s4d1_A_imag
    la    a3, s4d1_B
    la    a4, s4d1_C_real
    la    a5, s4d1_C_imag
    la    a6, s4d1_out
    li    a7, 4096
    call  s4d_layer

    # ── LAYER 4: GELU 1 ──
    la    a0, s4d1_out
    li    a1, 262144          # 4096 * 64
    call  gelu_vec

    # ── LAYER 5: S4D Layer 2 ──
    la    a0, s4d1_out        # output of gelu1 is input of s4d2
    la    a1, s4d2_A_real
    la    a2, s4d2_A_imag
    la    a3, s4d2_B
    la    a4, s4d2_C_real
    la    a5, s4d2_C_imag
    la    a6, s4d2_out
    li    a7, 4096
    call  s4d_layer

    # ── LAYER 6: GELU 2 ──
    la    a0, s4d2_out
    li    a1, 262144
    call  gelu_vec

    # ── LAYER 7: TakeLastTimestep (4096,64) -> (64,) ──
    la    a0, s4d2_out
    la    a1, tlts_out
    call  take_last_timestep

    # ── LAYER 8: FC Layer (64,) -> (4,) ──
    la    a0, tlts_out
    la    a1, fc_w
    la    a2, fc_b
    la    a3, fc_out
    li    a4, 64
    li    a5, 4
    li    a6, 1
    call  linear_layer

    # ── LAYER 9: Softmax (4,) -> (4,) ──
    la    a0, fc_out
    call  softmax

    # ── Signal completion to Whisper ──
    li    t0, 0xd0580000
    li    t1, 1
    sw    t1, 0(t0)

    # ── infinite loop to halt simulation ──
done:
    j done

# ──────────────────────────────────────────────────────────
# .data section — weights and buffers
# ──────────────────────────────────────────────────────────
.section .data
.align 4

# Input image — embed binary file (1 channel, 64x64 = 4096 floats)
input_image:   .incbin "testdata/input_00.bin"

# Weights — embedded from binary files
linear_w:      .incbin "weights/linear_w.bin"
linear_b:      .incbin "weights/linear_b.bin"
s4d1_A_real:   .incbin "weights/s4d1_A_real.bin"
s4d1_A_imag:   .incbin "weights/s4d1_A_imag.bin"
s4d1_B:        .incbin "weights/s4d1_B.bin"
s4d1_C_real:   .incbin "weights/s4d1_C_real.bin"
s4d1_C_imag:   .incbin "weights/s4d1_C_imag.bin"
s4d2_A_real:   .incbin "weights/s4d2_A_real.bin"
s4d2_A_imag:   .incbin "weights/s4d2_A_imag.bin"
s4d2_B:        .incbin "weights/s4d2_B.bin"
s4d2_C_real:   .incbin "weights/s4d2_C_real.bin"
s4d2_C_imag:   .incbin "weights/s4d2_C_imag.bin"
fc_w:          .incbin "weights/fc_w.bin"
fc_b:          .incbin "weights/fc_b.bin"

# Intermediate buffers
.section .bss
.align 4
hilbert_out:  .space 16384   # 4096 * 1 * 4 (C=1 channel)
linear_out:   .space 1048576 # 4096 * 64 * 4
s4d1_out:     .space 1048576
s4d2_out:     .space 1048576
tlts_out:     .space 256     # 64 * 4
fc_out:       .space 16      # 4 * 4

# Stack (8KB)
stack_bottom:
    .space 8192
stack_top:
