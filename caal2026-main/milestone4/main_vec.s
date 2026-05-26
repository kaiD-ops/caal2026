# =============================================================================
# main_vec.s  -  S4D Galaxy Classifier demo for RISC-V / VeeR-iSS
#                Milestone 4: Vectorized (RVV) implementation
#
# Same 9-stage pipeline as M3 main.s.
# Calls the same function names; the *_vec.s files provide vectorized bodies.
#
# Weight layout in model_weights.bin (total 84496 bytes):
#   hilbert_indices : 4096 * 4  = 16384 bytes  (int32)
#   uproject_weight :   64 * 4  =   256 bytes
#   uproject_bias   :   64 * 4  =   256 bytes
#   s4_1_log_dt     :   64 * 4  =   256 bytes
#   s4_1_log_A_real : 64*32*4   =  8192 bytes
#   s4_1_A_imag     : 64*32*4   =  8192 bytes
#   s4_1_C          : 64*32*2*4 = 16384 bytes  (interleaved re,im pairs)
#   s4_1_D          :   64 * 4  =   256 bytes
#   s4_2_log_dt     :   64 * 4  =   256 bytes
#   s4_2_log_A_real : 64*32*4   =  8192 bytes
#   s4_2_A_imag     : 64*32*4   =  8192 bytes
#   s4_2_C          : 64*32*2*4 = 16384 bytes
#   s4_2_D          :   64 * 4  =   256 bytes
#   fc_weight       :  4*64*4   =  1024 bytes
#   fc_bias         :   4  *4   =    16 bytes
# =============================================================================

#define STDOUT 0xd0580000

.section .text
.global _start

_start:
    lui     sp, %hi(stack_top)
    addi    sp, sp, %lo(stack_top)

    # Zero-initialize .bss section (required for bare metal)
    lui     t0, %hi(__bss_start)
    addi    t0, t0, %lo(__bss_start)
    lui     t1, %hi(__bss_end)
    addi    t1, t1, %lo(__bss_end)
    fmv.w.x ft0, zero
bss_zero_loop:
    bge     t0, t1, bss_zero_done
    fsw     ft0, 0(t0)
    addi    t0, t0, 4
    j       bss_zero_loop
bss_zero_done:

    # Step 1: Hilbert scan
    lui     a0, %hi(weights)
    addi    a0, a0, %lo(weights)

    lui     a1, %hi(sample_img)
    addi    a1, a1, %lo(sample_img)

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)
    call    hilbert_scan

    # Step 2: UProject linear  [4096,1] -> [4096,64]
    lui     a0, %hi(w_uproject_w)
    addi    a0, a0, %lo(w_uproject_w)

    lui     a1, %hi(w_uproject_b)
    addi    a1, a1, %lo(w_uproject_b)

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)

    lui     a3, %hi(buf_proj)
    addi    a3, a3, %lo(buf_proj)

    li      a4, 1
    li      a5, 64
    li      a6, 4096
    call    linear_layer

    # Step 3: S4D Layer 1
    lui     a0, %hi(w_s4_1_log_dt)
    addi    a0, a0, %lo(w_s4_1_log_dt)

    lui     a1, %hi(w_s4_1_log_A_real)
    addi    a1, a1, %lo(w_s4_1_log_A_real)

    lui     a2, %hi(w_s4_1_A_imag)
    addi    a2, a2, %lo(w_s4_1_A_imag)

    lui     a3, %hi(w_s4_1_C)
    addi    a3, a3, %lo(w_s4_1_C)

    lui     a4, %hi(w_s4_1_D)
    addi    a4, a4, %lo(w_s4_1_D)

    lui     a5, %hi(buf_proj)
    addi    a5, a5, %lo(buf_proj)

    lui     a6, %hi(buf_s4d1)
    addi    a6, a6, %lo(buf_s4d1)
    call    s4d_layer

    # DIAG: write 'A' then newline to console after s4d_layer1
    lui     t4, 0xd0580
    addi    t4, t4, 4
    li      t0, 65
    sb      t0, 0(t4)
    li      t0, 10
    sb      t0, 0(t4)

    # Step 4: GELU  (4096*64 = 262144 elements)
    lui     a0, %hi(buf_s4d1)
    addi    a0, a0, %lo(buf_s4d1)
    li      a1, 262144
    call    gelu_inplace

    # Step 5: S4D Layer 2
    lui     a0, %hi(w_s4_2_log_dt)
    addi    a0, a0, %lo(w_s4_2_log_dt)

    lui     a1, %hi(w_s4_2_log_A_real)
    addi    a1, a1, %lo(w_s4_2_log_A_real)

    lui     a2, %hi(w_s4_2_A_imag)
    addi    a2, a2, %lo(w_s4_2_A_imag)

    lui     a3, %hi(w_s4_2_C)
    addi    a3, a3, %lo(w_s4_2_C)

    lui     a4, %hi(w_s4_2_D)
    addi    a4, a4, %lo(w_s4_2_D)

    lui     a5, %hi(buf_s4d1)
    addi    a5, a5, %lo(buf_s4d1)

    lui     a6, %hi(buf_s4d2)
    addi    a6, a6, %lo(buf_s4d2)
    call    s4d_layer

    # DIAG: buf_s4d2[32] = element (t=0,h=32)
    lui     t4, 0xd0580
    addi    t4, t4, 4
    lui     t5, %hi(buf_s4d2)
    addi    t5, t5, %lo(buf_s4d2)
    lw      t5, 128(t5)      # offset 32*4=128
    li      t6, 28
dB: bltz t6, dBd
    srl t0, t5, t6
    andi t0, t0, 0xF
    li t1, 10
    blt t0, t1, dBa
    addi t0, t0, 87
    j dBs
dBa: addi t0, t0, 48
dBs: sb t0, 0(t4)
    addi t6, t6, -4
    j dB
dBd: li t0, 10
    sb t0, 0(t4)

    # Step 6: GELU 2
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)
    li      a1, 262144
    call    gelu_inplace

    # Step 7: TakeLastTimestep  [4096,64] -> [64]
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)

    lui     a1, %hi(buf_pooled)
    addi    a1, a1, %lo(buf_pooled)
    call    take_last_timestep

    # DIAG: Write "P" then buf_pooled[0] after take_last
    lui     t4, 0xd0580
    addi    t4, t4, 4
    li      t0, 80
    sb      t0, 0(t4)
    lui     t0, %hi(buf_pooled)
    addi    t0, t0, %lo(buf_pooled)
    lw      t1, 0(t0)
    li      t2, 28
dBp: bltz t2, dBpd
    srl t0, t1, t2
    andi t0, t0, 0xF
    li t3, 10
    blt t0, t3, dBpa
    addi t0, t0, 87
    j dBps
dBpa: addi t0, t0, 48
dBps: sb t0, 0(t4)
    addi t2, t2, -4
    j dBp
dBpd: li t0, 10
    sb t0, 0(t4)

    # Step 8: FC head  [64] -> [4]
    lui     a0, %hi(w_fc_weight)
    addi    a0, a0, %lo(w_fc_weight)

    lui     a1, %hi(w_fc_bias)
    addi    a1, a1, %lo(w_fc_bias)

    lui     a2, %hi(buf_pooled)
    addi    a2, a2, %lo(buf_pooled)

    lui     a3, %hi(buf_logits)
    addi    a3, a3, %lo(buf_logits)

    li      a4, 64
    li      a5, 4
    li      a6, 1
    call    linear_layer

    # DIAG: Write "B" then logits before softmax
    lui     t4, 0xd0580
    addi    t4, t4, 4
    li      t0, 66
    sb      t0, 0(t4)
    lui     t0, %hi(buf_logits)
    addi    t0, t0, %lo(buf_logits)
    lw      t1, 0(t0)
    li      t2, 28
dB2: bltz t2, dB2d
    srl t0, t1, t2
    andi t0, t0, 0xF
    li t3, 10
    blt t0, t3, dB2a
    addi t0, t0, 87
    j dB2s
dB2a: addi t0, t0, 48
dB2s: sb t0, 0(t4)
    addi t2, t2, -4
    j dB2
dB2d: li t0, 10
    sb t0, 0(t4)

    # Step 9: Softmax
    lui     a0, %hi(buf_logits)
    addi    a0, a0, %lo(buf_logits)
    li      a1, 4
    call    softmax_inplace

    # Output probabilities to VeeR log
    lui     t0, %hi(buf_logits)
    addi    t0, t0, %lo(buf_logits)
    flw     fa0, 0(t0)
    flw     fa1, 4(t0)
    flw     fa2, 8(t0)
    flw     fa3, 12(t0)

    lui     t1, %hi(output_probs)
    addi    t1, t1, %lo(output_probs)
    fsw     fa0, 0(t1)
    fsw     fa1, 4(t1)
    fsw     fa2, 8(t1)
    fsw     fa3, 12(t1)

    # Argmax
    flt.s   t0, fa1, fa0
    li      t2, 0
    beqz    t0, 1f
    fmv.s   fa0, fa1
    li      t2, 1
1:  flt.s   t0, fa2, fa0
    beqz    t0, 2f
    fmv.s   fa0, fa2
    li      t2, 2
2:  flt.s   t0, fa3, fa0
    beqz    t0, 3f
    li      t2, 3
3:  lui     t0, %hi(output_pred)
    sw      t2, %lo(output_pred)(t0)

    # Write hex to console
    lui     s0, %hi(buf_logits)
    addi    s0, s0, %lo(buf_logits)
    lui     s1, 0xd0580
    addi    s1, s1, 4

    lw      s2, 0(s0)
    li      s3, 28
mhex0: bltz s3, mhex0d
    srl t0, s2, s3
    andi t0, t0, 0xF
    li t1, 10
    blt t0, t1, mh0d
    addi t0, t0, 87
    j mh0s
mh0d: addi t0, t0, 48
mh0s: sb t0, 0(s1)
    addi s3, s3, -4
    j mhex0
mhex0d: li t0, 32
    sb t0, 0(s1)

    lw      s2, 4(s0)
    li      s3, 28
mhex1: bltz s3, mhex1d
    srl t0, s2, s3
    andi t0, t0, 0xF
    li t1, 10
    blt t0, t1, mh1d
    addi t0, t0, 87
    j mh1s
mh1d: addi t0, t0, 48
mh1s: sb t0, 0(s1)
    addi s3, s3, -4
    j mhex1
mhex1d: li t0, 32
    sb t0, 0(s1)

    lw      s2, 8(s0)
    li      s3, 28
mhex2: bltz s3, mhex2d
    srl t0, s2, s3
    andi t0, t0, 0xF
    li t1, 10
    blt t0, t1, mh2d
    addi t0, t0, 87
    j mh2s
mh2d: addi t0, t0, 48
mh2s: sb t0, 0(s1)
    addi s3, s3, -4
    j mhex2
mhex2d: li t0, 32
    sb t0, 0(s1)

    lw      s2, 12(s0)
    li      s3, 28
mhex3: bltz s3, mhex3d
    srl t0, s2, s3
    andi t0, t0, 0xF
    li t1, 10
    blt t0, t1, mh3d
    addi t0, t0, 87
    j mh3s
mh3d: addi t0, t0, 48
mh3s: sb t0, 0(s1)
    addi s3, s3, -4
    j mhex3
mhex3d: li t0, 10
    sb t0, 0(s1)
_finish:
    lui     x3, 0xd0580
    addi    x3, x3, 0
    addi    x5, x0, 0xff
    sb      x5, 0(x3)
    beq     x0, x0, _finish
.rept 100
    nop
.endr

# =============================================================================
.section .data
.align 2

weights:
.incbin "../model_weights.bin"

.set OFF_HILBERT_IDX,   0
.set OFF_UPROJECT_W,    16384
.set OFF_UPROJECT_B,    16640
.set OFF_S4_1_LOG_DT,   16896
.set OFF_S4_1_LOG_AR,   17152
.set OFF_S4_1_AIMAG,    25344
.set OFF_S4_1_C,        33536
.set OFF_S4_1_D,        49920
.set OFF_S4_2_LOG_DT,   50176
.set OFF_S4_2_LOG_AR,   50432
.set OFF_S4_2_AIMAG,    58624
.set OFF_S4_2_C,        66816
.set OFF_S4_2_D,        83200
.set OFF_FC_W,          83456
.set OFF_FC_B,          84480

w_uproject_w      = weights + OFF_UPROJECT_W
w_uproject_b      = weights + OFF_UPROJECT_B
w_s4_1_log_dt     = weights + OFF_S4_1_LOG_DT
w_s4_1_log_A_real = weights + OFF_S4_1_LOG_AR
w_s4_1_A_imag     = weights + OFF_S4_1_AIMAG
w_s4_1_C          = weights + OFF_S4_1_C
w_s4_1_D          = weights + OFF_S4_1_D
w_s4_2_log_dt     = weights + OFF_S4_2_LOG_DT
w_s4_2_log_A_real = weights + OFF_S4_2_LOG_AR
w_s4_2_A_imag     = weights + OFF_S4_2_AIMAG
w_s4_2_C          = weights + OFF_S4_2_C
w_s4_2_D          = weights + OFF_S4_2_D
w_fc_weight       = weights + OFF_FC_W
w_fc_bias         = weights + OFF_FC_B

.align 2
sample_img:
.incbin "../test_data/sample_09_input.bin"

.align 2
output_probs:   .space 16
output_pred:    .space 4

.section .bss
.align 2
buf_hilbert:    .space 16384
buf_proj:       .space 1048576
buf_s4d1:       .space 1048576
buf_s4d2:       .space 1048576
buf_pooled:     .space 256
buf_logits:     .space 16
                .space 8192
stack_top:
