# =============================================================================
# main_vec.s  –  S4D Galaxy Classifier demo: RISC-V Vector (RVV) version
#
# Milestone 4 entry point.  Loads model weights and a sample input image,
# runs the complete 9-stage RVV-vectorized forward pass, and writes the
# output class probabilities to the VeeR-iSS console I/O port so they
# can be extracted and validated by the test harness.
#
# Pipeline (all layers vectorized except Softmax, see softmax_vec.s):
#   1. hilbert_scan        (RVV: vluxei32 gather)
#   2. linear_layer        (RVV: vfmul + vfredusum dot product)
#   3. s4d_layer  [L1]     (RVV: Phase-2 kernel build + Phase-3 convolution)
#   4. gelu_inplace        (RVV: vectorized polynomial + scalar tanhf)
#   5. s4d_layer  [L2]     (same as step 3)
#   6. gelu_inplace        (same as step 4)
#   7. take_last_timestep  (RVV: single vle32+vse32 trip)
#   8. linear_layer  [FC]  (RVV: same as step 2)
#   9. softmax_inplace     (SCALAR: N=4 elements, vectorization gives no gain)
#
# Weight layout in model_weights.bin (total 84496 bytes) — unchanged from M3:
#   hilbert_indices : 4096 * 4  = 16384  (int32)
#   uproject_weight :   64 * 4  =   256  (float [64,1])
#   uproject_bias   :   64 * 4  =   256
#   s4_1_log_dt     :   64 * 4  =   256
#   s4_1_log_A_real : 64*32*4   =  8192
#   s4_1_A_imag     : 64*32*4   =  8192
#   s4_1_C          : 64*32*2*4 = 16384  (interleaved re/im)
#   s4_1_D          :   64 * 4  =   256
#   s4_2_log_dt     :   64 * 4  =   256
#   s4_2_log_A_real : 64*32*4   =  8192
#   s4_2_A_imag     : 64*32*4   =  8192
#   s4_2_C          : 64*32*2*4 = 16384
#   s4_2_D          :   64 * 4  =   256
#   fc_weight       :  4*64*4   =  1024
#   fc_bias         :   4  *4   =    16
# =============================================================================

#define STDOUT 0xd0580000

.section .text
.global _start

_start:
    lui     sp, %hi(stack_top)
    addi    sp, sp, %lo(stack_top)

    # ── 1. Hilbert Scan ───────────────────────────────────────────────────────
    lui     a0, %hi(weights)
    addi    a0, a0, %lo(weights)        # a0 = &hilbert_indices[0]

    lui     a1, %hi(sample_img)
    addi    a1, a1, %lo(sample_img)

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)
    call    hilbert_scan                # RVV gather

    # ── 2. UProject  [4096,1] → [4096,64] ───────────────────────────────────
    lui     a0, %hi(w_uproject_w)
    addi    a0, a0, %lo(w_uproject_w)

    lui     a1, %hi(w_uproject_b)
    addi    a1, a1, %lo(w_uproject_b)

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)

    lui     a3, %hi(buf_proj)
    addi    a3, a3, %lo(buf_proj)

    li      a4, 1                       # in_dim  = 1
    li      a5, 64                      # out_dim = 64
    li      a6, 4096                    # seq_len
    call    linear_layer                # RVV dot-product

    # ── 3. S4D Layer 1 ───────────────────────────────────────────────────────
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
    call    s4d_layer                   # RVV vectorized

    # ── 4. GELU 1  (262144 elements) ─────────────────────────────────────────
    lui     a0, %hi(buf_s4d1)
    addi    a0, a0, %lo(buf_s4d1)
    li      a1, 262144
    call    gelu_inplace                # RVV polynomial, scalar tanhf

    # ── 5. S4D Layer 2 ───────────────────────────────────────────────────────
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

    # ── 6. GELU 2 ────────────────────────────────────────────────────────────
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)
    li      a1, 262144
    call    gelu_inplace

    # ── 7. TakeLastTimestep  [4096,64] → [64] ────────────────────────────────
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)

    lui     a1, %hi(buf_pooled)
    addi    a1, a1, %lo(buf_pooled)
    call    take_last_timestep          # RVV single-pass copy

    # ── 8. FC head  [64] → [4] ───────────────────────────────────────────────
    lui     a0, %hi(w_fc_weight)
    addi    a0, a0, %lo(w_fc_weight)

    lui     a1, %hi(w_fc_bias)
    addi    a1, a1, %lo(w_fc_bias)

    lui     a2, %hi(buf_pooled)
    addi    a2, a2, %lo(buf_pooled)

    lui     a3, %hi(buf_logits)
    addi    a3, a3, %lo(buf_logits)

    li      a4, 64                      # in_dim
    li      a5, 4                       # out_dim
    li      a6, 1                       # seq_len = 1
    call    linear_layer

    # ── 9. Softmax (N=4, scalar) ─────────────────────────────────────────────
    lui     a0, %hi(buf_logits)
    addi    a0, a0, %lo(buf_logits)
    li      a1, 4
    call    softmax_inplace

    # ── Output: load probabilities into FP registers so VeeR log records them
    lui     t0, %hi(buf_logits)
    addi    t0, t0, %lo(buf_logits)
    flw     fa0, 0(t0)                  # P(Smooth Round)
    flw     fa1, 4(t0)                  # P(Smooth Cigar)
    flw     fa2, 8(t0)                  # P(Edge-on Disk)
    flw     fa3, 12(t0)                 # P(Unbarred Spiral)

    # Copy to output_probs for Python harness
    lui     t1, %hi(output_probs)
    addi    t1, t1, %lo(output_probs)
    fsw     fa0,  0(t1)
    fsw     fa1,  4(t1)
    fsw     fa2,  8(t1)
    fsw     fa3, 12(t1)

    # Argmax → output_pred
    li      t2, 0
    flt.s   t0, fa0, fa1               # t0 = (fa0 < fa1)?
    beqz    t0, 1f
    fmv.s   fa0, fa1
    li      t2, 1
1:  flt.s   t0, fa0, fa2               # t0 = (fa0 < fa2)?
    beqz    t0, 2f
    fmv.s   fa0, fa2
    li      t2, 2
2:  flt.s   t0, fa0, fa3               # t0 = (fa0 < fa3)?
    beqz    t0, 3f
    li      t2, 3
3:  lui     t0, %hi(output_pred)
    sw      t2, %lo(output_pred)(t0)

    # ── Write raw hex of each probability to VeeR console I/O ────────────────
    lui     s0, %hi(buf_logits)
    addi    s0, s0, %lo(buf_logits)
    lui     s1, 0xd0580
    addi    s1, s1, 4                   # console I/O at 0xd0580004

    # Helper macro-expansion: dump 4 float words as 8-hex-digit strings
    # separated by spaces, terminated by newline
    li      s3, 0                       # word index

mv_dump_loop:
    li      t6, 4
    bge     s3, t6, mv_dump_done
    slli    t0, s3, 2
    add     t0, s0, t0
    lw      s2, 0(t0)                   # raw bits of probability
    li      s4, 28                      # bit position (MSN first)
mv_hex_loop:
    bltz    s4, mv_hex_done
    srl     t0, s2, s4
    andi    t0, t0, 0xF
    li      t1, 10
    blt     t0, t1, mv_digit
    addi    t0, t0, 87                  # 'a'-10
    j       mv_store
mv_digit:
    addi    t0, t0, 48                  # '0'
mv_store:
    sb      t0, 0(s1)
    addi    s4, s4, -4
    j       mv_hex_loop
mv_hex_done:
    li      t0, 32                      # space
    sb      t0, 0(s1)
    addi    s3, s3, 1
    j       mv_dump_loop

mv_dump_done:
    li      t0, 10                      # newline
    sb      t0, 0(s1)

    # ── Halt: VeeR-iSS exit sequence ─────────────────────────────────────────
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
# Data section
# =============================================================================
.section .data
.align 2

weights:
.incbin "../model_weights.bin"

# Weight offsets (identical to M3)
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

# =============================================================================
# BSS: intermediate activation buffers (same sizes as M3)
# =============================================================================
.section .bss
.align 4

buf_hilbert:    .space  16384       # [4096,1] * 4
buf_proj:       .space 1048576      # [4096,64] * 4
buf_s4d1:       .space 1048576      # [4096,64] * 4
buf_s4d2:       .space 1048576      # [4096,64] * 4
buf_pooled:     .space    256       # [64] * 4
buf_logits:     .space     16       # [4] * 4

                .space  8192        # 8 KB stack
stack_top:
