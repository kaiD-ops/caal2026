# =============================================================================
# main.s  –  S4D Galaxy Classifier demo for RISC-V / VeeR-iSS
#
# Loads model weights and sample_09 input from .data section (.incbin),
# runs the complete 9-stage forward pass, and writes class probabilities
# to the VeeR log so they can be extracted and validated.
#
# Pipeline:
#   1. hilbert_scan
#   2. linear_layer  (UProject)
#   3. s4d_layer     (layer 1)
#   4. gelu_inplace
#   5. s4d_layer     (layer 2)
#   6. gelu_inplace
#   7. take_last_timestep
#   8. linear_layer  (FC head)
#   9. softmax_inplace
#
# Weight layout in model_weights.bin (total 84496 bytes):
#   hilbert_indices : 4096 * 4  = 16384 bytes  (int32)
#   uproject_weight :   64 * 4  =   256 bytes  (float, shape [64,1])
#   uproject_bias   :   64 * 4  =   256 bytes
#   s4_1_log_dt     :   64 * 4  =   256 bytes
#   s4_1_log_A_real : 64*32*4   =  8192 bytes
#   s4_1_A_imag     : 64*32*4   =  8192 bytes
#   s4_1_C          : 64*32*2*4 = 16384 bytes
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
    # Set up stack pointer
    lui     sp, %hi(stack_top)
    addi    sp, sp, %lo(stack_top)

    # -----------------------------------------------------------------------
    # Step 1: Hilbert scan  (C_IN=1, so just reorder 4096 pixels)
    # hilbert_scan(hilbert_indices, img, buf_hilbert)
    # -----------------------------------------------------------------------
    lui     a0, %hi(weights)
    addi    a0, a0, %lo(weights)    # a0 = &hilbert_indices[0]

    lui     a1, %hi(sample_img)
    addi    a1, a1, %lo(sample_img) # a1 = input image

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)
    call    hilbert_scan

    # -----------------------------------------------------------------------
    # Step 2: UProject linear layer  [4096,1] -> [4096,64]
    # Weights start at offset 16384
    # linear_layer(W, b, in, out, in_dim=1, out_dim=64, seq_len=4096)
    # -----------------------------------------------------------------------
    lui     a0, %hi(w_uproject_w)
    addi    a0, a0, %lo(w_uproject_w)

    lui     a1, %hi(w_uproject_b)
    addi    a1, a1, %lo(w_uproject_b)

    lui     a2, %hi(buf_hilbert)
    addi    a2, a2, %lo(buf_hilbert)

    lui     a3, %hi(buf_proj)
    addi    a3, a3, %lo(buf_proj)

    li      a4, 1                   # in_dim  = C_IN = 1
    li      a5, 64                  # out_dim = D_MODEL
    li      a6, 4096                # seq_len = SEQ_LEN
    call    linear_layer

    # -----------------------------------------------------------------------
    # Step 3: S4D Layer 1
    # s4d_layer(log_dt, log_A_real, A_imag, C_mat, D_vec, in, out)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Step 4: GELU  (SEQ_LEN * D_MODEL = 4096*64 = 262144 elements)
    # -----------------------------------------------------------------------
    lui     a0, %hi(buf_s4d1)
    addi    a0, a0, %lo(buf_s4d1)
    li      a1, 262144
    call    gelu_inplace

    # -----------------------------------------------------------------------
    # Step 5: S4D Layer 2
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Step 6: GELU 2
    # -----------------------------------------------------------------------
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)
    li      a1, 262144
    call    gelu_inplace

    # -----------------------------------------------------------------------
    # Step 7: TakeLastTimestep  [4096,64] -> [64]
    # -----------------------------------------------------------------------
    lui     a0, %hi(buf_s4d2)
    addi    a0, a0, %lo(buf_s4d2)

    lui     a1, %hi(buf_pooled)
    addi    a1, a1, %lo(buf_pooled)
    call    take_last_timestep

    # -----------------------------------------------------------------------
    # Step 8: FC head  [64] -> [4]
    # linear_layer(fc_weight, fc_bias, pooled, logits, 64, 4, 1)
    # -----------------------------------------------------------------------
    lui     a0, %hi(w_fc_weight)
    addi    a0, a0, %lo(w_fc_weight)

    lui     a1, %hi(w_fc_bias)
    addi    a1, a1, %lo(w_fc_bias)

    lui     a2, %hi(buf_pooled)
    addi    a2, a2, %lo(buf_pooled)

    lui     a3, %hi(buf_logits)
    addi    a3, a3, %lo(buf_logits)

    li      a4, 64                  # in_dim  = D_MODEL
    li      a5, 4                   # out_dim = N_CLASSES
    li      a6, 1                   # seq_len = 1 (single vector)
    call    linear_layer

    # -----------------------------------------------------------------------
    # Step 9: Softmax
    # -----------------------------------------------------------------------
    lui     a0, %hi(buf_logits)
    addi    a0, a0, %lo(buf_logits)
    li      a1, 4                   # N_CLASSES
    call    softmax_inplace

    # -----------------------------------------------------------------------
    # Output: write probabilities to memory so VeeR log shows them
    # Load each probability into a register — VeeR log records register values
    # -----------------------------------------------------------------------
    lui     t0, %hi(buf_logits)
    addi    t0, t0, %lo(buf_logits)
    flw     fa0, 0(t0)              # P(Smooth Round)
    flw     fa1, 4(t0)              # P(Smooth Cigar)
    flw     fa2, 8(t0)              # P(Edge-on Disk)
    flw     fa3, 12(t0)             # P(Unbarred Spiral)

    # Store to a known memory address so Python can read from log
    lui     t1, %hi(output_probs)
    addi    t1, t1, %lo(output_probs)
    fsw     fa0, 0(t1)
    fsw     fa1, 4(t1)
    fsw     fa2, 8(t1)
    fsw     fa3, 12(t1)

    # Find argmax and store predicted class
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
3:  # t2 = predicted class (0-3)
    lui     t0, %hi(output_pred)
    sw      t2, %lo(output_pred)(t0)

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
# Data section: weights, input image, intermediate buffers
# =============================================================================
.section .data
.align 2

# Model weights embedded directly from binary file
weights:
.incbin "../model_weights.bin"

# Offsets into weights (computed from layout table)
.set W_BASE,            weights
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

# Convenience labels pointing into the weights blob
w_uproject_w    = weights + OFF_UPROJECT_W
w_uproject_b    = weights + OFF_UPROJECT_B
w_s4_1_log_dt   = weights + OFF_S4_1_LOG_DT
w_s4_1_log_A_real = weights + OFF_S4_1_LOG_AR
w_s4_1_A_imag   = weights + OFF_S4_1_AIMAG
w_s4_1_C        = weights + OFF_S4_1_C
w_s4_1_D        = weights + OFF_S4_1_D
w_s4_2_log_dt   = weights + OFF_S4_2_LOG_DT
w_s4_2_log_A_real = weights + OFF_S4_2_LOG_AR
w_s4_2_A_imag   = weights + OFF_S4_2_AIMAG
w_s4_2_C        = weights + OFF_S4_2_C
w_s4_2_D        = weights + OFF_S4_2_D
w_fc_weight     = weights + OFF_FC_W
w_fc_bias       = weights + OFF_FC_B

# Input image (sample_09, true label = 0 = Smooth Round)
.align 2
sample_img:
.incbin "../test_data/sample_09_input.bin"

# Output storage
.align 2
output_probs:   .space 16           # 4 floats
output_pred:    .space 4            # 1 int32

# =============================================================================
# BSS: intermediate activation buffers
# =============================================================================
.section .bss
.align 2

buf_hilbert:    .space 16384        # SEQ_LEN * C_IN * 4      = 4096*1*4
buf_proj:       .space 1048576      # SEQ_LEN * D_MODEL * 4   = 4096*64*4
buf_s4d1:       .space 1048576      # SEQ_LEN * D_MODEL * 4
buf_s4d2:       .space 1048576      # SEQ_LEN * D_MODEL * 4
buf_pooled:     .space 256          # D_MODEL * 4              = 64*4
buf_logits:     .space 16           # N_CLASSES * 4            = 4*4

# Stack (8KB)
                .space 8192
stack_top:
