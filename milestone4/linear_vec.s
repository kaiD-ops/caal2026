# =============================================================================
# linear_vec.s  –  RVV-vectorized Linear (FC) layer
#
# Computes Y = X * W^T + b for a sequence of input vectors.
# Shape:  X [seq_len, in_dim]  →  Y [seq_len, out_dim]
# W [out_dim, in_dim],  b [out_dim]
#
# Vectorization strategy
# ──────────────────────
# Outer loops: t (timestep) and o (output neuron) are kept scalar.
# Inner loop : dot product over in_dim is vectorized.
#   • vfmv.v.f  initialises accumulator to bias value
#   • vfmacc.vv  fuses multiply-accumulate over chunks of in_dim
#   • vfredusum.vs  reduces the partial dot-product vector to a scalar
#
# For the UProject call (in_dim=1, out_dim=64, seq_len=4096) the entire
# 64-element dot-product strip fits in one vector operation.
#
# Signature (unchanged from M3):
#   void linear_layer(float* W, float* b, float* in, float* out,
#                     int in_dim, int out_dim, int seq_len)
#   a0=W  a1=b  a2=in  a3=out  a4=in_dim  a5=out_dim  a6=seq_len
# =============================================================================

.section .text
.global linear_layer

linear_layer:
    addi    sp, sp, -44
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)

    mv      s0, a0          # W
    mv      s1, a1          # b
    mv      s2, a2          # in  (current row pointer)
    mv      s3, a3          # out (current row pointer)
    mv      s4, a4          # in_dim
    mv      s5, a5          # out_dim
    mv      s6, a6          # seq_len

    # Pre-compute byte strides
    slli    s7, s4, 2       # in_row_bytes  = in_dim  * 4
    slli    s8, s5, 2       # out_row_bytes = out_dim * 4

ll_t:
    beqz    s6, ll_done

    # ── inner: for each output neuron o ─────────────────────────────────────
    mv      t3, s5          # o countdown
    mv      t4, s1          # b pointer
    mv      t5, s3          # out pointer for this timestep
    mv      t6, s0          # W pointer = &W[0][0]

ll_o:
    beqz    t3, ll_t_next

    # acc = b[o]  (scalar → broadcast later via reduction init)
    flw     fa0, 0(t4)      # fa0 = bias

    # Vectorized dot product: acc += sum_i W[o][i] * in[t][i]
    mv      t0, s4          # remaining in_dim elements
    mv      t1, t6          # walking W row pointer
    mv      t2, s2          # walking in row pointer

    # Zero-accumulate vector register set (v0 used as zero-init reduction)
    # We accumulate into fa0 using scalar reduction after vector fmacc strips

    # Use v16 as partial sum accumulator, init to 0.0
    vsetvli t0, s4, e32, m4, ta, ma    # vl = min(s4, VLMAX)
    # We will strip-mine but keep fa0 = bias, accumulate into fa0
    # after each strip via vfredusum into v0[0]

    mv      t0, s4          # re-init remaining

ll_dot:
    beqz    t0, ll_dot_done

    vsetvli t1, t0, e32, m4, ta, ma    # t1 = this strip's vl

    # Load strip of W[o] and in[t]
    vle32.v  v0, (t6)       # v0 = W[o][i..i+vl-1]
    vle32.v  v4, (t2)       # v4 = in[t][i..i+vl-1]

    # Multiply into v8
    vfmul.vv v8, v0, v4     # v8 = W * in element-wise

    # Reduce v8 into scalar using vfredusum; init sum vector v12 = 0
    vfmv.v.f v12, fa7       # scratch — overwritten; use zero
    fmv.w.x  ft0, zero
    vfmv.s.f v12, ft0       # v12[0] = 0.0  (reduction identity)
    vfredusum.vs v12, v8, v12  # v12[0] = sum(v8)

    # Extract scalar and accumulate into fa0
    vfmv.f.s ft0, v12       # ft0 = partial dot product
    fadd.s  fa0, fa0, ft0   # fa0 += partial

    # Advance pointers
    slli    t5, t1, 2       # bytes = vl * 4
    add     t6, t6, t5      # W pointer forward
    add     t2, t2, t5      # in pointer forward
    sub     t0, t0, t1      # remaining -= vl
    j       ll_dot

ll_dot_done:
    # Store result; restore t5 (out pointer)
    mv      t5, s3          # base out for this timestep
    sub     t6, s5, t3      # o index = out_dim - countdown
    slli    t6, t6, 2
    add     t6, t5, t6      # &out[t][o]
    fsw     fa0, 0(t6)

    # Advance W to next row W[o+1]
    add     t6, s0, zero    # recompute: W base + o_next * in_dim * 4
    sub     t6, s5, t3      # o_done
    addi    t6, t6, 1       # o_next = o_done + 1  ... wait, use countdown
    # Simpler: t6 = W_base + (out_dim - t3 + 1) * in_dim * 4
    # But we already advanced t6 inside ll_dot — reset for next o
    sub     t0, s5, t3      # o = out_dim - t3
    addi    t0, t0, 1       # next o
    mul     t0, t0, s4      # next_o * in_dim
    slli    t0, t0, 2
    add     t6, s0, t0      # W row pointer for next o

    addi    t4, t4, 4       # b pointer++
    addi    t3, t3, -1
    j       ll_o

ll_t_next:
    add     s2, s2, s7      # in  += in_row_bytes
    add     s3, s3, s8      # out += out_row_bytes
    addi    s6, s6, -1
    j       ll_t

ll_done:
    lw      ra,  0(sp)
    lw      s0,  4(sp)
    lw      s1,  8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    lw      s4, 20(sp)
    lw      s5, 24(sp)
    lw      s6, 28(sp)
    lw      s7, 32(sp)
    lw      s8, 36(sp)
    lw      s9, 40(sp)
    addi    sp, sp, 44
    ret
