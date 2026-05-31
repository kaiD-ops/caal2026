

.section .text
.global linear_layer

linear_layer:
    addi    sp, sp, -48
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
    # 40(sp) and 44(sp) are padding slots keeping the stack 16-byte aligned

    mv      s0, a0          # W base pointer
    mv      s1, a1          # b base pointer
    mv      s2, a2          # in base matrix pointer
    mv      s3, a3          # out base matrix pointer
    mv      s4, a4          # in_dim
    mv      s5, a5          # out_dim
    mv      s6, a6          # seq_len

    # Pre-compute row stride offsets
    slli    s7, s4, 2       # in_row_bytes  = in_dim  * 4
    slli    s8, s5, 2       # out_row_bytes = out_dim * 4

ll_t:
    beqz    s6, ll_done

    # Setup scalar tracking values for this timestep row
    mv      t3, s5          # t3 = o (output neuron loop countdown)
    mv      t4, s1          # t4 = current b pointer
    mv      t5, s3          # t5 = current out pointer allocation
    mv      t6, s0          # t6 = current W row base pointer

ll_o:
    beqz    t3, ll_t_next

    # Initialize a vector register to all 0.0s to serve as our running row accumulator
    # Use e32, m1 for safe and highly compatible register allocation.
    vsetvli t0, s4, e32, m1, ta, ma
    vmv.v.i v0, 0           # Clear v0 tracking state (0 bits == 0.0f)

    mv      t0, s4          # t0 = remaining elements in this row
    mv      t2, s2          # t2 = refresh walking input row pointer

ll_dot:
    beqz    t0, ll_dot_done

    vsetvli t1, t0, e32, m1, tu, ma   # t1 = active vl; tu preserves partial sums in tail across strips

    # Load matching segments of W and input vector
    vle32.v  v1, (t6)       # v1 = W[o][i ... i+vl-1]
    vle32.v  v2, (t2)       # v2 = in[t][i ... i+vl-1]

    # Accumulate element-wise products directly into our running vector accumulator v0
    vfmacc.vv v0, v1, v2

    # Advance chunk segment tracking pointers
    slli    t4, t1, 2       # Use safe temp t4 (will be refreshed after dot product loop)
    add     t6, t6, t4      # W segment pointer forward
    add     t2, t2, t4      # input segment pointer forward
    sub     t0, t0, t1      # remaining row counter -= vl
    j       ll_dot

ll_dot_done:
    # Perform a SINGLE vector-to-scalar reduction now that the whole row is compiled
    # Initialize a temporary reduction landing vector register to 0.0
    vsetvli t0, s4, e32, m1, ta, ma
    fmv.w.x ft0, zero
    vfmv.s.f v1, ft0
    vfredusum.vs v1, v0, v1 # v1[0] = final vector dot-product sum

    # Extract the total scalar sum, then add our bias value
    vfmv.f.s fa0, v1        # fa0 = sum(W * X)
    mv      t4, s1          # Refresh base b pointer
    sub     t0, s5, t3      # Calculate current neuron index: o = out_dim - countdown
    slli    t1, t0, 2       # Offset in bytes = o * 4
    add     t4, t4, t1      # &b[o]
    flw     ft1, 0(t4)      # ft1 = b[o]
    fadd.s  fa0, fa0, ft1   # fa0 = final output value = sum + bias

    # Store result into &out[t][o]
    add     t1, t5, t1      # t1 = out pointer + (o * 4)
    fsw     fa0, 0(t1)

    # Recompute next W row base pointer cleanly without multicycle integer multiplies
    sub     t0, s5, t3      # current o
    addi    t0, t0, 1       # next o
    mul     t0, t0, s4      # next_o * in_dim
    slli    t0, t0, 2       # byte offset
    add     t6, s0, t0      # t6 = address of W[o+1][0]

    addi    t3, t3, -1      # Decrement output neuron countdown
    j       ll_o

ll_t_next:
    add     s2, s2, s7      # Move input base pointer to next row vector
    add     s3, s3, s8      # Move output base pointer to next row vector
    addi    s6, s6, -1      # Decrement timestep countdown
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
    addi    sp, sp, 48
    ret
