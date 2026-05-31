
.section .text
.global linear_layer

linear_layer:
    addi    sp, sp, -28
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    mv      s0, a0          # W base
    mv      s1, a1          # b base
    mv      s2, a2          # input base
    mv      s3, a3          # output base
    mv      s4, a4          # in_dim
    mv      s5, a5          # out_dim
    mv      s6, a6          # seq_len

ll_t:
    beqz    s6, ll_done
    mv      t3, s5          # output neuron counter
    mv      t4, s1          # bias pointer (reset per timestep)
    mv      t5, s3          # output row pointer

ll_o:
    beqz    t3, ll_t_next

    flw     fa0, 0(t4)      # fa0 = bias[out_neuron]

    # W row for this output neuron = W + (out_idx) * in_dim * 4
    sub     t6, s5, t3      # out_idx = s5 - t3
    mul     t6, t6, s4      # out_idx * in_dim
    slli    t6, t6, 2       # * 4 bytes
    add     t6, s0, t6      # W row pointer

    mv      t0, s2          # input row pointer
    mv      t1, s4          # in_dim counter

ll_i:
    beqz    t1, ll_i_done
    flw     fa1, 0(t6)      # W[out][in]
    flw     fa2, 0(t0)      # X[in]
    fmadd.s fa0, fa1, fa2, fa0
    addi    t6, t6, 4
    addi    t0, t0, 4
    addi    t1, t1, -1
    j       ll_i

ll_i_done:
    fsw     fa0, 0(t5)      # out[out_neuron] = dot + bias
    addi    t4, t4, 4       # next bias
    addi    t5, t5, 4       # next output slot
    addi    t3, t3, -1
    j       ll_o

ll_t_next:
    slli    t0, s4, 2       # in_dim * 4
    add     s2, s2, t0      # next input row
    slli    t0, s5, 2       # out_dim * 4
    add     s3, s3, t0      # next output row
    addi    s6, s6, -1
    j       ll_t

ll_done:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 28
    ret
