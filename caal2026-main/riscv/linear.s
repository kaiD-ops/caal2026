# =============================================================================
# linear.s - Linear (fully-connected) layer
# Y = X*W^T + b, handles both sequence and vector inputs
# void linear_layer(float* W, float* b, float* in, float* out,
#                   int in_dim, int out_dim, int seq_len)
# a0=W, a1=b, a2=in, a3=out, a4=in_dim, a5=out_dim, a6=seq_len
# =============================================================================
.section .text
.global linear_layer
linear_layer:
    addi    sp, sp, -28
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mv      s6, a6
ll_t:
    beqz    s6, ll_done
    mv      t3, s5
    mv      t4, s1
    mv      t5, s3
ll_o:
    beqz    t3, ll_t_next
    flw     fa0, 0(t4)
    sub     t6, s5, t3
    mul     t6, t6, s4
    slli    t6, t6, 2
    add     t6, s0, t6
    mv      t0, s2
    mv      t1, s4
ll_i:
    beqz    t1, ll_i_done
    flw     fa1, 0(t6)
    flw     fa2, 0(t0)
    fmadd.s fa0, fa1, fa2, fa0
    addi    t6, t6, 4
    addi    t0, t0, 4
    addi    t1, t1, -1
    j       ll_i
ll_i_done:
    fsw     fa0, 0(t5)
    addi    t4, t4, 4
    addi    t5, t5, 4
    addi    t3, t3, -1
    j       ll_o
ll_t_next:
    slli    t0, s4, 2
    add     s2, s2, t0
    slli    t0, s5, 2
    add     s3, s3, t0
    addi    s6, s6, -1
    j       ll_t
ll_done:
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 28
    ret
