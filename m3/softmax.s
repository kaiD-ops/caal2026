# =============================================================================
# softmax.s - Softmax activation
#
# In-place softmax: x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
# Uses numerically stable formulation (subtract max before exp).
# Calls expf from math.s.
#
# void softmax_inplace(float* x, int n)
# a0=x, a1=n
# MSE target: < 1e-8
# =============================================================================
.section .text
.global softmax_inplace

softmax_inplace:
    addi    sp, sp, -20
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    mv      s0, a0          # base ptr
    mv      s1, a1          # n

    # --- Find max ---
    flw     fa0, 0(s0)      # max = x[0]
    li      t0, 1
sm_max:
    bge     t0, s1, sm_max_done
    slli    t1, t0, 2
    add     t1, s0, t1
    flw     ft0, 0(t1)
    flt.s   t2, fa0, ft0
    beqz    t2, sm_max_skip
    fmv.s   fa0, ft0
sm_max_skip:
    addi    t0, t0, 1
    j       sm_max
sm_max_done:
    fmv.s   fs0, fa0        # fs0 = max (callee-saved)
    fmv.w.x fa3, zero       # sum = 0.0

    # --- Compute exp(x[i] - max) in-place, accumulate sum ---
    li      s2, 0
sm_exp:
    bge     s2, s1, sm_exp_done
    slli    t0, s2, 2
    add     s3, s0, t0      # s3 = &x[s2] (callee-saved)
    flw     fa0, 0(s3)
    fsub.s  fa0, fa0, fs0   # x[i] - max
    call    expf
    fsw     fa0, 0(s3)      # x[i] = exp(x[i]-max)
    fadd.s  fa3, fa3, fa0   # sum += x[i]
    addi    s2, s2, 1
    j       sm_exp
sm_exp_done:

    # --- Normalize ---
    li      s2, 0
sm_norm:
    bge     s2, s1, sm_norm_done
    slli    t0, s2, 2
    add     t0, s0, t0
    flw     fa0, 0(t0)
    fdiv.s  fa0, fa0, fa3
    fsw     fa0, 0(t0)
    addi    s2, s2, 1
    j       sm_norm
sm_norm_done:
    lw      s0,  4(sp)
    lw      s1,  8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 20
    ret
