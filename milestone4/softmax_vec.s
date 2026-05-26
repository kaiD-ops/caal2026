# =============================================================================
# softmax_vec.s  –  Softmax activation  (scalar implementation retained)
#
# VECTORIZATION DECISION:  Softmax is NOT vectorized for this milestone.
#
# Justification:
#   The softmax operates on only N=4 elements (one per galaxy class).
#   The RISC-V Vector Extension has a minimum vector length of 4 elements
#   for e32, so at most one vsetvli / vle32 / vse32 trip would execute —
#   identical overhead to 4 scalar loads/stores.  Furthermore:
#     • The reduction (max-finding, sum) requires two full passes and two
#       fredumax/vfredusum calls whose scalar overhead dominates at N=4.
#     • exp() must be called per element regardless; four scalar calls
#       are not slower than four calls inside a strip-mined vector loop.
#     • The dynamic instruction count contribution of softmax is negligible
#       compared to the S4D and linear layers (<0.001% of total).
#   Therefore scalar softmax is retained and the implementation effort is
#   directed at the actual hotspots (S4D recurrence, linear projection).
#
# Signature (unchanged from M3):
#   void softmax_inplace(float* x, int n)
#   a0 = x (in-place),  a1 = n (element count, = 4 here)
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

    # ── Find maximum ────────────────────────────────────────────────────────
    flw     fa0, 0(s0)      # max = x[0]
    li      t0, 1
sm_max_loop:
    bge     t0, s1, sm_max_done
    slli    t1, t0, 2
    add     t1, s0, t1
    flw     ft0, 0(t1)
    flt.s   t2, fa0, ft0
    beqz    t2, sm_max_skip
    fmv.s   fa0, ft0
sm_max_skip:
    addi    t0, t0, 1
    j       sm_max_loop
sm_max_done:
    fmv.s   fs0, fa0        # fs0 = max

    # ── exp(x - max) and accumulate sum ─────────────────────────────────────
    fmv.w.x fa3, zero       # sum = 0
    li      s2, 0
    mv      s3, s0
sm_exp_loop:
    bge     s2, s1, sm_exp_done
    flw     fa0, 0(s3)
    fsub.s  fa0, fa0, fs0
    call    expf
    fsw     fa0, 0(s3)
    fadd.s  fa3, fa3, fa0
    addi    s3, s3, 4
    addi    s2, s2, 1
    j       sm_exp_loop
sm_exp_done:

    # ── Normalize ────────────────────────────────────────────────────────────
    li      s2, 0
    mv      s3, s0
sm_norm_loop:
    bge     s2, s1, sm_norm_done
    flw     fa0, 0(s3)
    fdiv.s  fa0, fa0, fa3
    fsw     fa0, 0(s3)
    addi    s3, s3, 4
    addi    s2, s2, 1
    j       sm_norm_loop
sm_norm_done:

    lw      ra,  0(sp)
    lw      s0,  4(sp)
    lw      s1,  8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    addi    sp, sp, 20
    ret
