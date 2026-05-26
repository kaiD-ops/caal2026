# =============================================================================
# softmax_vec.s  -  Partially vectorized softmax_inplace
#
# Phase 1 (max):  vectorized with vfredmax.vs (strip-mined)
# Phase 2 (exp):  scalar loop (expf is scalar; subtract max vectorly)
# Phase 3 (norm): vectorized with vfdiv.vf (strip-mined)
#
# void softmax_inplace(float* x, int n)
# a0=x, a1=n
# =============================================================================
.section .text
.global softmax_inplace
softmax_inplace:
    addi    sp, sp, -28
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    fsw     fs0, 20(sp)
    fsw     fs1, 24(sp)

    mv      s0, a0          # base ptr
    mv      s1, a1          # n

    # Phase 1: find max (vectorized)
    flw     fa0, 0(s0)      # initial max = x[0]
    vfmv.s.f v1, fa0        # v1[0] = initial max

    mv      t0, s0
    addi    t0, t0, 4       # ptr = &x[1]
    mv      t1, s1
    addi    t1, t1, -1      # remaining = n-1

sm_max_loop:
    beqz    t1, sm_max_done
    vsetvli a0, t1, e32, m1, ta, ma
    vle32.v v0, (t0)
    vfredmax.vs v1, v0, v1  # v1[0] = max(v1[0], elems of v0)
    sub     t1, t1, a0
    slli    t2, a0, 2
    add     t0, t0, t2
    j       sm_max_loop

sm_max_done:
    vfmv.f.s fs0, v1        # fs0 = max (callee-saved, survives expf calls)

    # Phase 2: exp(x[i] - max), scalar, accumulate sum
    # Use s3 (ptr) and fs1 (sum): both callee-saved, survive call expf
    fmv.w.x fa0, zero
    fmv.s   fs1, fa0        # fs1 = sum = 0.0
    li      s2, 0
    mv      s3, s0          # s3 = ptr (callee-saved, survives expf)

sm_exp_loop:
    bge     s2, s1, sm_exp_done
    flw     fa0, 0(s3)
    fsub.s  fa0, fa0, fs0   # x - max
    call    expf             # fa0 = exp(x-max); s3 and fs1 preserved
    fsw     fa0, 0(s3)
    fadd.s  fs1, fs1, fa0   # sum += exp(x-max)
    addi    s3, s3, 4
    addi    s2, s2, 1
    j       sm_exp_loop

sm_exp_done:
    # Phase 3: divide all by sum (vectorized)
    mv      t0, s0
    mv      t1, s1

sm_norm_loop:
    beqz    t1, sm_norm_done
    vsetvli a0, t1, e32, m1, ta, ma
    vle32.v v0, (t0)
    vfdiv.vf v0, v0, fs1    # divide by sum (fs1 = scalar sum)
    vse32.v v0, (t0)
    sub     t1, t1, a0
    slli    t2, a0, 2
    add     t0, t0, t2
    j       sm_norm_loop

sm_norm_done:
    flw     fs1, 24(sp)
    flw     fs0, 20(sp)
    lw      s3, 16(sp)
    lw      s2, 12(sp)
    lw      s1,  8(sp)
    lw      s0,  4(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 28
    ret
