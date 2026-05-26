.section .text
.global gelu_inplace
gelu_inplace:
    addi    sp, sp, -48
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    fsw     fs0, 24(sp)
    fsw     fs1, 28(sp)
    fsw     fs2, 32(sp)
    fsw     fs3, 36(sp)
    fsw     fs4, 40(sp)
    fsw     fs5, 44(sp)

    mv      s0, a0          # ptr
    mv      s1, a1          # n

    # Load constants in float callee-saved regs
    lui     t0, %hi(gelu_c1)
    flw     fs3, %lo(gelu_c1)(t0)   # 0.044715
    lui     t0, %hi(gelu_c2)
    flw     fs4, %lo(gelu_c2)(t0)   # sqrt(2/pi)
    lui     t0, %hi(gelu_half)
    flw     fs5, %lo(gelu_half)(t0) # 0.5
    lui     t0, %hi(math_one)
    flw     fs0, %lo(math_one)(t0)  # 1.0
    lui     t0, %hi(math_two)
    flw     fs1, %lo(math_two)(t0)  # 2.0
    lui     t0, %hi(const_pos_clip)
    flw     fs2, %lo(const_pos_clip)(t0)  # 8.0 (for tanh clipping)

gelu_batch:
    li      t0, 8
    blt     s1, t0, gelu_tail   # fewer than 8 left -> scalar tail

    # ---- vector phase 1: compute tanh arguments ----
    vsetvli zero, t0, e32, m1, ta, ma  # VL=8
    vle32.v v0, (s0)                   # v0 = x[0..7]

    vfmul.vv v1, v0, v0                # v1 = x^2
    vfmul.vv v1, v1, v0                # v1 = x^3
    vfmul.vf v1, v1, fs3               # v1 = x^3 * c1
    vfadd.vv v1, v1, v0                # v1 = x + x^3*c1
    vfmul.vf v1, v1, fs4               # v1 = (x + x^3*c1) * c2  (tanh arg)

    # Store tanh args to BSS buffer
    lui     t0, %hi(gelu_tanh_in)
    addi    t0, t0, %lo(gelu_tanh_in)
    vse32.v v1, (t0)
    mv      s2, t0                      # s2 = gelu_tanh_in

    # Prepare output buffer pointer
    lui     t0, %hi(gelu_tanh_out)
    addi    s3, t0, %lo(gelu_tanh_out)  # s3 = gelu_tanh_out
    li      s4, 8                       # counter

    # ---- scalar phase 2: custom tanh using expf (friend's method) ----
gelu_tanh_loop:
    beqz    s4, gelu_tanh_done
    flw     fa0, 0(s2)                  # load tanh argument
    
    # Clip to [-8, 8] like friend's code
    flt.s   t4, fs2, fa0                # if 8.0 < arg
    bnez    t4, tanh_clip_pos
    fneg.s  ft0, fs2                    # -8.0
    flt.s   t4, fa0, ft0                # if arg < -8.0
    bnez    t4, tanh_clip_neg

    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    fmul.s  fa0, fa0, fs1               # 2 * x
    call    expf                        # e^(2x)
    fmv.s   ft0, fa0                    # save e^(2x)
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)      # 1.0
    fsub.s  ft2, ft0, ft1               # e^(2x) - 1
    fadd.s  ft3, ft0, ft1               # e^(2x) + 1
    fdiv.s  fa0, ft2, ft3               # tanh = (e^(2x)-1)/(e^(2x)+1)
    j       tanh_store

tanh_clip_pos:
    fmv.s   fa0, fs0                    # tanh = 1.0
    j       tanh_store

tanh_clip_neg:
    fneg.s  fa0, fs0                    # tanh = -1.0

tanh_store:
    fsw     fa0, 0(s3)
    addi    s2, s2, 4
    addi    s3, s3, 4
    addi    s4, s4, -1
    j       gelu_tanh_loop

gelu_tanh_done:
    # ---- vector phase 3: x * 0.5 * (1 + tanh) ----
    li      t0, 8
    vsetvli zero, t0, e32, m1, ta, ma
    vle32.v v0, (s0)                    # reload original x values
    lui     t0, %hi(gelu_tanh_out)
    addi    t0, t0, %lo(gelu_tanh_out)
    vle32.v v1, (t0)                    # load tanh results

    vfadd.vf v1, v1, fs0                # 1 + tanh
    vfmul.vf v1, v1, fs5                # * 0.5
    vfmul.vv v0, v0, v1                 # x * 0.5 * (1+tanh)
    vse32.v  v0, (s0)                   # store

    addi    s0, s0, 32                  # advance 8*4 bytes
    addi    s1, s1, -8
    j       gelu_batch

# ================================================================
# Scalar tail for remaining < 8 elements (uses same tanh method)
# ================================================================
gelu_tail:
    beqz    s1, gelu_done
    flw     fa0, 0(s0)                  # load x

    # Compute tanh argument: (x + c1*x^3) * c2
    fmul.s  ft0, fa0, fa0
    fmul.s  ft0, ft0, fa0
    fmul.s  ft0, ft0, fs3
    fadd.s  ft0, fa0, ft0
    fmul.s  ft0, ft0, fs4

    # Clip to [-8, 8] like friend's code
    fmv.s   fa0, ft0
    flt.s   t4, fs2, fa0                # if 8.0 < arg
    bnez    t4, tail_tanh_clip_pos
    fneg.s  ft1, fs2                    # -8.0
    flt.s   t4, fa0, ft1                # if arg < -8.0
    bnez    t4, tail_tanh_clip_neg

    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    fmul.s  fa0, fa0, fs1               # 2 * x
    call    expf
    fmv.s   ft0, fa0
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fsub.s  ft2, ft0, ft1
    fadd.s  ft3, ft0, ft1
    fdiv.s  ft0, ft2, ft3
    j       tail_tanh_done

tail_tanh_clip_pos:
    fmv.s   ft0, fs0                    # tanh = 1.0
    j       tail_tanh_done

tail_tanh_clip_neg:
    fneg.s  ft0, fs0                    # tanh = -1.0

tail_tanh_done:
    # GELU: 0.5 * x * (1 + tanh)
    fadd.s  ft0, ft0, fs0               # 1 + tanh
    flw     ft1, 0(s0)                  # reload x
    fmul.s  ft0, ft1, ft0
    fmul.s  ft0, ft0, fs5
    fsw     ft0, 0(s0)

    addi    s0, s0, 4
    addi    s1, s1, -1
    j       gelu_tail

gelu_done:
    flw     fs5, 44(sp)
    flw     fs4, 40(sp)
    flw     fs3, 36(sp)
    flw     fs2, 32(sp)
    flw     fs1, 28(sp)
    flw     fs0, 24(sp)
    lw      s4, 20(sp)
    lw      s3, 16(sp)
    lw      s2, 12(sp)
    lw      s1,  8(sp)
    lw      s0,  4(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 48
    ret

.section .data
.align 2
gelu_c1:        .float 0.044715
gelu_c2:        .float 0.79788456080
gelu_half:      .float 0.5
math_one:       .float 1.0
math_two:       .float 2.0
const_pos_clip: .float 8.0

.section .bss
.align 2
gelu_tanh_in:   .space 32   # 8 float tanh arguments
gelu_tanh_out:  .space 32   # 8 float tanh results