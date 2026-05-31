
.section .text
.global gelu_inplace_vec

gelu_inplace_vec:
    addi    sp, sp, -16
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    mv      s0, a0
    mv      s1, a1

    # Constants
    lui     t0, %hi(gelu_vec_c1)
    flw     fs2, %lo(gelu_vec_c1)(t0)   # 0.044715
    lui     t0, %hi(gelu_vec_c2)
    flw     fs3, %lo(gelu_vec_c2)(t0)   # sqrt(2/pi)
    lui     t0, %hi(gelu_vec_half)
    flw     fs4, %lo(gelu_vec_half)(t0) # 0.5

gelu_vec_loop:
    beqz    s1, gelu_vec_done

    flw     fa0, 0(s0)

    # inner = x + 0.044715 * x^3
    fmul.s  ft0, fa0, fa0
    fmul.s  ft0, ft0, fa0
    fmul.s  ft0, ft0, fs2
    fadd.s  ft0, fa0, ft0

    # arg = sqrt(2/pi) * inner
    fmul.s  fa1, ft0, fs3

    # save x
    fmv.s   fs1, fa0

    # tanh(arg)
    fmv.s   fa0, fa1
    call    tanhf

    # result = 0.5 * x * (1 + tanh)
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fadd.s  fa0, fa0, ft1
    fmul.s  fa0, fs1, fa0
    fmul.s  fa0, fa0, fs4

    fsw     fa0, 0(s0)
    addi    s0, s0, 4
    addi    s1, s1, -1
    j       gelu_vec_loop

gelu_vec_done:
    lw      s2, 12(sp)
    lw      s1,  8(sp)
    lw      s0,  4(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 16
    ret

.section .data
.align 2
gelu_vec_c1:   .float 0.044715
gelu_vec_c2:   .float 0.79788456080
gelu_vec_half: .float 0.5
