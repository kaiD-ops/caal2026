.section .text
.global gelu_inplace
gelu_inplace:
    addi    sp, sp, -16
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)
    sw      s2, 12(sp)
    mv      s0, a0          # ptr (callee-saved)
    mv      s1, a1          # n (callee-saved)
gelu_loop:
    beqz    s1, gelu_done
    flw     fa0, 0(s0)
    # save x in fs1 (callee-saved float reg)
    fmv.s   fs1, fa0
    fmul.s  ft0, fa0, fa0
    fmul.s  ft0, ft0, fa0
    lui     t0, %hi(gelu_c1)
    flw     ft1, %lo(gelu_c1)(t0)
    fmul.s  ft0, ft0, ft1
    fadd.s  ft0, fa0, ft0
    lui     t0, %hi(gelu_c2)
    flw     ft1, %lo(gelu_c2)(t0)
    fmul.s  fa0, ft0, ft1
    call    tanhf           # fa0 = tanh(inner), s0/s1/fs1 preserved
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fadd.s  fa0, fa0, ft1   # 1 + tanh
    fmul.s  fa0, fs1, fa0   # x * (1+tanh)  using saved x in fs1
    lui     t0, %hi(gelu_half)
    flw     ft1, %lo(gelu_half)(t0)
    fmul.s  fa0, fa0, ft1   # * 0.5
    fsw     fa0, 0(s0)      # s0 still valid!
    addi    s0, s0, 4
    addi    s1, s1, -1
    j       gelu_loop
gelu_done:
    lw      s0, 4(sp)
    lw      s1, 8(sp)
    lw      s2, 12(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 16
    ret

.section .data
.align 2
gelu_c1:   .float 0.044715
gelu_c2:   .float 0.79788456080
gelu_half: .float 0.5
