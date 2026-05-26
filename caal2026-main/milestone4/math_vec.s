# =============================================================================
# math_vec.s  -  Scalar float math helpers (same as M3; transcendentals
#                cannot be meaningfully vectorized with per-element accuracy)
# Routines: expf, cosf, sinf, tanhf
# =============================================================================

.section .text

.global expf
expf:
    lui     t0, %hi(exp_max)
    flw     ft1, %lo(exp_max)(t0)
    flt.s   t1, ft1, fa0
    beqz    t1, 1f
    lui     t0, %hi(exp_inf_bits)
    lw      t1, %lo(exp_inf_bits)(t0)
    fmv.w.x fa0, t1
    ret
1:  lui     t0, %hi(exp_min)
    flw     ft1, %lo(exp_min)(t0)
    flt.s   t1, fa0, ft1
    beqz    t1, 2f
    fmv.w.x fa0, zero
    ret
2:
    lui     t0, %hi(exp_log2e)
    flw     ft1, %lo(exp_log2e)(t0)
    fmul.s  ft2, fa0, ft1
    fcvt.w.s t2, ft2, rne
    fcvt.s.w ft3, t2

    lui     t0, %hi(exp_ln2)
    flw     ft4, %lo(exp_ln2)(t0)
    fmul.s  ft5, ft3, ft4
    fsub.s  ft0, fa0, ft5

    lui     t0, %hi(exp_c5)
    flw     fa1, %lo(exp_c5)(t0)
    lui     t0, %hi(exp_c4)
    flw     fa2, %lo(exp_c4)(t0)
    fmadd.s fa1, ft0, fa1, fa2

    lui     t0, %hi(exp_c3)
    flw     fa2, %lo(exp_c3)(t0)
    fmadd.s fa1, ft0, fa1, fa2

    lui     t0, %hi(exp_c2)
    flw     fa2, %lo(exp_c2)(t0)
    fmadd.s fa1, ft0, fa1, fa2

    lui     t0, %hi(exp_c1)
    flw     fa2, %lo(exp_c1)(t0)
    fmadd.s fa1, ft0, fa1, fa2

    fmadd.s fa0, ft0, fa1, fa2

    addi    t2, t2, 127
    slli    t2, t2, 23
    fmv.w.x ft1, t2
    fmul.s  fa0, fa0, ft1
    ret

.global cosf
cosf:
    lui     t0, %hi(math_2pi)
    flw     ft1, %lo(math_2pi)(t0)
    lui     t0, %hi(math_inv2pi)
    flw     ft2, %lo(math_inv2pi)(t0)
    fmul.s  ft3, fa0, ft2
    fcvt.w.s t1, ft3, rdn
    fcvt.s.w ft3, t1
    fmul.s  ft3, ft3, ft1
    fsub.s  fa0, fa0, ft3

    lui     t0, %hi(math_pi)
    flw     ft1, %lo(math_pi)(t0)
    flt.s   t1, ft1, fa0
    beqz    t1, cos_reduced
    fsub.s  fa0, fa0, ft1
    fsub.s  fa0, fa0, ft1
cos_reduced:
    fmul.s  ft0, fa0, fa0
    lui     t0, %hi(cos_c3)
    flw     ft1, %lo(cos_c3)(t0)
    lui     t0, %hi(cos_c2)
    flw     ft2, %lo(cos_c2)(t0)
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(cos_c1)
    flw     ft2, %lo(cos_c1)(t0)
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(cos_c0)
    flw     ft2, %lo(cos_c0)(t0)
    fmadd.s fa0, ft0, ft1, ft2
    ret

.global sinf
sinf:
    fsgnjn.s ft7, fa0, fa0
    fabs.s  fa0, fa0

    lui     t0, %hi(math_2pi)
    flw     ft1, %lo(math_2pi)(t0)
    lui     t0, %hi(math_inv2pi)
    flw     ft2, %lo(math_inv2pi)(t0)
    fmul.s  ft3, fa0, ft2
    fcvt.w.s t1, ft3, rdn
    fcvt.s.w ft3, t1
    fmul.s  ft3, ft3, ft1
    fsub.s  fa0, fa0, ft3

    lui     t0, %hi(math_pi)
    flw     ft1, %lo(math_pi)(t0)
    flt.s   t1, ft1, fa0
    beqz    t1, sin_reduced
    fsub.s  fa0, fa0, ft1
    fsub.s  fa0, fa0, ft1
sin_reduced:
    fmul.s  ft0, fa0, fa0
    lui     t0, %hi(sin_c3)
    flw     ft1, %lo(sin_c3)(t0)
    lui     t0, %hi(sin_c2)
    flw     ft2, %lo(sin_c2)(t0)
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(sin_c1)
    flw     ft2, %lo(sin_c1)(t0)
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(sin_c0)
    flw     ft2, %lo(sin_c0)(t0)
    fmadd.s ft1, ft0, ft1, ft2
    fmul.s  fa0, fa0, ft1

    fsgnjx.s fa0, fa0, ft7
    ret

.global tanhf
tanhf:
    addi    sp, sp, -12
    sw      ra, 0(sp)
    fsw     fs1, 4(sp)
    fsw     fs2, 8(sp)
    fmv.s   fs2, fa0
    fabs.s  ft0, fa0
    lui     t0, %hi(tanh_sat)
    flw     ft1, %lo(tanh_sat)(t0)
    flt.s   t1, ft0, ft1
    beqz    t1, tanh_sat_case
    lui     t0, %hi(math_two)
    flw     ft0, %lo(math_two)(t0)
    fmul.s  fa0, fs2, ft0
    call    expf
    fmv.s   fs1, fa0
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  fa0, fs1, ft0
    fadd.s  ft1, fs1, ft0
    fdiv.s  fa0, fa0, ft1
    j       tanh_done
tanh_sat_case:
    lui     t0, %hi(math_one)
    flw     fa0, %lo(math_one)(t0)
    fsgnj.s fa0, fa0, fs2
tanh_done:
    flw     fs2, 8(sp)
    flw     fs1, 4(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 12
    ret

.section .data
.align 2

exp_max:       .float  88.0
exp_min:       .float -88.0
exp_inf_bits:  .word   0x7f800000
exp_log2e:     .float  1.44269504089
exp_ln2:       .float  0.69314718056
exp_c1:        .float  1.0
exp_c2:        .float  0.5
exp_c3:        .float  0.16666667
exp_c4:        .float  0.04166667
exp_c5:        .float  0.00833333

cos_c0:        .float  1.0
cos_c1:        .float -0.5
cos_c2:        .float  0.04166667
cos_c3:        .float -0.00138889

sin_c0:        .float  1.0
sin_c1:        .float -0.16666667
sin_c2:        .float  0.00833333
sin_c3:        .float -0.00019841

.global math_one
.global math_two
.global math_pi
math_pi:       .float  3.14159265359
math_2pi:      .float  6.28318530718
math_inv2pi:   .float  0.15915494309
math_one:      .float  1.0
math_two:      .float  2.0

tanh_sat:      .float  4.0
