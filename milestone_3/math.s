# ============================================================
# math_simple.s - Working FPU math library for your simulator
# All float args in fa0-fa3, float return in fa0
# ============================================================
.section .text

# ─── exp_f: Simplified exponential approximation ─────────
.global exp_f
exp_f:
    # Clamp input to [-2, 2] for stability
    li    t0, 0x40000000    # 2.0
    fmv.w.x ft0, t0
    fmin.s fa0, fa0, ft0
    
    li    t0, 0xc0000000    # -2.0
    fmv.w.x ft0, t0
    fmax.s fa0, fa0, ft0
    
    # result = 1.0
    li    t0, 0x3f800000
    fmv.w.x ft0, t0
    
    # result += x
    fadd.s ft0, ft0, fa0
    
    # x²/2
    fmul.s ft1, fa0, fa0    # x²
    li    t0, 0x3f000000    # 0.5
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2    # x²/2
    fadd.s ft0, ft0, ft1    # result += x²/2
    
    # x³/6
    fmul.s ft1, ft1, fa0    # x³/2
    li    t0, 0x3e2aaaab    # 1/3 ≈ 0.33333334
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2    # x³/6
    fadd.s fa0, ft0, ft1    # result += x³/6
    
    ret

# ─── sin_f: Sine approximation (Taylor series) ───────────
.global sin_f
sin_f:
    fmv.s ft0, fa0           # x
    fmul.s ft1, fa0, fa0     # x²
    fmul.s ft2, ft1, fa0     # x³
    
    li    t0, 0x3e2aaaab     # 1/6 ≈ 0.16666667
    fmv.w.x ft3, t0
    fmul.s ft2, ft2, ft3     # x³/6
    
    fsub.s fa0, ft0, ft2     # x - x³/6
    ret

# ─── cos_f: Cosine approximation (Taylor series) ─────────
.global cos_f
cos_f:
    li    t0, 0x3f800000     # 1.0
    fmv.w.x ft0, t0
    
    fmul.s ft1, fa0, fa0     # x²
    li    t0, 0x3f000000     # 0.5
    fmv.w.x ft2, t0
    fmul.s ft1, ft1, ft2     # x²/2
    
    fsub.s fa0, ft0, ft1     # 1 - x²/2
    ret

# ─── complex_mul ─────────────────────────────────────────
.global complex_mul
complex_mul:
    fmul.s ft0, fa0, fa2     # re1*re2
    fmul.s ft1, fa1, fa3     # im1*im2
    fsub.s ft2, ft0, ft1     # re_out
    fmul.s ft0, fa0, fa3     # re1*im2
    fmul.s ft1, fa1, fa2     # im1*re2
    fadd.s ft3, ft0, ft1     # im_out
    fmv.s fa0, ft2
    fmv.s fa1, ft3
    ret

# ─── tanh_f: tanh(x) using exp_f ─────────────────────────
.global tanh_f
tanh_f:
    addi  sp, sp, -16
    sw    ra, 12(sp)
    
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    fadd.s fa0, fa0, fa0     # 2x
    call  exp_f
    li    t0, 0x3f800000     # 1.0
    fmv.w.x ft0, t0
    fsub.s fa1, fa0, ft0
    fadd.s fa2, fa0, ft0
    fdiv.s fa0, fa1, fa2
    
    lw    ra, 12(sp)
    addi  sp, sp, 16
    ret
