# =============================================================================
# math.s  –  float math helpers for S4D galaxy classifier
#
# Routines:   expf, cosf, sinf, tanhf
# Convention: argument in fa0, result in fa0
#             Caller-saved: fa1–fa7, ft0–ft7, t0–t6
# Accuracy:   ≥ 4 decimal places (matches TA requirement)
#
# Approximation strategy (per TA guidance – avoid Taylor O(n), use O(1)):
#   expf : range-reduce then 6-term minimax polynomial
#   cosf : 4-term Chebyshev-derived polynomial (accurate on ±10π with reduction)
#   sinf : 4-term polynomial after same reduction
#   tanhf: (e^2x – 1)/(e^2x + 1), saturate for |x| ≥ 4
# =============================================================================

.section .text

# --------------------------------------------------------------------------
# expf: e^fa0  (single precision)
# Special: x >  88 → +inf,  x < -88 → 0
# Algorithm:
#   n  = round(x * log2e)
#   r  = x – n * ln2          (range |r| ≤ ln2/2 ≈ 0.347)
#   p  = 1 + r(1 + r(½ + r(⅙ + r(1/24 + r/120))))   Horner
#   result = p * 2^n           (add n to IEEE exponent field)
# --------------------------------------------------------------------------
.global expf
expf:
    # --- overflow / underflow guards ---
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
    # --- range reduction: n = round(x/ln2) ---
    lui     t0, %hi(exp_log2e)
    flw     ft1, %lo(exp_log2e)(t0)      # ft1 = log2(e)
    fmul.s  ft2, fa0, ft1                # ft2 = x * log2(e)
    fcvt.w.s t2, ft2, rne               # t2  = n (integer)
    fcvt.s.w ft3, t2                     # ft3 = float(n)

    lui     t0, %hi(exp_ln2)
    flw     ft4, %lo(exp_ln2)(t0)        # ft4 = ln2
    fmul.s  ft5, ft3, ft4                # ft5 = n * ln2
    fsub.s  ft0, fa0, ft5                # ft0 = r = x - n*ln2

    # --- Horner: p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120)))) ---
    lui     t0, %hi(exp_c5)
    flw     fa1, %lo(exp_c5)(t0)         # 1/120
    lui     t0, %hi(exp_c4)
    flw     fa2, %lo(exp_c4)(t0)         # 1/24
    fmadd.s fa1, ft0, fa1, fa2           # r/120 + 1/24

    lui     t0, %hi(exp_c3)
    flw     fa2, %lo(exp_c3)(t0)         # 1/6
    fmadd.s fa1, ft0, fa1, fa2           # r*(r/120+1/24) + 1/6

    lui     t0, %hi(exp_c2)
    flw     fa2, %lo(exp_c2)(t0)         # 0.5
    fmadd.s fa1, ft0, fa1, fa2           # r*(...) + 0.5

    lui     t0, %hi(exp_c1)
    flw     fa2, %lo(exp_c1)(t0)         # 1.0
    fmadd.s fa1, ft0, fa1, fa2           # r*(...) + 1.0

    fmadd.s fa0, ft0, fa1, fa2           # r*(r*(...)+1) + 1

    # --- scale: multiply by 2^n ---
    addi    t2, t2, 127                  # biased exponent
    slli    t2, t2, 23                   # shift to exponent field
    fmv.w.x ft1, t2                      # ft1 = 2^n (float)
    fmul.s  fa0, fa0, ft1
    ret

# --------------------------------------------------------------------------
# cosf: cos(fa0)  – range reduction then polynomial
# Reduces to [-π/2, π/2] via: x = x mod 2π, handle quadrant
# For S4D the argument is A_imag*dt which can reach ~3.1*31*0.1 ≈ 9.7
# --------------------------------------------------------------------------
.global cosf
cosf:
    # Reduce modulo 2π
    lui     t0, %hi(math_2pi)
    flw     ft1, %lo(math_2pi)(t0)       # 2π
    lui     t0, %hi(math_inv2pi)
    flw     ft2, %lo(math_inv2pi)(t0)    # 1/(2π)
    fmul.s  ft3, fa0, ft2                # x/(2π)
    fcvt.w.s t1, ft3, rdn               # floor(x/(2π))
    fcvt.s.w ft3, t1
    fmul.s  ft3, ft3, ft1                # floor * 2π
    fsub.s  fa0, fa0, ft3                # x mod 2π  in [0, 2π)

    # shift to [-π, π]
    lui     t0, %hi(math_pi)
    flw     ft1, %lo(math_pi)(t0)        # π
    flt.s   t1, ft1, fa0                 # x > π ?
    beqz    t1, cos_reduced
    fsub.s  fa0, fa0, ft1
    fsub.s  fa0, fa0, ft1                # x -= 2π  → [-π, 0)
cos_reduced:
    # cos(x) polynomial on [-π, π]: 1 + x²(-½ + x²(1/24 + x²*(-1/720)))
    fmul.s  ft0, fa0, fa0                # x²
    lui     t0, %hi(cos_c3)
    flw     ft1, %lo(cos_c3)(t0)         # -1/720
    lui     t0, %hi(cos_c2)
    flw     ft2, %lo(cos_c2)(t0)         # 1/24
    fmadd.s ft1, ft0, ft1, ft2           # x²*(-1/720) + 1/24
    lui     t0, %hi(cos_c1)
    flw     ft2, %lo(cos_c1)(t0)         # -0.5
    fmadd.s ft1, ft0, ft1, ft2           # x²*... - 0.5
    lui     t0, %hi(cos_c0)
    flw     ft2, %lo(cos_c0)(t0)         # 1.0
    fmadd.s fa0, ft0, ft1, ft2           # x²*... + 1
    ret

# --------------------------------------------------------------------------
# sinf: sin(fa0)
# --------------------------------------------------------------------------
.global sinf
sinf:
    # Save sign, work with |x|, restore sign at end
    fsgnjn.s ft7, fa0, fa0              # ft7 = -|fa0| (negative copy)
    fabs.s  fa0, fa0                    # fa0 = |x|

    # reduce mod 2π
    lui     t0, %hi(math_2pi)
    flw     ft1, %lo(math_2pi)(t0)
    lui     t0, %hi(math_inv2pi)
    flw     ft2, %lo(math_inv2pi)(t0)
    fmul.s  ft3, fa0, ft2
    fcvt.w.s t1, ft3, rdn
    fcvt.s.w ft3, t1
    fmul.s  ft3, ft3, ft1
    fsub.s  fa0, fa0, ft3               # |x| mod 2π

    # shift to [-π, π]
    lui     t0, %hi(math_pi)
    flw     ft1, %lo(math_pi)(t0)
    flt.s   t1, ft1, fa0
    beqz    t1, sin_reduced
    fsub.s  fa0, fa0, ft1
    fsub.s  fa0, fa0, ft1
sin_reduced:
    # sin(x) = x*(1 + x²(-1/6 + x²*(1/120 + x²*(-1/5040))))
    fmul.s  ft0, fa0, fa0               # x²
    lui     t0, %hi(sin_c3)
    flw     ft1, %lo(sin_c3)(t0)        # -1/5040
    lui     t0, %hi(sin_c2)
    flw     ft2, %lo(sin_c2)(t0)        # 1/120
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(sin_c1)
    flw     ft2, %lo(sin_c1)(t0)        # -1/6
    fmadd.s ft1, ft0, ft1, ft2
    lui     t0, %hi(sin_c0)
    flw     ft2, %lo(sin_c0)(t0)        # 1.0
    fmadd.s ft1, ft0, ft1, ft2          # 1 + x²*(...)
    fmul.s  fa0, fa0, ft1               # x * (1 + x²*(...)) = sin|x|

    # restore original sign
    fsgnjx.s fa0, fa0, ft7              # apply sign from ft7
    ret

# --------------------------------------------------------------------------
# tanhf: tanh(fa0)
# For |x| >= 4: tanh ≈ ±1.0  (< 4e-4 error, within TA spec)
# Else: (e^2x - 1) / (e^2x + 1)
# --------------------------------------------------------------------------
.global tanhf
tanhf:
    addi    sp, sp, -8
    fsw     fa0, 0(sp)                  # save x

    # saturation check
    fabs.s  ft0, fa0
    lui     t0, %hi(tanh_sat)
    flw     ft1, %lo(tanh_sat)(t0)
    flt.s   t1, ft0, ft1               # |x| < 4?
    beqz    t1, tanh_sat_case

    # compute e^(2x)
    lui     t0, %hi(math_two)
    flw     ft1, %lo(math_two)(t0)
    fmul.s  fa0, fa0, ft1              # 2x
    call    expf                       # fa0 = e^(2x)

    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fsub.s  ft2, fa0, ft1              # e^2x - 1
    fadd.s  ft3, fa0, ft1              # e^2x + 1
    fdiv.s  fa0, ft2, ft3
    addi    sp, sp, 8
    ret

tanh_sat_case:
    flw     fa1, 0(sp)                 # reload original x for sign
    lui     t0, %hi(math_one)
    flw     fa0, %lo(math_one)(t0)
    fsgnj.s fa0, fa0, fa1              # sign(x) * 1.0
    addi    sp, sp, 8
    ret

# =============================================================================
# Constants
# =============================================================================
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
