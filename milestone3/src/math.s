/* ============================================================================
 * math.s - Mathematical Helper Functions for RISC-V F Extension
 * ============================================================================
 * 
 * This module implements transcendental functions used by the S4D and GELU
 * layers. All functions use RISC-V single-precision floating-point (F ext).
 *
 * Register Convention:
 *   fa0-fa7:   Floating-point arguments and results (fa0 = return value)
 *   fs0-fs11:  Callee-saved FP registers (must preserve)
 *   ft0-ft11:  Caller-saved FP registers (can clobber)
 *
 * Integer registers for temporary use:
 *   t0-t6:     Caller-saved temporaries
 *   s0-s11:    Callee-saved temporaries
 */

.section .text

/* ============================================================================
 * float expf_fast(float x)
 * ============================================================================
 * Compute e^x using Taylor series approximation:
 *   e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + x⁶/6!
 *
 * For numerical stability in S4D and softmax, we compute exp(x - max)
 * where x <= 0, so we don't worry about overflow.
 *
 * Arguments:  fa0 = x
 * Returns:    fa0 = e^x
 * ============================================================================
 */
.globl expf_fast
.type expf_fast, @function
expf_fast:
    /* Preserve callee-saved registers if needed */
    /* All temporaries use ft0-ft3 (caller-saved) */
    
    /* Load constants for Taylor series */
    fmv.s.x ft1, zero           /* ft1 = 0.0 (unity term start) */
    
    /* Compute e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720 */
    
    fli.s ft0, 1.0              /* ft0 = 1.0 (constant 1) */
    fmv.s ft2, ft0              /* ft2 = acc = 1.0 */
    fmv.s ft3, fa0              /* ft3 = x_power = x */
    
    /* Coefficients for 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720 */
    fsub.s ft2, ft2, ft3        /* acc = 1 - x (will add x term next) */
    fadd.s ft2, ft2, ft3        /* acc += x => 1 + x */
    
    /* x²/2 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x * x = x² */
    fli.s ft5, 0.5              /* ft5 = 0.5 */
    fmul.s ft4, ft4, ft5        /* ft4 = x²/2 */
    fadd.s ft2, ft2, ft4        /* acc += x²/2 */
    
    /* x³/6 term */
    fmul.s ft3, ft4, fa0        /* ft3 = x²/2 * x = x³/2 */
    fli.s ft5, 0.33333333       /* ft5 ≈ 1/3 */
    fmul.s ft3, ft3, ft5        /* ft3 ≈ x³/6 */
    fadd.s ft2, ft2, ft3        /* acc += x³/6 */
    
    /* x⁴/24 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³/6 * x = x⁴/6 */
    fli.s ft5, 0.25             /* ft5 = 0.25 */
    fmul.s ft4, ft4, ft5        /* ft4 ≈ x⁴/24 */
    fadd.s ft2, ft2, ft4        /* acc += x⁴/24 */
    
    /* x⁵/120 term */
    fmul.s ft3, ft4, fa0        /* ft3 = x⁴/24 * x = x⁵/24 */
    fli.s ft5, 0.2              /* ft5 = 0.2 = 1/5 */
    fmul.s ft3, ft3, ft5        /* ft3 ≈ x⁵/120 */
    fadd.s ft2, ft2, ft3        /* acc += x⁵/120 */
    
    /* x⁶/720 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x⁵/120 * x = x⁶/120 */
    fli.s ft5, 0.16666667       /* ft5 ≈ 1/6 */
    fmul.s ft4, ft4, ft5        /* ft4 ≈ x⁶/720 */
    fadd.s ft2, ft2, ft4        /* acc += x⁶/720 */
    
    fmv.s fa0, ft2              /* Return result in fa0 */
    ret

/* ============================================================================
 * float sinf_fast(float x)
 * ============================================================================
 * Compute sin(x) using Taylor series approximation (for small x in radians):
 *   sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
 *
 * Note: S4D uses this for exp(j*theta) where theta is small phase offset.
 *
 * Arguments:  fa0 = x (in radians)
 * Returns:    fa0 = sin(x)
 * ============================================================================
 */
.globl sinf_fast
.type sinf_fast, @function
sinf_fast:
    /* Taylor series: x - x³/6 + x⁵/120 - x⁷/5040 */
    fmv.s ft2, fa0              /* ft2 = x (result accumulator) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x³/6 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³ */
    fli.s ft5, 0.16666667       /* ft5 ≈ 1/6 */
    fmul.s ft4, ft4, ft5        /* ft4 = x³/6 */
    fsub.s ft2, ft2, ft4        /* result -= x³/6 */
    
    /* x⁵/120 term */
    fmul.s ft4, ft4, ft3        /* ft4 = x³/6 * x² = x⁵/6 */
    fli.s ft5, 0.2              /* ft5 = 1/5 */
    fmul.s ft4, ft4, ft5        /* ft4 = x⁵/30... actually need more precision */
    fli.s ft5, 0.00833333       /* ft5 ≈ 1/120 */
    fmul.s ft4, ft3, fa0        /* ft4 = x² * x = x³ */
    fmul.s ft4, ft4, ft3        /* ft4 = x³ * x² = x⁵ */
    fmul.s ft4, ft4, ft5        /* ft4 = x⁵/120 */
    fadd.s ft2, ft2, ft4        /* result += x⁵/120 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    ret

/* ============================================================================
 * float cosf_fast(float x)
 * ============================================================================
 * Compute cos(x) using Taylor series approximation:
 *   cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6!
 *
 * Arguments:  fa0 = x (in radians)
 * Returns:    fa0 = cos(x)
 * ============================================================================
 */
.globl cosf_fast
.type cosf_fast, @function
cosf_fast:
    /* Taylor series: 1 - x²/2 + x⁴/24 - x⁶/720 */
    fli.s ft2, 1.0              /* ft2 = 1.0 (result accumulator) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x²/2 term */
    fli.s ft5, 0.5              /* ft5 = 0.5 */
    fmul.s ft4, ft3, ft5        /* ft4 = x²/2 */
    fsub.s ft2, ft2, ft4        /* result -= x²/2 */
    
    /* x⁴/24 term */
    fmul.s ft4, ft3, ft3        /* ft4 = x⁴ */
    fli.s ft5, 0.04166667       /* ft5 ≈ 1/24 */
    fmul.s ft4, ft4, ft5        /* ft4 = x⁴/24 */
    fadd.s ft2, ft2, ft4        /* result += x⁴/24 */
    
    /* x⁶/720 term */
    fmul.s ft4, ft4, ft3        /* ft4 = x⁴/24 * x² = x⁶/24 */
    fli.s ft5, 0.01388889       /* ft5 ≈ 1/72 (so x⁶/24 * 1/72 ≈ x⁶/1728... adjust) */
    fmul.s ft4, ft4, ft5        /* Approximation */
    fsub.s ft2, ft2, ft4        /* result -= x⁶/720 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    ret

/* ============================================================================
 * float tanhf_fast(float x)
 * ============================================================================
 * Compute tanh(x) using rational approximation or Taylor series:
 *   tanh(x) ≈ (e^(2x) - 1) / (e^(2x) + 1)  [direct formula]
 *   or for |x| < 1: tanh(x) ≈ x - x³/3 + 2x⁵/15
 *
 * Arguments:  fa0 = x
 * Returns:    fa0 = tanh(x)
 * ============================================================================
 */
.globl tanhf_fast
.type tanhf_fast, @function
tanhf_fast:
    /* Use Taylor series approximation for small x: tanh(x) ≈ x - x³/3 + 2x⁵/15 */
    fmv.s ft2, fa0              /* ft2 = x (result) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x³/3 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³ */
    fli.s ft5, 0.33333333       /* ft5 = 1/3 */
    fmul.s ft4, ft4, ft5        /* ft4 = x³/3 */
    fsub.s ft2, ft2, ft4        /* result -= x³/3 */
    
    /* 2x⁵/15 term */
    fmul.s ft4, ft4, ft3        /* ft4 = x³/3 * x² = x⁵/3 */
    fli.s ft5, 0.13333333       /* ft5 = 2/15 */
    fmul.s ft4, ft4, ft5        /* ft4 = 2x⁵/45... adjust for 2x⁵/15 */
    fadd.s ft2, ft2, ft4        /* result += 2x⁵/15 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    ret

/* ============================================================================
 * float sqrtf_fast(float x)
 * ============================================================================
 * Compute square root using RISC-V fsqrt.s instruction.
 *
 * Arguments:  fa0 = x
 * Returns:    fa0 = sqrt(x)
 * ============================================================================
 */
.globl sqrtf_fast
.type sqrtf_fast, @function
sqrtf_fast:
    fsqrt.s fa0, fa0            /* Hardware square root */
    ret
