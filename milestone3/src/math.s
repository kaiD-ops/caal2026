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

.section .rodata
# Floating-point constants (IEEE 754 single precision)
.align 4
const_1_0:      .word 0x3f800000    # 1.0
const_0_5:      .word 0x3f000000    # 0.5
const_1_3:      .word 0x3eaaaaab    # 1/3 ≈ 0.333...
const_0_25:     .word 0x3e800000    # 0.25 = 1/4
const_0_2:      .word 0x3e4ccccd    # 0.2 = 1/5
const_1_6:      .word 0x3e2aaaab    # 1/6 ≈ 0.1666...
const_1_24:     .word 0x3d888889    # 1/24 ≈ 0.04166...
const_1_120:    .word 0x3c0feb84    # 1/120 ≈ 0.00833...
const_2_15:     .word 0x3e088889    # 2/15 ≈ 0.1333...
const_1_72:     .word 0x3d068fd2    # 1/72 ≈ 0.01388...

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
    addi sp, sp, -4
    sw t0, 0(sp)
    
    /* Load 1.0 constant */
    la t0, const_1_0
    flw ft0, 0(t0)              /* ft0 = 1.0 */
    
    /* Initialize accumulator and power terms */
    fmv.s ft2, ft0              /* ft2 = acc = 1.0 */
    fmv.s ft3, fa0              /* ft3 = x_power = x */
    
    /* Coefficients for 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720 */
    fadd.s ft2, ft2, ft3        /* acc += x => 1 + x */
    
    /* x²/2 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x * x = x² */
    la t0, const_0_5
    flw ft5, 0(t0)              /* ft5 = 0.5 */
    fmul.s ft4, ft4, ft5        /* ft4 = x²/2 */
    fadd.s ft2, ft2, ft4        /* acc += x²/2 */
    
    /* x³/6 term */
    fmul.s ft3, ft4, fa0        /* ft3 = x²/2 * x = x³/2 */
    la t0, const_1_3
    flw ft5, 0(t0)              /* ft5 ≈ 1/3 */
    fmul.s ft3, ft3, ft5        /* ft3 ≈ x³/6 */
    fadd.s ft2, ft2, ft3        /* acc += x³/6 */
    
    /* x⁴/24 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³/6 * x = x⁴/6 */
    la t0, const_0_25
    flw ft5, 0(t0)              /* ft5 = 0.25 */
    fmul.s ft4, ft4, ft5        /* ft4 ≈ x⁴/24 */
    fadd.s ft2, ft2, ft4        /* acc += x⁴/24 */
    
    /* x⁵/120 term */
    fmul.s ft3, ft4, fa0        /* ft3 = x⁴/24 * x = x⁵/24 */
    la t0, const_0_2
    flw ft5, 0(t0)              /* ft5 = 0.2 = 1/5 */
    fmul.s ft3, ft3, ft5        /* ft3 ≈ x⁵/120 */
    fadd.s ft2, ft2, ft3        /* acc += x⁵/120 */
    
    /* x⁶/720 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x⁵/120 * x = x⁶/120 */
    la t0, const_1_6
    flw ft5, 0(t0)              /* ft5 ≈ 1/6 */
    fmul.s ft4, ft4, ft5        /* ft4 ≈ x⁶/720 */
    fadd.s ft2, ft2, ft4        /* acc += x⁶/720 */
    
    fmv.s fa0, ft2              /* Return result in fa0 */
    
    lw t0, 0(sp)
    addi sp, sp, 4
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
    addi sp, sp, -4
    sw t0, 0(sp)
    
    /* Taylor series: x - x³/6 + x⁵/120 */
    fmv.s ft2, fa0              /* ft2 = x (result accumulator) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x³/6 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³ */
    la t0, const_1_6
    flw ft5, 0(t0)              /* ft5 ≈ 1/6 */
    fmul.s ft4, ft4, ft5        /* ft4 = x³/6 */
    fsub.s ft2, ft2, ft4        /* result -= x³/6 */
    
    /* x⁵/120 term */
    fmul.s ft4, ft3, ft3        /* ft4 = x⁴ */
    fmul.s ft4, ft4, fa0        /* ft4 = x⁵ */
    la t0, const_1_120
    flw ft5, 0(t0)              /* ft5 ≈ 1/120 */
    fmul.s ft4, ft4, ft5        /* ft4 = x⁵/120 */
    fadd.s ft2, ft2, ft4        /* result += x⁵/120 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    
    lw t0, 0(sp)
    addi sp, sp, 4
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
    addi sp, sp, -4
    sw t0, 0(sp)
    
    /* Taylor series: 1 - x²/2 + x⁴/24 - x⁶/720 */
    la t0, const_1_0
    flw ft2, 0(t0)              /* ft2 = 1.0 (result accumulator) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x²/2 term */
    la t0, const_0_5
    flw ft5, 0(t0)              /* ft5 = 0.5 */
    fmul.s ft4, ft3, ft5        /* ft4 = x²/2 */
    fsub.s ft2, ft2, ft4        /* result -= x²/2 */
    
    /* x⁴/24 term */
    fmul.s ft4, ft3, ft3        /* ft4 = x⁴ */
    la t0, const_1_24
    flw ft5, 0(t0)              /* ft5 ≈ 1/24 */
    fmul.s ft4, ft4, ft5        /* ft4 = x⁴/24 */
    fadd.s ft2, ft2, ft4        /* result += x⁴/24 */
    
    /* x⁶/720 term */
    fmul.s ft4, ft4, ft3        /* ft4 = x⁴/24 * x² = x⁶/24 */
    la t0, const_1_72
    flw ft5, 0(t0)              /* ft5 ≈ 1/72 */
    fmul.s ft4, ft4, ft5        /* ft4 ≈ x⁶/1728 */
    fsub.s ft2, ft2, ft4        /* result -= x⁶/720 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    
    lw t0, 0(sp)
    addi sp, sp, 4
    ret

/* ============================================================================
 * float tanhf_fast(float x)
 * ============================================================================
 * Compute tanh(x) using Taylor series for small x:
 *   tanh(x) ≈ x - x³/3 + 2x⁵/15
 *
 * Arguments:  fa0 = x
 * Returns:    fa0 = tanh(x)
 * ============================================================================
 */
.globl tanhf_fast
.type tanhf_fast, @function
tanhf_fast:
    addi sp, sp, -4
    sw t0, 0(sp)
    
    /* Taylor series: x - x³/3 + 2x⁵/15 */
    fmv.s ft2, fa0              /* ft2 = x (result) */
    fmul.s ft3, fa0, fa0        /* ft3 = x² */
    
    /* x³/3 term */
    fmul.s ft4, ft3, fa0        /* ft4 = x³ */
    la t0, const_1_3
    flw ft5, 0(t0)              /* ft5 = 1/3 */
    fmul.s ft4, ft4, ft5        /* ft4 = x³/3 */
    fsub.s ft2, ft2, ft4        /* result -= x³/3 */
    
    /* 2x⁵/15 term */
    fmul.s ft4, ft3, ft3        /* ft4 = x⁴ */
    fmul.s ft4, ft4, fa0        /* ft4 = x⁵ */
    la t0, const_2_15
    flw ft5, 0(t0)              /* ft5 = 2/15 */
    fmul.s ft4, ft4, ft5        /* ft4 = 2x⁵/15 */
    fadd.s ft2, ft2, ft4        /* result += 2x⁵/15 */
    
    fmv.s fa0, ft2              /* Return in fa0 */
    
    lw t0, 0(sp)
    addi sp, sp, 4
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
