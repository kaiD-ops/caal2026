/* ============================================================================
 * softmax_inplace.s - Softmax Activation (In-Place)
 * ============================================================================
 *
 * Converts logits to probabilities using softmax:
 *   softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
 *
 * Function Signature:
 *   void softmax_inplace(float *x, int n)
 *
 * Arguments (RISC-V calling convention):
 *   a0 = float *x  (input/output buffer, n floats)
 *   a1 = int n     (number of elements, typically 4 for N_CLASSES)
 *
 * Algorithm:
 *   1. Find max value in x[0..n-1]
 *   2. For each i: x[i] = exp(x[i] - max)   (numerator computation)
 *   3. Sum all exponentials
 *   4. For each i: x[i] /= sum             (normalization)
 *
 * Note: We modify the array in-place. For numerical stability, we subtract
 * the maximum before exponentiation to avoid overflow.
 *
 * Register Usage:
 *   t0:  loop counter
 *   t1:  address calculation
 *   t2:  maximum value (as float, in fs3)
 *   fs0: temporary float
 *   fs1: temporary float
 *   fs2: accumulator for sum
 *   fs3: maximum value
 */

.extern expf_fast

.section .rodata
.align 4
const_0_0:      .word 0x00000000    # 0.0
const_1e7:      .word 0x33d6e3f7    # 1e-7
const_1_0:      .word 0x3f800000    # 1.0

.section .text
.globl softmax_inplace
.type softmax_inplace, @function

softmax_inplace:
    /* Arguments:
       a0 = float *x
       a1 = int n */
    
    /* Return immediately if n <= 0 */
    ble a1, zero, .softmax_done
    
    /* ========== Phase 1: Find maximum value ==========
       mx = x[0] */
    flw fs3, 0(a0)              /* fs3 = x[0] (max value) */
    
    /* Loop through remaining elements to find max */
    li t0, 1                    /* loop counter = 1 */
    
.softmax_find_max_loop:
    bge t0, a1, .softmax_max_found
    
    /* Load x[t0] */
    slli t1, t0, 2              /* t1 = t0 * 4 (byte offset) */
    add t1, a0, t1              /* t1 = &x[t0] */
    flw fs0, 0(t1)              /* fs0 = x[t0] */
    
    /* Update max if x[t0] > mx */
    fle.s t2, fs0, fs3          /* t2 = (fs0 <= fs3) ? 1 : 0 */
    beq t2, zero, .softmax_update_max  /* if NOT (x[t0] <= max), update max */
    j .softmax_skip_max_update
    
.softmax_update_max:
    fmv.s fs3, fs0              /* fs3 = fs0 (new max) */
    
.softmax_skip_max_update:
    addi t0, t0, 1
    j .softmax_find_max_loop
    
.softmax_max_found:
    /* ========== Phase 2: Compute exp(x[i] - max) and sum ==========
       sum = 0 */
    la t2, const_0_0
    flw fs2, 0(t2)              /* fs2 = 0.0 (sum accumulator) */
    li t0, 0                    /* loop counter = 0 */
    
.softmax_exp_sum_loop:
    bge t0, a1, .softmax_sum_found
    
    /* Calculate address of x[t0] */
    slli t1, t0, 2              /* t1 = t0 * 4 (byte offset) */
    add t1, a0, t1              /* t1 = &x[t0] */
    
    /* Load x[t0] */
    flw fs0, 0(t1)              /* fs0 = x[t0] */
    
    /* Compute x[t0] - max */
    fsub.s fs0, fs0, fs3        /* fs0 = x[t0] - max */
    
    /* Call expf_fast(x[t0] - max)
       Move argument to fa0 and call */
    fmv.s fa0, fs0              /* fa0 = x[t0] - max */
    jal ra, expf_fast           /* call expf_fast, result in fa0 */
    
    /* Store result back to x[t0] */
    fsw fa0, 0(t1)              /* x[t0] = exp(x[t0] - max) */
    
    /* Add to sum */
    fadd.s fs2, fs2, fa0        /* fs2 += exp(x[t0] - max) */
    
    addi t0, t0, 1
    j .softmax_exp_sum_loop
    
.softmax_sum_found:
    /* ========== Phase 3: Normalize by sum ==========
       for i in 0..n-1: x[i] /= sum */
    
    /* If sum is effectively zero, handle gracefully */
    la t2, const_1e7
    flw fs0, 0(t2)              /* fs0 = 1e-7 small epsilon for safety */
    flt.s t2, fs2, fs0           /* t2 = (sum < epsilon) ? 1 : 0 */
    beq t2, zero, .softmax_divide_loop
    la t2, const_1_0
    flw fs2, 0(t2)              /* fs2 = 1.0 if sum too small, use 1.0 to avoid divide by zero */
    
.softmax_divide_loop:
    li t0, 0                    /* loop counter = 0 */
    
.softmax_normalize_loop:
    bge t0, a1, .softmax_done
    
    /* Calculate address of x[t0] */
    slli t1, t0, 2              /* t1 = t0 * 4 (byte offset) */
    add t1, a0, t1              /* t1 = &x[t0] */
    
    /* Load x[t0] (which is exp(x[t0] - max)) */
    flw fs0, 0(t1)              /* fs0 = x[t0] */
    
    /* Divide by sum */
    fdiv.s fs0, fs0, fs2        /* fs0 /= sum */
    
    /* Store result */
    fsw fs0, 0(t1)              /* x[t0] = fs0 */
    
    addi t0, t0, 1
    j .softmax_normalize_loop
    
.softmax_done:
    ret
