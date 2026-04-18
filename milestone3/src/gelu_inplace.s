/* ============================================================================
 * gelu_inplace.s - GELU Activation (In-Place)
 * ============================================================================
 *
 * Gaussian Error Linear Unit:
 *   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * This is a good approximation for transformers that avoids computing
 * the error function directly.
 *
 * Function Signature:
 *   void gelu_inplace(float *x, int n)
 *
 * Arguments (RISC-V calling convention):
 *   a0 = float *x  (input/output buffer, n floats)
 *   a1 = int n     (number of elements)
 *
 * Algorithm:
 *   for i in 0..n-1:
 *       Let c1 = sqrt(2/π) ≈ 0.7978845608
 *       Let c2 = 0.044715
 *       Let cdf = tanh(c1 * (x[i] + c2 * x[i]³))
 *       x[i] = 0.5 * x[i] * (1 + cdf)
 *
 * Register Usage:
 *   t0:  loop counter
 *   t1:  address calculation
 *   fs0: input value
 *   fs1: x³ (x * x * x)
 *   fs2: x + c2*x³
 *   fs3: tanh result
 */

.extern tanhf_fast

.section .rodata
.align 4
const_c2:       .word 0x3d36f08e    # 0.044715
const_c1:       .word 0x3f4c422f    # ≈ 0.7978845608 (sqrt(2/π))
const_1_0:      .word 0x3f800000    # 1.0
const_0_5:      .word 0x3f000000    # 0.5

.section .text
.globl gelu_inplace
.type gelu_inplace, @function

gelu_inplace:
    /* Arguments:
       a0 = float *x
       a1 = int n */
    
    /* Return immediately if n <= 0 */
    ble a1, zero, .gelu_done
    
    /* Constants */
    /* c1 = sqrt(2/π) ≈ 0.7978845608 */
    /* c2 = 0.044715 */
    /* 0.5 = 0.5 (for final scaling) */
    /* 1.0 = 1.0 (for tanh calculation) */
    
    li t0, 0                    /* loop counter */
    
.gelu_loop:
    bge t0, a1, .gelu_done
    
    /* Load x[t0] */
    slli t1, t0, 2              /* t1 = t0 * 4 (byte offset) */
    add t1, a0, t1              /* t1 = &x[t0] */
    flw fs0, 0(t1)              /* fs0 = x[t0] */
    
    /* Compute x³ */
    fmul.s fs1, fs0, fs0        /* fs1 = x² */
    fmul.s fs1, fs1, fs0        /* fs1 = x³ */
    
    /* Compute c2 * x³ where c2 = 0.044715 */
    la t2, const_c2
    flw fs2, 0(t2)              /* fs2 = 0.044715 */
    fmul.s fs2, fs2, fs1        /* fs2 = 0.044715 * x³ */
    
    /* Compute x + c2*x³ */
    fadd.s fs2, fs0, fs2        /* fs2 = x + 0.044715*x³ */
    
    /* Multiply by c1 = sqrt(2/π) ≈ 0.7978845608 */
    la t2, const_c1
    flw fs3, 0(t2)              /* fs3 ≈ sqrt(2/π) */
    fmul.s fs2, fs2, fs3        /* fs2 = sqrt(2/π) * (x + 0.044715*x³) */
    
    /* Call tanhf_fast
       Move argument to fa0 and call */
    fmv.s fa0, fs2              /* fa0 = argument to tanh */
    jal ra, tanhf_fast          /* call tanhf_fast, result in fa0 */
    
    /* Compute 1 + tanh(...) */
    la t2, const_1_0
    flw fs2, 0(t2)              /* fs2 = 1.0 */
    fadd.s fs2, fa0, fs2        /* fs2 = 1 + tanh(...) */
    
    /* Compute x * (1 + tanh(...)) */
    fmul.s fs2, fs0, fs2        /* fs2 = x * (1 + tanh(...)) */
    
    /* Compute 0.5 * x * (1 + tanh(...)) */
    la t2, const_0_5
    flw fs3, 0(t2)              /* fs3 = 0.5 */
    fmul.s fs2, fs2, fs3        /* fs2 = 0.5 * x * (1 + tanh(...)) */
    
    /* Store result */
    fsw fs2, 0(t1)              /* x[t0] = fs2 */
    
    addi t0, t0, 1
    j .gelu_loop
    
.gelu_done:
    ret
