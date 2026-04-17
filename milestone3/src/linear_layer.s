/* ============================================================================
 * linear_layer.s - Fully Connected Linear Layer (Matrix-Vector Multiplication)
 * ============================================================================
 *
 * Performs batch matrix-vector multiplication with bias:
 *   out[t, o] = bias[o] + sum_i(weight[o, i] * in[t, i])
 *
 * This is used for both the input projection (C_IN -> D_MODEL) and
 * the classification head (D_MODEL -> N_CLASSES).
 *
 * Function Signature:
 *   void linear_layer(const float *weight, const float *bias,
 *                     const float *in, float *out,
 *                     int in_dim, int out_dim, int seq_len)
 *
 * Arguments (RISC-V calling convention):
 *   a0 = const float *weight [out_dim, in_dim] row-major (out_dim*in_dim floats)
 *   a1 = const float *bias   [out_dim] (out_dim floats)
 *   a2 = const float *in     [seq_len, in_dim] row-major (seq_len*in_dim floats)
 *   a3 = float *out          [seq_len, out_dim] row-major (seq_len*out_dim floats)
 *   a4 = int in_dim          (input dimension, typically 1 or 64)
 *   a5 = int out_dim         (output dimension, typically 64 or 4)
 *   a6 = int seq_len         (sequence length, typically 4096)
 *
 * Algorithm (translated from C):
 *   for t in 0..seq_len-1:                      // timestep
 *       for o in 0..out_dim-1:                  // output neuron
 *           acc = bias[o]
 *           for i in 0..in_dim-1:               // input feature
 *               acc += weight[o*in_dim + i] * in[t*in_dim + i]
 *           out[t*out_dim + o] = acc
 *
 * Optimization: For this network, in_dim is typically small (1 for input projection,
 * 64 for FC layer), so the innermost loop is very short. Register allocation
 * prioritizes minimizing store operations.
 *
 * Register Usage:
 *   t0:  outer loop counter t (timestep)
 *   t1:  middle loop counter o (output neuron)
 *   t2:  inner loop counter i (input feature)
 *   t3:  temporary for address calculation
 *   t4:  temporary for address calculation
 *   fs0: accumulator for dot product
 *   fs1: temporary float values
 *   fs2: temporary float values
 */

.section .text
.globl linear_layer
.type linear_layer, @function

linear_layer:
    /* Arguments:
       a0 = weight, a1 = bias, a2 = in, a3 = out
       a4 = in_dim, a5 = out_dim, a6 = seq_len */
    
    /* Outer loop: for t in 0..seq_len-1 */
    li t0, 0                    /* t = 0 */
    
.linear_t_loop:
    bge t0, a6, .linear_done    /* if t >= seq_len, we're done */
    
    /* Middle loop: for o in 0..out_dim-1 */
    li t1, 0                    /* o = 0 */
    
.linear_o_loop:
    bge t1, a5, .linear_next_t  /* if o >= out_dim, move to next t */
    
    /* Load bias[o] as initial accumulator */
    slli t3, t1, 2              /* t3 = o * 4 (byte offset for float) */
    add t4, a1, t3              /* t4 = &bias[o] */
    flw fs0, 0(t4)              /* fs0 = bias[o] */
    
    /* Inner loop: for i in 0..in_dim-1 (accumulate dot product) */
    li t2, 0                    /* i = 0 */
    
.linear_i_loop:
    bge t2, a4, .linear_store   /* if i >= in_dim, store result */
    
    /* Load weight[o*in_dim + i] */
    mul t3, t1, a4              /* t3 = o * in_dim */
    add t3, t3, t2              /* t3 = o*in_dim + i (linear index) */
    slli t3, t3, 2              /* t3 *= 4 (byte offset) */
    add t4, a0, t3              /* t4 = &weight[o*in_dim + i] */
    flw fs1, 0(t4)              /* fs1 = weight[o*in_dim + i] */
    
    /* Load in[t*in_dim + i] */
    mul t3, t0, a4              /* t3 = t * in_dim */
    add t3, t3, t2              /* t3 = t*in_dim + i (linear index) */
    slli t3, t3, 2              /* t3 *= 4 (byte offset) */
    add t4, a2, t3              /* t4 = &in[t*in_dim + i] */
    flw fs2, 0(t4)              /* fs2 = in[t*in_dim + i] */
    
    /* Multiply weight * input and add to accumulator */
    fmadd.s fs0, fs1, fs2, fs0  /* fs0 += fs1 * fs2 */
    
    /* Increment i and continue inner loop */
    addi t2, t2, 1
    j .linear_i_loop
    
.linear_store:
    /* Store out[t*out_dim + o] = fs0 */
    mul t3, t0, a5              /* t3 = t * out_dim */
    add t3, t3, t1              /* t3 = t*out_dim + o (linear index) */
    slli t3, t3, 2              /* t3 *= 4 (byte offset) */
    add t4, a3, t3              /* t4 = &out[t*out_dim + o] */
    fsw fs0, 0(t4)              /* out[t*out_dim + o] = fs0 */
    
    /* Increment o and continue middle loop */
    addi t1, t1, 1
    j .linear_o_loop
    
.linear_next_t:
    /* Increment t and continue outer loop */
    addi t0, t0, 1
    j .linear_t_loop
    
.linear_done:
    ret
