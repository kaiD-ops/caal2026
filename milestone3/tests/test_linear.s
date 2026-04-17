/* ============================================================================
 * test_linear.s - Test Harness for Linear Layer
 * ============================================================================
 *
 * Tests the linear layer with a small example:
 * - Input: 2x3 (2 timesteps, 3 features)
 * - Weight: 2x3 (2 outputs, 3 inputs) 
 * - Bias: 2
 * - Output: 2x2 (2 timesteps, 2 outputs)
 */

.extern linear_layer

.section .data

/* Test input: 2 timesteps x 3 features */
.align 4
test_input:
    /* Timestep 0: [1.0, 2.0, 3.0] */
    .float 1.0, 2.0, 3.0
    /* Timestep 1: [4.0, 5.0, 6.0] */
    .float 4.0, 5.0, 6.0

/* Test weights: 2 outputs x 3 inputs (row-major) */
.align 4
test_weights:
    /* Output 0: [0.1, 0.2, 0.3] */
    .float 0.1, 0.2, 0.3
    /* Output 1: [0.4, 0.5, 0.6] */
    .float 0.4, 0.5, 0.6

/* Test biases: 2 outputs */
.align 4
test_biases:
    .float 0.5, 1.0

/* Output buffer: 2 timesteps x 2 outputs */
.align 4
test_output:
    .space 16  /* 4 floats x 4 bytes */

.section .text
.globl main

main:
    la sp, 0x80001000          /* Initialize stack */
    
    /* Call linear_layer(weight, bias, in, out, in_dim, out_dim, seq_len) */
    la a0, test_weights        /* weight */
    la a1, test_biases         /* bias */
    la a2, test_input          /* in */
    la a3, test_output         /* out */
    li a4, 3                   /* in_dim = 3 */
    li a5, 2                   /* out_dim = 2 */
    li a6, 2                   /* seq_len = 2 */
    
    jal ra, linear_layer
    
    /* Expected output:
       Timestep 0:
         out[0] = 0.5 + 1.0*0.1 + 2.0*0.2 + 3.0*0.3 = 0.5 + 0.1 + 0.4 + 0.9 = 1.9
         out[1] = 1.0 + 1.0*0.4 + 2.0*0.5 + 3.0*0.6 = 1.0 + 0.4 + 1.0 + 1.8 = 4.2
       Timestep 1:
         out[0] = 0.5 + 4.0*0.1 + 5.0*0.2 + 6.0*0.3 = 0.5 + 0.4 + 1.0 + 1.8 = 3.7
         out[1] = 1.0 + 4.0*0.4 + 5.0*0.5 + 6.0*0.6 = 1.0 + 1.6 + 2.5 + 3.6 = 8.7
    */
    
    /* Halt and allow inspection */
.halt_loop:
    j .halt_loop
