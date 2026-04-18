/* ============================================================================
 * main.s - RISC-V S4D Galaxy Classifier Demo
 * ============================================================================
 *
 * This program demonstrates the full S4D inference pipeline on a test image.
 * It performs the following steps:
 * 1. Load model weights from parameter file
 * 2. Load test image
 * 3. Run forward pass through the entire network
 * 4. Output predicted class and probabilities
 *
 * Register Conventions:
 *   gp register is used for data segment access
 *   sp is preserved for stack
 */

#include "nn.h"

/* External function declarations */
.extern hilbert_scan
.extern linear_layer
.extern s4d_layer
.extern gelu_inplace
.extern take_last_timestep
.extern softmax_inplace
.extern expf_fast

/* Data segment: Model parameters and test data */
.section .data

/* Placeholder for model parameters
   In a real implementation, these would be loaded from a binary file */
.align 4
model_params:
    .space 1000000  /* Allocate space for ModelParams structure (~1MB) */

/* Intermediate buffers for pipeline */
.align 4
buffer_after_hilbert:  /* [4096] floats */
    .space 16384

.align 4
buffer_after_uproject:  /* [4096, 64] floats */
    .space 1048576

.align 4
buffer_after_s4d_1:  /* [4096, 64] floats */
    .space 1048576

.align 4
buffer_after_gelu_1:  /* [4096, 64] floats */
    .space 1048576

.align 4
buffer_after_s4d_2:  /* [4096, 64] floats */
    .space 1048576

.align 4
buffer_after_gelu_2:  /* [4096, 64] floats */
    .space 1048576

.align 4
buffer_last_timestep:  /* [64] floats */
    .space 256

.align 4
buffer_output_logits:  /* [4] floats */
    .space 16

.align 4
output_probs:  /* [4] floats - final output */
    .space 16

/* Test image: 64x64 grayscale (placeholder - would be loaded from file) */
.align 4
test_image:
    .space 16384

/* Messages for output */
msg_start:     .asciz "S4D Galaxy Classifier - RISC-V\n"
msg_logits:    .asciz "Logits: "
msg_probs:     .asciz "Probabilities: "
msg_class:     .asciz "Predicted Class: "
msg_newline:   .asciz "\n"

.section .text
.globl main

main:
    /* Initialize stack */
    li sp, 0x80800000
    li gp, 0x80000000      /* Initialize global pointer for data access */
    
    /* ============================================================================
     * FORWARD PASS: S4D Galaxy Classifier
     * ============================================================================
     */
    
    /* Step 1: Hilbert Scan - reorder 64x64 image along Hilbert curve */
    la a0, model_params           /* ModelParams* */
    la a1, test_image             /* input image (64x64 = 4096 floats) */
    la a2, buffer_after_hilbert   /* output (4096 floats) */
    jal ra, hilbert_scan
    
    /* Step 2: Input Projection (U-project) - Linear layer C_IN=1 -> D_MODEL=64 */
    /* Note: This is a sequence operation, apply to all 4096 sequence positions */
    la a0, model_params
    lw a0, 64(a0)                 /* Offset to uproject_weight in ModelParams */
    lw a1, 68(a0)                 /* uproject_bias */
    la a2, buffer_after_hilbert   /* input: [4096] */
    la a3, buffer_after_uproject  /* output: [4096, 64] */
    li t0, 4096                   /* seq_len */
    li t1, 1                       /* in_dim */
    li t2, 64                      /* out_dim (D_MODEL) */
    jal ra, linear_layer
    
    /* Step 3: S4D Layer 1 */
    la a0, model_params
    la a1, buffer_after_uproject  /* input [4096, 64] */
    la a2, buffer_after_s4d_1     /* output [4096, 64] */
    jal ra, s4d_layer
    
    /* Step 4: GELU activation in-place on Layer 1 output */
    la a0, buffer_after_s4d_1     /* pointer to data */
    li a1, 262144                  /* size: 4096 * 64 floats */
    jal ra, gelu_inplace
    
    /* Step 5: S4D Layer 2 */
    la a0, model_params
    la a1, buffer_after_s4d_1     /* input [4096, 64] */
    la a2, buffer_after_s4d_2     /* output [4096, 64] */
    jal ra, s4d_layer
    
    /* Step 6: GELU activation in-place on Layer 2 output */
    la a0, buffer_after_s4d_2
    li a1, 262144
    jal ra, gelu_inplace
    
    /* Step 7: Take last timestep - extract final sequence output [64] */
    la a0, buffer_after_s4d_2       /* input [4096, 64] */
    la a1, buffer_last_timestep     /* output [64] */
    li a2, 4096                     /* seq_len */
    li a3, 64                       /* d_model */
    jal ra, take_last_timestep
    
    /* Step 8: FC Head - Linear layer D_MODEL=64 -> N_CLASSES=4 */
    la a0, model_params
    lw a0, 256(a0)                 /* fc_weight offset */
    lw a1, 512(a0)                 /* fc_bias offset */
    la a2, buffer_last_timestep    /* input [64] */
    la a3, buffer_output_logits    /* output [4] */
    li t0, 64                       /* in_dim */
    li t1, 4                        /* out_dim (N_CLASSES) */
    li t2, 1                        /* seq_len (single sample) */
    jal ra, linear_layer
    
    /* Step 9: Softmax - convert logits to probabilities */
    la a0, buffer_output_logits    /* input/output [4] floats */
    li a1, 4                        /* size */
    jal ra, softmax_inplace
    
    /* ============================================================================
     * EXIT CLEANLY
     * ============================================================================
     * Use ECALL to invoke the exit() syscall
     * RISC-V ABI: a0 contains exit code (0 = success)
     */
    
    li a0, 0                        /* exit code: 0 = success */
    li a7, 93                       /* syscall number for exit() */
    ecall                           /* invoke syscall - program terminates here */
    
    /* This code should never be reached */
    j .                             /* if ecall fails, spin forever */

/* ============================================================================
 * Utility Functions (simplified - VeeR-iSS environment dependent)
 * ============================================================================
 */

.globl simple_puts
simple_puts:
    /* Print null-terminated string in a0 */
    /* Placeholder - actual implementation depends on VeeR-iSS setup */
    ret

.globl simple_puti
simple_puti:
    /* Print integer in a0 */
    /* Placeholder - actual implementation depends on VeeR-iSS setup */
    ret

.globl simple_putf
simple_putf:
    /* Print float in fa0 */
    /* Placeholder - actual implementation depends on VeeR-iSS setup */
    ret
