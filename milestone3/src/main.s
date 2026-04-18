/* ============================================================================
 * main.s - Test Hilbert Scan Layer
 * ============================================================================
 */

#include "nn.h"

.extern hilbert_scan

/* Test data - small to avoid initialization overhead */
.section .bss

/* Small test image: 10 pixels */
.align 4
test_img:
    .space 40        /* 10 * 4 bytes */

/* Small output: 10 pixels */
.align 4
test_out:
    .space 40

/* Minimal ModelParams (just the indices part) */
.align 4
test_params:
    .space 100       /* small structure */

.section .text
.globl main

main:
    /* Initialize stack */
    lui sp, 0x80800
    
    /* Call hilbert_scan(test_params, test_img, test_out) */
    la a0, test_params
    la a1, test_img
    la a2, test_out
    jal ra, hilbert_scan
    
    /* If we reach here, hilbert_scan completed */
    /* Spin forever */
    j spin

spin:
    j spin

.globl simple_puts
simple_puts:
    ret

.globl simple_puti
simple_puti:
    ret

.globl simple_putf
simple_putf:
    ret
