/* ============================================================================
 * main.s - Test Hilbert Scan with Small Data
 * ============================================================================
 */

#include "nn.h"

.extern hilbert_scan_test

/* Test data */
.section .bss

/* 10 hilbert indices */
.align 4
test_indices:
    .space 40        /* 10 * 4 bytes */

/* 10-pixel input image */
.align 4
test_img:
    .space 40        /* 10 * 4 bytes */

/* 10-pixel output */
.align 4
test_out:
    .space 40        /* 10 * 4 bytes */

.section .text
.globl main

main:
    /* Initialize stack */
    lui sp, 0x80800
    
    /* Initialize test data in registers (quick setup) */
    la a0, test_indices          /* indices array */
    la a1, test_img              /* input image */
    la a2, test_out              /* output */
    li a3, 10                    /* num_pixels = 10 */
    
    /* Call hilbert_scan_test(indices, img, out, 10) */
    jal ra, hilbert_scan_test
    
    /* If we reach here, test completed */
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
