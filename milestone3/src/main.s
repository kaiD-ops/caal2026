/* ============================================================================
 * main.s - RISC-V S4D Galaxy Classifier Demo (Simplified)
 * ============================================================================
 */

#include "nn.h"

/* External function declarations */
.extern hilbert_scan

/* Minimal data section */
.section .data

/* Loop count for testing */
.align 4
loop_count:
    .word 100

.section .text
.globl main

main:
    /* Initialize stack */
    li sp, 0x80800000
    
    /* Load loop count from data section */
    la t0, loop_count
    lw t0, 0(t0)
    
test_loop:
    addi t0, t0, -1
    bne t0, zero, test_loop
    
    /* Done - spin forever */
    j spin

spin:
    j spin

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
