/* ============================================================================
 * test_hilbert.s - Minimal test (no data, no BSS, text-only)
 * ============================================================================
 */

.section .text
.globl main

main:
    /* No data initialization - pure text section only */
    
    /* Loop 10 times with tight code */
    addi t0, zero, 10
    
loop:
    addi t1, zero, 1000
inner:
    addi t1, t1, -1
    bne t1, zero, inner
    
    addi t0, t0, -1
    bne t0, zero, loop
    
    /* Spin forever */
    j spin

spin:
    j spin
