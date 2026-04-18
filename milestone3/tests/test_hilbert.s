/* ============================================================================
 * test_hilbert.s - Test Harness for Hilbert Scan Layer (SIMPLE TEST)
 * ============================================================================
 *
 * This test harness:
 * 1. Does NOT call hilbert_scan yet
 * 2. Just loops a few times to test the framework
 * 3. Spins forever for inspection
 *
 * To run:
 *   make build-test-hilbert
 *   whisper --configfile veer/whisper.json bin/test_hilbert
 */

.section .text
.globl main

main:
    /* Initialize stack */
    lui sp, 0x80800
    
    /* Simple test loop: 100 iterations */
    addi t0, zero, 100
    
test_loop:
    addi t0, t0, -1
    bne t0, zero, test_loop
    
    /* Done - spin forever */
    j spin

spin:
    j spin
