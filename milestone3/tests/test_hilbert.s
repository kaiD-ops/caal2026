/* ============================================================================
 * test_hilbert.s - Test Harness for Hilbert Scan Layer
 * ============================================================================
 *
 * This test harness:
 * 1. Sets up a small test case (4x4 image, not full 64x64)
 * 2. Calls hilbert_scan
 * 3. Outputs results for verification
 *
 * To run:
 *   make build-test-hilbert
 *   veer-iss bin/test_hilbert
 */

.extern hilbert_scan

.section .data

/* Small test image: 4x4 = 16 pixels */
.align 4
test_img:
    .float 1.0, 2.0, 3.0, 4.0
    .float 5.0, 6.0, 7.0, 8.0
    .float 9.0, 10.0, 11.0, 12.0
    .float 13.0, 14.0, 15.0, 16.0

/* Small hilbert indices for 4x4: maps position d to flat 2D index */
.align 4
hilbert_indices:
    .int 0, 1, 5, 4
    .int 2, 3, 6, 7
    .int 8, 9, 13, 12
    .int 10, 11, 14, 15
    /* Extended to 4096 with zeros for test */
    .fill 4080, 4, 0

/* ModelParams structure (minimal for this test) */
.align 4
test_params:
    .word hilbert_indices  /* offset 0: pointer to indices */
    .space 1000           /* rest of structure, mostly unused */

/* Output buffer */
.align 4
test_output:
    .space 64              /* 16 floats = 64 bytes for output */

.section .text
.globl main

main:
    la sp, 0x80001000       /* Initialize stack */
    
    /* Call hilbert_scan(test_params, test_img, test_output) */
    la a0, test_params
    la a1, test_img
    la a2, test_output
    jal ra, hilbert_scan
    
    /* TODO: Output results (depends on VeeR-iSS I/O) */
    /* For now, just loop forever so we can inspect memory */
    
.halt_loop:
    j .halt_loop
