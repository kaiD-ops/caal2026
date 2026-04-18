/* ============================================================================
 * hilbert_scan_test.s - Test version of Hilbert Scan
 * ============================================================================
 * 
 * Simplified version that processes a configurable number of pixels instead
 * of hardcoding 4096.
 *
 * Function Signature:
 *   void hilbert_scan_test(const int32_t *indices, const float *img, 
 *                          float *out, int num_pixels)
 *
 * Arguments:
 *   a0 = pointer to hilbert_indices array
 *   a1 = pointer to input image
 *   a2 = pointer to output array
 *   a3 = number of pixels to process
 */

.section .text
.globl hilbert_scan_test
.type hilbert_scan_test, @function

hilbert_scan_test:
    /* a0 = int32_t *indices
       a1 = float *img
       a2 = float *out
       a3 = num_pixels (loop count) */
    
    li t0, 0                    /* d = 0 (loop counter) */
    
.hs_outer_loop:
    /* Check if d < num_pixels */
    bge t0, a3, .hs_done
    
    /* Load indices[d] */
    slli t1, t0, 2              /* t1 = d * 4 (byte offset for int32_t) */
    add t1, a0, t1              /* t1 = &indices[d] */
    lw t2, 0(t1)                /* t2 = flat_idx = indices[d] */
    
    /* Load img[flat_idx] */
    slli t1, t2, 2              /* t1 = flat_idx * 4 (byte offset for float32) */
    add t1, a1, t1              /* t1 = &img[flat_idx] */
    flw ft0, 0(t1)              /* ft0 = img[flat_idx] */
    
    /* Store to out[d] */
    slli t1, t0, 2              /* t1 = d * 4 (byte offset for float32) */
    add t1, a2, t1              /* t1 = &out[d] */
    fsw ft0, 0(t1)              /* out[d] = ft0 */
    
    /* Increment d and continue */
    addi t0, t0, 1
    j .hs_outer_loop
    
.hs_done:
    ret
