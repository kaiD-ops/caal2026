/* ============================================================================
 * hilbert_scan.s - Hilbert Curve Reordering Layer
 * ============================================================================
 *
 * This layer reorders pixels from a 64x64 image according to a pre-computed
 * Hilbert curve index. No floating-point arithmetic is required.
 *
 * Function Signature:
 *   void hilbert_scan(const ModelParams *p, const float *img, float *out)
 *
 * Arguments:
 *   a0 = pointer to ModelParams structure
 *        (contains hilbert_indices[SEQ_LEN] at offset 0)
 *   a1 = pointer to input image [C_IN, 64, 64] flattened as [C_IN*4096] floats
 *   a2 = pointer to output [SEQ_LEN, C_IN] = [4096, 1] flattened as [4096] floats
 *
 * Constants (from nn.h):
 *   SEQ_LEN = 4096
 *   C_IN = 1
 *
 * Algorithm (translated from C):
 *   for d in 0..SEQ_LEN-1:
 *       flat2d = p->hilbert_indices[d]
 *       for c in 0..C_IN-1:
 *           out[d*C_IN + c] = img[c*4096 + flat2d]
 *
 * Memory Layout:
 *   - hilbert_indices: offset 0 in ModelParams, array of int32_t (4 bytes each)
 *   - img: array of float32 (4 bytes each), indexed linearly
 *   - out: array of float32 (4 bytes each), indexed linearly
 *
 * Register Usage:
 *   t0:  loop counter d (dimension index)
 *   t1:  loop counter c (channel index)
 *   t2:  flat2d index
 *   t3:  temporary calculation
 *   t4:  address calculation
 *   t5:  loaded float value
 */

.section .text
.globl hilbert_scan
.type hilbert_scan, @function

hilbert_scan:
    /* a0 = ModelParams *p (required for hilbert_indices)
       a1 = float *img (input image)
       a2 = float *out (output array) */
    
    /* Constants for this layer */
    li t0, 0                    /* d = 0 (outer loop counter) */
    li t6, 4096                 /* SEQ_LEN = 4096 */
    li t5, 1                    /* C_IN = 1 (inner loop counter) */
    
.hilbert_outer_loop:
    /* Check if d < SEQ_LEN */
    bge t0, t6, .hilbert_done
    
    /* Load hilbert_indices[d] */
    /* hilbert_indices is at offset 0 in ModelParams */
    slli t3, t0, 2              /* t3 = d * 4 (offset in bytes for int32_t array) */
    add t3, a0, t3              /* t3 = &p->hilbert_indices[d] */
    lw t2, 0(t3)                /* t2 = flat2d = p->hilbert_indices[d] */
    
    /* Inner loop: for c in 0..C_IN-1 */
    li t1, 0                    /* c = 0 */
    
.hilbert_inner_loop:
    /* Check if c < C_IN (which is 1) */
    bge t1, t5, .hilbert_next_d
    
    /* out[d*C_IN + c] = img[c*4096 + flat2d] */
    
    /* Calculate source index: c*4096 + flat2d */
    li t3, 4096
    mul t4, t1, t3              /* t4 = c * 4096 */
    add t4, t4, t2              /* t4 = c*4096 + flat2d (linear index) */
    slli t4, t4, 2              /* t4 *= 4 (convert to byte offset for float32) */
    add t4, a1, t4              /* t4 = &img[c*4096 + flat2d] */
    flw ft0, 0(t4)              /* ft0 = img[c*4096 + flat2d] */
    
    /* Calculate destination index: d*C_IN + c */
    mul t3, t0, t5              /* t3 = d * C_IN */
    add t3, t3, t1              /* t3 = d*C_IN + c (linear index) */
    slli t3, t3, 2              /* t3 *= 4 (convert to byte offset for float32) */
    add t3, a2, t3              /* t3 = &out[d*C_IN + c] */
    fsw ft0, 0(t3)              /* out[d*C_IN + c] = ft0 */
    
    /* Increment c and continue inner loop */
    addi t1, t1, 1
    j .hilbert_inner_loop
    
.hilbert_next_d:
    /* Increment d and continue outer loop */
    addi t0, t0, 1
    j .hilbert_outer_loop
    
.hilbert_done:
    ret
