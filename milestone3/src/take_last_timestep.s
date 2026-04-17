/* ============================================================================
 * take_last_timestep.s - Extract Final Timestep from Sequence
 * ============================================================================
 *
 * Copies the last timestep from a sequence [SEQ_LEN, D_MODEL] to output [D_MODEL].
 *
 * Function Signature:
 *   void take_last_timestep(const float *in, float *out)
 *
 * Arguments (RISC-V calling convention):
 *   a0 = const float *in   [SEQ_LEN, D_MODEL] flattened as [SEQ_LEN*D_MODEL] floats
 *   a1 = float *out        [D_MODEL] (D_MODEL floats)
 *
 * Constants (from nn.h):
 *   SEQ_LEN = 4096
 *   D_MODEL = 64
 *
 * Algorithm:
 *   // Copy D_MODEL floats from in[(SEQ_LEN-1)*D_MODEL] to out[0]
 *   memcpy(out, &in[(SEQ_LEN-1)*D_MODEL], D_MODEL*sizeof(float))
 *
 * Optimized Implementation:
 *   For D_MODEL=64, we simply copy 64 floats from the last row.
 *   Last row starts at offset: (SEQ_LEN - 1) * D_MODEL * 4 bytes
 *                            = 4095 * 64 * 4 = 1,048,320 bytes
 *
 * Register Usage:
 *   t0:  loop counter (0 to D_MODEL-1)
 *   t1:  source address calculation
 *   t2:  destination address
 *   ft0: temporary for float loads/stores
 */

.section .text
.globl take_last_timestep
.type take_last_timestep, @function

take_last_timestep:
    /* Arguments:
       a0 = const float *in
       a1 = float *out */
    
    /* Calculate offset to last timestep:
       (SEQ_LEN - 1) * D_MODEL * 4 bytes = 4095 * 64 * 4 */
    li t0, 4095                 /* t0 = SEQ_LEN - 1 = 4095 */
    li t1, 64                   /* t1 = D_MODEL = 64 */
    mul t0, t0, t1              /* t0 = 4095 * 64 = 262,080 (in float units) */
    slli t0, t0, 2              /* t0 *= 4 (convert to byte offset) */
    add t1, a0, t0              /* t1 = &in[(SEQ_LEN-1)*D_MODEL] */
    
    /* Copy loop: 64 floats from in to out */
    li t0, 0                    /* loop counter */
    li t2, 64                   /* number of floats to copy */
    
.tlt_copy_loop:
    bge t0, t2, .tlt_done       /* if copied all D_MODEL=64 floats, done */
    
    /* Load from in */
    flw ft0, 0(t1)              /* ft0 = *in_ptr */
    
    /* Store to out */
    fsw ft0, 0(a1)              /* *out_ptr = ft0 */
    
    /* Advance pointers (4 bytes per float) */
    addi t1, t1, 4              /* in_ptr += 4 */
    addi a1, a1, 4              /* out_ptr += 4 */
    
    /* Increment counter and continue */
    addi t0, t0, 1
    j .tlt_copy_loop
    
.tlt_done:
    ret
