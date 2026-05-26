# =============================================================================
# take_last_vec.s  –  RVV-vectorized TakeLastTimestep
#
# Extracts the final row (timestep 4095) from a [4096, 64] buffer.
# The scalar version iterates 64 times with individual flw/fsw pairs.
# The vector version copies all 64 floats in a single vle32 / vse32 pair
# (64 * 4 = 256 bytes, fits in one vector operation for VLEN ≥ 256,
# and in two strips for VLEN = 128).
#
# Signature (unchanged from M3):
#   void take_last_timestep(float* in, float* out)
#   a0 = in   [4096][64] float buffer
#   a1 = out  [64]  float destination
# =============================================================================

.section .text
.global take_last_timestep

take_last_timestep:
    # Compute byte offset to row 4095:  4095 * 64 * 4 = 1048320 bytes
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     a0, a0, t0              # a0 → last row

    li      t0, 64                  # 64 elements to copy
    mv      t1, a0
    mv      t2, a1

tl_vec_loop:
    beqz    t0, tl_vec_done

    vsetvli t3, t0, e32, m4, ta, ma    # t3 = vl

    vle32.v  v0, (t1)              # load strip
    vse32.v  v0, (t2)              # store strip

    slli    t4, t3, 2
    add     t1, t1, t4
    add     t2, t2, t4
    sub     t0, t0, t3
    j       tl_vec_loop

tl_vec_done:
    ret
