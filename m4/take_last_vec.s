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

    # Set vector configuration for e32 elements using an m4 group grouping allocation
    vsetvli t3, t0, e32, m4, ta, ma    # t3 = elements granted (vl)

    # FIX: Use vector register v4 instead of v0 to avoid architectural conflicts 
    # with the vector mask tracking registers across multi-register allocations.
    vle32.v  v4, (t1)              # Load vector data strip into v4-v7
    vse32.v  v4, (t2)              # Store vector data strip out of v4-v7

    # Advance pointers and track element updates
    slli    t4, t3, 2              # Convert element count to byte offset (vl * 4)
    add     t1, t1, t4             # Advance input pointer
    add     t2, t2, t4             # Advance output destination pointer
    sub     t0, t0, t3             # Remaining elements loop decrement
    j       tl_vec_loop

tl_vec_done:
    ret