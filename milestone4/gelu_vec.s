# =============================================================================
# gelu_vec.s  –  RVV-vectorized GELU activation
#
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
#
# Vectorization strategy
# ──────────────────────
# The transcendental tanhf call is unavoidable per element.
# We vectorize the polynomial part (x + c1*x³) and the final scaling
# (0.5*x*(1+tanh)) using vector arithmetic, reducing the integer/address
# overhead by a factor of ≈VLMAX.
#
# The tanhf calls are done element-by-element (scalar loop over the loaded
# vector strip) because our math.s tanhf is a scalar routine.
# This still saves all the load/store/pointer-advance/branch overhead.
#
# Signature (unchanged from M3):
#   void gelu_inplace(float* x, int n)
#   a0 = x  (in-place),  a1 = n (element count)
# =============================================================================

.section .text
.global gelu_inplace

gelu_inplace:
    addi    sp, sp, -48
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    # Save float callee-saved registers
    fsw     fs0, 20(sp)
    fsw     fs1, 24(sp)
    fsw     fs2, 28(sp)
    fsw     fs3, 32(sp)
    fsw     fs4, 36(sp)
    fsw     fs5, 40(sp)
    fsw     fs6, 44(sp)

    mv      s0, a0          # ptr (current position)
    mv      s1, a1          # remaining count

    # Load constants
    lui     t0, %hi(gelu_c1)
    flw     fs3, %lo(gelu_c1)(t0)   # fs3 = 0.044715
    lui     t0, %hi(gelu_c2)
    flw     fs4, %lo(gelu_c2)(t0)   # fs4 = sqrt(2/π) ≈ 0.79788456
    lui     t0, %hi(math_one)
    flw     fs5, %lo(math_one)(t0)  # fs5 = 1.0
    lui     t0, %hi(gelu_half)
    flw     fs6, %lo(gelu_half)(t0) # fs6 = 0.5

    # We need a small stack buffer for the tanh strip
    # Allocate on stack: up to 8 floats (32 bytes) for tmp storage
    addi    sp, sp, -128            # 32 floats * 4 = 128 bytes for tmp
    mv      s2, sp                  # s2 = tmp buffer base

gelu_vec_loop:
    beqz    s1, gelu_vec_done

    # How many elements in this strip?
    # Use m1/e32 so that vl ≤ 32 (fits in tmp buffer easily)
    vsetvli s3, s1, e32, m1, ta, ma   # s3 = vl  (overwrites s3 constant!)
    # Note: we will reload constants inside the element loop below

    # Load x[0..vl-1] from memory
    vle32.v  v0, (s0)              # v0 = x strip

    # Compute x³ = x * x * x
    vfmul.vv v1, v0, v0            # v1 = x²
    vfmul.vv v1, v1, v0            # v1 = x³

    # Load c1 = 0.044715 into scalar and broadcast
    lui     t0, %hi(gelu_c1)
    flw     ft0, %lo(gelu_c1)(t0)
    vfmv.v.f v2, ft0               # v2 = broadcast(c1)
    vfmul.vv v1, v1, v2            # v1 = c1 * x³

    # inner = x + c1*x³
    vfadd.vv v1, v0, v1            # v1 = x + c1*x³

    # arg = c2 * inner
    lui     t0, %hi(gelu_c2)
    flw     ft0, %lo(gelu_c2)(t0)
    vfmv.v.f v2, ft0               # broadcast c2
    vfmul.vv v1, v1, v2            # v1 = c2*(x + c1*x³)

    # Store arg strip to tmp buffer so scalar loop can call tanhf
    vse32.v  v1, (s2)

    # Scalar loop: compute tanh(arg) for each element, store back to tmp
    mv      t1, s3                  # vl count
    mv      t2, s2                  # tmp pointer

gelu_tanh_loop:
    beqz    t1, gelu_tanh_done
    flw     fa0, 0(t2)
    call    tanhf
    fsw     fa0, 0(t2)
    addi    t2, t2, 4
    addi    t1, t1, -1
    j       gelu_tanh_loop

gelu_tanh_done:
    # Reload tanh results from tmp into v3
    vsetvli zero, s3, e32, m1, ta, ma
    vle32.v  v3, (s2)              # v3 = tanh(arg)

    # Reload x (still in v0, unchanged)
    # final = 0.5 * x * (1 + tanh(arg))
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    vfmv.v.f v4, ft0
    vfadd.vv v3, v3, v4            # v3 = 1 + tanh

    vfmul.vv v3, v0, v3            # v3 = x * (1 + tanh)

    lui     t0, %hi(gelu_half)
    flw     ft0, %lo(gelu_half)(t0)
    vfmv.v.f v4, ft0
    vfmul.vv v3, v3, v4            # v3 = 0.5 * x * (1 + tanh)

    # Store back in-place
    vse32.v  v3, (s0)

    # Advance pointer and counter
    slli    t0, s3, 2
    add     s0, s0, t0             # ptr += vl * 4
    sub     s1, s1, s3             # remaining -= vl
    j       gelu_vec_loop

gelu_vec_done:
    addi    sp, sp, 128            # free tmp buffer

    lw      ra,  0(sp)
    lw      s0,  4(sp)
    lw      s1,  8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    lw      s4, 20(sp)
    flw     fs0, 20(sp)
    flw     fs1, 24(sp)
    flw     fs2, 28(sp)
    flw     fs3, 32(sp)
    flw     fs4, 36(sp)
    flw     fs5, 40(sp)
    flw     fs6, 44(sp)
    addi    sp, sp, 48
    ret

.section .data
.align 2
gelu_c1:    .float 0.044715
gelu_c2:    .float 0.79788456080
gelu_half:  .float 0.5
