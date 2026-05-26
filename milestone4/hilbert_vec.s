# =============================================================================
# hilbert_vec.s  –  RVV-vectorized Hilbert Scan
#
# Reorders image pixels according to pre-computed Hilbert curve indices.
# The scalar version issues one flw/fsw pair per pixel (4096 iterations).
# The vector version loads a strip of indices, gathers floats via a
# stride-1 gather idiom (vluxei32.v), and stores them in order.
#
# Signature (unchanged from M3):
#   void hilbert_scan(int32_t* indices, float* img, float* out)
#   a0 = indices   (int32, 4096 entries)
#   a1 = img       (float*, base of source image, byte-addressed)
#   a2 = out       (float*, destination, contiguous)
#
# RVV notes:
#   • Element width: e32  (32-bit int/float)
#   • The index array holds *element* indices (not byte offsets);
#     we scale by 4 using vsll.vi before vluxei32.v.
#   • vsetvli / strip-mining handles any VLEN.
# =============================================================================

.section .text
.global hilbert_scan

hilbert_scan:
    li      t0, 4096                # total elements remaining
    mv      t1, a0                  # walking index pointer
    mv      t2, a2                  # walking output pointer

hs_vec_loop:
    beqz    t0, hs_vec_done

    # Set vector length for up to t0 e32 elements
    vsetvli t3, t0, e32, m4, ta, ma   # t3 = vl actually granted

    # Load a strip of 32-bit Hilbert indices from t1
    vle32.v  v0, (t1)              # v0 = raw element indices

    # Convert element indices → byte offsets: off = idx << 2
    vsll.vi  v4, v0, 2             # v4 = byte offsets into img

    # Gather: out[i] = img[ index[i] ]  using byte-offset gather
    vluxei32.v  v8, (a1), v4      # v8 = gathered pixel values

    # Store gathered values sequentially to output
    vse32.v  v8, (t2)

    # Advance pointers and counter
    slli    t4, t3, 2              # bytes = vl * 4
    add     t1, t1, t4             # indices += vl
    add     t2, t2, t4             # out     += vl
    sub     t0, t0, t3             # remaining -= vl
    j       hs_vec_loop

hs_vec_done:
    ret
