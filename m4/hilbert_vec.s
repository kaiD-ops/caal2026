

.section .text
.global hilbert_scan

hilbert_scan:
    li      t0, 4096                # total elements remaining
    mv      t1, a0                  # walking index pointer
    mv      t2, a2                  # walking output pointer

hs_vec_loop:
    beqz    t0, hs_vec_done

    
    # hardware compatibility across all store variations (vse32.v).
    vsetvli t3, t0, e32, m1, ta, ma   # t3 = vl actually granted

    
    vle32.v  v0, (t1)              # v0 = float-encoded indices
    vfcvt.rtz.x.f.v v0, v0         # v0 = (int32) index   (truncate toward zero)

    # Convert element indices → byte offsets: off = idx << 2
    vsll.vi  v1, v0, 2             # Use v1 (safe m1 register) to calculate offsets

    # Gather: out[i] = img[ index[i] ] using byte-offset gather
    vluxei32.v  v2, (a1), v1       # v2 = gathered pixel values

    # Store gathered values sequentially to output destination
    vse32.v  v2, (t2)

    
    slli    t4, t3, 2              # bytes = vl * 4
    add     t1, t1, t4             # indices pointer forward
    add     t2, t2, t4             # out pointer forward
    sub     t0, t0, t3             # remaining element count -= vl
    j       hs_vec_loop

hs_vec_done:
    ret
