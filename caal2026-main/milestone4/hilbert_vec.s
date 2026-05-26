t
# =============================================================================
.section .text
.global hilbert_scan
hilbert_scan:
    li      t2, 512             # 4096 / 8 = 512 iterations
    li      t3, 8
    vsetvli zero, t3, e32, m1, ta, ma  # VL=8, SEW=32, LMUL=1

hs_loop:
    beqz    t2, hs_done

    vle32.v  v1, (a0)           # load 8 int32 Hilbert indices
    vsll.vi  v1, v1, 2          # scale to byte offsets (* 4)
    vluxei32.v v0, (a1), v1    # gather 8 floats: v0[i] = img[v1[i]]
    vse32.v  v0, (a2)           # store 8 reordered pixels

    addi    a0, a0, 32          # indices += 8 * 4 bytes
    addi    a2, a2, 32          # out    += 8 * 4 bytes
    addi    t2, t2, -1
    j       hs_loop

hs_done:
    ret
