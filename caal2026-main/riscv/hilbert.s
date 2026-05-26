# =============================================================================
# hilbert.s - Hilbert Scan layer
# Reorders image pixels using pre-computed Hilbert indices
# void hilbert_scan(int32_t* indices, float* img, float* out)
# a0=indices, a1=img, a2=out
# =============================================================================
.section .text
.global hilbert_scan
hilbert_scan:
    li      t0, 4096
hs_loop:
    beqz    t0, hs_done
    lw      t4, 0(a0)       # hilbert_index[d]
    slli    t5, t4, 2        # * 4 bytes
    add     t6, a1, t5       # &img[index]
    flw     fa0, 0(t6)
    fsw     fa0, 0(a2)
    addi    a0, a0, 4
    addi    a2, a2, 4
    addi    t0, t0, -1
    j       hs_loop
hs_done:
    ret
