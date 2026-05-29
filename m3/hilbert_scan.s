# =============================================================================
# hilbert_scan.s - Hilbert Scan layer
#
# Reorders image pixels using pre-computed Hilbert indices.
# Pure integer loads/stores, no floating point except for the flw/fsw copies.
#
# void hilbert_scan(int32_t* indices, float* img, float* out)
# a0=indices, a1=img, a2=out
# MSE target: < 1e-12 (exact copy, no arithmetic)
# =============================================================================
.section .text
.global hilbert_scan

hilbert_scan:
    li      t0, 4096
    mv      t1, a0              # indices ptr
    mv      t2, a1              # img ptr
    mv      t3, a2              # out ptr
hs_loop:
    beqz    t0, hs_done
    lw      t4, 0(t1)           # t4 = hilbert_index[i]
    slli    t5, t4, 2           # t5 = index * 4 (byte offset)
    add     t6, t2, t5          # t6 = &img[index]
    flw     fa0, 0(t6)          # fa0 = img[index]
    fsw     fa0, 0(t3)          # out[i] = img[index]
    addi    t1, t1, 4           # advance indices ptr
    addi    t3, t3, 4           # advance out ptr
    addi    t0, t0, -1
    j       hs_loop
hs_done:
    ret
