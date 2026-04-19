# Hilbert Scan - Corrected RISC-V Implementation
# Input:
#   a0 = pointer to hilbert_indices[4096]
#   a1 = input image (C_IN x 4096)
#   a2 = output (4096 x C_IN)
#   a3 = C_IN

.text
.globl hilbert_scan
.align 2

hilbert_scan:

    # -------------------------
    # Save callee-saved regs
    # -------------------------
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)
    sw s7, 28(sp)

    # -------------------------
    # Setup registers
    # -------------------------
    mv s0, a0        # hilbert_indices base
    mv s1, a1        # input image
    mv s2, a2        # output
    mv s3, a3        # C_IN

    li s4, 0         # d = 0
    li s7, 4096      # loop limit

    li t0, 4096      # image stride (H*W)

# =====================================================
# OUTER LOOP: d = 0..4095
# =====================================================
outer_loop:

    bge s4, s7, done

    # -----------------------------------------
    # flat2d = hilbert_indices[d]
    # -----------------------------------------
    slli t1, s4, 2       # d * 4
    add t1, s0, t1       # &hilbert_indices[d]
    lw s6, 0(t1)         # s6 = flat2d

    li s5, 0             # c = 0

# =====================================================
# INNER LOOP: c = 0..C_IN-1
# =====================================================
inner_loop:

    bge s5, s3, inner_done

    # -----------------------------------------
    # input index = c * 4096 + flat2d
    # -----------------------------------------
    mul t2, s5, t0       # c * 4096
    add t2, t2, s6       # + flat2d
    slli t2, t2, 2       # *4 bytes
    add t2, s1, t2       # address

    lw t3, 0(t2)         # load input value

    # -----------------------------------------
    # output index = d * C_IN + c
    # -----------------------------------------
    mul t4, s4, s3
    add t4, t4, s5
    slli t4, t4, 2
    add t4, s2, t4

    sw t3, 0(t4)

    # next channel
    addi s5, s5, 1
    j inner_loop

# =====================================================
inner_done:
    addi s4, s4, 1
    j outer_loop

# =====================================================
done:

    # -------------------------
    # Restore registers
    # -------------------------
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    addi sp, sp, 32

    ret
