# RISC-V Assembly Implementation of Hilbert Scan Layer
# 
# Function: hilbert_scan
# Purpose: Reorders pixels from 2D images into 1D sequences following Hilbert curve order
#
# Arguments:
#   a0 = pointer to ModelParams (contains hilbert_indices[4096])
#   a1 = pointer to input image, shape (C_IN, 64, 64) flattened as (C_IN * 4096) floats
#   a2 = pointer to output sequence, shape (4096, C_IN) flattened as (4096 * C_IN) floats
#   a3 = C_IN (number of channels, typically 1)
#
# Input layout:   img[c * 4096 + flat2d]  (channel-major: C, H, W)
# Output layout:  out[d * C_IN + c]        (sequence-major: L, C)
#
# Algorithm:
#   for d = 0 to 4095:
#       flat2d = hilbert_indices[d]
#       for c = 0 to C_IN-1:
#           out[d * C_IN + c] = img[c * 4096 + flat2d]
#
# This is pure integer indexing with no floating-point operations.
# Callee-saved registers used: s0, s1, s2, s3, s4, s5, s6, s7
#
# Register allocation:
#   s0 = ModelParams pointer (params)
#   s1 = input image pointer
#   s2 = output pointer
#   s3 = C_IN (channels)
#   s4 = d (outer loop counter, 0-4095)
#   s5 = c (inner loop counter, 0-C_IN-1)
#   s6 = flat2d (loaded from hilbert_indices[d])
#   s7 = 4096 (constant, size of image)

.section .text
.globl hilbert_scan
.align 2

hilbert_scan:
    # Save callee-saved registers
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)
    sw s7, 28(sp)

    # Initialize registers
    mv s0, a0           # s0 = params pointer
    mv s1, a1           # s1 = input image pointer
    mv s2, a2           # s2 = output pointer
    mv s3, a3           # s3 = C_IN (channels)
    li s7, 4096         # s7 = 4096 (image H*W)
    li s4, 0            # s4 = d = 0 (outer loop counter)

    # Constants for address calculation
    li t0, 4096         # t0 = img channel stride

    # Outer loop: for d = 0 to 4095
.outer_loop:
    bge s4, s7, .outer_end  # if d >= 4096, exit outer loop

    # Load hilbert_indices[d] from params
    # ModelParams structure assumed to have hilbert_indices at offset 0
    lw s6, (s0)         # Load base address of hilbert_indices from params
    slli t1, s4, 2      # t1 = d * 4 (byte offset)
    add t1, s6, t1      # t1 = &hilbert_indices[d]
    lw s6, 0(t1)        # s6 = hilbert_indices[d]

    # Inner loop: for c = 0 to C_IN-1
    li s5, 0            # s5 = c = 0

.inner_loop:
    bge s5, s3, .inner_end  # if c >= C_IN, exit inner loop

    # Calculate input offset: c * 4096 + flat2d
    # t2 = c * 4096
    mul t2, s5, t0          # t2 = c * 4096
    add t2, t2, s6          # t2 = c * 4096 + flat2d
    slli t2, t2, 2          # t2 = (c * 4096 + flat2d) * 4 (byte offset)
    add t2, s1, t2          # t2 = address of img[c * 4096 + flat2d]
    
    # Load from input
    lw t3, 0(t2)            # t3 = img[c * 4096 + flat2d] (as integer, 4 bytes)

    # Calculate output offset: d * C_IN + c
    # t4 = d * C_IN + c
    mul t4, s4, s3          # t4 = d * C_IN
    add t4, t4, s5          # t4 = d * C_IN + c
    slli t4, t4, 2          # t4 = (d * C_IN + c) * 4 (byte offset)
    add t4, s2, t4          # t4 = address of out[d * C_IN + c]
    
    # Store to output
    sw t3, 0(t4)            # out[d * C_IN + c] = img[c * 4096 + flat2d]

    # Increment c and continue inner loop
    addi s5, s5, 1
    j .inner_loop

.inner_end:
    # Increment d and continue outer loop
    addi s4, s4, 1
    j .outer_loop

.outer_end:
    # Restore callee-saved registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    addi sp, sp, 32

    # Return
    ret
