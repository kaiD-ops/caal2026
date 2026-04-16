# RISC-V Assembly Implementation of Take Last Timestep Layer
#
# Function: take_last_timestep
# Purpose: Extracts the final timestep from a sequence tensor for classification
#
# Arguments:
#   a0 = pointer to input sequence, shape (SEQ_LEN=4096, D_MODEL=64)
#   a1 = pointer to output vector, shape (D_MODEL=64,)
#
# The input is stored in row-major order (sequence first, features second).
# We need to extract the last row, which starts at offset:
#   offset = (SEQ_LEN - 1) * D_MODEL * 4 bytes
#   offset = (4096 - 1) * 64 * 4 = 4095 * 256 = 1048320 bytes
#
# Algorithm:
#   output[0..D_MODEL-1] = input[(D_MODEL * (SEQ_LEN - 1)) .. (D_MODEL * SEQ_LEN)]
#
# This is simple memcpy with calculated offset.
# D_MODEL = 64 floats = 256 bytes
#
# Callee-saved registers: s0, s1, s2

.section .text
.globl take_last_timestep
.align 2

take_last_timestep:
    # Save callee-saved registers for clarity
    addi sp, sp, -12
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)

    mov s0, a0          # s0 = input pointer
    mv s1, a1           # s1 = output pointer

    # Constants
    li s2, 64           # s2 = D_MODEL = 64
    
    # Calculate offset to last timestep
    # offset = (SEQ_LEN - 1) * D_MODEL * 4
    # offset = 4095 * 64 * 4 = 1048320 bytes
    # More efficiently: offset = (4095 * 256) = (4096 - 1) * 256
    
    # Calculate address of last row: input + (4095 * 256)
    li t0, 4095         # t0 = 4095 = SEQ_LEN - 1
    li t1, 256          # t1 = 256 = D_MODEL * 4
    mul t0, t0, t1      # t0 = 4095 * 256 (byte offset)
    add t0, s0, t0      # t0 = input pointer + offset (points to last row)

    # Copy D_MODEL=64 floats from input[last_row] to output
    # Loop 64 times, each iteration copies one float (4 bytes)
    li t2, 0            # t2 = counter

.copy_loop:
    bge t2, s2, .copy_end   # if counter >= D_MODEL, exit

    # Load one float from input
    slli t3, t2, 2          # t3 = counter * 4 (byte offset)
    add t3, t0, t3          # t3 = address of input[last_row + counter]
    flw f0, 0(t3)           # f0 = input value

    # Store one float to output
    add t3, s1, t3          # t3 = address of output[counter] (reuse t3)
    fsw f0, 0(t3)           # output[counter] = f0

    addi t2, t2, 1          # counter++
    j .copy_loop

.copy_end:
    # Restore callee-saved registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    addi sp, sp, 12

    ret
