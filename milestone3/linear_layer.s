# RISC-V Assembly Implementation of Linear Layer (Input Projection)
#
# Function: linear_layer
# Purpose: Fully-connected layer: out = in * W^T + b
#
# Arguments:
#   a0 = pointer to weight matrix, shape (out_dim, in_dim), row-major
#   a1 = pointer to bias vector, shape (out_dim,)
#   a2 = pointer to input tensor, shape (seq_len, in_dim)
#   a3 = pointer to output tensor, shape (seq_len, out_dim)
#   a4 = in_dim (input feature dimension)
#   a5 = out_dim (output feature dimension)
#   a6 = seq_len (number of timesteps / batch size)
#
# Algorithm (from C code):
#   for t = 0 to seq_len-1:
#       for o = 0 to out_dim-1:
#           acc = bias[o]
#           for i = 0 to in_dim-1:
#               acc += weight[o * in_dim + i] * input[t * in_dim + i]
#           output[t * out_dim + o] = acc
#
# This layer handles both:
#   - Sequence inputs (seq_len > 1): UProject maps (4096, 1) to (4096, 64)
#   - Vector inputs (seq_len = 1): FC head maps (64,) to (4,)
#
# Callee-saved registers: s0-s7, s10, s11
# We save/restore on the stack

.section .text
.globl linear_layer
.align 2

linear_layer:
    # Save callee-saved registers
    addi sp, sp, -48
    sw s0, 0(sp)   # weight ptr
    sw s1, 4(sp)   # bias ptr
    sw s2, 8(sp)   # input ptr
    sw s3, 12(sp)  # output ptr
    sw s4, 16(sp)  # in_dim
    sw s5, 20(sp)  # out_dim
    sw s6, 24(sp)  # seq_len
    sw s7, 28(sp)  # t (timestep counter)
    sw s10, 32(sp) # o (output neuron counter)
    sw s11, 36(sp) # i (input feature counter)
    sw ra, 40(sp)  # return address
    sw s8, 44(sp)  # padding for alignment

    # Initialize saved registers from arguments
    mv s0, a0       # s0 = weight ptr
    mv s1, a1       # s1 = bias ptr
    mv s2, a2       # s2 = input ptr
    mv s3, a3       # s3 = output ptr
    mv s4, a4       # s4 = in_dim
    mv s5, a5       # s5 = out_dim
    mv s6, a6       # s6 = seq_len

    li s7, 0        # s7 = t = 0 (timestep counter)

    # Outer loop: for t = 0 to seq_len-1
.t_loop:
    bge s7, s6, .t_end           # if t >= seq_len, exit

    # Calculate input pointer for this timestep: x_t = input + t * in_dim * 4
    mul t0, s7, s4                # t0 = t * in_dim
    slli t0, t0, 2                # t0 = t * in_dim * 4 (bytes)
    add t0, s2, t0                # t0 = input + t * in_dim * 4

    # Calculate output pointer for this timestep: y_t = output + t * out_dim * 4
    mul t1, s7, s5                # t1 = t * out_dim
    slli t1, t1, 2                # t1 = t * out_dim * 4 (bytes)
    add t1, s3, t1                # t1 = output + t * out_dim * 4

    li s10, 0                     # s10 = o = 0 (output neuron counter)

    # Middle loop: for o = 0 to out_dim-1
.o_loop:
    bge s10, s5, .o_end           # if o >= out_dim, exit

    # Load bias[o]
    slli t2, s10, 2               # t2 = o * 4 (bytes)
    add t2, s1, t2                # t2 = &bias[o]
    flw f0, 0(t2)                 # f0 = bias[o]

    li s11, 0                     # s11 = i = 0 (input feature counter)

    # Inner loop: for i = 0 to in_dim-1
.i_loop:
    bge s11, s4, .i_end           # if i >= in_dim, exit

    # Load weight[o * in_dim + i]
    mul t2, s10, s4               # t2 = o * in_dim
    add t2, t2, s11               # t2 = o * in_dim + i
    slli t2, t2, 2                # t2 = (o * in_dim + i) * 4 (bytes)
    add t2, s0, t2                # t2 = &weight[o * in_dim + i]
    flw f1, 0(t2)                 # f1 = weight[o * in_dim + i]

    # Load input[t * in_dim + i]
    slli t2, s11, 2               # t2 = i * 4 (bytes)
    add t2, t0, t2                # t2 = &input[t * in_dim + i]
    flw f2, 0(t2)                 # f2 = input[t * in_dim + i]

    # Multiply and accumulate: acc += weight * input
    fmul.s f1, f1, f2             # f1 = weight * input
    fadd.s f0, f0, f1             # f0 = acc + (weight * input)

    addi s11, s11, 1              # i++
    j .i_loop

.i_end:
    # Store accumulated value to output[t * out_dim + o]
    slli t2, s10, 2               # t2 = o * 4 (bytes)
    add t2, t1, t2                # t2 = &output[t * out_dim + o]
    fsw f0, 0(t2)                 # output[t * out_dim + o] = acc

    addi s10, s10, 1              # o++
    j .o_loop

.o_end:
    addi s7, s7, 1                # t++
    j .t_loop

.t_end:
    # Restore callee-saved registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    lw s10, 32(sp)
    lw s11, 36(sp)
    lw ra, 40(sp)
    lw s8, 44(sp)
    addi sp, sp, 48

    ret
