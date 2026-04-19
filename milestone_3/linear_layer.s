# ============================================================
# linear_layer.s
# Linear (fully-connected) projection
# out[i][j] = sum_k( in[i][k] * W[k][j] ) + bias[j]
#
# Arguments:
#   a0 = input ptr     (float32, shape batch x in_dim)
#   a1 = weight ptr    (float32, shape in_dim x out_dim, row-major)
#   a2 = bias ptr      (float32, shape out_dim)
#   a3 = output ptr    (float32, shape batch x out_dim)
#   a4 = in_dim        (e.g. C = number of channels)
#   a5 = out_dim       (e.g. 64)
#   a6 = batch         (e.g. 4096)
# ============================================================
.section .text
.global linear_layer
.global _start

_start:
    # Setup stack
    la   sp, _stack_start
    
    # Example test (modify as needed)
    la   a0, test_input
    la   a1, test_weight
    la   a2, test_bias
    la   a3, test_output
    li   a4, 4          # in_dim = 4
    li   a5, 3          # out_dim = 3
    li   a6, 2          # batch = 2
    
    call linear_layer
    
    # Signal completion to Whisper
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

linear_layer:
    # Save registers (s0-s11 + ra + floating-point)
    addi  sp, sp, -80
    sw    ra, 76(sp)
    sw    s0, 72(sp)
    sw    s1, 68(sp)
    sw    s2, 64(sp)
    sw    s3, 60(sp)
    sw    s4, 56(sp)
    sw    s5, 52(sp)
    sw    s6, 48(sp)
    sw    s7, 44(sp)
    sw    s8, 40(sp)
    sw    s9, 36(sp)
    sw    s10, 32(sp)
    sw    s11, 28(sp)
    # Save floating-point temporary registers if needed
    fsw   ft0, 24(sp)
    fsw   ft1, 20(sp)
    fsw   ft2, 16(sp)
    fsw   ft3, 12(sp)

    # Store arguments in saved registers
    mv    s0, a0        # input ptr
    mv    s1, a1        # weight ptr
    mv    s2, a2        # bias ptr
    mv    s3, a3        # output ptr
    mv    s4, a4        # in_dim
    mv    s5, a5        # out_dim
    mv    s6, a6        # batch

    # Validate inputs
    beqz  s0, error
    beqz  s1, error
    beqz  s2, error
    beqz  s3, error
    beqz  s4, error
    beqz  s5, error
    beqz  s6, error

    li    s7, 0          # i = 0 (batch row)
row_loop:
    bge   s7, s6, row_done

    li    s8, 0          # j = 0 (output col)
col_loop:
    bge   s8, s5, col_done

    # acc = bias[j]
    slli  t0, s8, 2
    add   t0, s2, t0
    flw   ft0, 0(t0)     # ft0 = acc = bias[j]

    li    s9, 0          # k = 0
    li    s10, 0         # Temporary for offset calculations
    li    s11, 0         # Temporary for offset calculations
    
inner_loop:
    bge   s9, s4, inner_done

    # in[i][k] -> offset (i * in_dim + k) * 4
    mul   t1, s7, s4
    add   t1, t1, s9
    slli  t1, t1, 2
    add   t1, s0, t1
    flw   ft1, 0(t1)     # ft1 = input[i][k]

    # W[k][j] -> offset (k * out_dim + j) * 4
    mul   t2, s9, s5
    add   t2, t2, s8
    slli  t2, t2, 2
    add   t2, s1, t2
    flw   ft2, 0(t2)     # ft2 = W[k][j]

    # acc += input * weight
    fmul.s  ft3, ft1, ft2
    fadd.s  ft0, ft0, ft3

    addi  s9, s9, 1
    j     inner_loop
inner_done:

    # output[i][j] = acc
    mul   t3, s7, s5
    add   t3, t3, s8
    slli  t3, t3, 2
    add   t3, s3, t3
    fsw   ft0, 0(t3)

    addi  s8, s8, 1
    j     col_loop
col_done:
    addi  s7, s7, 1
    j     row_loop

error:
    li    a0, -1
    j     row_done

row_done:
    # Restore registers
    lw    ra, 76(sp)
    lw    s0, 72(sp)
    lw    s1, 68(sp)
    lw    s2, 64(sp)
    lw    s3, 60(sp)
    lw    s4, 56(sp)
    lw    s5, 52(sp)
    lw    s6, 48(sp)
    lw    s7, 44(sp)
    lw    s8, 40(sp)
    lw    s9, 36(sp)
    lw    s10, 32(sp)
    lw    s11, 28(sp)
    flw   ft0, 24(sp)
    flw   ft1, 20(sp)
    flw   ft2, 16(sp)
    flw   ft3, 12(sp)
    addi  sp, sp, 80
    ret

# ============================================================
# Test data section
# ============================================================
.section .data
.align 4
test_input:
    .float 1.0, 2.0, 3.0, 4.0    # batch 0
    .float 5.0, 6.0, 7.0, 8.0    # batch 1

test_weight:
    .float 0.1, 0.2, 0.3         # k=0, j=0..2
    .float 0.4, 0.5, 0.6         # k=1
    .float 0.7, 0.8, 0.9         # k=2
    .float 1.0, 1.1, 1.2         # k=3

test_bias:
    .float 0.01, 0.02, 0.03

.section .bss
.align 4
test_output:
    .space 256         # Space for 2x3 floats
_stack_start:
    .space 8192
_stack_end:
