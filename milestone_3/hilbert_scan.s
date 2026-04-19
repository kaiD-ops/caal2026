# ============================================================
# hilbert_scan.s
# Transforms input (C, 64, 64) -> output (4096, C)
# For VeeR-ISS Whisper with tohost at 0xd0580000
# ============================================================

.section .text
.global _start
.global hilbert_scan
.global d2xy

_start:
    # Initialize stack pointer
    la   sp, _stack_start
    
    # Example test - modify these addresses as needed
    li   a0, 0x80001000    # input ptr (example)
    li   a1, 0x80002000    # output ptr (example)
    li   a2, 3             # C = 3 channels
    
    call hilbert_scan
    
    # Signal completion to Whisper (tohost at 0xd0580000)
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    
    # Halt
    j    .

# ============================================================
# hilbert_scan - Main transformation function
# ============================================================
hilbert_scan:
    addi  sp, sp, -48
    sw    ra, 44(sp)
    sw    s0, 40(sp)
    sw    s1, 36(sp)
    sw    s2, 32(sp)
    sw    s3, 28(sp)
    sw    s4, 24(sp)
    sw    s5, 20(sp)
    sw    s6, 16(sp)

    mv    s0, a0
    mv    s1, a1
    mv    s2, a2

    # Validate inputs
    beqz  s0, error
    beqz  s1, error
    beqz  s2, error

    li    s3, 0
pixel_loop:
    li    t0, 4096
    bge   s3, t0, success

    # Get (row, col) for this Hilbert index
    li    a0, 64
    mv    a1, s3
    call  d2xy
    mv    s5, a0
    mv    s6, a1

    li    s4, 0
channel_loop:
    bge   s4, s2, next_pixel

    # input[c][row][col] offset
    li    t0, 4096
    mul   t1, s4, t0
    li    t0, 64
    mul   t2, s5, t0
    add   t1, t1, t2
    add   t1, t1, s6
    slli  t1, t1, 2
    add   t1, s0, t1
    lw    t3, 0(t1)

    # output[pixel][c] offset
    mul   t4, s3, s2
    add   t4, t4, s4
    slli  t4, t4, 2
    add   t4, s1, t4
    sw    t3, 0(t4)

    addi  s4, s4, 1
    j     channel_loop

next_pixel:
    addi  s3, s3, 1
    j     pixel_loop

success:
    li    a0, 0
    j     done

error:
    li    a0, -1

done:
    lw    ra, 44(sp)
    lw    s0, 40(sp)
    lw    s1, 36(sp)
    lw    s2, 32(sp)
    lw    s3, 28(sp)
    lw    s4, 24(sp)
    lw    s5, 20(sp)
    lw    s6, 16(sp)
    addi  sp, sp, 48
    ret

# ============================================================
# d2xy - Hilbert curve index to (x,y) conversion
# Uses only basic RISC-V instructions
# ============================================================
d2xy:
    addi  sp, sp, -32
    sw    s0, 28(sp)
    sw    s1, 24(sp)
    sw    s2, 20(sp)
    sw    s3, 16(sp)
    sw    s4, 12(sp)
    sw    s5, 8(sp)
    sw    s6, 4(sp)
    
    mv    s0, a0         # s0 = N
    mv    s1, a1         # s1 = d
    li    s2, 0          # s2 = x
    li    s3, 0          # s3 = y
    li    s4, 1          # s4 = t
    
d2xy_loop:
    bge   s4, s0, d2xy_done
    
    # rx = (d >> 1) & 1
    srli  s5, s1, 1
    andi  s5, s5, 1      # s5 = rx
    
    # ry = (d ^ rx) & 1
    xor   s6, s1, s5
    andi  s6, s6, 1      # s6 = ry
    
    # if ry == 0
    bnez  s6, d2xy_skip_rot
    
    # if rx == 1
    beqz  s5, d2xy_skip_flip
    addi  t0, s4, -1
    sub   s2, t0, s2
    sub   s3, t0, s3
    
d2xy_skip_flip:
    # swap x and y
    mv    t0, s2
    mv    s2, s3
    mv    s3, t0
    
d2xy_skip_rot:
    # x += t * rx
    mul   t0, s4, s5
    add   s2, s2, t0
    
    # y += t * ry
    mul   t0, s4, s6
    add   s3, s3, t0
    
    # d >>= 2
    srli  s1, s1, 2
    
    # t <<= 1
    slli  s4, s4, 1
    
    j     d2xy_loop
    
d2xy_done:
    mv    a0, s2
    mv    a1, s3
    
    lw    s0, 28(sp)
    lw    s1, 24(sp)
    lw    s2, 20(sp)
    lw    s3, 16(sp)
    lw    s4, 12(sp)
    lw    s5, 8(sp)
    lw    s6, 4(sp)
    addi  sp, sp, 32
    ret

# ============================================================
# Stack section
# ============================================================
.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
