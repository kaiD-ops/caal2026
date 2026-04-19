# ============================================================
# take_last_timestep.s
# Extracts last row from (4096 x 64) matrix -> 64 floats
# Arguments:
#   a0 = input ptr  (float32, shape 4096 x 64)
#   a1 = output ptr (float32, shape 64)
# ============================================================
.section .text
.global take_last_timestep
.global _start

_start:
    # Setup stack
    la   sp, _stack_start
    
    # Example test
    la   a0, test_input     # 4096x64 test matrix
    la   a1, test_output    # output array of 64 floats
    call take_last_timestep
    
    # Signal completion to Whisper
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

take_last_timestep:
    # Save registers (s0-s1 + ra)
    addi  sp, sp, -32
    sw    ra, 28(sp)
    sw    s0, 24(sp)
    sw    s1, 20(sp)
    sw    s2, 16(sp)
    sw    s3, 12(sp)
    sw    s4, 8(sp)
    
    # Save floating-point registers if used
    fsw   ft0, 4(sp)
    fsw   ft1, 0(sp)

    # Store arguments
    mv    s0, a0        # input ptr
    mv    s1, a1        # output ptr

    # Validate inputs
    beqz  s0, error
    beqz  s1, error

    # offset to last row = 4095 * 64 * 4 bytes
    # 4095 * 256 = 4095 << 8 = 1,048,320 bytes
    li    t0, 4095
    li    t1, 256        # 64 floats * 4 bytes
    mul   t0, t0, t1    # byte offset to row 4095
    add   s0, s0, t0    # s0 now points to start of last row

    li    s2, 0          # i = 0
copy_loop:
    li    t3, 64
    bge   s2, t3, copy_done
    
    # Load float from input last row
    slli  t4, s2, 2      # i * 4 bytes
    add   t5, s0, t4
    flw   ft0, 0(t5)
    
    # Store to output
    add   t5, s1, t4
    fsw   ft0, 0(t5)
    
    addi  s2, s2, 1
    j     copy_loop

error:
    li    a0, -1
    j     copy_done

copy_done:
    # Restore registers
    flw   ft0, 4(sp)
    flw   ft1, 0(sp)
    lw    ra, 28(sp)
    lw    s0, 24(sp)
    lw    s1, 20(sp)
    lw    s2, 16(sp)
    lw    s3, 12(sp)
    lw    s4, 8(sp)
    addi  sp, sp, 32
    ret

# ============================================================
# Test data section
# ============================================================
.section .data
.align 4
test_input:
    # Create a simple test pattern
    # For brevity, just fill first and last rows with known values
    .rept 4095
        .rept 64
            .float 0.0
        .endr
    .endr
    # Last row (row 4095) with values 0.0 to 63.0
    .set i, 0
    .rept 64
        .float i
        .set i, i+1
    .endr

.section .bss
.align 4
test_output:
    .space 256          # 64 floats * 4 bytes
_stack_start:
    .space 8192
_stack_end:
