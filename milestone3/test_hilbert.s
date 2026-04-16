# RISC-V Test Harness for Hilbert Scan Layer
# 
# Purpose: Validate the hilbert_scan function with a small test case
# Test Case: 4x4 image with 1 channel (16 floats) following Hilbert curve
#
# The Hilbert curve for a 4x4 grid follows this order:
#   0  1 14 15
#   3  2 13 12
#   4  7  8 11
#   5  6  9 10
#
# So if input is row-major [0, 1, 2, ..., 15], the output following Hilbert order is:
#   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (indices in Hilbert order)

.section .data

# Hilbert indices for 4x4 grid (0-15 reordered by Hilbert curve)
hilbert_indices_4x4:
    .word 0      # d=0 -> (0,0) -> flat=0
    .word 1      # d=1 -> (1,0) -> flat=1
    .word 5      # d=2 -> (1,1) -> flat=5
    .word 4      # d=3 -> (0,1) -> flat=4
    .word 8      # d=4 -> (0,2) -> flat=8
    .word 9      # d=5 -> (1,2) -> flat=9
    .word 13     # d=6 -> (1,3) -> flat=13
    .word 12     # d=7 -> (0,3) -> flat=12
    .word 3      # d=8 -> (3,0) -> flat=3
    .word 2      # d=9 -> (2,0) -> flat=2
    .word 6      # d=10 -> (2,1) -> flat=6
    .word 7      # d=11 -> (3,1) -> flat=7
    .word 11     # d=12 -> (3,2) -> flat=11
    .word 10     # d=13 -> (2,2) -> flat=10
    .word 14     # d=14 -> (2,3) -> flat=14
    .word 15     # d=15 -> (3,3) -> flat=15

# Input image: 4x4 with 1 channel, stored as [0,1,2,3,4,5,...,15]
input_image_4x4:
    .word 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

# Output buffer (will store 16 floats)
output_buffer:
    .word 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# Mock ModelParams structure
mock_params:
    .word hilbert_indices_4x4

# Expected output (following Hilbert order)
expected_output_4x4:
    .word 0, 1, 5, 4, 8, 9, 13, 12, 3, 2, 6, 7, 11, 10, 14, 15

# String for output messages (for logging/debugging)
msg_start:
    .ascii "Starting Hilbert Scan Test...\n"
msg_start_len = . - msg_start

msg_pass:
    .ascii "TEST PASSED: Hilbert Scan output matches expected results\n"
msg_pass_len = . - msg_pass

msg_fail:
    .ascii "TEST FAILED: Output mismatch\n"
msg_fail_len = . - msg_fail

.section .text
.globl test_hilbert_scan
.align 2

# Main test function
test_hilbert_scan:
    # Save return address and local state
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)

    # Print start message (optional, depends on simulator support)
    # la a0, msg_start
    # li a1, msg_start_len
    # callsei syscall_write  (would need syscall wrapper)

    # Call hilbert_scan(params, input, output, c_in=1)
    la a0, mock_params         # a0 = &params (pointer to hilbert_indices)
    la a1, input_image_4x4     # a1 = input image
    la a2, output_buffer       # a2 = output buffer
    li a3, 1                   # a3 = C_IN = 1
    
    # Make the call
    jal ra, hilbert_scan

    # Verify output
    la s0, output_buffer       # s0 = output buffer
    la s1, expected_output_4x4 # s1 = expected output
    li s2, 16                  # s2 = num_words to check
    li t0, 0                   # t0 = loop counter

.verify_loop:
    bge t0, s2, .verify_end
    
    # Load actual and expected values
    slli t1, t0, 2
    add t1, s0, t1
    lw t2, 0(t1)               # t2 = output[i]
    
    add t1, s1, t1
    lw t3, 0(t1)               # t3 = expected[i]
    
    # Compare
    bne t2, t3, .verify_failed
    
    addi t0, t0, 1
    j .verify_loop

.verify_end:
    # Success path
    # Print pass message (optional)
    li a0, 0  # Exit code 0 (success)
    jal zero, .test_end

.verify_failed:
    # Failure path
    # Print fail message (optional)
    li a0, 1  # Exit code 1 (failure)
    jal zero, .test_end

.test_end:
    # Restore registers
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret
