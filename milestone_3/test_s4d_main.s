.section .text
.global _start

_start:
    la   sp, _stack_start
    
    # Test parameters
    la   a0, test_input      # input ptr
    la   a1, test_A_real     # A_real ptr
    la   a2, test_A_imag     # A_imag ptr
    la   a3, test_B          # B ptr
    la   a4, test_C_real     # C_real ptr
    la   a5, test_C_imag     # C_imag ptr
    la   a6, test_output     # output ptr
    li   a7, 2               # seq_len = 2
    
    call s4d_layer
    
    # Signal completion to Whisper
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

# Test data (seq_len=2, d_model=64)
.section .data
.align 4
test_input:
    .rept 128
        .float 1.0
    .endr
test_A_real:
    .rept 64
        .float -0.1
    .endr
test_A_imag:
    .rept 64
        .float 0.5
    .endr
test_B:
    .rept 64
        .float 1.0
    .endr
test_C_real:
    .rept 64
        .float 1.0
    .endr
test_C_imag:
    .rept 64
        .float 0.0
    .endr
test_output:
    .space 512

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
