# ============================================================
# minimal_test.s - Absolute minimal test
# ============================================================
.section .text
.global _start

_start:
    # Simple test: just set a register and loop
    li    t0, 0
    li    t1, 10
    
test_loop:
    addi  t0, t0, 1
    blt   t0, t1, test_loop
    
    # Exit successfully
    li    a0, 0
    li    a7, 10
    ecall
