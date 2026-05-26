# =============================================================================
# take_last.s - TakeLastTimestep layer
# Extracts final timestep from (SEQ_LEN, D_MODEL) buffer
# void take_last_timestep(float* in, float* out)
# a0=in, a1=out
# =============================================================================
.section .text
.global take_last_timestep
take_last_timestep:
    # offset = (4096-1)*64*4 = 1048320 bytes
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     a0, a0, t0      # point to last row
    li      t0, 64          # D_MODEL elements
tl_loop:
    beqz    t0, tl_done
    flw     ft0, 0(a0)
    fsw     ft0, 0(a1)
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    t0, t0, -1
    j       tl_loop
tl_done:
    ret
