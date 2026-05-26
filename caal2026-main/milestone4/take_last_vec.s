.section .text
.global take_last_timestep
take_last_timestep:
    # Offset to last row: (4096-1)*64*4 = 1048320 bytes
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1
    slli    t0, t0, 2
    add     a0, a0, t0          # a0 = &in[4095][0]

    li      t0, 64
    vsetvli zero, t0, e32, m8, ta, ma  # VL=64, LMUL=8
    vle32.v v0, (a0)            # load all 64 floats (one vector op)
    vse32.v v0, (a1)            # store all 64 floats
    ret
