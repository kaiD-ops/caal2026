
.section .text
.global take_last_timestep

take_last_timestep:
    li      t0, 4095
    li      t1, 64
    mul     t0, t0, t1      # 4095 * 64 = 262080
    slli    t0, t0, 2       # * 4 = 1048320 bytes
    add     a0, a0, t0      # point to last row

    li      t0, 64          # D_MODEL elements to copy
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
