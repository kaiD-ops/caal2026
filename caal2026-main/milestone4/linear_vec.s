.section .text
.global linear_forward
.global linear_vec_forward

linear_forward:
    addi    sp, sp, -64
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)
    sw      s10,44(sp)
    sw      s11,48(sp)
    fsw     fs0,52(sp)
    fsw     fs1,56(sp)
    fsw     fs2,60(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5
    mv      s6, a6

    slli    s7, s4, 2
    slli    s8, s5, 2

    li      t0, 0x3f800000
    fmv.w.x fs0, t0

    li      s9, 0
1:
    bge     s9, s6, 2f

    mul     t0, s9, s7
    add     s10, s2, t0
    mul     t0, s9, s8
    add     s11, s3, t0

    li      t3, 0
3:
    bge     t3, s5, 4f

    slli    t0, t3, 2
    add     t0, s1, t0
    flw     fs1, 0(t0)

    mul     t0, t3, s7
    add     t4, s0, t0

    mv      t0, s4
    mv      t1, s10
    mv      t2, t4
    
    vsetvli t5, t0, e32, m1, ta, ma
    vmv.v.i v2, 0

5:
    beqz    t0, 6f
    
    vsetvli t5, t0, e32, m1, ta, ma
    vle32.v v0, (t1)
    vle32.v v1, (t2)
    vfmacc.vv v2, v0, v1
    
    slli    t6, t5, 2
    add     t1, t1, t6
    add     t2, t2, t6
    sub     t0, t0, t5
    j       5b

6:
    vsetvli t5, s4, e32, m1, ta, ma
    vmv.v.i v3, 0
    vfredusum.vs v3, v2, v3
    vfmv.f.s fs2, v3
    
    fadd.s  fs2, fs2, fs1
    
    slli    t0, t3, 2
    add     t0, s11, t0
    fsw     fs2, 0(t0)

    addi    t3, t3, 1
    j       3b

4:
    addi    s9, s9, 1
    j       1b

2:
    flw     fs2,60(sp)
    flw     fs1,56(sp)
    flw     fs0,52(sp)
    lw      s11,48(sp)
    lw      s10,44(sp)
    lw      s9, 40(sp)
    lw      s8, 36(sp)
    lw      s7, 32(sp)
    lw      s6, 28(sp)
    lw      s5, 24(sp)
    lw      s4, 20(sp)
    lw      s3, 16(sp)
    lw      s2, 12(sp)
    lw      s1,  8(sp)
    lw      s0,  4(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 64
    ret

linear_vec_forward:
    addi    sp, sp, -48
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)
    sw      s10,44(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3
    mv      s4, a4
    mv      s5, a5

    slli    s6, s4, 2

    li      s7, 0
1:
    bge     s7, s5, 2f

    slli    t0, s7, 2
    add     t0, s2, t0
    flw     fs0, 0(t0)

    mul     t0, s7, s6
    add     s8, s1, t0

    mv      t0, s4
    mv      t1, s0
    mv      t2, s8
    
    vsetvli t5, t0, e32, m1, ta, ma
    vmv.v.i v2, 0

3:
    beqz    t0, 4f
    
    vsetvli t5, t0, e32, m1, ta, ma
    vle32.v v0, (t1)
    vle32.v v1, (t2)
    vfmacc.vv v2, v0, v1
    
    slli    t6, t5, 2
    add     t1, t1, t6
    add     t2, t2, t6
    sub     t0, t0, t5
    j       3b

4:
    vsetvli t5, s4, e32, m1, ta, ma
    vmv.v.i v3, 0
    vfredusum.vs v3, v2, v3
    vfmv.f.s fs1, v3
    
    fadd.s  fs1, fs1, fs0
    
    slli    t0, s7, 2
    add     t0, s3, t0
    fsw     fs1, 0(t0)

    addi    s7, s7, 1
    j       1b

2:
    lw      s10,44(sp)
    lw      s9, 40(sp)
    lw      s8, 36(sp)
    lw      s7, 32(sp)
    lw      s6, 28(sp)
    lw      s5, 24(sp)
    lw      s4, 20(sp)
    lw      s3, 16(sp)
    lw      s2, 12(sp)
    lw      s1,  8(sp)
    lw      s0,  4(sp)
    lw      ra,  0(sp)
    addi    sp, sp, 48
    ret