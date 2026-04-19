# ============================================================
# gelu.s
# GELU activation: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
# Arguments: a0 = ptr to float array, a1 = length (in-place)
# ============================================================
.section .text
.global gelu_vec
.global _start

_start:
    la   sp, _stack_start
    
    # Test with sample data
    la   a0, test_data
    li   a1, 4
    call gelu_vec
    
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

gelu_vec:
    addi  sp, sp, -48
    sw    ra, 44(sp)
    sw    s0, 40(sp)    # ptr
    sw    s1, 36(sp)    # length
    sw    s2, 32(sp)    # loop index
    fsw   fs0, 28(sp)
    fsw   fs1, 24(sp)
    fsw   fs2, 20(sp)
    fsw   fs3, 16(sp)

    mv    s0, a0
    mv    s1, a1
    li    s2, 0

    # Load constants
    lui   t0, %hi(gelu_c1)
    flw   ft0, %lo(gelu_c1)(t0)   # 0.044715
    fmv.s fs1, ft0
    
    lui   t0, %hi(gelu_c2)
    flw   ft0, %lo(gelu_c2)(t0)   # sqrt(2/pi)
    fmv.s fs2, ft0
    
    lui   t0, %hi(one_f_g)
    flw   ft0, %lo(one_f_g)(t0)   # 1.0
    fmv.s fs3, ft0
    
    lui   t0, %hi(half_f)
    flw   ft0, %lo(half_f)(t0)    # 0.5
    fmv.s ft0, ft0

gelu_loop:
    bge   s2, s1, gelu_done

    slli  t0, s2, 2
    add   t1, s0, t0
    flw   fs0, 0(t1)     # fs0 = x

    # compute x^3
    fmul.s ft1, fs0, fs0   # x^2
    fmul.s ft1, ft1, fs0   # x^3

    # 0.044715 * x^3
    fmul.s ft1, fs1, ft1

    # x + 0.044715*x^3
    fadd.s ft1, fs0, ft1

    # sqrt(2/pi) * (x + 0.044715*x^3)
    fmul.s fa0, fs2, ft1

    # tanh(...)
    call  tanh_f

    # 1 + tanh
    fadd.s fa0, fa0, fs3

    # 0.5 * x * (1 + tanh)
    fmul.s fa0, fa0, fs0
    lui   t2, %hi(half_f)
    flw   ft1, %lo(half_f)(t2)
    fmul.s fa0, fa0, ft1

    # store back
    add   t1, s0, t0
    fsw   fa0, 0(t1)

    addi  s2, s2, 1
    j     gelu_loop
gelu_done:
    lw    ra, 44(sp)
    lw    s0, 40(sp)
    lw    s1, 36(sp)
    lw    s2, 32(sp)
    flw   fs0, 28(sp)
    flw   fs1, 24(sp)
    flw   fs2, 20(sp)
    flw   fs3, 16(sp)
    addi  sp, sp, 48
    ret

.section .rodata
.align 2
gelu_c1: .float 0.044715
gelu_c2: .float 0.7978845608   # sqrt(2/pi)
one_f_g: .float 1.0
half_f:  .float 0.5

# ============================================================
# Test data
# ============================================================
.section .data
.align 4
test_data:
    .float -2.0, -1.0, 0.0, 1.0, 2.0

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
