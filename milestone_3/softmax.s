# ============================================================
# softmax.s - Simplified version
# Softmax over 4 floats, in-place
# Arguments: a0 = ptr to 4 floats
# ============================================================
.section .text
.global softmax
.global _start

_start:
    la   sp, _stack_start
    la   a0, test_data
    call softmax
    
    li   t0, 0xd0580000
    li   t1, 1
    sw   t1, 0(t0)
    j    .

# ============================================================
# Simplified exponential (just for softmax)
# Uses: e^x ≈ 2^(x/ln(2)) via integer manipulation
# ============================================================
exp_f:
    # exp(x) for softmax - simple and fast
    # Convert x to integer exponent approximation
    
    # Clamp input to avoid overflow
    li    t0, 0x41000000    # 8.0
    fmv.w.x ft0, t0
    flt.s t0, fa0, ft0
    bnez  t0, clamp_neg
    
    li    t0, 0xc1000000    # -8.0
    fmv.w.x ft0, t0
    flt.s t0, ft0, fa0
    bnez  t0, clamp_pos
    
    # Approximate: e^x = 2^(x * log2(e))
    li    t0, 0x3fb8aa3b    # 1.44269504 (log2(e))
    fmv.w.x ft0, t0
    fmul.s fa0, fa0, ft0    # x * log2(e)
    
    # Convert to integer and add bias
    fcvt.w.s t0, fa0, rtz
    addi    t0, t0, 127
    slli    t0, t0, 23
    li      t1, 0x3f800000   # 1.0
    add     t0, t0, t1
    fmv.w.x fa0, t0
    ret
    
clamp_neg:
    # For large negative, return 0
    fmv.w.x fa0, zero
    ret
    
clamp_pos:
    # For large positive, return large number
    li    t0, 0x461c3c00    # ~10000.0
    fmv.w.x fa0, t0
    ret

# ============================================================
# softmax - Main function (unrolled for clarity)
# ============================================================
softmax:
    addi  sp, sp, -48
    sw    ra, 44(sp)
    sw    s0, 40(sp)
    fsw   fs0, 36(sp)
    fsw   fs1, 32(sp)
    fsw   fs2, 28(sp)
    fsw   fs3, 24(sp)

    mv    s0, a0
    beqz  s0, error

    # Step 1: Find maximum value
    flw   fs0, 0(s0)        # max = x[0]
    flw   ft0, 4(s0)
    fmax.s fs0, fs0, ft0
    flw   ft0, 8(s0)
    fmax.s fs0, fs0, ft0
    flw   ft0, 12(s0)
    fmax.s fs0, fs0, ft0

    # Step 2: Compute exp(x-max) for each element and sum
    # Element 0
    flw   fa0, 0(s0)
    fsub.s fa0, fa0, fs0
    call  exp_f
    fsw   fa0, 0(s0)
    fmv.s fs1, fa0          # sum = exp0
    
    # Element 1
    flw   fa0, 4(s0)
    fsub.s fa0, fa0, fs0
    call  exp_f
    fsw   fa0, 4(s0)
    fadd.s fs1, fs1, fa0    # sum += exp1
    
    # Element 2
    flw   fa0, 8(s0)
    fsub.s fa0, fa0, fs0
    call  exp_f
    fsw   fa0, 8(s0)
    fadd.s fs1, fs1, fa0    # sum += exp2
    
    # Element 3
    flw   fa0, 12(s0)
    fsub.s fa0, fa0, fs0
    call  exp_f
    fsw   fa0, 12(s0)
    fadd.s fs1, fs1, fa0    # sum += exp3

    # Step 3: Divide each by sum
    flw   fa0, 0(s0)
    fdiv.s fa0, fa0, fs1
    fsw   fa0, 0(s0)
    
    flw   fa0, 4(s0)
    fdiv.s fa0, fa0, fs1
    fsw   fa0, 4(s0)
    
    flw   fa0, 8(s0)
    fdiv.s fa0, fa0, fs1
    fsw   fa0, 8(s0)
    
    flw   fa0, 12(s0)
    fdiv.s fa0, fa0, fs1
    fsw   fa0, 12(s0)

    li    a0, 0
    j     done

error:
    li    a0, -1

done:
    flw   fs0, 36(sp)
    flw   fs1, 32(sp)
    flw   fs2, 28(sp)
    flw   fs3, 24(sp)
    lw    ra, 44(sp)
    lw    s0, 40(sp)
    addi  sp, sp, 48
    ret

# ============================================================
# Test data
# ============================================================
.section .data
.align 4
test_data:
    .float 1.0, 2.0, 3.0, 4.0

.section .bss
.align 4
_stack_start:
    .space 8192
_stack_end:
