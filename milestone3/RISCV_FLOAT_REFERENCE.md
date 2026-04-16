# RISC-V Floating Point (RV32F/RV64F) Quick Reference

## Floating Point Registers

### Register File
- f0-f31: 32 floating-point registers (32-bit each for RV32F)
- fa0-fa7: Float arguments (f10-f17)
- fs0-fs11: Saved (callee-saved) float registers
- f0-f7: Return value registers (first float return)

### ABI Calling Convention
| Reg | Name | Purpose | Preserved |
|-----|------|---------|-----------|
| fa0-fa7 | Float args | Arguments 0-7 | Caller |
| fs0-fs11 | Float saved | Callee-saved | Callee |
| ft0-ft11 | Float temp | Temporary | Caller |

## Load/Store Instructions

| Instruction | Format | Operands | Description |
|-------------|--------|----------|-------------|
| `flw` | flw rd, offset(rs1) | rd=dest, offset=imm, rs1=base | Load float32 from memory |
| `fsw` | fsw rs2, offset(rs1) | rs2=src, offset=imm, rs1=base | Store float32 to memory |

### Example
```risc-v
flw f0, 0(a0)          # Load float from address in a0 into f0
fsw f1, 4(sp)          # Store f1 to stack (sp+4)
```

## Arithmetic Instructions

### Single-Precision (32-bit)

| Instruction | Format | Meaning | Notes |
|-------------|--------|---------|-------|
| `fadd.s` | fadd.s rd, rs1, rs2 | rd = rs1 + rs2 | Float add |
| `fsub.s` | fsub.s rd, rs1, rs2 | rd = rs1 - rs2 | Float subtract |
| `fmul.s` | fmul.s rd, rs1, rs2 | rd = rs1 * rs2 | Float multiply |
| `fdiv.s` | fdiv.s rd, rs1, rs2 | rd = rs1 / rs2 | Float divide (slow!) |
| `fsqrt.s` | fsqrt.s rd, rs1 | rd = √rs1 | Square root |
| `fabs.s` | fabs.s rd, rs1 | rd = \|rs1\| | Absolute value |
| `fneg.s` | fneg.s rd, rs1 | rd = -rs1 | Negate |
| `fmax.s` | fmax.s rd, rs1, rs2 | rd = max(rs1, rs2) | Maximum (NaN handling) |
| `fmin.s` | fmin.s rd, rs1, rs2 | rd = min(rs1, rs2) | Minimum (NaN handling) |
| `fmadd.s` | fmadd.s rd, rs1, rs2, rs3 | rd = rs1*rs2 + rs3 | Fused multiply-add (fast!) |
| `fmsub.s` | fmsub.s rd, rs1, rs2, rs3 | rd = rs1*rs2 - rs3 | Fused multiply-subtract |

### Example: Accumulate dot product
```risc-v
# Slow version: separate multiply and add
flw f1, 0(a0)          # f1 = weight[i]
flw f2, 0(a1)          # f2 = input[i]
fmul.s f3, f1, f2      # f3 = f1 * f2
fadd.s f0, f0, f3      # f0 = f0 + f3

# Fast version: fused multiply-add (recommended!)
flw f1, 0(a0)          # f1 = weight[i]
flw f2, 0(a1)          # f2 = input[i]
fmadd.s f0, f1, f2, f0 # f0 = f0 + (f1 * f2) - one instruction!
```

## Comparison & Branch Instructions

| Instruction | Format | Description |
|-------------|--------|-------------|
| `feq.s` | feq.s rd, rs1, rs2 | rd = (rs1 == rs2) ? 1 : 0 |
| `flt.s` | flt.s rd, rs1, rs2 | rd = (rs1 < rs2) ? 1 : 0 |
| `fle.s` | fle.s rd, rs1, rs2 | rd = (rs1 <= rs2) ? 1 : 0 |
| `beq` | beq rs1, rs2, label | Branch if equal (compare registers) |
| `bne` | bne rs1, rs2, label | Branch if not equal |
| `blt` | blt rs1, rs2, label | Branch if less than |
| `ble` | ble rs1, rs2, label | Branch if less than or equal |
| `bge` | bge rs1, rs2, label | Branch if greater than or equal |

### Example: Find maximum
```risc-v
# Compare f0 and f1, store max in f0
fmax.s f0, f0, f1      # Recommended: fmax handles NaN

# Alternative (manual comparison):
flt.s t0, f0, f1       # t0 = 1 if f0 < f1
beq t0, x0, skip_swap  # if f0 >= f1, skip
fmv.s f0, f1           # f0 = f1
skip_swap:
```

## Float-to-Integer Conversions

| Instruction | Format | Description |
|-------------|--------|-------------|
| `fcvt.w.s` | fcvt.w.s rd, rs1 | rd = (int32)rs1 (float→int32) |
| `fcvt.wu.s` | fcvt.wu.s rd, rs1 | rd = (uint32)rs1 (float→uint32) |
| `fcvt.s.w` | fcvt.s.w rd, rs1 | rd = (float)rs1 (int32→float) |
| `fcvt.s.wu` | fcvt.s.wu rd, rs1 | rd = (float)rs1 (uint32→float) |
| `fmv.x.s` | fmv.x.s rd, rs1 | rd = bit_cast(rs1) (float bits as int) |
| `fmv.s.x` | fmv.s.x rd, rs1 | rd = bit_cast(rs1) (int bits as float) |

### Example: Debug print float as hex
```risc-v
flw f0, 0(sp)          # Load float to examine
fmv.x.s t0, f0         # Move float bits to integer register
# Now t0 contains the bit pattern of f0 (can print as hex)
```

## Important Rounding Modes

Default: Round to nearest, ties to even (RNE)

| Mode (frm field) | Abbrev | Behavior |
|------------------|--------|----------|
| 000 | RNE | Round to nearest, ties to even |
| 001 | RTZ | Round towards zero |
| 010 | RDN | Round down (toward -∞) |
| 011 | RUP | Round up (toward +∞) |
| 100 | RMM | Round to nearest, ties to max magnitude |

Usually don't need to change - RNE is fine for our use.

## IEEE 754 Single Precision (float32)

### Bit Layout
```
[Sign: 1 bit][Exponent: 8 bits][Mantissa: 23 bits]
  bit 31          bits 30-23         bits 22-0
```

### Special Values
| Pattern | Value |
|---------|-------|
| 0x00000000 | +0.0 |
| 0x80000000 | -0.0 |
| 0x7f800000 | +∞ |
| 0xff800000 | -∞ |
| 0x7f800001-0x7fffffff | NaN (quiet) |
| 0xff800001-0xffffffff | NaN (signaling) |

### Precision
- Mantissa: 23 bits → ~7 decimal digits
- Exponent range: -126 to 127 (biased by 127)
- Smallest normalized: ~1.17e-38
- Largest: ~3.40e38

## Multiply-Accumulate Pattern (Most Common)

### Efficient Dot Product
```risc-v
# Compute: acc = sum(weight[i] * input[i])
# Loop: for i = 0 to in_dim-1

li s0, 0                # Counter i = 0
fcvt.s.w f0, x0         # f0 = 0.0 (accumulator)

dot_loop:
    bge s0, a4, dot_done # if i >= in_dim, exit
    
    # Offset for weight[i]
    slli t0, s0, 2      # t0 = i * 4 (bytes)
    add t0, a0, t0      # t0 = &weight[i]
    
    # Load weight and input
    flw f1, 0(t0)       # f1 = weight[i]
    flw f2, (t1, s0)    # Alternative: need to calculate offset
    
    # Multiply-accumulate
    fmadd.s f0, f1, f2, f0 # f0 = f0 + (f1 * f2)
    
    addi s0, s0, 1      # i++
    j dot_loop

dot_done:
    # f0 now contains the dot product
```

## Numerically Stable Softmax

```risc-v
# Step 1: Find maximum
fmax.s f10, f0, f1  # f10 = max(f0, f1)
fmax.s f10, f10, f2 # f10 = max(f10, f2)
fmax.s f10, f10, f3 # f10 = max(all)

# Step 2: Shift and exp
fsub.s fa0, f0, f10 # fa0 = f0 - max
jal ra, exp         # fa0 = exp(fa0)
fsw fa0, 0(a0)      # Store exp[0]

# ... repeat for other values ...

# Step 3: Normalize (divide by sum)
fdiv.s f1, f1, f0   # f1 = f1 / sum
fsw f1, 0(a1)       # Store normalized
```

## Common Mistakes

### ❌ Mistake 1: Forgetting memory layout
```risc-v
# WRONG: treat sequence as contiguous in memory
lw t0, 0(a0)        # This loads an integer (not float!)
lw t1, 4(a0)        # Loads next integer

# RIGHT: use floating-point instructions
flw f0, 0(a0)       # Load float at offset 0
flw f1, 4(a0)       # Load float at offset 4 (next float32)
```

### ❌ Mistake 2: Not saving/restoring callee-saved registers
```risc-v
# WRONG: use fs0 without saving
linear_layer:
    # ... use fs0 ...
    ret             # fs0 is corrupted for caller!

# RIGHT: save and restore
linear_layer:
    addi sp, sp, -4
    sw fs0, 0(sp)
    # ... use fs0 ...
    lw fs0, 0(sp)
    addi sp, sp, 4
    ret
```

### ❌ Mistake 3: Integer vs float registers
```risc-v
# WRONG: load float into integer register
lw t0, 0(a0)        # t0 now has garbage bits (as integer)

# RIGHT: load into float register
flw f0, 0(a0)       # f0 has correct float value
```

### ❌ Mistake 4: Accumulating tiny errors with repeated multiplication
```risc-v
# SLOW & INACCURATE (computing powers by repeated multiply):
# exp(x*t) computed as exp_t = exp_prev * exp_step
# Errors accumulate!

# BETTER (recompute each time):
# exp(x*t) computed directly from x*t each iteration
fmul.s f1, f_x, f_t  # Fresh computation of x*t
jal ra, exp          # exp(x*t) - no accumulated error
```

## Performance Tips

1. **Use fmadd.s instead of fmul.s + fadd.s** (2 cycles → 1 cycle)
2. **Minimize fdiv.s** (divides are slow: 10+ cycles) - rescale instead if possible
3. **Prefetch next values** before waiting for slow operations
4. **Avoid dependency chains** - parallelize independent operations
5. **Use fmax.s/fmin.s** instead of comparison + branch when possible

## Testing Floating Point Code

### Check for special values:
```risc-v
# Print if NaN or Inf (useful for debugging)
# IEEE 754: Inf = 0x7f800000, NaN = any exponent=255 with non-zero mantissa
fmv.x.s t0, f0          # Get float bits as integer
li t1, 0x7f800000       # Inf mask
bne t0, t1, not_inf     # Branch if not inf
# ... handle inf ...
```

### Unit test pattern:
```risc-v
# Load test input
flw f0, 0(a0)
# Call function  
jal ra, exp
# Store output for comparison
fsw f0, 0(a1)
# Load expected
flw f1, 0(a2)
# Compute error
fsub.s f2, f0, f1
# Check if within tolerance (e.g., 1e-6)
```

---

**Remember**: Always test with a range of inputs:
- Normal values (0-10 range)
- Large values (100+)
- Small values (1e-6 to 1e-3)
- Negative values
- Special cases (0, very large, very small)
