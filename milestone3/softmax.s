# RISC-V Assembly Implementation of Softmax Activation
#
# Function: softmax_inplace
# Purpose: Converts logits to probability distribution via softmax
#
# Arguments:
#   a0 = pointer to 4 floats (logits), will be overwritten with probabilities
#
# Algorithm (numerically stable version):
#   1. Find max value among the 4 inputs
#   2. Subtract max from each input to prevent overflow
#   3. Apply exp to each shifted value
#   4. Sum all exp values
#   5. Divide each exp value by the sum => probabilities that sum to 1.0
#
# Formula: softmax(x_i) = exp(x_i - max_x) / sum_j(exp(x_j - max_x))
#
# The shift by max_x preserves the softmax property (shift-invariant)
# while preventing numerical overflow.
#
# This function requires an exp() routine from the math library.
# Assumed interface:
#   - fa0 = input value (single-precision float)
#   - call ra, exp
#   - fa0 = output (exp of input)
#
# Callee-saved registers: s0, s1, s2, s3

.section .text
.globl softmax_inplace
.align 2

softmax_inplace:
    # Save callee-saved registers
    addi sp, sp, -20
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw ra, 16(sp)

    mv s0, a0           # s0 = pointer to 4 floats (logits)

    # ===== STEP 1: Find maximum value among 4 inputs =====
    flw f0, 0(s0)       # f0 = logits[0]
    flw f1, 4(s0)       # f1 = logits[1]
    flw f2, 8(s0)       # f2 = logits[2]
    flw f3, 12(s0)      # f3 = logits[3]

    # Find max(logits[0], logits[1])
    fmax.s f0, f0, f1   # f0 = max(logits[0], logits[1])

    # Find max(f0, logits[2])
    fmax.s f0, f0, f2   # f0 = max(max(logits[0], logits[1]), logits[2])

    # Find max(f0, logits[3])
    fmax.s f0, f0, f3   # f0 = max of all 4 values

    # Store max in s1 (using floating-point register across calls is risky)
    # We use an alternative: keep max in f10 (callee-saved)
    fmv.s f10, f0       # f10 = max_logit (save for later use)

    # ===== STEP 2: Subtract max from each input and apply exp =====
    # This also initializes the exp array in place
    # Reload logits and subtract max, then exp

    # logits[0]
    flw f0, 0(s0)       # f0 = logits[0]
    fsub.s fa0, f0, f10 # fa0 = logits[0] - max
    jal ra, exp         # fa0 = exp(logits[0] - max)
    fsw fa0, 0(s0)      # logits[0] = exp(logits[0] - max)
    fmv.s f0, fa0       # f0 = exp(logits[0] - max) for accumulation

    # logits[1]
    flw f1, 4(s0)       # f1 = logits[1]
    fsub.s fa0, f1, f10 # fa0 = logits[1] - max
    jal ra, exp         # fa0 = exp(logits[1] - max)
    fsw fa0, 4(s0)      # logits[1] = exp(logits[1] - max)
    fadd.s f0, f0, fa0  # f0 = exp[0] + exp[1]

    # logits[2]
    flw f2, 8(s0)       # f2 = logits[2]
    fsub.s fa0, f2, f10 # fa0 = logits[2] - max
    jal ra, exp         # fa0 = exp(logits[2] - max)
    fsw fa0, 8(s0)      # logits[2] = exp(logits[2] - max)
    fadd.s f0, f0, fa0  # f0 = exp[0] + exp[1] + exp[2]

    # logits[3]
    flw f3, 12(s0)      # f3 = logits[3]
    fsub.s fa0, f3, f10 # fa0 = logits[3] - max
    jal ra, exp         # fa0 = exp(logits[3] - max)
    fsw fa0, 12(s0)     # logits[3] = exp(logits[3] - max)
    fadd.s f0, f0, fa0  # f0 = sum of all 4 exp values

    # ===== STEP 4: Normalize by sum =====
    # f0 now contains the sum of all exp values

    # logits[0] = exp[0] / sum
    flw f1, 0(s0)       # f1 = exp[0]
    fdiv.s f1, f1, f0   # f1 = exp[0] / sum
    fsw f1, 0(s0)       # logits[0] = exp[0] / sum

    # logits[1] = exp[1] / sum
    flw f1, 4(s0)       # f1 = exp[1]
    fdiv.s f1, f1, f0   # f1 = exp[1] / sum
    fsw f1, 4(s0)       # logits[1] = exp[1] / sum

    # logits[2] = exp[2] / sum
    flw f1, 8(s0)       # f1 = exp[2]
    fdiv.s f1, f1, f0   # f1 = exp[2] / sum
    fsw f1, 8(s0)       # logits[2] = exp[2] / sum

    # logits[3] = exp[3] / sum
    flw f1, 12(s0)      # f1 = exp[3]
    fdiv.s f1, f1, f0   # f1 = exp[3] / sum
    fsw f1, 12(s0)      # logits[3] = exp[3] / sum

    # Restore registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw ra, 16(sp)
    addi sp, sp, 20

    ret


# PLACEHOLDER: exp() function stub
# This will be provided by Member 2 in math.s
# Expected interface:
#   Input: fa0 = x (single-precision float)
#   Output: fa0 = exp(x)
#
# Member 2 will implement this using Taylor series or optimized libm routine

.globl exp
.align 2
# exp stub - to be replaced by Member 2
exp:
    # PLACEHOLDER: This is a stub
    # Member 2 will fill in the actual implementation
    # For now, return input (obviously wrong, just for testing)
    ret
