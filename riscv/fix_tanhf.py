with open('math.s', 'r') as f:
    content = f.read()

old = '''tanhf:
    addi    sp, sp, -8
    fsw     fa0, 0(sp)                  # save x
    # saturation check
    fabs.s  ft0, fa0
    lui     t0, %hi(tanh_sat)
    flw     ft1, %lo(tanh_sat)(t0)
    flt.s   t1, ft0, ft1               # |x| < 4?
    beqz    t1, tanh_sat_case
    # compute e^(2x)
    lui     t0, %hi(math_two)
    flw     ft1, %lo(math_two)(t0)
    fmul.s  fa0, fa0, ft1              # 2x
    call    expf                       # fa0 = e^(2x)
    lui     t0, %hi(math_one)
    flw     ft1, %lo(math_one)(t0)
    fsub.s  ft2, fa0, ft1              # e^2x - 1
    fadd.s  ft3, fa0, ft1              # e^2x + 1
    fdiv.s  fa0, ft2, ft3
    addi    sp, sp, 8
    ret
tanh_sat_case:
    flw     fa1, 0(sp)                 # reload original x for sign
    lui     t0, %hi(math_one)
    flw     fa0, %lo(math_one)(t0)
    fsgnj.s fa0, fa0, fa1              # sign(x) * 1.0
    addi    sp, sp, 8'''

new = '''tanhf:
    # Save ra and original x using callee-saved float regs
    addi    sp, sp, -8
    sw      ra, 0(sp)
    fsw     fs2, 4(sp)                 # save fs2
    fmv.s   fs2, fa0                   # fs2 = original x (callee-saved)

    # saturation check: if |x| >= 4, return sign(x)
    fabs.s  ft0, fa0
    lui     t0, %hi(tanh_sat)
    flw     ft1, %lo(tanh_sat)(t0)
    flt.s   t1, ft0, ft1              # t1=1 if |x| < 4
    beqz    t1, tanh_sat_case         # if |x| >= 4, saturate

    # compute tanh(x) = (e^2x - 1)/(e^2x + 1)
    lui     t0, %hi(math_two)
    flw     ft0, %lo(math_two)(t0)
    fmul.s  fa0, fs2, ft0             # fa0 = 2x
    call    expf                      # fa0 = e^(2x), fs2 still valid

    # now compute (e^2x - 1)/(e^2x + 1)
    fmv.s   fs1, fa0                  # fs1 = e^2x (callee-saved)
    lui     t0, %hi(math_one)
    flw     ft0, %lo(math_one)(t0)
    fsub.s  fa0, fs1, ft0             # e^2x - 1
    fadd.s  ft1, fs1, ft0             # e^2x + 1
    fdiv.s  fa0, fa0, ft1             # result
    j       tanh_done

tanh_sat_case:
    # return sign(x) * 1.0
    lui     t0, %hi(math_one)
    flw     fa0, %lo(math_one)(t0)
    fsgnj.s fa0, fa0, fs2             # copy sign of original x

tanh_done:
    flw     fs2, 4(sp)
    lw      ra, 0(sp)
    addi    sp, sp, 8'''

if old in content:
    content = content.replace(old, new)
    print("tanhf fixed!")
else:
    print("ERROR: not found")
    idx = content.find("tanhf:")
    print(repr(content[idx:idx+200]))

with open('math.s', 'w') as f:
    f.write(content)
