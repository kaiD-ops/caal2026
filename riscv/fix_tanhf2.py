with open('math.s', 'r') as f:
    lines = f.readlines()

# Find tanhf function and replace it
start = None
end = None
for i, line in enumerate(lines):
    if line.strip() == 'tanhf:':
        start = i
    if start and i > start and line.strip() == 'ret' and end is None:
        # Find the second ret (end of tanh_sat_case)
        for j in range(i+1, min(i+10, len(lines))):
            if 'ret' in lines[j] or '# ====' in lines[j]:
                end = j
                break
        if end is None:
            end = i + 1
        break

print(f"Found tanhf at lines {start}-{end}")
print("Current tanhf:")
for i in range(start, min(end+2, len(lines))):
    print(f"  {i}: {lines[i]}", end='')

new_tanhf = [
    'tanhf:\n',
    '    addi    sp, sp, -12\n',
    '    sw      ra, 0(sp)\n',
    '    fsw     fs1, 4(sp)\n',
    '    fsw     fs2, 8(sp)\n',
    '    fmv.s   fs2, fa0\n',
    '    fabs.s  ft0, fa0\n',
    '    lui     t0, %hi(tanh_sat)\n',
    '    flw     ft1, %lo(tanh_sat)(t0)\n',
    '    flt.s   t1, ft0, ft1\n',
    '    beqz    t1, tanh_sat_case\n',
    '    lui     t0, %hi(math_two)\n',
    '    flw     ft0, %lo(math_two)(t0)\n',
    '    fmul.s  fa0, fs2, ft0\n',
    '    call    expf\n',
    '    fmv.s   fs1, fa0\n',
    '    lui     t0, %hi(math_one)\n',
    '    flw     ft0, %lo(math_one)(t0)\n',
    '    fsub.s  fa0, fs1, ft0\n',
    '    fadd.s  ft1, fs1, ft0\n',
    '    fdiv.s  fa0, fa0, ft1\n',
    '    j       tanh_done\n',
    'tanh_sat_case:\n',
    '    lui     t0, %hi(math_one)\n',
    '    flw     fa0, %lo(math_one)(t0)\n',
    '    fsgnj.s fa0, fa0, fs2\n',
    'tanh_done:\n',
    '    flw     fs2, 8(sp)\n',
    '    flw     fs1, 4(sp)\n',
    '    lw      ra, 0(sp)\n',
    '    addi    sp, sp, 12\n',
    '    ret\n',
]

# Find actual end (look for next function or section)
actual_end = end
for i in range(end, len(lines)):
    if '# ===' in lines[i] or ('.section' in lines[i] and i > start+5):
        actual_end = i
        break

print(f"\nReplacing lines {start} to {actual_end}")
lines = lines[:start] + new_tanhf + lines[actual_end:]

with open('math.s', 'w') as f:
    f.writelines(lines)

print("Done!")
