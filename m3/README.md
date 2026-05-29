# Milestone 3 — Scalar RISC-V Galaxy Classifier

**ISA:** RV32IMF · **ABI:** ilp32f · **Toolchain:** riscv32-unknown-elf GCC 15.2.0

## What This Is

Full forward-pass inference of an S4D sequence model for galaxy morphology classification.
Implemented entirely in bare-metal RISC-V assembly. No OS, no libc, no runtime.

- **Input:** 4096-pixel single-channel galaxy image
- **Model:** Hilbert scan → UProject → S4D×2 → GELU×2 → TakeLastTimestep → FC → Softmax
- **Output:** 4-class probability vector `[SmoothRound, SmoothCigar, EdgeOnDisk, UnbarredSpiral]`
- **Parameters:** SEQ_LEN=4096, D_MODEL=64, N_STATES=32, 84,496 weight bytes

## Files

| File | Purpose |
|------|---------|
| `main.s` | Pipeline driver: `.incbin` weights+sample, 9-stage calls, MMIO hex output, halt |
| `math.s` | Scalar transcendentals: `expf`, `cosf`, `sinf`, `tanhf` (Horner polynomial, range-reduced) |
| `hilbert_scan.s` | Hilbert curve pixel permutation (integer-index gather, 4096 elements) |
| `linear_layer.s` | FC layer: Y = X·Wᵀ + b (triple-nested scalar loop, `fmadd.s` dot product) |
| `s4d_layer.s` | S4D recurrence: Phase-1 transcendental precompute, Phase-2 kernel build, Phase-3 causal conv |
| `gelu.s` | In-place GELU activation (scalar `tanhf` per element) |
| `take_last_timestep.s` | Temporal pooling: copy row 4095 of [4096×64] buffer → [64] |
| `softmax.s` | Numerically-stable softmax (max-shift, exp, normalize) |
| `build.sh` | Build + simulate script |
| `veer/link.ld` | Linker script: code@0x80000000, data@0xf0040000, stack 32KB |
| `veer/whisper.json` | Whisper config: misa=0x40001126 (no V bit), 256 MB region |

## Build & Run

```bash
export PATH=/opt/riscv32imfcv/bin:$PATH
cd m3
bash build.sh          # assemble → link → hex → simulate (sample 09)
```

Manual:
```bash
# Assemble
for SRC in math.s hilbert_scan.s linear_layer.s s4d_layer.s \
           gelu.s take_last_timestep.s softmax.s main.s; do
    riscv32-unknown-elf-as -march=rv32imf -mabi=ilp32f -g \
        -o build/obj/${SRC%.s}.o $SRC
done

# Link
riscv32-unknown-elf-ld -T veer/link.ld -o build/exe/s4d.exe build/obj/*.o

# Simulate
/home/kai/VeeR-ISS/build-Linux/whisper \
    --configfile veer/whisper.json \
    --tohost 0xd0580000 --consoleio 0xd0580004 \
    --consoleoutfile build/logs/galaxy_probs.txt \
    --maxinst 20000000000 \
    build/exe/s4d.exe
```

## Output

Console writes 4 big-endian IEEE 754 hex words before halting:
```
3f66d583 3c339568 3d170d81 3d4eb4ee
→ [0.9017, 0.0110, 0.0369, 0.0505]  pred=0 (SmoothRound) ✓
```

Whisper success message: `Error: Failed stop: write to to-host: 255` — this is the expected clean exit.

## Performance (sample 09)

- Dynamic instructions: **16,003,496,500**
- Simulation rate: 43.3M inst/s
- Wall time: ~370s under Whisper ISS

## Memory Layout

```
0x80000000   .text         (code)
0xd0580000   .data.io      (MMIO: halt at +0, stdout at +4)
0xf0040000   .data/.bss    (weights 84KB incbin + activation buffers ~3MB)
```

Activation buffers (BSS): `buf_hilbert`=16KB, `buf_proj`/`buf_s4d1`/`buf_s4d2`=1MB each,
`buf_pooled`=256B, `buf_logits`=16B.

## Calling Convention

ILP32F ABI. Integer args: `a0–a6`. FP args: `fa0–fa7`. Return: `a0`/`fa0`.
Callee-saved: `s0–s11`, `fs0–fs11`. Caller-saved: `t0–t6`, `ft0–ft11`, `a0–a7`, `fa0–fa7`.
All layer functions use in-place modification via pointer arguments.
