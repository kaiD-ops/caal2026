#!/bin/bash
# build_vec.sh - Build and run M4 vectorized RISC-V implementation
# Usage: bash build_vec.sh [--run] [--test]
#   (no args)  : build only
#   --run      : build + run with whisper
#   --test     : build + run with --profileinst

set -e

AS="riscv32-unknown-elf-gcc"
ASFLAGS="-march=rv32gcv -mabi=ilp32f -nostdlib"
LD="riscv32-unknown-elf-ld"
LDFLAGS="-T veer/link.ld -m elf32lriscv"
OBJCOPY="riscv32-unknown-elf-objcopy"
WHISPER="whisper"
BUILD="build"

SRCS="main_vec.s linear_vec.s hilbert_vec.s take_last_vec.s \
      gelu_vec.s softmax_vec.s s4d_vec.s math_vec.s"

mkdir -p "$BUILD"

echo "=== Assembling M4 vectorized sources ==="
OBJS=""
for src in $SRCS; do
    obj="${BUILD}/${src%.s}.o"
    echo "  AS $src -> $obj"
    $AS $ASFLAGS -c "$src" -o "$obj"
    OBJS="$OBJS $obj"
done

echo "=== Linking ==="
$LD $LDFLAGS -o "${BUILD}/galaxy_vec.elf" $OBJS
echo "  -> ${BUILD}/galaxy_vec.elf"

echo "=== Generating HEX ==="
$OBJCOPY -O ihex "${BUILD}/galaxy_vec.elf" "${BUILD}/galaxy_vec.hex"
echo "  -> ${BUILD}/galaxy_vec.hex"

if [[ "$1" == "--test" ]]; then
    echo "=== Running with profiling ==="
    $WHISPER --configfile veer/whisper.json \
             --tohost 0xd0580000 --consoleio 0xd0580004 \
             -s 0x80000000 \
             --profileinst "${BUILD}/galaxy_vec_prof.log" \
             --consoleoutfile "${BUILD}/galaxy_vec_probs.txt" \
             "${BUILD}/galaxy_vec.hex"
    echo "Profile: ${BUILD}/galaxy_vec_prof.log"
    python3 count_instructions.py "${BUILD}/galaxy_vec_prof.log"

elif [[ "$1" == "--run" ]] || [[ -z "$1" ]]; then
    echo "=== Running inference ==="
    $WHISPER --configfile veer/whisper.json \
             --tohost 0xd0580000 --consoleio 0xd0580004 \
             -s 0x80000000 \
             --consoleoutfile "${BUILD}/galaxy_vec_probs.txt" \
             "${BUILD}/galaxy_vec.hex"
    echo "Output: ${BUILD}/galaxy_vec_probs.txt"
fi

echo "=== Build complete ==="
