#!/usr/bin/env bash
# =============================================================================
# build.sh - Build and run M3 scalar RISC-V pipeline
#
# Usage: ./build.sh [--sample N] [--clean] [--build-only]
# =============================================================================

set -e

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BLDDIR="$SRCDIR/build"

GCC_PREFIX="riscv32-unknown-elf"
ARCH="-march=rv32imf -mabi=ilp32f"
LINK="$SRCDIR/veer/link.ld"
WHISPER="${WHISPER:-/home/kai/VeeR-ISS/build-Linux/whisper}"
WHISPER_CFG="$SRCDIR/veer/whisper.json"

export PATH="/opt/riscv32imfcv/bin:$PATH"

SAMPLE=9
CLEAN=0
BUILD_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --sample) SAMPLE="$2"; shift 2 ;;
        --clean)  CLEAN=1 ;;
        --build-only) BUILD_ONLY=1 ;;
    esac
done

if [[ $CLEAN -eq 1 ]]; then
    echo "[clean] Removing $BLDDIR"
    rm -rf "$BLDDIR"
    exit 0
fi

mkdir -p "$BLDDIR/exe" "$BLDDIR/hex" "$BLDDIR/logs" "$BLDDIR/obj"

SOURCES=(
    "$SRCDIR/math.s"
    "$SRCDIR/hilbert_scan.s"
    "$SRCDIR/linear_layer.s"
    "$SRCDIR/s4d_layer.s"
    "$SRCDIR/gelu.s"
    "$SRCDIR/take_last_timestep.s"
    "$SRCDIR/softmax.s"
    "$SRCDIR/main.s"
)

OBJS=()
for SRC in "${SOURCES[@]}"; do
    BASE=$(basename "$SRC" .s)
    OBJ="$BLDDIR/obj/${BASE}.o"
    echo "[as]  $(basename $SRC) -> $OBJ"
    $GCC_PREFIX-gcc $ARCH -nostdlib -c "$SRC" -o "$OBJ"
    OBJS+=("$OBJ")
done

ELF="$BLDDIR/exe/galaxy.elf"
HEX="$BLDDIR/hex/galaxy.hex"

echo "[ld]  linking -> $ELF"
riscv32-unknown-elf-ld -T "$LINK" -m elf32lriscv -o "$ELF" "${OBJS[@]}"

echo "[hex] $ELF -> $HEX"
$GCC_PREFIX-objcopy -O ihex "$ELF" "$HEX"

echo "[ok]  Build complete: $HEX"

if [[ $BUILD_ONLY -eq 1 ]]; then
    exit 0
fi

SAMPLE_PAD=$(printf "%02d" "$SAMPLE")
ELF_S="$BLDDIR/exe/galaxy.elf"
PROBS="$BLDDIR/logs/galaxy_probs.txt"

echo ""
echo "[sim] Running whisper on sample_${SAMPLE_PAD} (s4d layers ~ a few min)..."
# --tohost lets the program halt the simulator; --consoleio + --consoleoutfile
# capture the bytes the program writes to 0xd0580004 (the 4 class probabilities
# as big-endian hex words).
"$WHISPER" --configfile "$WHISPER_CFG" \
           --tohost 0xd0580000 \
           --consoleio 0xd0580004 \
           --consoleoutfile "$PROBS" \
           "$ELF_S"

echo ""
echo "=== Class probabilities ==="
for hw in $(cat "$PROBS"); do
    python3 -c "
import struct
v = struct.unpack('>f', bytes.fromhex('$hw'))[0]
print(f'  0x$hw = {v:.6f}')
" 2>/dev/null || echo "  $hw"
done

echo ""
echo "Classes: [0]=SmoothRound  [1]=SmoothCigar  [2]=EdgeOnDisk  [3]=UnbarredSpiral"
echo "Probs file: $PROBS"
