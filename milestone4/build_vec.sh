#!/usr/bin/env bash
# =============================================================================
# build_vec.sh  –  Build and run the M4 RVV-vectorized S4D pipeline
#
# Usage:
#   ./build_vec.sh            # Compile + link + simulate (default sample_09)
#   ./build_vec.sh -c         # Clean build artifacts
#   ./build_vec.sh -e         # Re-run last compiled binary (skip compile)
#   ./build_vec.sh -s <N>     # Run sample N (0-based, default 9)
#
# Requires:
#   • riscv32-unknown-elf-as / riscv32-unknown-elf-ld on PATH
#     (built with --with-arch=rv32imfcv, --with-abi=ilp32f)
#   • whisper (VeeR-iSS) on PATH
#   • model_weights.bin  one level up  (../model_weights.bin)
#   • test_data/sample_NN_input.bin   (one level up, ../test_data/)
#
# Environment assumptions match the riscv-env-setup README.
# =============================================================================

set -e

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BLDDIR="$SRCDIR/build_vec"
VEER="$SRCDIR/../veer"        # link.ld and whisper.json live here

AS="riscv32-unknown-elf-as"
LD="riscv32-unknown-elf-ld"
SIM="whisper"

ARCH_FLAGS="-march=rv32imfcv -mabi=ilp32f"
AS_FLAGS="$ARCH_FLAGS -g"
LD_FLAGS="-T $VEER/link.ld"

SAMPLE=9
CLEAN=0
EXEC_ONLY=0

# --- Parse arguments ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CLEAN=1; shift ;;
        -e) EXEC_ONLY=1; shift ;;
        -s) SAMPLE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $CLEAN -eq 1 ]]; then
    echo "[clean] Removing $BLDDIR"
    rm -rf "$BLDDIR"
    exit 0
fi

mkdir -p "$BLDDIR/obj" "$BLDDIR/hex" "$BLDDIR/exe" "$BLDDIR/logs"

# Pad sample number to 2 digits
SAMPLE_PAD=$(printf "%02d" "$SAMPLE")
HEX="$BLDDIR/hex/s4d_vec.hex"
EXE="$BLDDIR/exe/s4d_vec.exe"
LOG="$BLDDIR/logs/s4d_vec_sample${SAMPLE_PAD}.txt"

if [[ $EXEC_ONLY -eq 0 ]]; then
    echo "[info] Target sample: sample_${SAMPLE_PAD}"
    echo "[info] Architecture: rv32imfcv / ilp32f"

    # Patch main_vec.s to load the correct sample (sed in-place copy)
    MAIN_TMP="$BLDDIR/main_vec_s${SAMPLE_PAD}.s"
    sed "s/sample_09_input/sample_${SAMPLE_PAD}_input/" \
        "$SRCDIR/main_vec.s" > "$MAIN_TMP"

    # Source files
    SOURCES=(
        "$MAIN_TMP"
        "$SRCDIR/math_vec.s"
        "$SRCDIR/hilbert_vec.s"
        "$SRCDIR/linear_vec.s"
        "$SRCDIR/s4d_vec.s"
        "$SRCDIR/gelu_vec.s"
        "$SRCDIR/take_last_vec.s"
        "$SRCDIR/softmax_vec.s"
    )

    OBJS=()
    for SRC in "${SOURCES[@]}"; do
        BASE=$(basename "$SRC" .s)
        OBJ="$BLDDIR/obj/${BASE}.o"
        echo "[as]  $SRC → $OBJ"
        $AS $AS_FLAGS -o "$OBJ" "$SRC"
        OBJS+=("$OBJ")
    done

    echo "[ld]  linking → $EXE"
    $LD $LD_FLAGS -o "$EXE" "${OBJS[@]}"

    echo "[hex] generating → $HEX"
    riscv32-unknown-elf-objcopy -O ihex "$EXE" "$HEX"
fi

echo "[sim] whisper --configfile $VEER/whisper.json --log $LOG $HEX"
$SIM --configfile "$VEER/whisper.json" --log "$LOG" "$HEX"

echo ""
echo "=== Simulation complete ==="
echo "Log: $LOG"

# --- Extract and display probabilities from log --------------------------------
echo ""
echo "=== Output probabilities (hex from console) ==="
grep -oP '[0-9a-f]{8}' "$LOG" | tail -4 | while read HEX; do
    python3 -c "
import struct
b = bytes.fromhex('$HEX')
v = struct.unpack('>f', b)[0]
print(f'  0x{\"$HEX\"} = {v:.6f}')
" 2>/dev/null || echo "  $HEX"
done

echo ""
echo "Classes: [0]=SmoothRound  [1]=SmoothCigar  [2]=EdgeOnDisk  [3]=UnbarredSpiral"
