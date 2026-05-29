#!/usr/bin/env bash
# Clean build+run of the working M4 RVV pipeline (no debug instrumentation).
# Uses the self-consistent source set the exe is built from:
#   main_vec_s09.s (.incbin weights + sample, M3-named layer calls)
#   + math_vec hilbert_vec linear_vec s4d_vec gelu_vec take_last_vec softmax_vec
# gelu/softmax expose *_vec symbols, so main's gelu_inplace/softmax_inplace
# calls are rewritten to the _vec names at assemble time.
set -e
cd /mnt/d/caal2026/m4
AS=/opt/riscv32imfcv/bin/riscv32-unknown-elf-as
LD=/opt/riscv32imfcv/bin/riscv32-unknown-elf-ld
OC=/opt/riscv32imfcv/bin/riscv32-unknown-elf-objcopy
W=/home/kai/VeeR-ISS/build-Linux/whisper
ARCH="-march=rv32imfcv -mabi=ilp32f"
SAMPLE=${1:-9}
SP=$(printf "%02d" "$SAMPLE")
mkdir -p build_vec/exe build_vec/hex build_vec/logs build_vec/obj

# patch sample label + gelu/softmax call names
sed -e "s/sample_09_input/sample_${SP}_input/" \
    -e "s/call[[:space:]]\+gelu_inplace\b/call gelu_inplace_vec/" \
    -e "s/call[[:space:]]\+softmax_inplace\b/call softmax_inplace_vec/" \
    build_vec/main_vec_s09.s > build_vec/main_run_s${SP}.s

SRCS="build_vec/main_run_s${SP}.s math_vec.s hilbert_vec.s linear_vec.s s4d_vec.s gelu_vec.s take_last_vec.s softmax_vec.s"
OBJS=""
for s in $SRCS; do o=build_vec/obj/$(basename ${s%.s}).o; $AS $ARCH -g -o "$o" "$s"; OBJS="$OBJS $o"; done
$LD -T veer/link.ld -o build_vec/exe/s4d_vec.exe $OBJS
$OC -O ihex build_vec/exe/s4d_vec.exe build_vec/hex/s4d_vec.hex
echo "[built] build_vec/exe/s4d_vec.exe"

LOG=build_vec/logs/probs_sample${SP}.txt
rm -f "$LOG"
timeout 400 $W --configfile veer/whisper.json --tohost 0xd0580000 --consoleio 0xd0580004 \
   --consoleoutfile "$LOG" --maxinst 4000000000 build_vec/exe/s4d_vec.exe 2>&1 | tail -2 || true
echo "=== probabilities (sample ${SP}) ==="
for hw in $(cat "$LOG"); do
  python3 -c "import struct;print('  %.6f'%struct.unpack('>f',bytes.fromhex('$hw'))[0])" 2>/dev/null || echo "  $hw"
done
echo "classes: 0=SmoothRound 1=SmoothCigar 2=EdgeOnDisk 3=UnbarredSpiral"
