#!/usr/bin/env bash
# Build+run M4 RVV vectorized on all 12 samples; collect probs + argmax.
cd /mnt/d/caal2026/m4
AS=/opt/riscv32imfcv/bin/riscv32-unknown-elf-as
LD=/opt/riscv32imfcv/bin/riscv32-unknown-elf-ld
W=/home/kai/VeeR-ISS/build-Linux/whisper
ARCH="-march=rv32imfcv -mabi=ilp32f"
mkdir -p build_vec/exe build_vec/obj build_vec/logs
RESULT=build_vec/logs/all_samples_vec.txt
: > "$RESULT"

for N in $(seq 0 11); do
  SP=$(printf "%02d" "$N")
  sed -e "s/sample_09_input/sample_${SP}_input/" \
      -e "s/call[[:space:]]\+gelu_inplace\b/call gelu_inplace_vec/" \
      -e "s/call[[:space:]]\+softmax_inplace\b/call softmax_inplace_vec/" \
      build_vec/main_vec_s09.s > build_vec/main_run_s${SP}.s
  SRCS="build_vec/main_run_s${SP}.s math_vec.s hilbert_vec.s linear_vec.s s4d_vec.s gelu_vec.s take_last_vec.s softmax_vec.s"
  OBJS=""
  for s in $SRCS; do o=build_vec/obj/$(basename ${s%.s}).o; $AS $ARCH -g -o "$o" "$s" 2>/dev/null; OBJS="$OBJS $o"; done
  $LD -T veer/link.ld -o build_vec/exe/s4d_s${SP}.exe $OBJS 2>/dev/null
  LOG=build_vec/logs/probs_sample${SP}.txt
  rm -f "$LOG"
  timeout 400 $W --configfile veer/whisper.json --tohost 0xd0580000 --consoleio 0xd0580004 \
     --consoleoutfile "$LOG" --maxinst 4000000000 build_vec/exe/s4d_s${SP}.exe >/dev/null 2>&1
  python3 -c "
import struct,sys
try:
    ws=open('$LOG').read().split()
    p=[struct.unpack('>f',bytes.fromhex(w))[0] for w in ws]
    pred=max(range(len(p)),key=lambda i:p[i])
    print('sample_$SP  pred=%d  probs=[%s]'%(pred,', '.join('%.4f'%x for x in p)))
except Exception as e:
    print('sample_$SP  ERROR', e)
" | tee -a "$RESULT"
done
echo "=== DONE ==="
cat "$RESULT"
