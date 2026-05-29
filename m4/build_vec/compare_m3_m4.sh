#!/usr/bin/env bash
# Build+run M3 (scalar) and M4 (RVV) on the SAME model (model_weights.bin) and
# SAME sample; report dynamic instruction count + wall time + probs for each.
set -e
AS=/opt/riscv32imfcv/bin/riscv32-unknown-elf-as
LD=/opt/riscv32imfcv/bin/riscv32-unknown-elf-ld
W=/home/kai/VeeR-ISS/build-Linux/whisper
SAMPLE=${1:-9}; SP=$(printf "%02d" "$SAMPLE")
M3=/mnt/d/caal2026/m3
M4=/mnt/d/caal2026/m4
OUT=/mnt/d/caal2026/m4/build_vec/logs
mkdir -p "$OUT" /tmp/cmp

decode(){ for hw in $(cat "$1"); do python3 -c "import struct;print('%.4f'%struct.unpack('>f',bytes.fromhex('$hw'))[0])"; done | paste -sd' '; }

# ---------- M3 scalar ----------
cd "$M3"
sed "s/sample_09_input/sample_${SP}_input/" main.s > /tmp/cmp/m3_main.s
M3SRC="math.s hilbert_scan.s linear_layer.s s4d_layer.s gelu.s take_last_timestep.s softmax.s /tmp/cmp/m3_main.s"
O=""; for s in $M3SRC; do o=/tmp/cmp/m3_$(basename ${s%.s}).o; $AS -march=rv32imf -mabi=ilp32f -o "$o" "$s" 2>/dev/null; O="$O $o"; done
$LD -T veer/link.ld -o /tmp/cmp/m3.exe $O 2>/dev/null
echo "[M3 built]"
rm -f /tmp/cmp/m3_probs.txt
M3LINE=$(timeout 600 $W --configfile veer/whisper.json --tohost 0xd0580000 --consoleio 0xd0580004 \
   --consoleoutfile /tmp/cmp/m3_probs.txt --maxinst 20000000000 /tmp/cmp/m3.exe 2>&1 | grep -i "Retired")

# ---------- M4 RVV (same model_weights.bin, int indices -> no vfcvt) ----------
cd "$M4"
sed -e "s#\"../model_params/weights.bin\"#\"../model_weights.bin\"#" \
    -e "s/sample_09_input/sample_${SP}_input/" \
    -e "s/call[[:space:]]\+gelu_inplace\b/call gelu_inplace_vec/" \
    -e "s/call[[:space:]]\+softmax_inplace\b/call softmax_inplace_vec/" \
    build_vec/main_vec_s09.s > /tmp/cmp/m4_main.s
grep -v "vfcvt.rtz.x.f.v" hilbert_vec.s > /tmp/cmp/m4_hilbert.s    # int indices: drop float->int cvt
M4SRC="/tmp/cmp/m4_main.s math_vec.s /tmp/cmp/m4_hilbert.s linear_vec.s s4d_vec.s gelu_vec.s take_last_vec.s softmax_vec.s"
O=""; for s in $M4SRC; do o=/tmp/cmp/m4_$(basename ${s%.s}).o; $AS -march=rv32imfcv -mabi=ilp32f -o "$o" "$s" 2>/dev/null; O="$O $o"; done
$LD -T veer/link.ld -o /tmp/cmp/m4.exe $O 2>/dev/null
echo "[M4 built]"
rm -f /tmp/cmp/m4_probs.txt
M4LINE=$(timeout 600 $W --configfile veer/whisper.json --tohost 0xd0580000 --consoleio 0xd0580004 \
   --consoleoutfile /tmp/cmp/m4_probs.txt --maxinst 20000000000 /tmp/cmp/m4.exe 2>&1 | grep -i "Retired")

echo "=================== sample_${SP} ==================="
echo "M3 scalar : $M3LINE"
echo "          probs: $(decode /tmp/cmp/m3_probs.txt)"
echo "M4 RVV    : $M4LINE"
echo "          probs: $(decode /tmp/cmp/m4_probs.txt)"
