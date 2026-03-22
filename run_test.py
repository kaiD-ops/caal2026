
import os, sys, re, subprocess, argparse

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--weights",  default="model_weights.bin")

parser.add_argument("--test-dir", default="test_data")

parser.add_argument("--bin",      default="c_implementation/galaxy_test")

args = parser.parse_args()

if not os.path.isfile(args.bin):

    print(f"[ERROR] Test binary not found: {args.bin}")

    print("  Build it first:  cd c_implementation && make galaxy_test")

    sys.exit(1)

if not os.path.isfile(args.weights):

    print(f"[ERROR] Weights file not found: {args.weights}")

    print("  Generate it first:  python generate_test_data.py")

    sys.exit(1)

test_dir  = Path(args.test_dir)

input_bins = sorted(test_dir.glob("sample_*_input.bin"))

if not input_bins:

    print(f"[ERROR] No sample_*_input.bin files found in {test_dir}")

    sys.exit(1)

prefixes = [str(p).replace("_input.bin", "") for p in input_bins]

print(f"Found {len(prefixes)} test samples in {test_dir}/")

print(f"Using binary : {args.bin}")

print(f"Using weights: {args.weights}")

print("=" * 70)

LAYER_NAMES = ["hilbert_scan", "uproject", "s4d_layer_1", "gelu_1",

               "s4d_layer_2", "gelu_2", "take_last", "fc_head", "softmax"]

layer_mse_sum  = {k: 0.0 for k in LAYER_NAMES}

layer_mae_sum  = {k: 0.0 for k in LAYER_NAMES}

layer_pass_cnt = {k: 0   for k in LAYER_NAMES}

layer_total    = {k: 0   for k in LAYER_NAMES}

sample_pass = 0

sample_fail = 0

pred_correct = 0

pred_total   = 0

for prefix in prefixes:

    result = subprocess.run([args.bin, prefix, args.weights],

                            capture_output=True, text=True)

    output   = result.stdout

    all_pass = (result.returncode == 0)

    if all_pass: sample_pass += 1

    else:        sample_fail += 1

    for line in output.splitlines():

        m = re.match(

            r"\s+(\S+)\s+n=\d+\s+MSE=([\d.eE+\-]+)\s+MAE=([\d.eE+\-]+)\s+\S+=\S+\s+(\w+)",

            line)

        if m:

            name = m.group(1)

            mse  = float(m.group(2))

            mae  = float(m.group(3))

            ok   = m.group(4) == "PASS"

            if name in layer_mse_sum:

                layer_mse_sum[name]  += mse

                layer_mae_sum[name]  += mae

                layer_pass_cnt[name] += int(ok)

                layer_total[name]    += 1

    m = re.search(r"Predicted:\s+(\d+)\s+\|\s+True:\s+(\d+)", output)

    if m:

        pred_total += 1

        if m.group(1) == m.group(2): pred_correct += 1

    status = "PASS" if all_pass else "FAIL"

    print(f"  {os.path.basename(prefix):20s}  {status}")

print("\n" + "=" * 70)

print(f"Results: {sample_pass}/{len(prefixes)} samples passed")

print(f"Prediction accuracy: {pred_correct}/{pred_total} ({100*pred_correct/max(pred_total,1):.1f}%)")

print()

print(f"{'Layer':<20}  {'Avg MSE':>12}  {'Avg MAE':>12}  {'Pass':>8}")

print("-" * 60)

for name in LAYER_NAMES:

    n = layer_total[name]

    if n == 0: continue

    print(f"  {name:<18}  {layer_mse_sum[name]/n:>12.3e}  {layer_mae_sum[name]/n:>12.3e}  {layer_pass_cnt[name]:>3}/{n}")

print("=" * 70)

if sample_fail > 0:

    print(f"\n[FAIL] {sample_fail} sample(s) failed.")

    sys.exit(1)

else:

    print("\n[PASS] All samples passed.")

    sys.exit(0)

