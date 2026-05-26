#!/bin/bash
#  Full performance benchmarking script
# Run from repo root: bash run_benchmark.sh

set -e
cd c_implementation

echo "=== Building all optimization levels ==="
gcc -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I. -O0    -o galaxy_bench_O0    ../benchmark.c nn.c -lm
gcc -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I. -O1    -o galaxy_bench_O1    ../benchmark.c nn.c -lm
gcc -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I. -O2    -o galaxy_bench_O2    ../benchmark.c nn.c -lm
gcc -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I. -O3    -o galaxy_bench_O3    ../benchmark.c nn.c -lm
gcc -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I. -Ofast -o galaxy_bench_Ofast ../benchmark.c nn.c -lm

echo ""
echo "=== Running benchmarks ==="

for opt in O0 O1 O2 O3 Ofast; do
    echo ""
    echo "---------- -$opt ----------"
    ./galaxy_bench_$opt ../test_data/sample_09_input.bin ../model_weights.bin
done > ../benchmark_results.txt 2>&1

# Also generate assembly dumps
echo ""
echo "=== Generating assembly dumps ==="
gcc -std=c11 -S -O0 nn.c -o nn_O0.s
gcc -std=c11 -S -O2 nn.c -o nn_O2.s
gcc -std=c11 -S -O3 nn.c -o nn_O3.s

echo "Done! Results in benchmark_results.txt"
