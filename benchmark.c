/*
 * benchmark.c - Comprehensive performance benchmarking for S4D galaxy classifier
 * Measures per-layer timing, total inference time, and memory footprint
 * across all compiler optimization levels.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "nn.h"

#define N_ITER 100   /* number of iterations for timing */

/* High-resolution timer in seconds */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Per-layer intermediate buffers (same as nn.c) */
static float bench_hilbert[SEQ_LEN * C_IN];
static float bench_proj   [SEQ_LEN * D_MODEL];
static float bench_s4d1   [SEQ_LEN * D_MODEL];
static float bench_s4d2   [SEQ_LEN * D_MODEL];
static float bench_pooled [D_MODEL];
static float bench_logits [N_CLASSES];
static float bench_probs  [N_CLASSES];

int main(int argc, char *argv[])
{
    const char *img_path     = argc >= 2 ? argv[1] : "../test_data/sample_09_input.bin";
    const char *weights_path = argc >= 3 ? argv[2] : "../model_weights.bin";

    /* Load weights */
    static ModelParams params;
    if (load_weights(weights_path, &params) != 0) return 1;

    /* Load image */
    float img[C_IN * 64 * 64];
    FILE *f = fopen(img_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", img_path); return 1; }
    fread(img, sizeof(float), C_IN * 64 * 64, f);
    fclose(f);

    printf("=== S4D Galaxy Classifier - Performance Benchmark ===\n");
    printf("Image: %s\n", img_path);
    printf("Iterations: %d\n\n", N_ITER);

    /* Per-layer timing*/
    double t_hilbert = 0, t_proj = 0, t_s4d1 = 0, t_gelu1 = 0;
    double t_s4d2 = 0, t_gelu2 = 0, t_pool = 0, t_fc = 0, t_softmax = 0;
    double t_total = 0;

    /* Arrays for std dev computation */
    double iter_times[N_ITER];

    /* warm up */
    for (int w = 0; w < 5; w++) {
        forward(&params, img, bench_probs);
    }

    /* per-layer timing */
    double t0, t1;
    for (int it = 0; it < N_ITER; it++) {
        double iter_start = now();

        t0 = now(); hilbert_scan(&params, img, bench_hilbert);                                          t1 = now(); t_hilbert += t1 - t0;
        t0 = now(); linear_layer((const float*)params.uproject_weight, params.uproject_bias, bench_hilbert, bench_proj, C_IN, D_MODEL, SEQ_LEN); t1 = now(); t_proj += t1 - t0;
        t0 = now(); memcpy(bench_s4d1, bench_proj, SEQ_LEN * D_MODEL * sizeof(float));
                    s4d_layer(params.s4_1_log_dt, (const float*)params.s4_1_log_A_real, (const float*)params.s4_1_A_imag, (const float*)params.s4_1_C, params.s4_1_D, bench_proj, bench_s4d1); t1 = now(); t_s4d1 += t1 - t0;
        t0 = now(); gelu_inplace(bench_s4d1, SEQ_LEN * D_MODEL);                                        t1 = now(); t_gelu1 += t1 - t0;
        t0 = now(); s4d_layer(params.s4_2_log_dt, (const float*)params.s4_2_log_A_real, (const float*)params.s4_2_A_imag, (const float*)params.s4_2_C, params.s4_2_D, bench_s4d1, bench_s4d2); t1 = now(); t_s4d2 += t1 - t0;
        t0 = now(); gelu_inplace(bench_s4d2, SEQ_LEN * D_MODEL);                                        t1 = now(); t_gelu2 += t1 - t0;
        t0 = now(); take_last_timestep(bench_s4d2, bench_pooled);                                       t1 = now(); t_pool += t1 - t0;
        t0 = now(); linear_layer((const float*)params.fc_weight, params.fc_bias, bench_pooled, bench_logits, D_MODEL, N_CLASSES, 1); t1 = now(); t_fc += t1 - t0;
        t0 = now(); memcpy(bench_probs, bench_logits, N_CLASSES * sizeof(float));
                    softmax_inplace(bench_probs, N_CLASSES);                                             t1 = now(); t_softmax += t1 - t0;

        iter_times[it] = now() - iter_start;
        t_total += iter_times[it];
    }

    /* compute std dev of total */
    double mean = t_total / N_ITER;
    double var = 0;
    for (int i = 0; i < N_ITER; i++) var += (iter_times[i] - mean) * (iter_times[i] - mean);
    double stddev = sqrt(var / N_ITER);

    /* print per-layer results */
    printf("--- Per-Layer Timing (mean over %d iterations) ---\n", N_ITER);
    printf("%-20s  %10s  %8s\n", "Layer", "Mean (ms)", "% Total");
    printf("%-20s  %10s  %8s\n", "-----", "---------", "-------");

    double layers[] = {t_hilbert, t_proj, t_s4d1, t_gelu1, t_s4d2, t_gelu2, t_pool, t_fc, t_softmax};
    const char *names[] = {"hilbert_scan", "uproject", "s4d_layer_1", "gelu_1", "s4d_layer_2", "gelu_2", "take_last", "fc_head", "softmax"};
    int nlayers = 9;

    for (int i = 0; i < nlayers; i++) {
        double ms = layers[i] / N_ITER * 1000.0;
        double pct = layers[i] / t_total * 100.0;
        printf("  %-18s  %10.3f  %7.2f%%\n", names[i], ms, pct);
    }

    printf("\n--- Total Inference ---\n");
    printf("  Mean:    %.3f ms\n", mean * 1000.0);
    printf("  Std dev: %.3f ms\n", stddev * 1000.0);
    printf("  Throughput: %.2f images/sec\n", 1.0 / mean);

    /* Memory footprint */
    printf("\n--- Memory Footprint ---\n");
    size_t param_bytes = sizeof(ModelParams);
    size_t buf_bytes   = sizeof(bench_hilbert) + sizeof(bench_proj) +
                         sizeof(bench_s4d1) + sizeof(bench_s4d2) +
                         sizeof(bench_pooled) + sizeof(bench_logits);
    size_t kernel_bytes = SEQ_LEN * sizeof(float); /* s4d_kernel static buffer */
    size_t total_bytes  = param_bytes + buf_bytes + kernel_bytes;

    printf("  Model parameters:  %6.1f KB\n", param_bytes / 1024.0);
    printf("  Activation buffers:%6.1f KB\n", buf_bytes   / 1024.0);
    printf("  S4D kernel buffer: %6.1f KB\n", kernel_bytes/ 1024.0);
    printf("  Total:             %6.1f KB\n", total_bytes / 1024.0);

    return 0;
}
