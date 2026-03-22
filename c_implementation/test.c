#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn.h"

#define TOL_EXACT_MSE   1e-12f
#define TOL_EXACT_MAE   1e-10f
#define TOL_LINEAR_MSE  1e-8f
#define TOL_LINEAR_MAE  1e-6f
#define TOL_S4D_MSE     1e-7f
#define TOL_S4D_MAE     1e-4f
#define TOL_GELU_MSE    1e-7f
#define TOL_GELU_MAE    1e-4f
#define TOL_SOFTMAX_MSE 1e-8f
#define TOL_SOFTMAX_MAE 1e-4f

static float compute_mse(const float *a, const float *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i]-b[i]; sum += d*d; }
    return (float)(sum / n);
}

static float compute_mae(const float *a, const float *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i]-b[i]; sum += d<0?-d:d; }
    return (float)(sum / n);
}

static float compute_max(const float *a, const float *b, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; i++) { float d=a[i]-b[i]; if(d<0)d=-d; if(d>mx)mx=d; }
    return mx;
}

static float *load_bin(const char *path, int n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  [MISSING] %s\n", path); return NULL; }
    float *buf = (float *)malloc(n * sizeof(float));
    fread(buf, sizeof(float), n, f);
    fclose(f);
    return buf;
}

static int check(const char *name, const float *c, const float *ref,
                 int n, float mse_tol, float mae_tol)
{
    float mse = compute_mse(c, ref, n);
    float mae = compute_mae(c, ref, n);
    float mx  = compute_max(c, ref, n);
    int pass  = (mse <= mse_tol) && (mae <= mae_tol);
    printf("  %-18s  n=%-8d  MSE=%.2e  MAE=%.2e  MaxErr=%.2e  %s\n",
           name, n, mse, mae, mx, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("    [DBG] C  : "); for(int i=0;i<4&&i<n;i++) printf("%.6f ",c[i]);   printf("\n");
        printf("    [DBG] ref: "); for(int i=0;i<4&&i<n;i++) printf("%.6f ",ref[i]); printf("\n");
    }
    return pass;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sample_prefix> [weights.bin]\n", argv[0]);
        return 1;
    }
    const char *prefix       = argv[1];
    const char *weights_path = (argc >= 3) ? argv[2] : "../model_weights.bin";

    static ModelParams params;
    if (load_weights(weights_path, &params) != 0) return 1;

    char path[512];
    snprintf(path, sizeof(path), "%s_input.bin", prefix);
    float *img = load_bin(path, C_IN * 64 * 64);
    if (!img) return 1;

    int true_label = -1;
    snprintf(path, sizeof(path), "%s_label.txt", prefix);
    FILE *lf = fopen(path, "r");
    if (lf) { fscanf(lf, "%d", &true_label); fclose(lf); }

    printf("============================================================\n");
    printf("Validation: %s   (true label: %d)\n", prefix, true_label);
    printf("============================================================\n");

    int all_pass = 1;

    static float c_hilbert[SEQ_LEN * C_IN];
    static float c_linear [SEQ_LEN * D_MODEL];
    static float c_s4d1   [SEQ_LEN * D_MODEL];
    static float c_gelu1  [SEQ_LEN * D_MODEL];
    static float c_s4d2   [SEQ_LEN * D_MODEL];
    static float c_gelu2  [SEQ_LEN * D_MODEL];
    static float c_pooled [D_MODEL];
    static float c_logits [N_CLASSES];
    float        c_probs  [N_CLASSES];

    hilbert_scan(&params, img, c_hilbert);
    snprintf(path, sizeof(path), "%s_hilbert.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*C_IN);
      if (ref) { all_pass &= check("hilbert_scan", c_hilbert, ref, SEQ_LEN*C_IN, TOL_EXACT_MSE, TOL_EXACT_MAE); free(ref); } }

    linear_layer((const float*)params.uproject_weight, params.uproject_bias,
                 c_hilbert, c_linear, C_IN, D_MODEL, SEQ_LEN);
    snprintf(path, sizeof(path), "%s_linear.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*D_MODEL);
      if (ref) { all_pass &= check("uproject", c_linear, ref, SEQ_LEN*D_MODEL, TOL_LINEAR_MSE, TOL_LINEAR_MAE); free(ref); } }

    s4d_layer(params.s4_1_log_dt, (const float*)params.s4_1_log_A_real,
              (const float*)params.s4_1_A_imag, (const float*)params.s4_1_C,
              params.s4_1_D, c_linear, c_s4d1);
    snprintf(path, sizeof(path), "%s_s4d1.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*D_MODEL);
      if (ref) { all_pass &= check("s4d_layer_1", c_s4d1, ref, SEQ_LEN*D_MODEL, TOL_S4D_MSE, TOL_S4D_MAE); free(ref); } }

    memcpy(c_gelu1, c_s4d1, SEQ_LEN*D_MODEL*sizeof(float));
    gelu_inplace(c_gelu1, SEQ_LEN*D_MODEL);
    snprintf(path, sizeof(path), "%s_gelu1.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*D_MODEL);
      if (ref) { all_pass &= check("gelu_1", c_gelu1, ref, SEQ_LEN*D_MODEL, TOL_GELU_MSE, TOL_GELU_MAE); free(ref); } }

    s4d_layer(params.s4_2_log_dt, (const float*)params.s4_2_log_A_real,
              (const float*)params.s4_2_A_imag, (const float*)params.s4_2_C,
              params.s4_2_D, c_gelu1, c_s4d2);
    snprintf(path, sizeof(path), "%s_s4d2.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*D_MODEL);
      if (ref) { all_pass &= check("s4d_layer_2", c_s4d2, ref, SEQ_LEN*D_MODEL, TOL_S4D_MSE, TOL_S4D_MAE); free(ref); } }

    memcpy(c_gelu2, c_s4d2, SEQ_LEN*D_MODEL*sizeof(float));
    gelu_inplace(c_gelu2, SEQ_LEN*D_MODEL);
    snprintf(path, sizeof(path), "%s_gelu2.bin", prefix);
    { float *ref = load_bin(path, SEQ_LEN*D_MODEL);
      if (ref) { all_pass &= check("gelu_2", c_gelu2, ref, SEQ_LEN*D_MODEL, TOL_GELU_MSE, TOL_GELU_MAE); free(ref); } }

    take_last_timestep(c_gelu2, c_pooled);
    snprintf(path, sizeof(path), "%s_pooled.bin", prefix);
    { float *ref = load_bin(path, D_MODEL);
      if (ref) { all_pass &= check("take_last", c_pooled, ref, D_MODEL, TOL_EXACT_MSE, TOL_EXACT_MAE); free(ref); } }

    linear_layer((const float*)params.fc_weight, params.fc_bias,
                 c_pooled, c_logits, D_MODEL, N_CLASSES, 1);
    snprintf(path, sizeof(path), "%s_logits.bin", prefix);
    { float *ref = load_bin(path, N_CLASSES);
      if (ref) { all_pass &= check("fc_head", c_logits, ref, N_CLASSES, TOL_LINEAR_MSE, TOL_LINEAR_MAE); free(ref); } }

    memcpy(c_probs, c_logits, N_CLASSES*sizeof(float));
    softmax_inplace(c_probs, N_CLASSES);
    snprintf(path, sizeof(path), "%s_probs.bin", prefix);
    { float *ref = load_bin(path, N_CLASSES);
      if (ref) { all_pass &= check("softmax", c_probs, ref, N_CLASSES, TOL_SOFTMAX_MSE, TOL_SOFTMAX_MAE); free(ref); } }

    int pred = argmax(c_probs, N_CLASSES);
    printf("\nPredicted: %d  |  True: %d  |  %s\n",
           pred, true_label,
           (true_label < 0 || pred == true_label) ? "CORRECT" : "WRONG");
    printf("Probabilities: ");
    for (int i = 0; i < N_CLASSES; i++) printf("%.4f ", c_probs[i]);
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Overall: %s\n", all_pass ? "ALL LAYERS PASSED" : "SOME LAYERS FAILED");
    printf("============================================================\n\n");

    free(img);
    return all_pass ? 0 : 1;
}
