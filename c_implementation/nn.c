#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float buf_hilbert[SEQ_LEN * C_IN];
static float buf_proj   [SEQ_LEN * D_MODEL];
static float buf_s4d1   [SEQ_LEN * D_MODEL];
static float buf_s4d2   [SEQ_LEN * D_MODEL];
static float buf_pooled [D_MODEL];
static float buf_logits [N_CLASSES];
static float s4d_kernel [SEQ_LEN];

int load_weights(const char *path, ModelParams *params)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open: %s\n", path); return -1; }
    size_t n;
    n = fread(params->hilbert_indices, sizeof(int32_t), SEQ_LEN, f);
    if (n != SEQ_LEN) goto err;
    n = fread(params->uproject_weight, sizeof(float), D_MODEL * C_IN, f);
    if (n != (size_t)(D_MODEL * C_IN)) goto err;
    n = fread(params->uproject_bias, sizeof(float), D_MODEL, f);
    if (n != D_MODEL) goto err;
    n = fread(params->s4_1_log_dt, sizeof(float), D_MODEL, f);
    if (n != D_MODEL) goto err;
    n = fread(params->s4_1_log_A_real, sizeof(float), D_MODEL * D_STATE, f);
    if (n != (size_t)(D_MODEL * D_STATE)) goto err;
    n = fread(params->s4_1_A_imag, sizeof(float), D_MODEL * D_STATE, f);
    if (n != (size_t)(D_MODEL * D_STATE)) goto err;
    n = fread(params->s4_1_C, sizeof(float), D_MODEL * D_STATE * 2, f);
    if (n != (size_t)(D_MODEL * D_STATE * 2)) goto err;
    n = fread(params->s4_1_D, sizeof(float), D_MODEL, f);
    if (n != D_MODEL) goto err;
    n = fread(params->s4_2_log_dt, sizeof(float), D_MODEL, f);
    if (n != D_MODEL) goto err;
    n = fread(params->s4_2_log_A_real, sizeof(float), D_MODEL * D_STATE, f);
    if (n != (size_t)(D_MODEL * D_STATE)) goto err;
    n = fread(params->s4_2_A_imag, sizeof(float), D_MODEL * D_STATE, f);
    if (n != (size_t)(D_MODEL * D_STATE)) goto err;
    n = fread(params->s4_2_C, sizeof(float), D_MODEL * D_STATE * 2, f);
    if (n != (size_t)(D_MODEL * D_STATE * 2)) goto err;
    n = fread(params->s4_2_D, sizeof(float), D_MODEL, f);
    if (n != D_MODEL) goto err;
    n = fread(params->fc_weight, sizeof(float), N_CLASSES * D_MODEL, f);
    if (n != (size_t)(N_CLASSES * D_MODEL)) goto err;
    n = fread(params->fc_bias, sizeof(float), N_CLASSES, f);
    if (n != N_CLASSES) goto err;
    fclose(f);
    return 0;
err:
    fprintf(stderr, "[ERROR] Unexpected EOF in %s\n", path);
    fclose(f);
    return -1;
}

/*
 * hilbert_scan
 * ------------
 * Reorders pixels from a 2D image into a 1D sequence following the
 * Hilbert curve traversal order. This preserves spatial locality:
 * pixels that are close in 2D space stay close in the 1D sequence,
 * which helps the S4 model capture spatial patterns effectively.
 *
 * The pre-loaded hilbert_indices array maps each sequence position d
 * to a flat 2D index computed as: flat2d = row * 64 + col
 * So for sequence position d:
 *   flat2d = hilbert_indices[d]   <- pre-computed Hilbert curve index
 *   row    = flat2d / 64          <- which row in the 64x64 image
 *   col    = flat2d % 64          <- which column in the 64x64 image
 *
 * Input layout:  img[c * 64 * 64 + flat2d]  (channel-major: C, H, W)
 * Output layout: out[d * C_IN + c]           (sequence-major: L, C)
 *
 * Parameters:
 *   p   - model params containing pre-loaded hilbert_indices[4096]
 *   img - input image of shape (C_IN, 64, 64) flattened row-major
 *   out - output sequence of shape (SEQ_LEN, C_IN) = (4096, 1)
 *
 * Validation: produces exact match with PyTorch reference (MSE = 0.0)
 * because this is pure index reordering with no floating point math.
 */
void hilbert_scan(const ModelParams *p, const float *img, float *out)
{
    int d, c;
    for (d = 0; d < SEQ_LEN; d++) {
        /* hilbert_indices[d] gives the flat 2D position (row*64 + col)
         * for the d-th step along the Hilbert curve.
         * row = flat2d / 64,  col = flat2d % 64 */
        int flat2d = p->hilbert_indices[d];
        for (c = 0; c < C_IN; c++)
            /* img is stored channel-major: pixel at (c, row, col)
             * lives at index c*4096 + flat2d */
            out[d * C_IN + c] = img[c * 64 * 64 + flat2d];
    }
}

/*
 * linear_layer
 * ------------
 * Implements an affine transformation: out = in * W^T + b
 * This is the standard fully-connected / linear layer used in two places:
 *   1. UProject: maps each pixel value (C_IN=1) to d_model=64 features
 *      Called with seq_len=SEQ_LEN=4096, in_dim=1,  out_dim=64
 *   2. FC head:  maps the pooled 64-dim vector to 4 class logits
 *      Called with seq_len=1,            in_dim=64, out_dim=4
 *
 * Weight layout: row-major (D_out, D_in)
 *   weight[o * in_dim + i] = W[o][i]
 *   This means each row o of the weight matrix corresponds to one
 *   output neuron, and the dot product with the input gives out[o].
 *
 * Handles both:
 *   - Sequence inputs (seq_len > 1): processes each timestep independently
 *     Input shape:  (seq_len, in_dim)
 *     Output shape: (seq_len, out_dim)
 *   - Vector inputs (seq_len = 1): single affine transform
 *     Input shape:  (in_dim,)
 *     Output shape: (out_dim,)
 *
 * Parameters:
 *   weight  - weight matrix of shape (out_dim, in_dim), row-major
 *   bias    - bias vector of shape (out_dim,)
 *   in      - input of shape (seq_len, in_dim)
 *   out     - output of shape (seq_len, out_dim)
 *   in_dim  - input feature dimension
 *   out_dim - output feature dimension
 *   seq_len - number of timesteps (use 1 for plain vector input)
 *
 * Validation: uproject achieves MSE=0.0 vs PyTorch (exact match)
 */
void linear_layer(const float *weight, const float *bias,
                  const float *in, float *out,
                  int in_dim, int out_dim, int seq_len)
{
    int t, o, i;
    /* loop over timesteps - works for both sequence (seq_len>1)
     * and single vector (seq_len=1) inputs */
    for (t = 0; t < seq_len; t++) {
        const float *x_t = in  + t * in_dim;   /* input at timestep t */
        float       *y_t = out + t * out_dim;  /* output at timestep t */
        for (o = 0; o < out_dim; o++) {
            /* bias initialises the accumulator - avoids a separate add */
            float acc = bias[o];
            /* weight row o: weight[o * in_dim + 0 .. in_dim-1]
             * row-major layout means W[o][i] = weight[o*in_dim + i] */
            const float *w_o = weight + o * in_dim;
            for (i = 0; i < in_dim; i++)
                acc += w_o[i] * x_t[i];  /* dot product W[o] . x_t */
            y_t[o] = acc;
        }
    }
}

void s4d_layer(const float *log_dt, const float *log_A_real,
               const float *A_imag, const float *C_mat,
               const float *D_vec, const float *in, float *out)
{
    int h, n, t, j;
    float A_bar_r[D_STATE], A_bar_i[D_STATE];
    float Ct_r[D_STATE],    Ct_i[D_STATE];
    float pow_r[D_STATE],   pow_i[D_STATE];

    for (h = 0; h < D_MODEL; h++) {
        float dt = expf(log_dt[h]);

        for (n = 0; n < D_STATE; n++) {
            float Ar = -expf(log_A_real[h * D_STATE + n]);
            float Ai =  A_imag[h * D_STATE + n];
            float dtAr = Ar * dt;
            float dtAi = Ai * dt;
            float mag  = expf(dtAr);
            A_bar_r[n] = mag * cosf(dtAi);
            A_bar_i[n] = mag * sinf(dtAi);
            float em1r = A_bar_r[n] - 1.0f;
            float em1i = A_bar_i[n];
            float Cr = C_mat[(h * D_STATE + n) * 2 + 0];
            float Ci = C_mat[(h * D_STATE + n) * 2 + 1];
            float num_r = Cr * em1r - Ci * em1i;
            float num_i = Cr * em1i + Ci * em1r;
            float Amag2 = Ar * Ar + Ai * Ai;
            Ct_r[n] = (num_r * Ar + num_i * Ai) / Amag2;
            Ct_i[n] = (num_i * Ar - num_r * Ai) / Amag2;
        }

        for (n = 0; n < D_STATE; n++) { pow_r[n] = 1.0f; pow_i[n] = 0.0f; }

        for (t = 0; t < SEQ_LEN; t++) {
            float kval = 0.0f;
            for (n = 0; n < D_STATE; n++)
                kval += 2.0f * (Ct_r[n] * pow_r[n] - Ct_i[n] * pow_i[n]);
            s4d_kernel[t] = kval;
            for (n = 0; n < D_STATE; n++) {
                float nr = pow_r[n] * A_bar_r[n] - pow_i[n] * A_bar_i[n];
                float ni = pow_r[n] * A_bar_i[n] + pow_i[n] * A_bar_r[n];
                pow_r[n] = nr; pow_i[n] = ni;
            }
        }

        float Dh = D_vec[h];
        for (t = 0; t < SEQ_LEN; t++) {
            float acc = Dh * in[t * D_MODEL + h];
            for (j = 0; j <= t; j++)
                acc += s4d_kernel[j] * in[(t - j) * D_MODEL + h];
            out[t * D_MODEL + h] = acc;
        }
    }
}

void gelu_inplace(float *x, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        float v = x[i];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void softmax_inplace(float *x, int n)
{
    int i;
    float mx = x[0];
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (i = 0; i < n; i++) x[i] /= sum;
}

void take_last_timestep(const float *in, float *out)
{
    memcpy(out, in + (SEQ_LEN - 1) * D_MODEL, D_MODEL * sizeof(float));
}

int argmax(const float *x, int n)
{
    int best = 0, i;
    for (i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

void forward(const ModelParams *p, const float *img, float *probs)
{
    hilbert_scan(p, img, buf_hilbert);
    linear_layer((const float *)p->uproject_weight, p->uproject_bias,
                 buf_hilbert, buf_proj, C_IN, D_MODEL, SEQ_LEN);
    s4d_layer(p->s4_1_log_dt, (const float *)p->s4_1_log_A_real,
              (const float *)p->s4_1_A_imag, (const float *)p->s4_1_C,
              p->s4_1_D, buf_proj, buf_s4d1);
    gelu_inplace(buf_s4d1, SEQ_LEN * D_MODEL);
    s4d_layer(p->s4_2_log_dt, (const float *)p->s4_2_log_A_real,
              (const float *)p->s4_2_A_imag, (const float *)p->s4_2_C,
              p->s4_2_D, buf_s4d1, buf_s4d2);
    gelu_inplace(buf_s4d2, SEQ_LEN * D_MODEL);
    take_last_timestep(buf_s4d2, buf_pooled);
    linear_layer((const float *)p->fc_weight, p->fc_bias,
                 buf_pooled, buf_logits, D_MODEL, N_CLASSES, 1);
    memcpy(probs, buf_logits, N_CLASSES * sizeof(float));
    softmax_inplace(probs, N_CLASSES);
}
