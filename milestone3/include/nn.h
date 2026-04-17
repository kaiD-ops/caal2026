#ifndef NN_H
#define NN_H

#include <stdint.h>

/* Network Dimensions */
#define SEQ_LEN    4096  /* Sequence length (number of pixels in Hilbert sequence) */
#define C_IN       1     /* Input channels (grayscale image) */
#define D_MODEL    64    /* Model hidden dimension */
#define D_STATE    32    /* S4D state dimension */
#define N_CLASSES  4     /* Number of galaxy classes */

/* Parameter Structure - matches binary weight file layout exactly */
typedef struct {
    /* Hilbert indexing: pre-computed Hilbert curve indices */
    int32_t hilbert_indices[SEQ_LEN];  /* [4096] mapping sequence pos to flat 2D index */
    
    /* Input projection (U-project): linear layer C_IN -> D_MODEL */
    float uproject_weight[D_MODEL * C_IN];  /* [64, 1] row-major */
    float uproject_bias[D_MODEL];            /* [64] */
    
    /* S4D Layer 1 parameters */
    float s4_1_log_dt[D_MODEL];              /* [64] log(dt) values */
    float s4_1_log_A_real[D_MODEL * D_STATE]; /* [64, 32] real part of A */
    float s4_1_A_imag[D_MODEL * D_STATE];    /* [64, 32] imaginary part of A */
    float s4_1_C[D_MODEL * D_STATE * 2];     /* [64, 32, 2] complex C matrix */
    float s4_1_D[D_MODEL];                   /* [64] feedthrough */
    
    /* S4D Layer 2 parameters */
    float s4_2_log_dt[D_MODEL];              /* [64] */
    float s4_2_log_A_real[D_MODEL * D_STATE]; /* [64, 32] */
    float s4_2_A_imag[D_MODEL * D_STATE];    /* [64, 32] */
    float s4_2_C[D_MODEL * D_STATE * 2];     /* [64, 32, 2] */
    float s4_2_D[D_MODEL];                   /* [64] */
    
    /* FC Head (classification): linear layer D_MODEL -> N_CLASSES */
    float fc_weight[N_CLASSES * D_MODEL];   /* [4, 64] row-major */
    float fc_bias[N_CLASSES];                /* [4] */
} ModelParams;

/* Function Declarations - Assembly Routines */

/* hilbert_scan: Reorder pixels along Hilbert curve */
void hilbert_scan(const ModelParams *p, const float *img, float *out);

/* linear_layer: Affine transformation (batch matrix-vector multiplication) */
void linear_layer(const float *weight, const float *bias,
                  const float *in, float *out,
                  int in_dim, int out_dim, int seq_len);

/* s4d_layer: S4D state-space layer */
void s4d_layer(const float *log_dt, const float *log_A_real,
               const float *A_imag, const float *C_mat,
               const float *D_vec, const float *in, float *out);

/* gelu_inplace: GELU activation (in-place) */
void gelu_inplace(float *x, int n);

/* softmax_inplace: Softmax activation (in-place) */
void softmax_inplace(float *x, int n);

/* take_last_timestep: Extract final timestep (simple copy) */
void take_last_timestep(const float *in, float *out);

/* forward: Complete inference pipeline */
void forward(const ModelParams *p, const float *img, float *probs);

/* Math functions - must be implemented in math.s */
float expf_fast(float x);
float sinf_fast(float x);
float cosf_fast(float x);
float tanhf_fast(float x);
float sqrtf_fast(float x);

#endif
