#ifndef NN_H
#define NN_H

#include <stdint.h>

#define SEQ_LEN    4096
#define D_MODEL    64
#define D_STATE    32
#define N_CLASSES  4
#define C_IN       1

typedef struct {
    int32_t hilbert_indices[SEQ_LEN];
    float uproject_weight[D_MODEL][C_IN];
    float uproject_bias[D_MODEL];
    float s4_1_log_dt[D_MODEL];
    float s4_1_log_A_real[D_MODEL][D_STATE];
    float s4_1_A_imag[D_MODEL][D_STATE];
    float s4_1_C[D_MODEL][D_STATE][2];
    float s4_1_D[D_MODEL];
    float s4_2_log_dt[D_MODEL];
    float s4_2_log_A_real[D_MODEL][D_STATE];
    float s4_2_A_imag[D_MODEL][D_STATE];
    float s4_2_C[D_MODEL][D_STATE][2];
    float s4_2_D[D_MODEL];
    float fc_weight[N_CLASSES][D_MODEL];
    float fc_bias[N_CLASSES];
} ModelParams;

int  load_weights(const char *path, ModelParams *params);
void hilbert_scan(const ModelParams *p, const float *img, float *out);
void linear_layer(const float *weight, const float *bias,
                  const float *in, float *out,
                  int in_dim, int out_dim, int seq_len);
void s4d_layer(const float *log_dt, const float *log_A_real,
               const float *A_imag, const float *C_mat,
               const float *D_vec, const float *in, float *out);
void gelu_inplace(float *x, int n);
void softmax_inplace(float *x, int n);
void take_last_timestep(const float *in, float *out);
void forward(const ModelParams *p, const float *img, float *probs);
int  argmax(const float *x, int n);

#endif
