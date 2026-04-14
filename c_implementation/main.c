/*
 * main.c - Galaxy Classifier Demo Program
 * ========================================
 * Standalone bare-metal C inference demo for the S4D galaxy classifier.
 * No external libraries, no PyTorch, no numpy - pure C11 with only
 * standard library (stdio, stdlib) and our own nn.h/nn.c.
 *
 * Usage:
 *   ./galaxy_infer <image.bin> [weights.bin]
 *
 *   image.bin   - binary file of C_IN*64*64 float32 values (channel-major)
 *   weights.bin - model parameters file (default: ../model_weights.bin)
 *
 * Design decisions (bare-metal constraints):
 *   - Static ModelParams struct: all weights in one compile-time struct,
 *     no heap allocation for parameters (mirrors RISC-V data section)
 *   - Single malloc for image only: avoids heap for all inference buffers
 *   - Direct binary weight loading: single fread() into flat struct layout
 *   - No abstraction layers: main calls forward() directly, no wrappers
 *
 * Output format matches milestone spec:
 *   Class probabilities for all 4 galaxy morphology classes
 *   Predicted class label (argmax of probabilities)
 */
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

static const char *CLASS_NAMES[N_CLASSES] = {
    "Smooth Round",
    "Smooth Cigar",
    "Edge-on Disk",
    "Unbarred Spiral"
};

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.bin> [weights.bin]\n", argv[0]);
        fprintf(stderr, "  image.bin  : %d floats (C_IN=%d, 64x64)\n",
                C_IN * 64 * 64, C_IN);
        return 1;
    }

    const char *img_path     = argv[1];
    const char *weights_path = (argc >= 3) ? argv[2] : "../model_weights.bin";

    static ModelParams params;
    printf("Loading model weights from: %s\n", weights_path);
    if (load_weights(weights_path, &params) != 0) return 1;
    printf("Model weights loaded successfully.\n\n");

    int img_floats = C_IN * 64 * 64;
    float *img = (float *)malloc(img_floats * sizeof(float));
    if (!img) { fprintf(stderr, "Out of memory\n"); return 1; }

    FILE *f = fopen(img_path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open image: %s\n", img_path);
        free(img); return 1;
    }
    size_t n = fread(img, sizeof(float), img_floats, f);
    fclose(f);
    if ((int)n != img_floats) {
        fprintf(stderr, "[ERROR] Image too short: expected %d floats, got %zu\n",
                img_floats, n);
        free(img); return 1;
    }
    printf("Running inference on: %s\n", img_path);

    float probs[N_CLASSES];
    forward(&params, img, probs);

    printf("\nClass Probabilities:\n");
    for (int i = 0; i < N_CLASSES; i++)
        printf("  Class %d (%-16s): %.4f\n", i, CLASS_NAMES[i], probs[i]);

    int pred = argmax(probs, N_CLASSES);
    printf("\nPredicted Class: %d (%s)\n", pred, CLASS_NAMES[pred]);

    free(img);
    return 0;
}
