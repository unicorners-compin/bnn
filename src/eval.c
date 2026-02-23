#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t input[BNN_INPUT_SIZE];
    float scalar_score = 0.0f;
    float avx2_score = 0.0f;

    memset(input, 0x5A, sizeof(input));

    bnn_init();

    if (bnn_score_1088_on_backend(input, &scalar_score, BNN_BACKEND_SCALAR) != 0) {
        fprintf(stderr, "scalar inference failed\n");
        return 1;
    }

    if (bnn_backend_supported(BNN_BACKEND_AVX2)) {
        if (bnn_score_1088_on_backend(input, &avx2_score, BNN_BACKEND_AVX2) != 0) {
            fprintf(stderr, "avx2 inference failed\n");
            return 1;
        }
    } else {
        avx2_score = scalar_score;
    }

    printf("default_backend=%s\n", bnn_backend_name(bnn_backend_active()));
    printf("scalar_score=%.1f\n", scalar_score);
    printf("avx2_score=%.1f\n", avx2_score);
    printf("scalar_avx2_match=%s\n", (scalar_score == avx2_score) ? "yes" : "no");

    return (scalar_score == avx2_score) ? 0 : 2;
}
