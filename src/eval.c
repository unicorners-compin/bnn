#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t input[BNN_INPUT_SIZE];
    float scalar_score = 0.0f;
    float avx2_score = 0.0f;
    float avx512_score = 0.0f;
    int avx2_match = 1;
    int avx512_match = 1;

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

    if (bnn_backend_supported(BNN_BACKEND_AVX512)) {
        if (bnn_score_1088_on_backend(input, &avx512_score, BNN_BACKEND_AVX512) != 0) {
            fprintf(stderr, "avx512 inference failed\n");
            return 1;
        }
    } else {
        avx512_score = scalar_score;
    }

    avx2_match = (scalar_score == avx2_score) ? 1 : 0;
    avx512_match = (scalar_score == avx512_score) ? 1 : 0;

    printf("default_backend=%s\n", bnn_backend_name(bnn_backend_active()));
    printf("active_model_id=%u\n", bnn_active_model_id());
    printf("scalar_score=%.1f\n", scalar_score);
    printf("avx2_score=%.1f\n", avx2_score);
    printf("avx512_score=%.1f\n", avx512_score);
    printf("scalar_avx2_match=%s\n", avx2_match ? "yes" : "no");
    printf("scalar_avx512_match=%s\n", avx512_match ? "yes" : "no");

    return (avx2_match && avx512_match) ? 0 : 2;
}
