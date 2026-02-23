#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t input[BNN_INPUT_SIZE];
    float s0 = 0.0f;
    float s1 = 0.0f;

    memset(input, 0x5A, sizeof(input));

    bnn_init();
    if (bnn_score_1088(input, &s0) != 0) {
        fprintf(stderr, "inference failed on first run\n");
        return 1;
    }
    if (bnn_score_1088(input, &s1) != 0) {
        fprintf(stderr, "inference failed on second run\n");
        return 1;
    }

    printf("backend=%s\n", bnn_backend_name(bnn_backend_active()));
    printf("score_run0=%.1f\n", s0);
    printf("score_run1=%.1f\n", s1);
    printf("deterministic=%s\n", (s0 == s1) ? "yes" : "no");

    return (s0 == s1) ? 0 : 2;
}
