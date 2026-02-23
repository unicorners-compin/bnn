#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void) {
    enum { kWarmup = 100000, kRuns = 1000000 };
    uint8_t input[BNN_INPUT_SIZE];
    float score = 0.0f;
    uint64_t t0;
    uint64_t t1;
    int i;

    memset(input, 0xA5, sizeof(input));

    bnn_init();

    for (i = 0; i < kWarmup; ++i) {
        bnn_score_1088(input, &score);
        input[(size_t)i % BNN_INPUT_SIZE] ^= (uint8_t)(i & 0xFF);
    }

    t0 = now_ns();
    for (i = 0; i < kRuns; ++i) {
        bnn_score_1088(input, &score);
        input[(size_t)i % BNN_INPUT_SIZE] ^= (uint8_t)((i * 13) & 0xFF);
    }
    t1 = now_ns();

    printf("backend=%s\n", bnn_backend_name(bnn_backend_active()));
    printf("runs=%d\n", kRuns);
    printf("avg_latency_ns=%.2f\n", (double)(t1 - t0) / (double)kRuns);
    printf("avg_latency_us=%.4f\n", ((double)(t1 - t0) / (double)kRuns) / 1000.0);
    printf("last_score=%.1f\n", score);

    return 0;
}
