#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a;
    const uint64_t y = *(const uint64_t *)b;
    if (x < y) {
        return -1;
    }
    if (x > y) {
        return 1;
    }
    return 0;
}

static void run_bench_backend(bnn_backend_t backend, int runs, int warmup) {
    uint8_t input[BNN_INPUT_SIZE];
    uint64_t *samples;
    uint64_t total_begin;
    uint64_t total_end;
    float score = 0.0f;
    int i;

    if (!bnn_backend_supported(backend)) {
        printf("backend=%s unsupported\n", bnn_backend_name(backend));
        return;
    }

    samples = (uint64_t *)malloc((size_t)runs * sizeof(uint64_t));
    if (samples == NULL) {
        fprintf(stderr, "malloc failed for benchmark samples\n");
        return;
    }

    memset(input, 0xA5, sizeof(input));

    if (bnn_force_backend(backend) != 0) {
        fprintf(stderr, "cannot force backend=%s\n", bnn_backend_name(backend));
        free(samples);
        return;
    }

    for (i = 0; i < warmup; ++i) {
        bnn_score_1088(input, &score);
        input[(size_t)i % BNN_INPUT_SIZE] ^= (uint8_t)(i & 0xFF);
    }

    total_begin = now_ns();
    for (i = 0; i < runs; ++i) {
        uint64_t t0 = now_ns();
        bnn_score_1088(input, &score);
        uint64_t t1 = now_ns();

        samples[i] = t1 - t0;
        input[(size_t)i % BNN_INPUT_SIZE] ^= (uint8_t)((i * 13) & 0xFF);
    }
    total_end = now_ns();

    qsort(samples, (size_t)runs, sizeof(samples[0]), cmp_u64);

    printf("backend=%s\n", bnn_backend_name(backend));
    printf("runs=%d\n", runs);
    printf("avg_latency_ns=%.2f\n", (double)(total_end - total_begin) / (double)runs);
    printf("avg_latency_us=%.4f\n", ((double)(total_end - total_begin) / (double)runs) / 1000.0);
    printf("p50_ns=%llu\n", (unsigned long long)samples[(size_t)runs * 50 / 100]);
    printf("p99_ns=%llu\n", (unsigned long long)samples[(size_t)runs * 99 / 100]);
    printf("last_score=%.1f\n", score);

    bnn_clear_forced_backend();
    free(samples);
}

int main(void) {
    enum { kWarmup = 200000, kRuns = 500000 };

    bnn_init();

    run_bench_backend(BNN_BACKEND_SCALAR, kRuns, kWarmup);
    run_bench_backend(BNN_BACKEND_AVX2, kRuns, kWarmup);
    run_bench_backend(BNN_BACKEND_AVX512, kRuns, kWarmup);

    return 0;
}
