#include "bnn_infer.h"

#include <immintrin.h>

#include <stddef.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

static bnn_backend_t g_backend = BNN_BACKEND_SCALAR;

static uint64_t load_u64_unaligned(const void *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static unsigned popcnt64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcountll(v);
#else
    unsigned c = 0;
    while (v != 0u) {
        c += (unsigned)(v & 1u);
        v >>= 1u;
    }
    return c;
#endif
}

static bnn_backend_t detect_backend(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) {
        return BNN_BACKEND_AVX512;
    }
    if (__builtin_cpu_supports("avx2")) {
        return BNN_BACKEND_AVX2;
    }
#endif
#endif
    return BNN_BACKEND_SCALAR;
}

int bnn_init(void) {
    g_backend = detect_backend();
    return 0;
}

bnn_backend_t bnn_backend_active(void) {
    return g_backend;
}

const char *bnn_backend_name(bnn_backend_t backend) {
    switch (backend) {
        case BNN_BACKEND_AVX512:
            return "avx512";
        case BNN_BACKEND_AVX2:
            return "avx2";
        case BNN_BACKEND_SCALAR:
        default:
            return "scalar";
    }
}

static float score_scalar(const uint8_t input[BNN_INPUT_SIZE]) {
    const size_t nwords = BNN_INPUT_SIZE / sizeof(uint64_t);
    const uint8_t *p = input;
    unsigned total = 0u;
    size_t i;

    for (i = 0; i < nwords; ++i) {
        total += popcnt64(load_u64_unaligned(p));
        p += sizeof(uint64_t);
    }

    return (float)total;
}

static float score_avx2(const uint8_t input[BNN_INPUT_SIZE]) {
#if defined(__AVX2__)
    const size_t nwords = BNN_INPUT_SIZE / sizeof(uint64_t);
    const __m256i all_ones = _mm256_set1_epi64x(-1);
    const uint8_t *p = input;
    unsigned total = 0u;
    size_t i = 0;

    for (; i + 3 < nwords; i += 4) {
        __m256i v = _mm256_loadu_si256((const __m256i *)p);
        __m256i x = _mm256_xor_si256(v, all_ones);
        uint64_t lanes[4];
        _mm256_storeu_si256((__m256i *)lanes, x);
        total += popcnt64(~lanes[0]);
        total += popcnt64(~lanes[1]);
        total += popcnt64(~lanes[2]);
        total += popcnt64(~lanes[3]);
        p += 4 * sizeof(uint64_t);
    }

    for (; i < nwords; ++i) {
        total += popcnt64(load_u64_unaligned(p));
        p += sizeof(uint64_t);
    }

    return (float)total;
#else
    (void)input;
    return 0.0f;
#endif
}

static float score_avx512(const uint8_t input[BNN_INPUT_SIZE]) {
    /* AVX-512 kernel will be implemented in ISSUE-0003. */
    return score_avx2(input);
}

int bnn_score_1088(const uint8_t input[BNN_INPUT_SIZE], float *out_score) {
    if (input == NULL || out_score == NULL) {
        return -1;
    }

    switch (g_backend) {
        case BNN_BACKEND_AVX512:
            *out_score = score_avx512(input);
            break;
        case BNN_BACKEND_AVX2:
            *out_score = score_avx2(input);
            break;
        case BNN_BACKEND_SCALAR:
        default:
            *out_score = score_scalar(input);
            break;
    }

    return 0;
}
