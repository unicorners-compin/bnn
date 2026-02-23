#include "bnn_infer.h"

#include <immintrin.h>

#include <stddef.h>
#include <string.h>

#define BNN_WORDS (BNN_INPUT_SIZE / sizeof(uint64_t))
#define BNN_BITS (BNN_INPUT_SIZE * 8u)

#if defined(__GNUC__) || defined(__clang__)
#define BNN_ALIGN64 __attribute__((aligned(64)))
#else
#define BNN_ALIGN64
#endif

static BNN_ALIGN64 uint64_t g_weights[BNN_WORDS];

static bnn_backend_t g_backend = BNN_BACKEND_SCALAR;
static int g_has_avx2 = 0;
static int g_has_avx512 = 0;
static int g_forced_backend = -1;

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

/*
 * Temporary deterministic weights.
 * ISSUE-0004 will replace this with exported model weights.
 */
static void init_weights(void) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    size_t i;

    for (i = 0; i < BNN_WORDS; ++i) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        g_weights[i] = s * 0x2545F4914F6CDD1DULL;
    }
}

static void detect_backend(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    g_has_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    g_has_avx512 = __builtin_cpu_supports("avx512f") ? 1 : 0;
#endif
#endif

    if (g_has_avx512) {
        g_backend = BNN_BACKEND_AVX512;
    } else if (g_has_avx2) {
        g_backend = BNN_BACKEND_AVX2;
    } else {
        g_backend = BNN_BACKEND_SCALAR;
    }
}

int bnn_init(void) {
    init_weights();
    detect_backend();
    return 0;
}

bnn_backend_t bnn_backend_active(void) {
    if (g_forced_backend >= 0) {
        return (bnn_backend_t)g_forced_backend;
    }
    return g_backend;
}

int bnn_backend_supported(bnn_backend_t backend) {
    switch (backend) {
        case BNN_BACKEND_SCALAR:
            return 1;
        case BNN_BACKEND_AVX2:
            return g_has_avx2;
        case BNN_BACKEND_AVX512:
            return g_has_avx512;
        default:
            return 0;
    }
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

int bnn_force_backend(bnn_backend_t backend) {
    if (!bnn_backend_supported(backend)) {
        return -1;
    }
    g_forced_backend = (int)backend;
    return 0;
}

void bnn_clear_forced_backend(void) {
    g_forced_backend = -1;
}

static float score_scalar(const uint8_t input[BNN_INPUT_SIZE]) {
    const uint8_t *p = input;
    unsigned matched = 0u;
    size_t i;

    for (i = 0; i < BNN_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor = ~(x ^ g_weights[i]);
        matched += popcnt64(xnor);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS);
}

static float score_avx2(const uint8_t input[BNN_INPUT_SIZE]) {
#if defined(__AVX2__)
    const __m256i all_ones = _mm256_set1_epi64x(-1);
    const uint8_t *p = input;
    unsigned matched = 0u;
    size_t i = 0;

    for (; i + 3 < BNN_WORDS; i += 4) {
        __m256i x = _mm256_loadu_si256((const __m256i *)p);
        __m256i w = _mm256_load_si256((const __m256i *)&g_weights[i]);
        __m256i xn = _mm256_xor_si256(x, w);
        __m256i xnor = _mm256_andnot_si256(xn, all_ones);
        uint64_t lanes[4];

        _mm256_storeu_si256((__m256i *)lanes, xnor);
        matched += popcnt64(lanes[0]);
        matched += popcnt64(lanes[1]);
        matched += popcnt64(lanes[2]);
        matched += popcnt64(lanes[3]);
        p += 4 * sizeof(uint64_t);
    }

    for (; i < BNN_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor = ~(x ^ g_weights[i]);
        matched += popcnt64(xnor);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS);
#else
    (void)input;
    return score_scalar(input);
#endif
}

static float score_avx512(const uint8_t input[BNN_INPUT_SIZE]) {
    /* AVX-512 kernel lands in ISSUE-0003; keep behavior identical for now. */
    return score_avx2(input);
}

int bnn_score_1088_on_backend(
    const uint8_t input[BNN_INPUT_SIZE],
    float *out_score,
    bnn_backend_t backend
) {
    if (input == NULL || out_score == NULL) {
        return -1;
    }
    if (!bnn_backend_supported(backend)) {
        return -2;
    }

    switch (backend) {
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

int bnn_score_1088(const uint8_t input[BNN_INPUT_SIZE], float *out_score) {
    bnn_backend_t backend = bnn_backend_active();
    return bnn_score_1088_on_backend(input, out_score, backend);
}
