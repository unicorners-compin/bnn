#include "bnn_infer.h"
#include "bnn_weights.h"

#include <immintrin.h>

#include <stddef.h>
#include <string.h>

#define BNN_WORDS (BNN_INPUT_SIZE / sizeof(uint64_t))
#define BNN_BITS (BNN_INPUT_SIZE * 8u)

static bnn_backend_t g_backend = BNN_BACKEND_SCALAR;
static int g_has_avx2 = 0;
static int g_has_avx512 = 0;
static int g_has_avx512_vpopcntdq = 0;
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

static void detect_backend(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    g_has_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    g_has_avx512 = __builtin_cpu_supports("avx512f") ? 1 : 0;
    g_has_avx512_vpopcntdq = __builtin_cpu_supports("avx512vpopcntdq") ? 1 : 0;
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
    if (BNN_MODEL_WORDS != BNN_WORDS) {
        return -2;
    }
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
        const uint64_t xnor = ~(x ^ BNN_WEIGHTS[i]);
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
        __m256i w = _mm256_loadu_si256((const __m256i *)&BNN_WEIGHTS[i]);
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
        const uint64_t xnor = ~(x ^ BNN_WEIGHTS[i]);
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
#if defined(__AVX512F__)
    const uint8_t *p = input;
    unsigned matched = 0u;
    size_t i = 0;

    for (; i + 7 < BNN_WORDS; i += 8) {
        __m512i x = _mm512_loadu_si512((const void *)p);
        __m512i w = _mm512_loadu_si512((const void *)&BNN_WEIGHTS[i]);
        __m512i xnor;

        xnor = _mm512_xor_si512(x, w);
        xnor = _mm512_xor_si512(xnor, _mm512_set1_epi64(-1));

#if defined(__AVX512VPOPCNTDQ__)
        if (g_has_avx512_vpopcntdq) {
            __m512i pc = _mm512_popcnt_epi64(xnor);
            uint64_t lanes[8];
            _mm512_storeu_si512((void *)lanes, pc);
            matched += (unsigned)lanes[0];
            matched += (unsigned)lanes[1];
            matched += (unsigned)lanes[2];
            matched += (unsigned)lanes[3];
            matched += (unsigned)lanes[4];
            matched += (unsigned)lanes[5];
            matched += (unsigned)lanes[6];
            matched += (unsigned)lanes[7];
        } else
#endif
        {
            uint64_t lanes[8];
            _mm512_storeu_si512((void *)lanes, xnor);
            matched += popcnt64(lanes[0]);
            matched += popcnt64(lanes[1]);
            matched += popcnt64(lanes[2]);
            matched += popcnt64(lanes[3]);
            matched += popcnt64(lanes[4]);
            matched += popcnt64(lanes[5]);
            matched += popcnt64(lanes[6]);
            matched += popcnt64(lanes[7]);
        }

        p += 8 * sizeof(uint64_t);
    }

    for (; i < BNN_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor_tail = ~(x ^ BNN_WEIGHTS[i]);
        matched += popcnt64(xnor_tail);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS);
#else
    return score_avx2(input);
#endif
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
