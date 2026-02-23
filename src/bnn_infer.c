#include "bnn_infer.h"
#include "bnn_weights.h"

#include <immintrin.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BNN_BITS (BNN_PAYLOAD_SIZE * 8u)
#define BNN_MAX_MODELS 16
#define BNN_WEIGHT_FILE_MAGIC 0x4D4E4E42u /* 'BNNM' little-endian */
#define BNN_WEIGHT_FILE_VERSION 1u

#ifndef BNN_MODEL_ID
#define BNN_MODEL_ID 0u
#endif

typedef struct bnn_model {
    int used;
    int owned;
    uint32_t model_id;
    int32_t bias;
    const uint64_t *weights;
} bnn_model_t;

typedef struct bnn_weight_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t model_id;
    uint32_t words;
    int32_t bias;
    uint32_t reserved;
} bnn_weight_file_header_t;

static bnn_backend_t g_backend = BNN_BACKEND_SCALAR;
static int g_has_avx2 = 0;
static int g_has_avx512 = 0;
static int g_has_avx512_vpopcntdq = 0;
static int g_forced_backend = -1;

static uint64_t g_route_mask = 0xFFu;
static uint8_t g_route_shift = 0u;
static uint32_t g_last_model_id = BNN_MODEL_ID;

static bnn_model_t g_models[BNN_MAX_MODELS];
static bnn_model_t *g_default_model = NULL;
static int g_models_ready = 0;

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

static void free_owned_models(void) {
    size_t i;
    for (i = 0; i < BNN_MAX_MODELS; ++i) {
        if (g_models[i].used && g_models[i].owned && g_models[i].weights != NULL) {
            free((void *)g_models[i].weights);
        }
        g_models[i].used = 0;
        g_models[i].owned = 0;
        g_models[i].model_id = 0u;
        g_models[i].bias = 0;
        g_models[i].weights = NULL;
    }
    g_default_model = NULL;
}

static bnn_model_t *find_model(uint32_t model_id) {
    size_t i;
    for (i = 0; i < BNN_MAX_MODELS; ++i) {
        if (g_models[i].used && g_models[i].model_id == model_id) {
            return &g_models[i];
        }
    }
    return NULL;
}

static int upsert_model(uint32_t model_id, int32_t bias, const uint64_t *weights, int owned) {
    bnn_model_t *m = find_model(model_id);
    size_t i;

    if (m == NULL) {
        for (i = 0; i < BNN_MAX_MODELS; ++i) {
            if (!g_models[i].used) {
                m = &g_models[i];
                break;
            }
        }
        if (m == NULL) {
            return -4;
        }
    } else if (m->owned && m->weights != NULL) {
        free((void *)m->weights);
    }

    m->used = 1;
    m->owned = owned;
    m->model_id = model_id;
    m->bias = bias;
    m->weights = weights;

    if (g_default_model == NULL || g_default_model->model_id == model_id) {
        g_default_model = m;
    }

    return 0;
}

static int init_models(void) {
    if (BNN_MODEL_WORDS != BNN_PAYLOAD_WORDS) {
        return -2;
    }
    free_owned_models();
    if (upsert_model(BNN_MODEL_ID, BNN_MODEL_BIAS, BNN_WEIGHTS, 0) != 0) {
        return -3;
    }
    g_default_model = find_model(BNN_MODEL_ID);
    g_last_model_id = BNN_MODEL_ID;
    g_models_ready = 1;
    return 0;
}

int bnn_init(void) {
    int rc;
    detect_backend();
    if (!g_models_ready) {
        rc = init_models();
        if (rc != 0) {
            return rc;
        }
    }
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

int bnn_set_route_rule(uint64_t mask, uint8_t shift) {
    if (shift >= 64u) {
        return -1;
    }
    g_route_mask = mask;
    g_route_shift = shift;
    return 0;
}

uint32_t bnn_pick_model_id(const uint8_t config[BNN_CONFIG_SIZE]) {
    uint64_t cfg = load_u64_unaligned(config);
    return (uint32_t)((cfg & g_route_mask) >> g_route_shift);
}

uint32_t bnn_active_model_id(void) {
    return g_last_model_id;
}

int bnn_set_default_model(uint32_t model_id) {
    bnn_model_t *m = find_model(model_id);
    if (m == NULL) {
        return -1;
    }
    g_default_model = m;
    return 0;
}

int bnn_load_model_file(const char *path) {
    FILE *fp;
    bnn_weight_file_header_t hdr;
    uint64_t *buf;
    size_t n;
    int rc;

    if (path == NULL) {
        return -1;
    }

    if (!g_models_ready) {
        rc = init_models();
        if (rc != 0) {
            return rc;
        }
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -2;
    }

    n = fread(&hdr, 1u, sizeof(hdr), fp);
    if (n != sizeof(hdr)) {
        fclose(fp);
        return -3;
    }
    if (hdr.magic != BNN_WEIGHT_FILE_MAGIC || hdr.version != BNN_WEIGHT_FILE_VERSION) {
        fclose(fp);
        return -4;
    }
    if (hdr.words != BNN_PAYLOAD_WORDS) {
        fclose(fp);
        return -5;
    }

    buf = (uint64_t *)malloc((size_t)hdr.words * sizeof(uint64_t));
    if (buf == NULL) {
        fclose(fp);
        return -6;
    }

    n = fread(buf, sizeof(uint64_t), hdr.words, fp);
    fclose(fp);
    if (n != hdr.words) {
        free(buf);
        return -7;
    }

    rc = upsert_model(hdr.model_id, hdr.bias, buf, 1);
    if (rc != 0) {
        free(buf);
        return rc;
    }

    return 0;
}

static float score_scalar_payload(const uint8_t payload[BNN_PAYLOAD_SIZE], const bnn_model_t *model) {
    const uint8_t *p = payload;
    unsigned matched = 0u;
    size_t i;

    for (i = 0; i < BNN_PAYLOAD_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor = ~(x ^ model->weights[i]);
        matched += popcnt64(xnor);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS + model->bias);
}

static float score_avx2_payload(const uint8_t payload[BNN_PAYLOAD_SIZE], const bnn_model_t *model) {
#if defined(__AVX2__)
    const __m256i all_ones = _mm256_set1_epi64x(-1);
    const uint8_t *p = payload;
    unsigned matched = 0u;
    size_t i = 0;

    for (; i + 3 < BNN_PAYLOAD_WORDS; i += 4) {
        __m256i x = _mm256_loadu_si256((const __m256i *)p);
        __m256i w = _mm256_loadu_si256((const __m256i *)&model->weights[i]);
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

    for (; i < BNN_PAYLOAD_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor = ~(x ^ model->weights[i]);
        matched += popcnt64(xnor);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS + model->bias);
#else
    return score_scalar_payload(payload, model);
#endif
}

static float score_avx512_payload(const uint8_t payload[BNN_PAYLOAD_SIZE], const bnn_model_t *model) {
#if defined(__AVX512F__)
    const uint8_t *p = payload;
    unsigned matched = 0u;
    size_t i = 0;

    for (; i + 7 < BNN_PAYLOAD_WORDS; i += 8) {
        __m512i x = _mm512_loadu_si512((const void *)p);
        __m512i w = _mm512_loadu_si512((const void *)&model->weights[i]);
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

    for (; i < BNN_PAYLOAD_WORDS; ++i) {
        const uint64_t x = load_u64_unaligned(p);
        const uint64_t xnor = ~(x ^ model->weights[i]);
        matched += popcnt64(xnor);
        p += sizeof(uint64_t);
    }

    return (float)((int)matched * 2 - (int)BNN_BITS + model->bias);
#else
    return score_avx2_payload(payload, model);
#endif
}

int bnn_score_1088_on_backend(
    const uint8_t input[BNN_INPUT_SIZE],
    float *out_score,
    bnn_backend_t backend
) {
    const uint8_t *config;
    const uint8_t *payload;
    uint32_t model_id;
    bnn_model_t *model;

    if (input == NULL || out_score == NULL) {
        return -1;
    }
    if (!bnn_backend_supported(backend)) {
        return -2;
    }
    if (!g_models_ready && bnn_init() != 0) {
        return -3;
    }

    config = input;
    payload = input + BNN_CONFIG_SIZE;
    model_id = bnn_pick_model_id(config);
    model = find_model(model_id);
    if (model == NULL) {
        model = g_default_model;
    }
    if (model == NULL) {
        return -4;
    }

    g_last_model_id = model->model_id;

    switch (backend) {
        case BNN_BACKEND_AVX512:
            *out_score = score_avx512_payload(payload, model);
            break;
        case BNN_BACKEND_AVX2:
            *out_score = score_avx2_payload(payload, model);
            break;
        case BNN_BACKEND_SCALAR:
        default:
            *out_score = score_scalar_payload(payload, model);
            break;
    }

    return 0;
}

int bnn_score_1088(const uint8_t input[BNN_INPUT_SIZE], float *out_score) {
    bnn_backend_t backend = bnn_backend_active();
    return bnn_score_1088_on_backend(input, out_score, backend);
}
