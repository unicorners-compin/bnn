#ifndef BNN_INFER_H
#define BNN_INFER_H

#include <stddef.h>
#include <stdint.h>

#define BNN_INPUT_SIZE 1088u
#define BNN_CONFIG_SIZE 64u
#define BNN_PAYLOAD_SIZE 1024u
#define BNN_PAYLOAD_WORDS (BNN_PAYLOAD_SIZE / 8u)

typedef enum bnn_backend {
    BNN_BACKEND_SCALAR = 0,
    BNN_BACKEND_AVX2 = 1,
    BNN_BACKEND_AVX512 = 2
} bnn_backend_t;

int bnn_init(void);
bnn_backend_t bnn_backend_active(void);
int bnn_backend_supported(bnn_backend_t backend);
const char *bnn_backend_name(bnn_backend_t backend);
int bnn_force_backend(bnn_backend_t backend);
void bnn_clear_forced_backend(void);

/* 模型管理与路由 */
int bnn_load_model_file(const char *path);
int bnn_set_default_model(uint32_t model_id);
int bnn_set_route_rule(uint64_t mask, uint8_t shift);
uint32_t bnn_pick_model_id(const uint8_t config[BNN_CONFIG_SIZE]);
uint32_t bnn_active_model_id(void);

/*
 * Returns 0 on success and stores the score in out_score.
 * Input must point to exactly 1088 bytes.
 */
int bnn_score_1088(const uint8_t input[BNN_INPUT_SIZE], float *out_score);
int bnn_score_1088_on_backend(
    const uint8_t input[BNN_INPUT_SIZE],
    float *out_score,
    bnn_backend_t backend
);

#endif
