#ifndef BNN_INFER_H
#define BNN_INFER_H

#include <stddef.h>
#include <stdint.h>

#define BNN_INPUT_SIZE 1088u
#define BNN_CONFIG_SIZE 64u
#define BNN_PAYLOAD_SIZE 1024u

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
