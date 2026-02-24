#include "bnn_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_backend(const char *s, bnn_backend_t *backend) {
    if (strcmp(s, "scalar") == 0) {
        *backend = BNN_BACKEND_SCALAR;
        return 0;
    }
    if (strcmp(s, "avx2") == 0) {
        *backend = BNN_BACKEND_AVX2;
        return 0;
    }
    if (strcmp(s, "avx512") == 0) {
        *backend = BNN_BACKEND_AVX512;
        return 0;
    }
    return -1;
}

int main(int argc, char **argv) {
    const char *path;
    bnn_backend_t backend = BNN_BACKEND_SCALAR;
    int use_forced_backend = 0;
    uint8_t input[BNN_INPUT_SIZE];
    float score = 0.0f;
    FILE *fp;
    size_t n;

    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: %s [--backend scalar|avx2|avx512] <input.bin>\n", argv[0]);
        return 2;
    }

    if (argc == 4) {
        if (strcmp(argv[1], "--backend") != 0 || parse_backend(argv[2], &backend) != 0) {
            fprintf(stderr, "invalid backend option\n");
            return 2;
        }
        use_forced_backend = 1;
        path = argv[3];
    } else {
        path = argv[1];
    }

    if (bnn_init() != 0) {
        fprintf(stderr, "bnn_init failed\n");
        return 1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }
    n = fread(input, 1, sizeof(input), fp);
    fclose(fp);

    if (n != sizeof(input)) {
        fprintf(stderr, "invalid input size: got %zu expected %u\n", n, (unsigned)BNN_INPUT_SIZE);
        return 1;
    }

    if (use_forced_backend) {
        if (bnn_score_1088_on_backend(input, &score, backend) != 0) {
            fprintf(stderr, "inference failed on backend=%s\n", bnn_backend_name(backend));
            return 1;
        }
    } else {
        if (bnn_score_1088(input, &score) != 0) {
            fprintf(stderr, "inference failed\n");
            return 1;
        }
    }

    printf("%.1f\n", score);
    return 0;
}
