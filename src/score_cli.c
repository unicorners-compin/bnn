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
    const char *path = NULL;
    const char *load_model_path = NULL;
    bnn_backend_t backend = BNN_BACKEND_SCALAR;
    int use_forced_backend = 0;
    int show_model = 0;
    int i;

    uint8_t input[BNN_INPUT_SIZE];
    float score = 0.0f;
    FILE *fp;
    size_t n;

    uint64_t route_mask = 0xFFu;
    uint8_t route_shift = 0u;
    int has_route_rule = 0;
    int has_default_model = 0;
    uint32_t default_model = 0u;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (parse_backend(argv[i + 1], &backend) != 0) {
                fprintf(stderr, "invalid backend option\n");
                return 2;
            }
            use_forced_backend = 1;
            i += 1;
        } else if (strcmp(argv[i], "--load-model") == 0 && i + 1 < argc) {
            load_model_path = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "--route-mask") == 0 && i + 1 < argc) {
            route_mask = strtoull(argv[i + 1], NULL, 0);
            has_route_rule = 1;
            i += 1;
        } else if (strcmp(argv[i], "--route-shift") == 0 && i + 1 < argc) {
            route_shift = (uint8_t)strtoul(argv[i + 1], NULL, 0);
            has_route_rule = 1;
            i += 1;
        } else if (strcmp(argv[i], "--default-model") == 0 && i + 1 < argc) {
            default_model = (uint32_t)strtoul(argv[i + 1], NULL, 0);
            has_default_model = 1;
            i += 1;
        } else if (strcmp(argv[i], "--show-model") == 0) {
            show_model = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }

    if (path == NULL) {
        fprintf(
            stderr,
            "usage: %s [--backend scalar|avx2|avx512] [--load-model file] [--route-mask m --route-shift s] [--default-model id] [--show-model] <input.bin>\n",
            argv[0]
        );
        return 2;
    }

    if (bnn_init() != 0) {
        fprintf(stderr, "bnn_init failed\n");
        return 1;
    }

    if (load_model_path != NULL) {
        if (bnn_load_model_file(load_model_path) != 0) {
            fprintf(stderr, "failed to load model file: %s\n", load_model_path);
            return 1;
        }
    }

    if (has_route_rule) {
        if (bnn_set_route_rule(route_mask, route_shift) != 0) {
            fprintf(stderr, "invalid route rule\n");
            return 1;
        }
    }

    if (has_default_model) {
        if (bnn_set_default_model(default_model) != 0) {
            fprintf(stderr, "default model not found: %u\n", default_model);
            return 1;
        }
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

    if (show_model) {
        printf("model_id=%u score=%.1f\n", bnn_active_model_id(), score);
    } else {
        printf("%.1f\n", score);
    }

    return 0;
}
