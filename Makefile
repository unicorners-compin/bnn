CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c11 -march=native -D_POSIX_C_SOURCE=200809L
LDFLAGS ?=

SRC_DIR := src
OBJ_DIR := build

COMMON_SRCS := $(SRC_DIR)/bnn_infer.c
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SRCS))

BENCH_BIN := bench
EVAL_BIN := eval
SCORE_BIN := score_cli

.PHONY: all clean

all: $(BENCH_BIN) $(EVAL_BIN) $(SCORE_BIN)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BENCH_BIN): $(SRC_DIR)/bench.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(EVAL_BIN): $(SRC_DIR)/eval.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(SCORE_BIN): $(SRC_DIR)/score_cli.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BENCH_BIN) $(EVAL_BIN) $(SCORE_BIN)
