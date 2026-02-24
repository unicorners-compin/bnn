#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

WORDS = 1088 // 8


def xorshift64star(state: int) -> int:
    state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
    return (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF


def gen_weights(seed: int, words: int):
    s = seed & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(words):
        s = xorshift64star(s)
        out.append(s)
    return out


def write_header(path: Path, weights):
    lines = []
    lines.append("#ifndef BNN_WEIGHTS_H")
    lines.append("#define BNN_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define BNN_MODEL_WORDS {len(weights)}u")
    lines.append("#define BNN_MODEL_BIAS 0")
    lines.append("")
    lines.append("static const uint64_t BNN_WEIGHTS[BNN_MODEL_WORDS] = {")

    for i, w in enumerate(weights):
        comma = "," if i + 1 < len(weights) else ""
        lines.append(f"    0x{w:016X}ULL{comma}")

    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description="Export deterministic BNN weights to C header and JSON")
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x9E3779B97F4A7C15)
    parser.add_argument("--header", default="src/bnn_weights.h")
    parser.add_argument("--json", default="docs/bnn_weights.json")
    args = parser.parse_args()

    weights = gen_weights(args.seed, WORDS)

    header_path = Path(args.header)
    json_path = Path(args.json)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    write_header(header_path, weights)
    json_path.write_text(
        json.dumps({"seed": args.seed, "words": WORDS, "weights": weights, "bias": 0}, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {header_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
