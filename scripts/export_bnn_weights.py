#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PAYLOAD_WORDS = 1024 // 8
MAGIC = 0x4D4E4E42  # 'BNNM'
VERSION = 1


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


def write_header(path: Path, model_id: int, bias: int, weights):
    lines = []
    lines.append("#ifndef BNN_WEIGHTS_H")
    lines.append("#define BNN_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define BNN_MODEL_ID {model_id}u")
    lines.append(f"#define BNN_MODEL_WORDS {len(weights)}u")
    lines.append(f"#define BNN_MODEL_BIAS {bias}")
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
    parser = argparse.ArgumentParser(description="导出 BNN 权重到 C 头文件/JSON/二进制模型文件")
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x9E3779B97F4A7C15)
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--header", default="src/bnn_weights.h")
    parser.add_argument("--json", default="docs/bnn_weights.json")
    parser.add_argument("--bin", default="models/model_0.bnnw")
    args = parser.parse_args()

    weights = gen_weights(args.seed, PAYLOAD_WORDS)

    header_path = Path(args.header)
    json_path = Path(args.json)
    bin_path = Path(args.bin)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    write_header(header_path, args.model_id, args.bias, weights)
    json_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "model_id": args.model_id,
                "words": PAYLOAD_WORDS,
                "weights": weights,
                "bias": args.bias,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with bin_path.open("wb") as f:
        f.write(MAGIC.to_bytes(4, "little"))
        f.write(VERSION.to_bytes(4, "little"))
        f.write(int(args.model_id).to_bytes(4, "little", signed=False))
        f.write(PAYLOAD_WORDS.to_bytes(4, "little"))
        f.write(int(args.bias).to_bytes(4, "little", signed=True))
        f.write((0).to_bytes(4, "little", signed=False))
        for w in weights:
            f.write(int(w).to_bytes(8, "little", signed=False))

    print(f"wrote {header_path}")
    print(f"wrote {json_path}")
    print(f"wrote {bin_path}")


if __name__ == "__main__":
    main()
