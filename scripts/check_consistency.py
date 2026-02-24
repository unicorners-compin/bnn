#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

INPUT_SIZE = 1088
BITS = INPUT_SIZE * 8


def popcnt64(x: int) -> int:
    return x.bit_count()


def score_python(data: bytes, weights: list[int], bias: int) -> float:
    matched = 0
    for i, w in enumerate(weights):
        chunk = data[i * 8 : (i + 1) * 8]
        x = int.from_bytes(chunk, byteorder="little", signed=False)
        xnor = (~(x ^ w)) & 0xFFFFFFFFFFFFFFFF
        matched += popcnt64(xnor)
    return float(matched * 2 - BITS + bias)


def score_c(score_bin: Path, data: bytes, backend: str) -> float:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        out = subprocess.check_output(
            [str(score_bin), "--backend", backend, str(tmp)],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return float(out)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Check Python/C score consistency")
    parser.add_argument("--weights-json", default="docs/bnn_weights.json")
    parser.add_argument("--score-bin", default="./score_cli")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260223)
    args = parser.parse_args()

    weight_obj = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
    weights = weight_obj["weights"]
    bias = int(weight_obj.get("bias", 0))

    rng = random.Random(args.seed)
    score_bin = Path(args.score_bin).resolve()
    if not score_bin.exists():
        raise FileNotFoundError(f"score binary not found: {score_bin}")

    mismatches = 0
    for i in range(args.samples):
        data = bytes(rng.getrandbits(8) for _ in range(INPUT_SIZE))
        py = score_python(data, weights, bias)
        c_scalar = score_c(score_bin, data, "scalar")

        if py != c_scalar:
            mismatches += 1
            print(f"mismatch at sample {i}: py={py} c_scalar={c_scalar}")

    print(f"samples={args.samples}")
    print(f"mismatches={mismatches}")
    print(f"result={'PASS' if mismatches == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
