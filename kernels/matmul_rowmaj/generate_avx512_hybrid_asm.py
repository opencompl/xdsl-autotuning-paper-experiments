import argparse
from pathlib import Path

from generate_avx512_kdot_asm import generate as generate_kdot
from generate_avx512_outer_asm import generate as generate_outer


def generate(m: int, n: int, k: int, fallback: str) -> str:
    if n == 1 or (n == 2 and m >= 7):
        return generate_kdot(m, n, k)
    if n <= 4:
        return generate_outer(m, n, k)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the measured AVX-512 small-GEMM assembly schedule"
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate(args.M, args.N, args.K, args.fallback.read_text()))


if __name__ == "__main__":
    main()
