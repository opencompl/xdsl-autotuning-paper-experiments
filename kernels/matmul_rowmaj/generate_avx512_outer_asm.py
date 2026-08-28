import argparse
from pathlib import Path

FIRST_ACCUMULATOR = 16
VECTOR_REGISTER_COUNT = 32


def _memory(base: str, offset: int) -> str:
    if offset == 0:
        return f"[{base}]"
    return f"[{base}+{offset}]"


def generate(m: int, n: int, k: int) -> str:
    if not 1 <= m <= 16:
        raise ValueError(f"M must be in [1, 16], got {m}")
    if not 2 <= n <= 4:
        raise ValueError(f"N must be in [2, 4], got {n}")
    if k != 64:
        raise ValueError(f"this proof of concept requires K=64, got {k}")

    accumulator_capacity = VECTOR_REGISTER_COUNT - FIRST_ACCUMULATOR
    chains_per_row = min(8, accumulator_capacity // m)
    assembly = [
        ".intel_syntax noprefix",
        ".text",
        ".p2align 4, 0x90",
        ".globl matmul",
        ".type matmul, @function",
        "matmul:",
        f"    mov eax, {(1 << n) - 1}",
        "    kmovb k1, eax",
    ]

    for row in range(m):
        first = FIRST_ACCUMULATOR + row * chains_per_row
        assembly.append(
            f"    vmovupd zmm{first} {{k1}}{{z}}, {_memory('rdx', row * n * 8)}"
        )
        for chain in range(1, chains_per_row):
            accumulator = first + chain
            assembly.append(
                f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
            )

    for k_index in range(k):
        assembly.append(
            f"    vmovupd zmm0 {{k1}}{{z}}, {_memory('rsi', k_index * n * 8)}"
        )
        chain = k_index % chains_per_row
        for row in range(m):
            accumulator = FIRST_ACCUMULATOR + row * chains_per_row + chain
            a_offset = (row * k + k_index) * 8
            assembly.append(
                f"    vfmadd231pd zmm{accumulator}, zmm0, "
                f"{_memory('rdi', a_offset)}{{1to8}}"
            )

    for row in range(m):
        first = FIRST_ACCUMULATOR + row * chains_per_row
        for chain in range(1, chains_per_row):
            accumulator = first + chain
            assembly.append(f"    vaddpd zmm{first}, zmm{first}, zmm{accumulator}")
        assembly.append(f"    vmovupd {_memory('rdx', row * n * 8)} {{k1}}, zmm{first}")

    assembly.extend(("    ret", ".size matmul, .-matmul"))
    return "\n".join(assembly) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a leaf AVX-512 outer-product f64 GEMM kernel"
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate(args.M, args.N, args.K))


if __name__ == "__main__":
    main()
