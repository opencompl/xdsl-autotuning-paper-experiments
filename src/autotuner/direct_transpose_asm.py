"""Generate AVX-512 transpose-tiled kernels for skinny row-major GEMMs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

Strategy = Literal["single", "multi"]

F64_BYTES = 8
ZMM_F64S = 8
K_TILE = 8


def _memory(displacement: int, base: str) -> str:
    if displacement == 0:
        return f"(%{base})"
    return f"{displacement}(%{base})"


def _vsib(displacement: int, base: str, index: str) -> str:
    if displacement == 0:
        return f"(%{base},%{index})"
    return f"{displacement}(%{base},%{index})"


def _shuffle(destination: int, left: int, right: int, immediate: int) -> str:
    return f"\tvshuff64x2\t${immediate:#x}, %zmm{right}, %zmm{left}, %zmm{destination}"


def _transpose_8x8() -> list[str]:
    """Transpose zmm0-zmm7 into zmm8-zmm15 using zmm0-zmm15."""

    lines: list[str] = []
    for pair in range(4):
        left = pair * 2
        right = left + 1
        temporary = 8 + pair * 2
        lines.extend(
            [
                f"\tvunpcklpd\t%zmm{right}, %zmm{left}, %zmm{temporary}",
                f"\tvunpckhpd\t%zmm{right}, %zmm{left}, %zmm{temporary + 1}",
            ]
        )

    # Each pair produces the low/high halves of two eventual columns.
    lines.extend(
        [
            _shuffle(0, 8, 10, 0x88),
            _shuffle(1, 9, 11, 0x88),
            _shuffle(2, 8, 10, 0xDD),
            _shuffle(3, 9, 11, 0xDD),
            _shuffle(4, 12, 14, 0x88),
            _shuffle(5, 13, 15, 0x88),
            _shuffle(6, 12, 14, 0xDD),
            _shuffle(7, 13, 15, 0xDD),
            _shuffle(8, 0, 4, 0x88),
            _shuffle(9, 1, 5, 0x88),
            _shuffle(10, 2, 6, 0x88),
            _shuffle(11, 3, 7, 0x88),
            _shuffle(12, 0, 4, 0xDD),
            _shuffle(13, 1, 5, 0xDD),
            _shuffle(14, 2, 6, 0xDD),
            _shuffle(15, 3, 7, 0xDD),
        ]
    )
    return lines


def generate_transpose_asm(*, m: int, n: int, k: int, strategy: Strategy) -> str:
    """Transpose 8x8 A tiles in registers, then vectorize across M."""

    if m not in (8, 16):
        raise ValueError("the transpose kernel requires M=8 or M=16")
    if n not in (2, 4):
        raise ValueError("the transpose kernel requires N=2 or N=4")
    if k <= 0 or k % K_TILE:
        raise ValueError(f"K must be a positive multiple of {K_TILE}")
    if strategy not in ("single", "multi"):
        raise ValueError(f"unknown strategy: {strategy}")

    blocks = m // ZMM_F64S
    accumulator_sets = 1 if strategy == "single" else 4

    def accumulator_register(column: int, accumulator_set: int) -> int:
        # zmm0-zmm15 are the transpose network.
        return 16 + column * accumulator_sets + accumulator_set

    lines = [
        "\t.section\t.rodata",
        "\t.p2align\t6",
        ".Lc_indices:",
        *(f"\t.quad\t{row * n * F64_BYTES}" for row in range(ZMM_F64S)),
        "\t.text",
        "\t.globl\tmatmul",
        "\t.p2align\t4",
        "\t.type\tmatmul,@function",
        "matmul:",
        f"\t# in-register 8x8 transpose: M={m}, N={n}, K={k}, "
        f"accumulator_sets={accumulator_sets}",
        "\t# rdi=A, rsi=B, rdx=C",
    ]

    for block in range(blocks):
        for column in range(n):
            for accumulator_set in range(accumulator_sets):
                accumulator = accumulator_register(column, accumulator_set)
                lines.append(
                    f"\tvpxord\t%zmm{accumulator}, %zmm{accumulator}, %zmm{accumulator}"
                )

        for k_base in range(0, k, K_TILE):
            for row in range(ZMM_F64S):
                a_displacement = ((block * ZMM_F64S + row) * k + k_base) * F64_BYTES
                lines.append(f"\tvmovupd\t{_memory(a_displacement, 'rdi')}, %zmm{row}")

            lines.extend(_transpose_8x8())

            for k_offset in range(K_TILE):
                k_index = k_base + k_offset
                accumulator_set = k_index % accumulator_sets
                for column in range(n):
                    b_displacement = (k_index * n + column) * F64_BYTES
                    accumulator = accumulator_register(column, accumulator_set)
                    lines.append(
                        f"\tvfmadd231pd\t{_memory(b_displacement, 'rsi')}"
                        f"{{1to8}}, %zmm{8 + k_offset}, %zmm{accumulator}"
                    )

        for column in range(n):
            accumulator = accumulator_register(column, 0)
            for accumulator_set in range(1, accumulator_sets):
                source = accumulator_register(column, accumulator_set)
                lines.append(
                    f"\tvaddpd\t%zmm{source}, %zmm{accumulator}, %zmm{accumulator}"
                )

            c_displacement = (block * ZMM_F64S * n + column) * F64_BYTES
            lines.extend(
                [
                    "\tvmovdqa64\t.Lc_indices(%rip), %zmm0",
                    "\tkxnorw\t%k1, %k1, %k1",
                    f"\tvgatherqpd\t{_vsib(c_displacement, 'rdx', 'zmm0')}, "
                    "%zmm1 {%k1}",
                    f"\tvaddpd\t%zmm1, %zmm{accumulator}, %zmm{accumulator}",
                    "\tkxnorw\t%k1, %k1, %k1",
                    f"\tvscatterqpd\t%zmm{accumulator}, "
                    f"{_vsib(c_displacement, 'rdx', 'zmm0')} {{%k1}}",
                ]
            )

    lines.extend(
        [
            "\tvzeroupper",
            "\tretq",
            "\t.size\tmatmul, .-matmul",
            '\t.section\t.note.GNU-stack,"",@progbits',
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=("f64",), required=True)
    parser.add_argument("--strategy", choices=("single", "multi"), required=True)
    args = parser.parse_args()

    assembly = generate_transpose_asm(
        m=args.m,
        n=args.n,
        k=args.k,
        strategy=args.strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assembly)


if __name__ == "__main__":
    main()
