"""Generate AVX-512 gather-across-M kernels for skinny row-major GEMMs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

Strategy = Literal["single", "multi"]

F64_BYTES = 8
ZMM_F64S = 8


def _vsib(displacement: int, base: str, index: str) -> str:
    if displacement == 0:
        return f"(%{base},%{index})"
    return f"{displacement}(%{base},%{index})"


def _memory(displacement: int, base: str) -> str:
    if displacement == 0:
        return f"(%{base})"
    return f"{displacement}(%{base})"


def _accumulator_sets(*, blocks: int, n: int, k: int, multi: bool) -> int:
    if not multi:
        return 1

    # The multi form reserves zmm0 and zmm27-zmm29 for independent gather
    # streams; zmm30 and zmm31 hold C and A indices.
    return min(4, k, 26 // (blocks * n))


def generate_gather_asm(*, m: int, n: int, k: int, strategy: Strategy) -> str:
    """Vectorize M with gathers and keep one vector accumulator per column."""

    if m not in (8, 16):
        raise ValueError("the gather kernel requires M=8 or M=16")
    if n not in (2, 4):
        raise ValueError("the gather kernel requires N=2 or N=4")
    if k <= 0:
        raise ValueError("K must be positive")
    if strategy not in ("single", "multi"):
        raise ValueError(f"unknown strategy: {strategy}")

    blocks = m // ZMM_F64S
    accumulator_sets = _accumulator_sets(
        blocks=blocks, n=n, k=k, multi=strategy == "multi"
    )
    gather_registers = (0, 29, 28, 27) if strategy == "multi" else (0,)

    def accumulator_register(block: int, column: int, accumulator_set: int) -> int:
        return 1 + (block * n + column) * accumulator_sets + accumulator_set

    lines = [
        "\t.section\t.rodata",
        "\t.p2align\t6",
        ".La_indices:",
        *(f"\t.quad\t{row * k * F64_BYTES}" for row in range(ZMM_F64S)),
        "\t.p2align\t6",
        ".Lc_indices:",
        *(f"\t.quad\t{row * n * F64_BYTES}" for row in range(ZMM_F64S)),
        "\t.text",
        "\t.globl\tmatmul",
        "\t.p2align\t4",
        "\t.type\tmatmul,@function",
        "matmul:",
        f"\t# gather across M: M={m}, N={n}, K={k}, "
        f"accumulator_sets={accumulator_sets}, "
        f"gather_streams={len(gather_registers)}",
        "\t# rdi=A, rsi=B, rdx=C",
        "\tvmovdqa64\t.La_indices(%rip), %zmm31",
        "\tvmovdqa64\t.Lc_indices(%rip), %zmm30",
    ]

    for block in range(blocks):
        for column in range(n):
            for accumulator_set in range(accumulator_sets):
                accumulator = accumulator_register(block, column, accumulator_set)
                lines.append(
                    f"\tvpxord\t%zmm{accumulator}, %zmm{accumulator}, %zmm{accumulator}"
                )

    for k_index in range(k):
        accumulator_set = k_index % accumulator_sets
        for block in range(blocks):
            gather_stream = (k_index * blocks + block) % len(gather_registers)
            gather_register = gather_registers[gather_stream]
            mask_register = gather_stream + 1
            a_displacement = (block * ZMM_F64S * k + k_index) * F64_BYTES
            lines.extend(
                [
                    f"\tkxnorw\t%k{mask_register}, %k{mask_register}, "
                    f"%k{mask_register}",
                    f"\tvgatherqpd\t{_vsib(a_displacement, 'rdi', 'zmm31')}, "
                    f"%zmm{gather_register} {{%k{mask_register}}}",
                ]
            )
            for column in range(n):
                b_displacement = (k_index * n + column) * F64_BYTES
                accumulator = accumulator_register(block, column, accumulator_set)
                lines.append(
                    f"\tvfmadd231pd\t{_memory(b_displacement, 'rsi')}{{1to8}}, "
                    f"%zmm{gather_register}, %zmm{accumulator}"
                )

    for block in range(blocks):
        for column in range(n):
            accumulator = accumulator_register(block, column, 0)
            for accumulator_set in range(1, accumulator_sets):
                source = accumulator_register(block, column, accumulator_set)
                lines.append(
                    f"\tvaddpd\t%zmm{source}, %zmm{accumulator}, %zmm{accumulator}"
                )

            c_displacement = (block * ZMM_F64S * n + column) * F64_BYTES
            c_address = _vsib(c_displacement, "rdx", "zmm30")
            lines.extend(
                [
                    "\tkxnorw\t%k1, %k1, %k1",
                    f"\tvgatherqpd\t{c_address}, %zmm0 {{%k1}}",
                    f"\tvaddpd\t%zmm0, %zmm{accumulator}, %zmm{accumulator}",
                    "\tkxnorw\t%k1, %k1, %k1",
                    f"\tvscatterqpd\t%zmm{accumulator}, {c_address} {{%k1}}",
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

    assembly = generate_gather_asm(
        m=args.m,
        n=args.n,
        k=args.k,
        strategy=args.strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assembly)


if __name__ == "__main__":
    main()
