"""Generate hand-written AVX-512 kernels for skinny row-major GEMMs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

Strategy = Literal["segmented-single", "segmented-multi", "outer-vl"]

F64_BYTES = 8
ZMM_F64S = 8
ZMM_REGISTERS = 32


def _memory(displacement: int, base: str) -> str:
    if displacement == 0:
        return f"(%{base})"
    return f"{displacement}(%{base})"


def _function_start(comment: str) -> list[str]:
    return [
        "\t.text",
        "\t.globl\tmatmul",
        "\t.p2align\t4",
        "\t.type\tmatmul,@function",
        "matmul:",
        f"\t# {comment}",
        "\t# rdi=A, rsi=B, rdx=C",
    ]


def _function_end() -> list[str]:
    return [
        "\tvzeroupper",
        "\tretq",
        "\t.size\tmatmul, .-matmul",
        '\t.section\t.note.GNU-stack,"",@progbits',
        "",
    ]


def _segmented_accumulator_sets(m: int, groups: int, multi: bool) -> int:
    if not multi:
        return 1

    # zmm0 is the current B slab, zmm1 is expanded A, and zmm31 holds the
    # permutation indices.  The remaining 29 registers are accumulators.
    return min(4, groups, 29 // m)


def _repeat_indices(n: int) -> tuple[int, ...]:
    k_values_per_group = ZMM_F64S // n
    return tuple(k_value for k_value in range(k_values_per_group) for _ in range(n))


def generate_segmented_asm(*, m: int, n: int, k: int, multi: bool) -> str:
    """Vectorize several K rows together and segment-reduce N outputs."""

    if m <= 0:
        raise ValueError("M must be positive")
    if n not in (2, 4):
        raise ValueError("segmented K reduction currently requires N=2 or N=4")

    k_values_per_group = ZMM_F64S // n
    if k <= 0 or k % k_values_per_group:
        raise ValueError(f"K must be a positive multiple of {k_values_per_group}")

    groups = k // k_values_per_group
    accumulator_sets = _segmented_accumulator_sets(m, groups, multi)
    if accumulator_sets == 0:
        raise ValueError("the requested M does not fit in the ZMM register file")

    def accumulator_register(row: int, accumulator_set: int) -> int:
        return 2 + row * accumulator_sets + accumulator_set

    lines = [
        "\t.section\t.rodata",
        "\t.p2align\t6",
        ".Lrepeat_indices:",
        *(f"\t.quad\t{index}" for index in _repeat_indices(n)),
        *_function_start(
            f"segmented K reduction: M={m}, N={n}, K={k}, "
            f"accumulator_sets={accumulator_sets}"
        ),
        "\tvmovdqa64\t.Lrepeat_indices(%rip), %zmm31",
    ]

    for group in range(groups):
        accumulator_set = group % accumulator_sets
        b_displacement = group * ZMM_F64S * F64_BYTES
        lines.append(f"\tvmovupd\t{_memory(b_displacement, 'rsi')}, %zmm0")

        for row in range(m):
            a_displacement = (row * k + group * k_values_per_group) * F64_BYTES
            a_register = "%ymm1" if n == 2 else "%xmm1"
            lines.extend(
                [
                    f"\tvmovupd\t{_memory(a_displacement, 'rdi')}, {a_register}",
                    "\tvpermpd\t%zmm1, %zmm31, %zmm1",
                ]
            )
            accumulator = accumulator_register(row, accumulator_set)
            instruction = "vmulpd" if group < accumulator_sets else "vfmadd231pd"
            lines.append(f"\t{instruction}\t%zmm0, %zmm1, %zmm{accumulator}")

    for row in range(m):
        accumulator = accumulator_register(row, 0)
        for accumulator_set in range(1, accumulator_sets):
            source = accumulator_register(row, accumulator_set)
            lines.append(
                f"\tvaddpd\t%zmm{source}, %zmm{accumulator}, %zmm{accumulator}"
            )

        c_displacement = row * n * F64_BYTES
        lines.extend(
            [
                f"\tvextractf64x4\t$1, %zmm{accumulator}, %ymm0",
                f"\tvaddpd\t%ymm{accumulator}, %ymm0, %ymm0",
            ]
        )
        if n == 2:
            lines.extend(
                [
                    "\tvextractf128\t$1, %ymm0, %xmm1",
                    "\tvaddpd\t%xmm1, %xmm0, %xmm0",
                    f"\tvaddpd\t{_memory(c_displacement, 'rdx')}, %xmm0, %xmm0",
                    f"\tvmovupd\t%xmm0, {_memory(c_displacement, 'rdx')}",
                ]
            )
        else:
            lines.extend(
                [
                    f"\tvaddpd\t{_memory(c_displacement, 'rdx')}, %ymm0, %ymm0",
                    f"\tvmovupd\t%ymm0, {_memory(c_displacement, 'rdx')}",
                ]
            )

    lines.extend(_function_end())
    return "\n".join(lines)


def _outer_accumulator_sets(m: int, k: int) -> int:
    # xmm0/ymm0 holds the current B row.  Use the other physical vector
    # registers for persistent accumulator chains.
    return min(4, k, 31 // m)


def generate_outer_vl_asm(*, m: int, n: int, k: int) -> str:
    """Generate a width-matched outer-product control kernel."""

    if m <= 0 or k <= 0:
        raise ValueError("M and K must be positive")
    if n not in (2, 4):
        raise ValueError("the width-matched outer kernel requires N=2 or N=4")

    accumulator_sets = _outer_accumulator_sets(m, k)
    if accumulator_sets == 0:
        raise ValueError("the requested M does not fit in the vector register file")

    register_name = "xmm" if n == 2 else "ymm"

    def accumulator_register(row: int, accumulator_set: int) -> int:
        return 1 + row * accumulator_sets + accumulator_set

    lines = _function_start(
        f"width-matched outer product: M={m}, N={n}, K={k}, "
        f"accumulator_sets={accumulator_sets}"
    )

    for row in range(m):
        for accumulator_set in range(accumulator_sets):
            accumulator = accumulator_register(row, accumulator_set)
            lines.append(
                f"\tvxorpd\t%{register_name}{accumulator}, "
                f"%{register_name}{accumulator}, %{register_name}{accumulator}"
            )

    for k_index in range(k):
        accumulator_set = k_index % accumulator_sets
        b_displacement = k_index * n * F64_BYTES
        lines.append(f"\tvmovupd\t{_memory(b_displacement, 'rsi')}, %{register_name}0")
        for row in range(m):
            a_displacement = (row * k + k_index) * F64_BYTES
            accumulator = accumulator_register(row, accumulator_set)
            lines.append(
                f"\tvfmadd231pd\t{_memory(a_displacement, 'rdi')}{{1to{n}}}, "
                f"%{register_name}0, %{register_name}{accumulator}"
            )

    for row in range(m):
        accumulator = accumulator_register(row, 0)
        for accumulator_set in range(1, accumulator_sets):
            source = accumulator_register(row, accumulator_set)
            lines.append(
                f"\tvaddpd\t%{register_name}{source}, %{register_name}{accumulator}, "
                f"%{register_name}{accumulator}"
            )

        c_displacement = row * n * F64_BYTES
        lines.extend(
            [
                f"\tvaddpd\t{_memory(c_displacement, 'rdx')}, "
                f"%{register_name}{accumulator}, %{register_name}{accumulator}",
                f"\tvmovupd\t%{register_name}{accumulator}, "
                f"{_memory(c_displacement, 'rdx')}",
            ]
        )

    lines.extend(_function_end())
    return "\n".join(lines)


def generate_skinny_asm(*, m: int, n: int, k: int, strategy: Strategy) -> str:
    if strategy == "segmented-single":
        return generate_segmented_asm(m=m, n=n, k=k, multi=False)
    if strategy == "segmented-multi":
        return generate_segmented_asm(m=m, n=n, k=k, multi=True)
    return generate_outer_vl_asm(m=m, n=n, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=("f64",), required=True)
    parser.add_argument(
        "--strategy",
        choices=("segmented-single", "segmented-multi", "outer-vl"),
        required=True,
    )
    args = parser.parse_args()

    assembly = generate_skinny_asm(
        m=args.m,
        n=args.n,
        k=args.k,
        strategy=args.strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assembly)


if __name__ == "__main__":
    main()
