"""Generate small hand-written AVX-512 kernels for direct K-dot experiments.

These kernels intentionally bypass the compiler pipeline.  They are temporary
experiments for deciding which schedules are worth expressing as nano-kernels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

Strategy = Literal["single", "multi"]

F64_BYTES = 8
F64S_PER_ZMM = 8
ZMM_REGISTERS = 32


def _memory(displacement: int, base: str) -> str:
    if displacement == 0:
        return f"(%{base})"
    return f"{displacement}(%{base})"


def _accumulator_sets(m: int, k_chunks: int, strategy: Strategy) -> int:
    if strategy == "single":
        return 1

    # B remains resident in one register per K chunk.  Use the remaining
    # registers for up to four independent accumulator chains per output row.
    available = ZMM_REGISTERS - k_chunks
    return min(4, k_chunks, available // m)


def generate_kdot_asm(*, m: int, n: int, k: int, strategy: Strategy) -> str:
    """Generate ``C += A @ B`` for row-major f64 matrices with N equal to one."""

    if m <= 0:
        raise ValueError("M must be positive")
    if n != 1:
        raise ValueError("the direct K-dot kernel currently requires N=1")
    if k <= 0 or k % F64S_PER_ZMM:
        raise ValueError("K must be a positive multiple of eight")

    k_chunks = k // F64S_PER_ZMM
    if k_chunks >= ZMM_REGISTERS:
        raise ValueError("K requires too many resident B registers")

    accumulator_sets = _accumulator_sets(m, k_chunks, strategy)
    if accumulator_sets == 0:
        raise ValueError("the requested M and K do not fit in the ZMM register file")

    register_demand = k_chunks + m * accumulator_sets
    if register_demand > ZMM_REGISTERS:
        raise AssertionError("internal register-pressure error")

    def accumulator_register(row: int, accumulator_set: int) -> int:
        return k_chunks + row * accumulator_sets + accumulator_set

    lines = [
        "\t.text",
        "\t.globl\tmatmul",
        "\t.p2align\t4",
        "\t.type\tmatmul,@function",
        "matmul:",
        f"\t# direct K-dot: M={m}, N=1, K={k}, accumulator_sets={accumulator_sets}",
        "\t# rdi=A, rsi=B, rdx=C; the benchmark provides 64-byte alignment",
    ]

    # Keep B resident and reuse it across every output row.
    for chunk in range(k_chunks):
        displacement = chunk * F64S_PER_ZMM * F64_BYTES
        lines.append(f"\tvmovapd\t{_memory(displacement, 'rsi')}, %zmm{chunk}")

    # Interleave rows and independent accumulator chains.  This spaces
    # dependent FMAs while retaining the compact eight-K-values-per-operation
    # schedule that makes the LLVM 1x1 kernel effective.
    for chunk in range(k_chunks):
        accumulator_set = chunk % accumulator_sets
        for row in range(m):
            a_displacement = (row * k + chunk * F64S_PER_ZMM) * F64_BYTES
            accumulator = accumulator_register(row, accumulator_set)
            instruction = "vmulpd" if chunk < accumulator_sets else "vfmadd231pd"
            lines.append(
                f"\t{instruction}\t{_memory(a_displacement, 'rdi')}, "
                f"%zmm{chunk}, %zmm{accumulator}"
            )

    # Merge independent chains, horizontally reduce K, and only then add C.
    # B is dead here, so zmm0/zmm1 and their narrower aliases are scratch.
    for row in range(m):
        accumulator = accumulator_register(row, 0)
        for accumulator_set in range(1, accumulator_sets):
            source = accumulator_register(row, accumulator_set)
            lines.append(
                f"\tvaddpd\t%zmm{source}, %zmm{accumulator}, %zmm{accumulator}"
            )

        c_displacement = row * F64_BYTES
        lines.extend(
            [
                f"\tvextractf64x4\t$1, %zmm{accumulator}, %ymm0",
                f"\tvaddpd\t%ymm{accumulator}, %ymm0, %ymm0",
                "\tvextractf128\t$1, %ymm0, %xmm1",
                "\tvaddpd\t%xmm1, %xmm0, %xmm0",
                "\tvshufpd\t$1, %xmm0, %xmm0, %xmm1",
                "\tvaddsd\t%xmm1, %xmm0, %xmm0",
                f"\tvaddsd\t{_memory(c_displacement, 'rdx')}, %xmm0, %xmm0",
                f"\tvmovsd\t%xmm0, {_memory(c_displacement, 'rdx')}",
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

    assembly = generate_kdot_asm(
        m=args.m,
        n=args.n,
        k=args.k,
        strategy=args.strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assembly)


if __name__ == "__main__":
    main()
