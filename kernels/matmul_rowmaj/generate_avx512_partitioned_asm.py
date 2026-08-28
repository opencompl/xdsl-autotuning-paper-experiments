import argparse
from pathlib import Path
from typing import Literal, cast

from generate_avx512_kdot_asm import _emit_horizontal_sums

VECTOR_LENGTH = 8
K_VECTORS = 8

TileKind = Literal["kdot", "gather", "outer"]
Strategy = Literal["equalized", "cost"]
Tile = tuple[TileKind, int]

PARTITIONS: dict[tuple[int, int, Strategy], tuple[Tile, ...]] = {
    (25, 2, "equalized"): (("gather", 9), ("gather", 8), ("gather", 8)),
    (25, 2, "cost"): (("gather", 12), ("gather", 12), ("outer", 1)),
    (26, 2, "equalized"): (("gather", 9), ("gather", 9), ("gather", 8)),
    (26, 2, "cost"): (("gather", 12), ("gather", 12), ("outer", 2)),
    (32, 1, "equalized"): (("kdot", 16), ("kdot", 16)),
    (32, 1, "cost"): (("kdot", 8), ("kdot", 8), ("kdot", 8), ("kdot", 8)),
}


def _memory(base: str, offset: int) -> str:
    if offset == 0:
        return f"[{base}]"
    return f"[{base}+{offset}]"


def _emit_n1_tile(assembly: list[str], row_start: int, rows: int, k: int) -> None:
    for k_vector in range(K_VECTORS):
        assembly.append(
            f"    vmovupd zmm{k_vector}, {_memory('rsi', k_vector * VECTOR_LENGTH * 8)}"
        )

    first_accumulator = 16
    for tile_row in range(rows):
        accumulator = first_accumulator + tile_row
        assembly.append(
            f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
        )

    for k_vector in range(K_VECTORS):
        for tile_row in range(rows):
            row = row_start + tile_row
            accumulator = first_accumulator + tile_row
            a_offset = (row * k + k_vector * VECTOR_LENGTH) * 8
            assembly.append(
                f"    vfmadd231pd zmm{accumulator}, zmm{k_vector}, "
                f"{_memory('rdi', a_offset)}"
            )

    _emit_horizontal_sums(
        assembly,
        [
            (first_accumulator + tile_row, (row_start + tile_row) * 8)
            for tile_row in range(rows)
        ],
        batch_size=8,
    )


def _emit_gather_n2_tile(
    assembly: list[str], row_start: int, rows: int, k: int
) -> None:
    if rows > 12:
        raise ValueError(f"an N=2 gather tile can contain at most 12 rows, got {rows}")

    assembly.extend(("    mov eax, 255", "    vmovdqu64 zmm0, [rip + .Lindices_n2]"))
    first_accumulator = 8
    for output in range(rows * 2):
        accumulator = first_accumulator + output
        assembly.append(
            f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
        )

    for k_vector in range(K_VECTORS):
        for column in range(2):
            b_offset = (k_vector * VECTOR_LENGTH * 2 + column) * 8
            assembly.extend(
                (
                    "    kmovb k1, eax",
                    (
                        f"    vgatherqpd zmm1 {{k1}}, "
                        f"{_memory('rsi + zmm0*8', b_offset)}"
                    ),
                )
            )
            for tile_row in range(rows):
                row = row_start + tile_row
                accumulator = first_accumulator + tile_row * 2 + column
                a_offset = (row * k + k_vector * VECTOR_LENGTH) * 8
                assembly.append(
                    f"    vfmadd231pd zmm{accumulator}, zmm1, "
                    f"{_memory('rdi', a_offset)}"
                )

    _emit_horizontal_sums(
        assembly,
        [
            (
                first_accumulator + tile_row * 2 + column,
                ((row_start + tile_row) * 2 + column) * 8,
            )
            for tile_row in range(rows)
            for column in range(2)
        ],
        batch_size=4,
    )


def _emit_outer_n2_tile(assembly: list[str], row_start: int, rows: int, k: int) -> None:
    first_accumulator = 16
    chains_per_row = min(8, 16 // rows)
    assembly.extend(("    mov eax, 3", "    kmovb k1, eax"))

    for tile_row in range(rows):
        first = first_accumulator + tile_row * chains_per_row
        c_offset = (row_start + tile_row) * 2 * 8
        assembly.append(
            f"    vmovupd zmm{first} {{k1}}{{z}}, {_memory('rdx', c_offset)}"
        )
        for chain in range(1, chains_per_row):
            accumulator = first + chain
            assembly.append(
                f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
            )

    for k_index in range(k):
        assembly.append(
            f"    vmovupd zmm0 {{k1}}{{z}}, {_memory('rsi', k_index * 2 * 8)}"
        )
        chain = k_index % chains_per_row
        for tile_row in range(rows):
            row = row_start + tile_row
            accumulator = first_accumulator + tile_row * chains_per_row + chain
            a_offset = (row * k + k_index) * 8
            assembly.append(
                f"    vfmadd231pd zmm{accumulator}, zmm0, "
                f"{_memory('rdi', a_offset)}{{1to8}}"
            )

    for tile_row in range(rows):
        first = first_accumulator + tile_row * chains_per_row
        for chain in range(1, chains_per_row):
            accumulator = first + chain
            assembly.append(f"    vaddpd zmm{first}, zmm{first}, zmm{accumulator}")
        c_offset = (row_start + tile_row) * 2 * 8
        assembly.append(f"    vmovupd {_memory('rdx', c_offset)} {{k1}}, zmm{first}")


def generate(m: int, n: int, k: int, strategy: Strategy) -> str:
    if k != VECTOR_LENGTH * K_VECTORS:
        raise ValueError(f"this proof of concept requires K=64, got {k}")

    try:
        partition = PARTITIONS[m, n, strategy]
    except KeyError as error:
        raise ValueError(
            f"unsupported proof-of-concept case M={m}, N={n}, strategy={strategy}"
        ) from error
    if sum(rows for _, rows in partition) != m:
        raise AssertionError(f"partition {partition} does not cover M={m}")

    assembly = [
        ".intel_syntax noprefix",
        ".text",
        ".p2align 4, 0x90",
        ".globl matmul",
        ".type matmul, @function",
        "matmul:",
    ]
    row_start = 0
    for kind, rows in partition:
        if kind == "kdot":
            _emit_n1_tile(assembly, row_start, rows, k)
        elif kind == "gather":
            _emit_gather_n2_tile(assembly, row_start, rows, k)
        else:
            _emit_outer_n2_tile(assembly, row_start, rows, k)
        row_start += rows

    assembly.extend(("    ret", ".size matmul, .-matmul"))
    if any(kind == "gather" for kind, _ in partition):
        assembly.extend(
            (
                ".section .rodata",
                ".p2align 6",
                ".Lindices_n2:",
                "    .quad "
                + ", ".join(str(index * 2) for index in range(VECTOR_LENGTH)),
            )
        )
    return "\n".join(assembly) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fused, explicitly partitioned AVX-512 f64 GEMM kernel"
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--strategy", choices=("equalized", "cost"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strategy = cast(Strategy, args.strategy)
    args.output.write_text(generate(args.M, args.N, args.K, strategy))


if __name__ == "__main__":
    main()
