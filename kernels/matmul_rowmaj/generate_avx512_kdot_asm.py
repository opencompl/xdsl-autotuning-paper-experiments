import argparse
from pathlib import Path

VECTOR_LENGTH = 8
K_VECTORS = 8
LAST_VECTOR_REGISTER = 31


def _memory(base: str, offset: int) -> str:
    if offset == 0:
        return f"[{base}]"
    return f"[{base}+{offset}]"


def _emit_horizontal_sums(
    assembly: list[str], outputs: list[tuple[int, int]], *, batch_size: int
) -> None:
    """Reduce independent accumulators in latency-hiding stages."""
    for batch_start in range(0, len(outputs), batch_size):
        batch = outputs[batch_start : batch_start + batch_size]
        for scratch, (accumulator, _) in enumerate(batch):
            assembly.append(f"    vextractf64x4 ymm{scratch}, zmm{accumulator}, 1")
        for scratch, (accumulator, _) in enumerate(batch):
            assembly.append(f"    vaddpd ymm{scratch}, ymm{accumulator}, ymm{scratch}")
        for scratch, _ in enumerate(batch):
            assembly.append(
                f"    vextractf128 xmm{batch_size + scratch}, ymm{scratch}, 1"
            )
        for scratch, _ in enumerate(batch):
            assembly.append(
                f"    vaddpd xmm{scratch}, xmm{scratch}, xmm{batch_size + scratch}"
            )
        for scratch, _ in enumerate(batch):
            assembly.append(
                f"    vshufpd xmm{batch_size + scratch}, xmm{scratch}, xmm{scratch}, 1"
            )
        for scratch, (accumulator, _) in enumerate(batch):
            assembly.append(
                f"    vaddsd xmm{accumulator}, xmm{scratch}, xmm{batch_size + scratch}"
            )
        for accumulator, c_offset in batch:
            assembly.append(
                f"    vaddsd xmm{accumulator}, xmm{accumulator}, "
                f"{_memory('rdx', c_offset)}"
            )
        for accumulator, c_offset in batch:
            assembly.append(f"    vmovsd {_memory('rdx', c_offset)}, xmm{accumulator}")


def _emit_n1_kernel(assembly: list[str], m: int, k: int) -> None:
    for k_vector in range(K_VECTORS):
        assembly.append(
            f"    vmovupd zmm{k_vector}, {_memory('rsi', k_vector * VECTOR_LENGTH * 8)}"
        )

    first_accumulator = 16
    for row in range(m):
        accumulator = first_accumulator + row
        assembly.append(
            f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
        )

    for k_vector in range(K_VECTORS):
        for row in range(m):
            accumulator = first_accumulator + row
            a_offset = (row * k + k_vector * VECTOR_LENGTH) * 8
            assembly.append(
                f"    vfmadd231pd zmm{accumulator}, zmm{k_vector}, "
                f"{_memory('rdi', a_offset)}"
            )

    _emit_horizontal_sums(
        assembly,
        [(first_accumulator + row, row * 8) for row in range(m)],
        batch_size=8,
    )


def _emit_gather_kernel(assembly: list[str], m: int, n: int, k: int) -> None:
    assembly.append("    mov eax, 255")

    reduction_batch_size = 4
    first_accumulator = 2 * reduction_batch_size
    accumulator_capacity = LAST_VECTOR_REGISTER - first_accumulator + 1
    rows_per_tile = accumulator_capacity // n

    for row_start in range(0, m, rows_per_tile):
        tile_rows = min(rows_per_tile, m - row_start)
        output_count = tile_rows * n
        assembly.append("    vmovdqu64 zmm0, [rip + .Lkdot_indices]")
        for output in range(output_count):
            accumulator = first_accumulator + output
            assembly.append(
                f"    vxorpd zmm{accumulator}, zmm{accumulator}, zmm{accumulator}"
            )

        for k_vector in range(K_VECTORS):
            for column in range(n):
                b_offset = (k_vector * VECTOR_LENGTH * n + column) * 8
                assembly.extend(
                    (
                        "    kmovb k1, eax",
                        (
                            f"    vgatherqpd zmm1 {{k1}}, "
                            f"{_memory('rsi + zmm0*8', b_offset)}"
                        ),
                    )
                )
                for tile_row in range(tile_rows):
                    row = row_start + tile_row
                    accumulator = first_accumulator + tile_row * n + column
                    a_offset = (row * k + k_vector * VECTOR_LENGTH) * 8
                    assembly.append(
                        f"    vfmadd231pd zmm{accumulator}, zmm1, "
                        f"{_memory('rdi', a_offset)}"
                    )

        outputs = [
            (
                first_accumulator + tile_row * n + column,
                ((row_start + tile_row) * n + column) * 8,
            )
            for tile_row in range(tile_rows)
            for column in range(n)
        ]
        _emit_horizontal_sums(assembly, outputs, batch_size=reduction_batch_size)


def generate(m: int, n: int, k: int) -> str:
    if not 1 <= m <= 16:
        raise ValueError(f"M must be in [1, 16], got {m}")
    if not 1 <= n <= 4:
        raise ValueError(f"N must be in [1, 4], got {n}")
    if k != VECTOR_LENGTH * K_VECTORS:
        raise ValueError(f"this proof of concept requires K=64, got {k}")

    assembly = [
        ".intel_syntax noprefix",
        ".text",
        ".p2align 4, 0x90",
        ".globl matmul",
        ".type matmul, @function",
        "matmul:",
    ]
    if n == 1:
        _emit_n1_kernel(assembly, m, k)
    else:
        _emit_gather_kernel(assembly, m, n, k)
    assembly.extend(("    ret", ".size matmul, .-matmul"))

    if n > 1:
        assembly.extend(
            (
                ".section .rodata",
                ".p2align 6",
                ".Lkdot_indices:",
                "    .quad " + ", ".join(str(i * n) for i in range(VECTOR_LENGTH)),
            )
        )

    return "\n".join(assembly) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a leaf AVX-512 K-vectorized f64 GEMM kernel"
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate(args.M, args.N, args.K))


if __name__ == "__main__":
    main()
