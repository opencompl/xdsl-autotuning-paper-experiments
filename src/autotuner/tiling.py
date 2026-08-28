from dataclasses import dataclass

from autotuner.nano_kernel import (
    GemmDescriptor,
    NanoKernel,
    ISAInfo,
    TileSizes,
)


@dataclass(frozen=True)
class BlockingRange:
    """A range covered by repeated tiles of one size."""

    extent: int
    tile_size: int

    @property
    def iterations(self) -> int:
        return self.extent // self.tile_size


@dataclass(frozen=True)
class TilingStrategy:
    """The M tile and one or two ranges that cover the N dimension."""

    m_tile_size: int
    n_ranges: tuple[BlockingRange, ...]


def _compute_equalized_n_ranges(n: int, max_n_tile: int) -> tuple[BlockingRange, ...]:
    """Reproduce LIBXSMM's equalized one- or two-range N decomposition."""
    number_of_chunks = (n - 1) // max_n_tile + 1
    larger_chunk_count = n % number_of_chunks
    smaller_tile = n // number_of_chunks
    larger_tile = min(smaller_tile + 1, max_n_tile)

    if larger_chunk_count == 0:
        return (BlockingRange(n, smaller_tile),)

    larger_extent = larger_chunk_count * larger_tile
    smaller_extent = (number_of_chunks - larger_chunk_count) * smaller_tile
    ranges = [BlockingRange(larger_extent, larger_tile)]
    if smaller_extent:
        ranges.append(BlockingRange(smaller_extent, smaller_tile))
    return tuple(ranges)


def compute_tiling_strategy(
    descriptor: GemmDescriptor,
    isa_info: ISAInfo,
    nano_kernel: NanoKernel,
) -> TilingStrategy:
    """Choose the current LIBXSMM M tile and equalized N decomposition."""
    if not nano_kernel.supports(descriptor, isa_info):
        raise ValueError("nano-kernel does not support the GEMM descriptor")

    vector_length = isa_info.vector_length(descriptor.datatype)
    max_m_candidate = min(
        descriptor.m, isa_info.register_capacity.vector * vector_length
    )
    m_tile_size = next(
        (
            m
            for m in range(max_m_candidate, 0, -1)
            if (
                nano_kernel.supports_tile(
                    descriptor, TileSizes(m, 1, descriptor.k), isa_info
                )
                and nano_kernel.register_usage(
                    descriptor, TileSizes(m, 1, descriptor.k), isa_info
                ).fits(isa_info.register_capacity)
            )
        ),
        None,
    )
    if m_tile_size is None:
        raise ValueError("could not find a legal M tile")

    max_n_candidate = min(descriptor.n, isa_info.register_capacity.vector)
    max_n_tile = next(
        (
            n
            for n in range(max_n_candidate, 0, -1)
            if (
                nano_kernel.supports_tile(
                    descriptor, TileSizes(m_tile_size, n, descriptor.k), isa_info
                )
                and nano_kernel.register_usage(
                    descriptor,
                    TileSizes(m_tile_size, n, descriptor.k),
                    isa_info,
                ).fits(isa_info.register_capacity)
            )
        ),
        None,
    )
    if max_n_tile is None:
        raise ValueError("could not find a legal N tile")

    return TilingStrategy(
        m_tile_size,
        _compute_equalized_n_ranges(descriptor.n, max_n_tile),
    )
