from dataclasses import dataclass
from typing import Protocol

from autotuner.nano_kernel import (
    GemmDescriptor,
    ISAInfo,
    NanoKernel,
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


class NTilePolicy(Protocol):
    """Choose the high-level decomposition of the N iteration space."""

    def n_ranges(
        self,
        descriptor: GemmDescriptor,
        m_tile_size: int,
        max_n_tile: int,
        isa_info: ISAInfo,
        nano_kernel: NanoKernel,
    ) -> tuple[BlockingRange, ...]: ...


def compute_equalized_n_ranges(n: int, max_n_tile: int) -> tuple[BlockingRange, ...]:
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


def compute_greedy_n_ranges(n: int, n_tile: int) -> tuple[BlockingRange, ...]:
    """Cover N with full tiles followed by one remainder range."""
    blocked_extent = n // n_tile * n_tile
    remainder = n - blocked_extent
    ranges: list[BlockingRange] = []
    if blocked_extent:
        ranges.append(BlockingRange(blocked_extent, n_tile))
    if remainder:
        ranges.append(BlockingRange(remainder, remainder))
    return tuple(ranges)


@dataclass(frozen=True)
class EqualizedNTilePolicy:
    """Use LIBXSMM's equalized one- or two-range decomposition."""

    def n_ranges(
        self,
        descriptor: GemmDescriptor,
        m_tile_size: int,
        max_n_tile: int,
        isa_info: ISAInfo,
        nano_kernel: NanoKernel,
    ) -> tuple[BlockingRange, ...]:
        return compute_equalized_n_ranges(descriptor.n, max_n_tile)


EQUALIZED_N_TILING = EqualizedNTilePolicy()


def compute_tiling_strategy(
    descriptor: GemmDescriptor,
    isa_info: ISAInfo,
    nano_kernel: NanoKernel,
    n_tiling: NTilePolicy | None = None,
) -> TilingStrategy:
    """Choose legal M blocking and apply the selected N-range policy."""
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

    n_tiling = EQUALIZED_N_TILING if n_tiling is None else n_tiling
    return TilingStrategy(
        m_tile_size,
        n_tiling.n_ranges(
            descriptor,
            m_tile_size,
            max_n_tile,
            isa_info,
            nano_kernel,
        ),
    )
