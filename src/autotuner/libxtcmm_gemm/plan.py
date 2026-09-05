"""Extract the LIBXSMM microkernel schedule as data via the shared tiling model.

The tiling / splitting decision is delegated to
``autotuner.tiling.compute_tiling_strategy``, ie. the SAME function the
``xsmm-tile-n-m`` xDSL pass uses to tile compxsmm/libxsmm GEMMs. So the schedule
matches the shipped kernel by construction: single source of truth, no
duplicated heuristics. This adapter only converts the LIBXSMM descriptor into
 the target-independent one, then returns the strategy as data
for the XTC scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag
from autotuner.nano_kernel import GemmDescriptor as NanoDescriptor
from autotuner.skx_nano_kernel import AVX512Info, SkxNanoKernel
from autotuner.tiling import BlockingRange, compute_tiling_strategy


@dataclass(frozen=True)
class XtcGemmPlan:
    """The LIBXSMM schedule, as data, in LIBXSMM dimension names.

    N, M and K are all expressed as tiers of ``tiling.BlockingRange``, where a
    range ``(extent, tile_size)`` is a rolled loop over ``extent // tile_size``
    blocks each holding an unrolled body of ``tile_size``:

    * ``n_ranges``: the 1-or-2 tier equalized split of N (broadcast columns).
    * ``m_ranges``: how M is covered by register blocks -- whole ``m_tile_size``
      blocks plus a remainder block when ``M % m_tile_size != 0``. Each tile is
      a register block (``ceil(tile_size / vector_length)`` zmm, the tail lane
      masked when ``tile_size`` is not a multiple of ``vector_length``).
    * ``k_ranges``: how the K reduction is covered -- a single fully-unrolled
      range for small K, otherwise a rolled loop of four-wide blocks plus an
      unrolled remainder.
    """

    vector_length: int
    n_ranges: tuple[BlockingRange, ...]
    m_ranges: tuple[BlockingRange, ...]
    k_ranges: tuple[BlockingRange, ...]


def _cover_ranges(extent: int, tile_size: int) -> tuple[BlockingRange, ...]:
    """Cover ``extent`` by whole ``tile_size`` blocks plus an optional remainder."""
    remainder = extent % tile_size
    full = extent - remainder
    ranges = []
    if full:
        ranges.append(BlockingRange(full, tile_size))
    if remainder:
        ranges.append(BlockingRange(remainder, remainder))
    return tuple(ranges)


def compute_plan(desc: GEMMDescriptor, arch: Arch) -> XtcGemmPlan:
    """Delegate the N/M tiling to the shared model and cover K like LIBXSMM."""
    assert arch == Arch.LIBXSMM_X86_AVX512_SKX, "libxtcmm-gemm supports skx only"
    isa_info = AVX512Info()
    datatype = desc.datatype.a.builtin_type
    vector_length = isa_info.vector_length(datatype)

    nano_desc = NanoDescriptor(
        m=desc.m,
        n=desc.n,
        k=desc.k,
        lda=desc.lda,
        ldb=desc.ldb,
        ldc=desc.ldc,
        datatype=datatype,
        aligned_a=bool(desc.flags & GEMMFlag.ALIGN_A),
        aligned_c=bool(desc.flags & GEMMFlag.ALIGN_C),
    )
    strategy = compute_tiling_strategy(nano_desc, isa_info, SkxNanoKernel())

    return XtcGemmPlan(
        vector_length=vector_length,
        n_ranges=strategy.n_ranges,
        m_ranges=_cover_ranges(desc.m, strategy.m_tile_size),
        k_ranges=_cover_ranges(desc.k, strategy.k_tile_size),
    )
