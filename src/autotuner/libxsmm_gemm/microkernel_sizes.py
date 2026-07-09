"""Enumerate the register-tile (microkernel) sizes the AVX512 GEMM generator supports.

For a plain dense F64/F32 GEMM on skylake (``LIBXSMM_X86_AVX512_SKX``) the generator
splits M into register blocks of ``m_blocking`` rows (``m_vector = ceil(m_blocking /
vector_length)`` zmm registers, 1..4) and N into blocks of ``n_blocking`` columns. The N
block size is capped by how many accumulator registers fit alongside the A/B registers in
the 32-wide zmm file. This module mirrors those decisions (reusing the generator's own
helper functions) to produce the exact list of supported ``(m_blocking, n_blocking)`` tiles,
which is the single source of truth for the microkernel benchmark sweep.

Run as ``python -m autotuner.libxsmm_gemm.microkernel_sizes`` to print the sizes.
"""

from __future__ import annotations

from autotuner.libxsmm_gemm.generator_common import MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking,
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype

# The only architecture the generator currently supports.
ARCH = Arch.LIBXSMM_X86_AVX512_SKX

# The largest register M-block: 4 zmm registers of ``vector_length`` rows each. For
# F64 that is 32 rows; the generator caps m_blocking at this (get_blocking_and_mask).
_MAX_M_BLOCKS = 4


def _make_descriptor(m: int, k: int, dt: Datatype) -> GEMMDescriptor:
    d = DescDatatype(dt, dt, dt, dt)
    # Leading dimensions do not influence blocking for a plain dense GEMM; use the
    # natural column-major minimums (n is irrelevant to the cap, so use m as a filler).
    return GEMMDescriptor(m, m, k, m, k, m, d, GEMMFlag(0), GEMMPrefetchType.NONE)


def _max_n_blocking(config: MicroKernelConfig, desc: GEMMDescriptor) -> int:
    """Mirror the register-budget reduction in the kernel wrapper.

    Start from the architecture cap (``get_max_n_blocking``), then shrink until the N
    accumulators plus the M-block registers plus one B register fit into the zmm file.
    """
    max_n = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        config, desc, ARCH
    )
    if max_n > 3:
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, ARCH, 0
        )
        init_m_blocks = (
            init_m_blocking + config.vector_length - 1
        ) // config.vector_length
        # F64/F32 on SKX takes the else-branch: init_m_blocks*n + init_m_blocks + 1.
        while (init_m_blocks * max_n + init_m_blocks + 1) > config.vector_reg_count:
            max_n -= 1
    assert max_n, "max_n_blocking collapsed to 0"
    return max_n


def _vector_length(dt: Datatype) -> int:
    desc = _make_descriptor(1, 1, dt)
    config = MicroKernelConfig()
    libxsmm_generator_gemm_init_micro_kernel_config(config, ARCH, desc, False)
    assert config.vector_length, "micro kernel config not populated"
    return config.vector_length


def nanokernel_for_m_blocking(m_blocking: int, dt: Datatype = Datatype.F64) -> str:
    """Return the AVX512 nanokernel emitted for an M register block of this width.

    Mirror of the dispatch in ``libxsmm_generator_gemm_avx512_kloop_kernel``: for a
    plain dense F32/F64 GEMM the choice reduces to ``m_vector == 1`` -> ``fsdbcst`` (B
    fused as a broadcast memory operand of the FMA), otherwise ``nofsdbcst`` (B
    broadcast into a register first). ``m_vector`` is the number of vector registers
    spanning the M block, i.e. ``ceil(m_blocking / vector_length)``.
    """
    vlen = _vector_length(dt)
    m_vector = -(-m_blocking // vlen)  # ceil(m_blocking / vlen)
    return "fsdbcst" if m_vector == 1 else "nofsdbcst"


def supported_microkernel_sizes(
    k: int = 64, dt: Datatype = Datatype.F64
) -> list[tuple[int, int]]:
    """Return the list of supported ``(m_blocking, n_blocking)`` microkernel tiles.

    ``m_blocking`` ranges over 1..(4*vector_length) rows (clean multiples of the vector
    length use no masking; the rest use an AVX512 tail mask). For each ``m_blocking``,
    ``n_blocking`` ranges 1..cap where cap is the register-budget limit for that M width.
    """
    sizes: list[tuple[int, int]] = []
    # vector_length depends only on arch+dtype, so compute it once from a probe config.
    probe_desc = _make_descriptor(1, k, dt)
    probe_config = MicroKernelConfig()
    libxsmm_generator_gemm_init_micro_kernel_config(
        probe_config, ARCH, probe_desc, False
    )
    assert probe_config.vector_length, "micro kernel config not populated"
    max_m = _MAX_M_BLOCKS * probe_config.vector_length

    for mb in range(1, max_m + 1):
        desc = _make_descriptor(mb, k, dt)
        config = MicroKernelConfig()
        libxsmm_generator_gemm_init_micro_kernel_config(config, ARCH, desc, False)
        cap = _max_n_blocking(config, desc)
        for nb in range(1, cap + 1):
            sizes.append((mb, nb))
    return sizes


def main() -> None:
    sizes = supported_microkernel_sizes()
    # Group by m_blocking for a compact summary.
    by_m: dict[int, int] = {}
    for mb, nb in sizes:
        by_m[mb] = max(by_m.get(mb, 0), nb)
    print(
        f"{len(sizes)} supported (m_blocking, n_blocking) microkernel tiles (F64/skx):"
    )
    for mb in sorted(by_m):
        print(f"  m_blocking={mb:2d}  n_blocking=1..{by_m[mb]}")


if __name__ == "__main__":
    main()
