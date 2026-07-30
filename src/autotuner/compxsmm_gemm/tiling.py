"""Tiling planner for the compxsmm GEMM generator.

Pure description of how ``..._kernel`` / ``..._kloop`` split M, N and K, with no code
emission. The emitting generator in ``generator_gemm_sse_avx_avx2_avx512`` consumes these
functions, so the plan and the loops that get emitted cannot drift apart.

Everything the planner does not change -- the blocking primitives, the descriptor, the
micro-kernel config -- is imported from the faithful libxsmm port rather than copied.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import NamedTuple

from autotuner.libxsmm_gemm.generator_common import (
    MicroKernelConfig,
    libxsmm_compute_equalized_blocking,
)
from autotuner.libxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking,
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


class NLoop(NamedTuple):
    """One hardware N-loop (an equalized-blocking tier).

    ``extent`` columns are covered in steps of ``n_blocking`` (tile count =
    ``extent // n_blocking``).
    """

    start: int
    n_blocking: int
    extent: int


class MLoop(NamedTuple):
    """One hardware M-loop group: ``n_tiles`` tiles of width ``m_blocking``.

    ``use_masking_a_c`` is the masking flag ``get_m_blocking`` set for this width.
    """

    start: int
    m_blocking: int
    n_tiles: int
    use_masking_a_c: bool


class KPhaseKind(StrEnum):
    """How a ``KPhase`` is emitted.

    ``LOOPED`` is a hardware K-loop of ``n_tiles`` tiles of ``size`` (``full`` marks the
    strategy-1 whole-K loop); ``UNROLLED`` is a single fully-unrolled tile; ``REMAINDER``
    is the strategy-3 tail tile.
    """

    LOOPED = "looped"
    UNROLLED = "unrolled"
    REMAINDER = "remainder"


class KPhase(NamedTuple):
    """One K emission phase."""

    kind: KPhaseKind
    size: int
    n_tiles: int
    extent: int
    full: bool


class KPlan(NamedTuple):
    k_blocking: int
    k_threshold: int
    strategy: int  # 1 = blocked loop, 2 = full unroll, 3 = blocked + remainder
    phases: list[KPhase]


def compxsmm_generator_gemm_sse_avx_avx2_avx512_compute_max_n_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch
) -> int:
    """N accumulator blocking (mirrors the top of ..._kernel).

    Starts from the architecture cap, then shrinks until the N accumulators plus the M
    block registers fit into the vector register file. The ``get_m_blocking(config, …,
    0)`` probe mutates ``config.use_masking_a_c`` as a side effect, exactly as the
    emitting kernel does just before the N loop, so callers get identical config state.
    """
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        config, desc, arch
    )
    if max_n_blocking > 3:
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, arch, 0
        )
        init_m_blocks = (
            init_m_blocking + config.vector_length - 1
        ) // config.vector_length
        is_Ai8_Bf16_gemm = (
            desc.datatype.a == Datatype.I8
            and desc.datatype.b == Datatype.F16
            and (desc.datatype.c in (Datatype.F16, Datatype.F32))
        )
        is_Ai8_Bbf16_gemm = (
            (desc.datatype.a == Datatype.I8 and not desc.is_Amxfp4_Bbf16_gemm())
            and (desc.datatype.b == Datatype.BF16)
            and (desc.datatype.c in (Datatype.BF16, Datatype.F32))
        )

        if is_Ai8_Bf16_gemm:
            raise NotImplementedError
        elif is_Ai8_Bbf16_gemm:
            raise NotImplementedError
        elif desc.is_Ai4_Bi8_gemm():
            raise NotImplementedError
        elif desc.is_Ai2_Bi8_gemm():
            raise NotImplementedError
        elif desc.is_Ai1_Bi8_gemm():
            raise NotImplementedError
        else:
            if Arch.LIBXSMM_X86_AVX2_SRF <= arch < Arch.LIBXSMM_X86_AVX512_SKX:
                while (
                    init_m_blocks * max_n_blocking + max_n_blocking + 1
                ) > config.vector_reg_count:
                    max_n_blocking -= 1
            else:
                while (
                    init_m_blocks * max_n_blocking + init_m_blocks + 1
                ) > config.vector_reg_count:
                    max_n_blocking -= 1

    assert max_n_blocking
    return max_n_blocking


def compxsmm_generator_gemm_sse_avx_avx2_avx512_iter_n_loops(
    n: int, max_n_blocking: int
) -> list[NLoop]:
    """The N split: one ``NLoop`` per equalized-blocking tier (mirrors ..._kernel)."""
    blocking = libxsmm_compute_equalized_blocking(n, max_n_blocking)
    n_N = [blocking.range_1, blocking.range_2]
    n_n = [blocking.block_1, blocking.block_2]
    assert n_N[0]

    loops: list[NLoop] = []
    n_done = 0
    n_count = 0
    while n_done != n:
        loops.append(NLoop(n_done, n_n[n_count], n_N[n_count]))
        n_done += n_N[n_count]
        n_count += 1
    return loops


def compxsmm_generator_gemm_sse_avx_avx2_avx512_iter_m_loops(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch
) -> Iterator[MLoop]:
    """The M split (mirrors the ``while m_done`` walk in ..._kernel).

    A generator so the per-group ``get_m_blocking`` calls — which mutate
    ``config.use_masking_a_c`` / ``config.c_vmove_nts_instruction`` — interleave with the
    caller's per-group work exactly as in the emitting kernel: at each ``yield`` the
    config reflects the group being yielded. One ``MLoop`` per emitted hardware M-loop.
    """
    m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
        config, desc, arch, 0
    )
    m_done = 0
    while m_done != desc.m:
        assert m_blocking
        m_done_old = m_done
        # Consume all full m_blocking chunks at once (one hardware M-loop).
        count = (desc.m - m_done_old) // m_blocking
        m_done = m_done_old + count * m_blocking
        if m_done != m_done_old:
            yield MLoop(m_done_old, m_blocking, count, config.use_masking_a_c)
        # Recompute to obtain the (smaller) remainder block width for the next group.
        m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, arch, m_blocking
        )


def compxsmm_generator_gemm_sse_avx_avx2_avx512_compute_k_plan(
    desc: GEMMDescriptor, arch: Arch
) -> KPlan:
    """The K split (mirrors the parameters + 3-strategy branch of ..._kloop).

    Returns the ``k_blocking``/``k_threshold`` and the emission phases without emitting
    or selecting the micro-kernel.
    """
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    is_Ai8_Bbf16_gemm = (
        desc.datatype.a == "I8"
        and not is_Amxfp4_Bbf16_gemm
        and desc.datatype.b == "BF16"
    )
    is_Ai4_Bf16_gemm = (
        (desc.flags & GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2)
        and desc.datatype.a == "I8"
        and desc.datatype.b == "F16"
        and (desc.datatype.c == "F16" or desc.datatype.c == "F32")
    )
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_i8_uu_ss_gemm = desc.datatype.ab == "I8" and (
        GEMMFlag.A_UNSIGNED in desc.flags == GEMMFlag.B_UNSIGNED in desc.flags
    )

    # a very simple k unrolling model
    k_blocking = 4
    k_threshold = 23

    if GEMMFlag.VNNI_A in desc.flags:
        # VNNI kernel should maintain the same amount of unrolled instructions
        raise NotImplementedError

    if is_i8_uu_ss_gemm and arch in (
        Arch.LIBXSMM_X86_AVX512_SKX,
        Arch.LIBXSMM_X86_AVX512_VL256_SKX,
    ):
        # for uu ss int 8 we need to limit the unrolling, software emulation code is very large
        k_blocking = 8
        k_threshold = 23

    if is_Ai8_Bbf16_gemm:
        raise NotImplementedError

    assert k_blocking <= k_threshold

    if is_Ai4_Bf16_gemm:
        k_blocking = 4
        k_threshold = 8
    if is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm:
        k_blocking = 32
        k_threshold = desc.k

    k = desc.k
    if not k % k_blocking and k_threshold < k:
        # 1. larger than the threshold and a multiple of the blocking parameter
        strategy = 1
        phases = [KPhase(KPhaseKind.LOOPED, k_blocking, k // k_blocking, k, True)]
    elif k <= k_threshold:
        # 2. fully unroll below the threshold
        strategy = 2
        phases = [KPhase(KPhaseKind.UNROLLED, k, 1, k, False)]
    else:
        # 3. largest possible blocking + remainder handling
        strategy = 3
        l_max_blocked_k = (k // k_blocking) * k_blocking
        phases: list[KPhase] = []
        if l_max_blocked_k > 0:
            phases.append(
                KPhase(
                    KPhaseKind.LOOPED,
                    k_blocking,
                    l_max_blocked_k // k_blocking,
                    l_max_blocked_k,
                    False,
                )
            )
        phases.append(
            KPhase(
                KPhaseKind.REMAINDER, k - l_max_blocked_k, 1, k - l_max_blocked_k, False
            )
        )

    return KPlan(k_blocking, k_threshold, strategy, phases)
