"""Fidelity tests for ``libxtcmm_gemm.plan.compute_plan``.

These only exercise the LIBXSMM decision functions (no ``xtc`` needed), so they
run under ``make pytest``. They pin the schedule extracted for the canonical
shape and check the register-resident scope guards.
"""

import pytest

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxtcmm_gemm.plan import compute_plan

SKX = Arch.LIBXSMM_X86_AVX512_SKX


def _desc(m: int, n: int, k: int, dt: Datatype) -> GEMMDescriptor:
    ddt = DescDatatype(dt, dt, dt, dt)
    return GEMMDescriptor(m, n, k, m, k, m, ddt, GEMMFlag(0), GEMMPrefetchType.NONE)


def test_plan_16x29x16_f64_matches_libxsmm():
    # The canonical 29x16x16.mlir shape: M=16 vectorized (2 zmm), N=29 split
    # 20(x10)+9(x9), K=16 fully unrolled.
    plan = compute_plan(_desc(16, 29, 16, Datatype.F64), SKX)
    assert plan.vector_length == 8
    assert [(r.extent, r.tile_size) for r in plan.m_ranges] == [(16, 16)]
    assert [(r.extent, r.tile_size) for r in plan.k_ranges] == [(16, 16)]
    assert [(r.extent, r.tile_size) for r in plan.n_ranges] == [(20, 10), (9, 9)]


def test_plan_single_range_f64():
    # N divides evenly -> a single equalized N tier.
    plan = compute_plan(_desc(16, 16, 16, Datatype.F64), SKX)
    assert plan.vector_length == 8
    assert len(plan.n_ranges) == 1


@pytest.mark.parametrize(
    "m,n,k,dt,vlen",
    [
        (16, 29, 16, Datatype.F64, 8),
        (16, 16, 16, Datatype.F64, 8),
        (16, 5, 16, Datatype.F64, 8),
        (16, 66, 16, Datatype.F64, 8),
        (32, 38, 16, Datatype.F32, 16),
        # Masked M (not a multiple of the vector length).
        (4, 16, 16, Datatype.F64, 8),
        (20, 29, 16, Datatype.F64, 8),
        (44, 16, 16, Datatype.F64, 8),
    ],
)
def test_plan_invariants(m, n, k, dt, vlen):
    plan = compute_plan(_desc(m, n, k, dt), SKX)
    assert plan.vector_length == vlen
    # N, M and K ranges each partition their dimension exactly, every tile
    # divides its extent, and every M tile fits the nano-kernel's 4 vector
    # registers (a tile not a multiple of the vector length has a masked tail).
    for dim, ranges in ((n, plan.n_ranges), (m, plan.m_ranges), (k, plan.k_ranges)):
        assert sum(r.extent for r in ranges) == dim
        for r in ranges:
            assert r.extent % r.tile_size == 0
            assert r.tile_size >= 1
    for r in plan.m_ranges:
        assert -(-r.tile_size // plan.vector_length) <= 4


def test_k_loop_ranges():
    # K <= 23 -> a single fully-unrolled range (no K loop).
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(16, 16, 16, Datatype.F64), SKX).k_ranges
    ] == [(16, 16)]
    # K > 23 and a multiple of the blocking factor -> one rolled loop of 4-wide
    # blocks.
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(16, 16, 100, Datatype.F64), SKX).k_ranges
    ] == [(100, 4)]
    # K > 23 not a multiple of the blocking factor -> a rolled loop over the
    # 4-wide blocks plus an unrolled remainder.
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(16, 16, 26, Datatype.F64), SKX).k_ranges
    ] == [(24, 4), (2, 2)]


def test_masked_m_supported():
    # M not a multiple of the vector length -> a single (masked) register block.
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(20, 16, 16, Datatype.F64), SKX).m_ranges
    ] == [(20, 20)]
    # A sub-vector M is one masked block; the tail lane covers the whole M.
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(4, 16, 16, Datatype.F64), SKX).m_ranges
    ] == [(4, 4)]
    # Full blocks plus a masked remainder (12 = one full + a masked 4).
    assert [
        (r.extent, r.tile_size)
        for r in compute_plan(_desc(44, 16, 16, Datatype.F64), SKX).m_ranges
    ] == [(32, 32), (12, 12)]


def test_multiple_m_blocks_accepted():
    # M=64 f64 -> m_tile_size=32, 64 % 32 == 0 -> a single tier of 2 whole blocks.
    plan = compute_plan(_desc(64, 16, 16, Datatype.F64), SKX)
    assert [(r.extent, r.tile_size) for r in plan.m_ranges] == [(64, 32)]


def test_m_remainder_split():
    # M=48 f64 -> m_tile_size=32 -> a full-blocks tier (32) + a remainder (16).
    plan = compute_plan(_desc(48, 16, 16, Datatype.F64), SKX)
    assert [(r.extent, r.tile_size) for r in plan.m_ranges] == [(32, 32), (16, 16)]
