from xdsl.dialects import builtin

from xdsl.dialects.x86 import registers as x86_registers

from autotuner.nano_kernel import GemmDescriptor, RegisterCount, TileSizes, VectorLayout
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel import (
    SKX_NANO_KERNELS,
    SkxNanoKernel,
    SkxNarrowNanoKernel,
    AVX512Info,
    get_skx_nano_kernel,
)
from autotuner.skx_narrow_fsdbcst_nano_kernel import SkxNarrowFsdbcstNanoKernel
from autotuner.skx_nofsdbcst_nano_kernel import SkxNofsdbcstNanoKernel
from autotuner.strategy import XSMM_STRATEGIES, get_xsmm_strategy
from autotuner.tiling import BlockingRange, TilingStrategy, compute_tiling_strategy


def _descriptor(
    *, m: int, n: int, k: int, datatype: builtin.Float32Type | builtin.Float64Type
) -> GemmDescriptor:
    return GemmDescriptor(
        m=m,
        n=n,
        k=k,
        lda=m,
        ldb=k,
        ldc=m,
        datatype=datatype,
        aligned_a=False,
        aligned_c=False,
    )


def test_register_count_fits() -> None:
    capacity = RegisterCount(general=16, vector=32, mask=8)
    assert RegisterCount(general=5, vector=32, mask=1).fits(capacity)
    assert not RegisterCount(general=5, vector=33, mask=1).fits(capacity)


def test_avx512_isa_info() -> None:
    assert AVX512Info().isa == "avx512"


def test_skx_nano_kernel_names() -> None:
    assert set(SKX_NANO_KERNELS) == {
        "libxsmm-skx",
        "libxsmm-skx-narrow",
        "libxsmm-skx-fsdbcst",
        "libxsmm-skx-narrow-fsdbcst",
        "libxsmm-skx-nofsdbcst",
    }
    for name, nano_kernel in SKX_NANO_KERNELS.items():
        assert nano_kernel.name == name
        assert get_skx_nano_kernel(name) is nano_kernel


def test_unknown_skx_nano_kernel() -> None:
    try:
        get_skx_nano_kernel("unknown")
    except ValueError as error:
        assert str(error) == (
            "unknown SKX nano-kernel 'unknown'; expected one of: "
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-narrow, "
            "libxsmm-skx-narrow-fsdbcst, libxsmm-skx-nofsdbcst"
        )
    else:
        raise AssertionError("expected an unknown nano-kernel to be rejected")


def test_xsmm_strategies_wrap_isa_and_nano_kernel_policy() -> None:
    assert set(XSMM_STRATEGIES) == set(SKX_NANO_KERNELS)
    strategy = get_xsmm_strategy("libxsmm-skx")
    assert strategy.isa_info.isa == "avx512"
    assert strategy.nano_kernel is get_skx_nano_kernel("libxsmm-skx")


def test_unknown_xsmm_strategy() -> None:
    try:
        get_xsmm_strategy("unknown")
    except ValueError as error:
        assert str(error) == (
            "unknown XSMM strategy 'unknown'; expected one of: "
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-narrow, "
            "libxsmm-skx-narrow-fsdbcst, libxsmm-skx-nofsdbcst"
        )
    else:
        raise AssertionError("expected an unknown strategy to be rejected")


def test_skx_register_usage() -> None:
    isa_info = AVX512Info()
    kernel = SkxNanoKernel()

    f64 = _descriptor(m=8, n=2, k=2, datatype=builtin.f64)
    assert kernel.register_usage(f64, TileSizes(8, 2, 2), isa_info) == RegisterCount(
        general=5, vector=6, mask=0
    )

    f32 = _descriptor(m=17, n=1, k=2, datatype=builtin.f32)
    assert kernel.register_usage(f32, TileSizes(17, 1, 2), isa_info) == RegisterCount(
        general=5, vector=5, mask=1
    )


def test_skx_fsdbcst_supported_tiles() -> None:
    isa_info = AVX512Info()
    kernel = SkxFsdbcstNanoKernel()
    descriptor = _descriptor(m=8, n=31, k=2, datatype=builtin.f64)

    assert kernel.supports_tile(descriptor, TileSizes(8, 28, 2), isa_info)
    assert kernel.register_usage(
        descriptor, TileSizes(8, 30, 2), isa_info
    ) == RegisterCount(general=5, vector=32, mask=0)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 30, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 31, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(16, 1, 2), isa_info)


def test_skx_nofsdbcst_supported_tiles() -> None:
    isa_info = AVX512Info()
    kernel = SkxNofsdbcstNanoKernel()
    descriptor = _descriptor(m=40, n=6, k=2, datatype=builtin.f64)

    assert kernel.supports_tile(descriptor, TileSizes(32, 6, 2), isa_info)
    assert kernel.register_usage(
        descriptor, TileSizes(40, 5, 2), isa_info
    ) == RegisterCount(general=5, vector=31, mask=0)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 5, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 6, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 1, 2), isa_info)


def test_skx_composite_retains_libxsmm_tiling_heuristics() -> None:
    isa_info = AVX512Info()
    kernel = SkxNanoKernel()
    f64 = _descriptor(m=40, n=29, k=2, datatype=builtin.f64)

    assert not SkxFsdbcstNanoKernel().supports_tile(f64, TileSizes(8, 29, 2), isa_info)
    assert not kernel.supports_tile(f64, TileSizes(8, 29, 2), isa_info)
    assert not SkxNofsdbcstNanoKernel().supports_tile(
        f64, TileSizes(40, 5, 2), isa_info
    )
    assert not kernel.supports_tile(f64, TileSizes(40, 5, 2), isa_info)


def test_single_n_range_tiling_strategy() -> None:
    strategy = compute_tiling_strategy(
        _descriptor(m=16, n=28, k=16, datatype=builtin.f64),
        AVX512Info(),
        SkxNanoKernel(),
    )
    assert strategy == TilingStrategy(
        m_tile_size=16,
        n_ranges=(BlockingRange(extent=28, tile_size=14),),
    )


def test_two_n_ranges_and_m_remainder_tiling_strategy() -> None:
    strategy = compute_tiling_strategy(
        _descriptor(m=70, n=38, k=128, datatype=builtin.f32),
        AVX512Info(),
        SkxNanoKernel(),
    )
    assert strategy == TilingStrategy(
        m_tile_size=64,
        n_ranges=(
            BlockingRange(extent=18, tile_size=6),
            BlockingRange(extent=20, tile_size=5),
        ),
    )


def test_avx512_vector_banks_run_narrowest_first() -> None:
    isa_info = AVX512Info()
    assert isa_info.vector_banks == (
        x86_registers.SSERegisterType,
        x86_registers.AVX2RegisterType,
        x86_registers.AVX512RegisterType,
    )
    assert isa_info.widest_vector_layout(builtin.f64) == VectorLayout(
        x86_registers.AVX512RegisterType, 8
    )
    assert isa_info.widest_vector_layout(builtin.f32) == VectorLayout(
        x86_registers.AVX512RegisterType, 16
    )


def test_narrowest_vector_layout_rounds_up_to_a_bank() -> None:
    isa_info = AVX512Info()
    # f64: 1-2 elements fit an xmm, 3-4 a ymm, 5-8 a zmm.
    assert [
        isa_info.narrowest_vector_layout(builtin.f64, lanes).lanes
        for lanes in range(1, 9)
    ] == [2, 2, 4, 4, 8, 8, 8, 8]
    # f32: four elements per xmm, eight per ymm, sixteen per zmm.
    assert [
        isa_info.narrowest_vector_layout(builtin.f32, lanes).lanes
        for lanes in (1, 4, 5, 8, 9, 16)
    ] == [4, 4, 8, 8, 16, 16]
    try:
        isa_info.narrowest_vector_layout(builtin.f64, 9)
    except ValueError as error:
        assert str(error) == "no vector bank holds 9 f64 elements"
    else:
        raise AssertionError("expected a too-wide lane count to be rejected")


def test_narrow_fsdbcst_narrows_the_bank_of_a_partial_m_vector() -> None:
    isa_info = AVX512Info()
    kernel = SkxNarrowFsdbcstNanoKernel()
    descriptor = _descriptor(m=3, n=13, k=64, datatype=builtin.f64)

    # A three-element M tile lands in a ymm, with one lane masked off; the
    # full-width kernel would mask five lanes of a zmm for the same tile.
    assert kernel.vector_layout(
        descriptor, TileSizes(3, 13, 64), isa_info
    ) == VectorLayout(x86_registers.AVX2RegisterType, 4)
    assert SkxFsdbcstNanoKernel().vector_layout(
        descriptor, TileSizes(3, 13, 64), isa_info
    ) == VectorLayout(x86_registers.AVX512RegisterType, 8)

    # An M tile that fills the widest bank is unaffected.
    full = _descriptor(m=8, n=13, k=64, datatype=builtin.f64)
    assert kernel.vector_layout(full, TileSizes(8, 13, 64), isa_info) == VectorLayout(
        x86_registers.AVX512RegisterType, 8
    )


def test_narrow_fsdbcst_masks_only_what_its_own_bank_leaves_over() -> None:
    isa_info = AVX512Info()
    descriptor = _descriptor(m=4, n=2, k=2, datatype=builtin.f64)

    # M exactly fills a ymm, so the narrow kernel needs no mask register at all,
    # where the full-width kernel masks the upper four lanes of a zmm.
    assert SkxNarrowFsdbcstNanoKernel().register_usage(
        descriptor, TileSizes(4, 2, 2), isa_info
    ) == RegisterCount(general=5, vector=6, mask=0)
    assert SkxFsdbcstNanoKernel().register_usage(
        descriptor, TileSizes(4, 2, 2), isa_info
    ) == RegisterCount(general=5, vector=6, mask=1)


def test_narrow_composite_narrows_one_vector_tiles_only() -> None:
    isa_info = AVX512Info()
    kernel = SkxNarrowNanoKernel()
    descriptor = _descriptor(m=40, n=6, k=2, datatype=builtin.f64)

    # One M vector: narrowed.
    assert kernel.vector_layout(
        descriptor, TileSizes(2, 6, 2), isa_info
    ) == VectorLayout(x86_registers.SSERegisterType, 2)
    # Several M vectors: LIBXSMM's full-width register-broadcast kernel.
    assert kernel.vector_layout(
        descriptor, TileSizes(32, 6, 2), isa_info
    ) == VectorLayout(x86_registers.AVX512RegisterType, 8)
    # Tile legality is unchanged from the full-width heuristic.
    for tile in (TileSizes(2, 6, 2), TileSizes(32, 6, 2), TileSizes(40, 5, 2)):
        assert kernel.supports_tile(
            descriptor, tile, isa_info
        ) == SkxNanoKernel().supports_tile(descriptor, tile, isa_info)


def test_narrow_composite_keeps_the_libxsmm_tiling_strategy() -> None:
    # The narrow kernel changes which register bank a tile lands in, not how the
    # GEMM is tiled, so both strategies must plan identically.
    for descriptor in (
        _descriptor(m=16, n=28, k=16, datatype=builtin.f64),
        _descriptor(m=70, n=38, k=128, datatype=builtin.f32),
        _descriptor(m=3, n=13, k=64, datatype=builtin.f64),
        _descriptor(m=12, n=1, k=64, datatype=builtin.f64),
    ):
        assert compute_tiling_strategy(
            descriptor, AVX512Info(), SkxNarrowNanoKernel()
        ) == compute_tiling_strategy(descriptor, AVX512Info(), SkxNanoKernel())
