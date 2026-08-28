from xdsl.dialects import builtin

from autotuner.avx512_kdot_nano_kernel import (
    Avx512KdotNanoKernel,
    KdotWithSkxFallbackNanoKernel,
)
from autotuner.nano_kernel import GemmDescriptor, RegisterCount, TileSizes
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel import (
    SKX_NANO_KERNELS,
    AVX512Info,
    SkxNanoKernel,
    get_skx_nano_kernel,
)
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
        "libxsmm-skx-fsdbcst",
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
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst"
        )
    else:
        raise AssertionError("expected an unknown nano-kernel to be rejected")


def test_xsmm_strategies_wrap_isa_and_nano_kernel_policy() -> None:
    assert set(XSMM_STRATEGIES) == {
        *SKX_NANO_KERNELS,
        "zen5-kdot",
        "zen5-kdot-greedy",
    }
    strategy = get_xsmm_strategy("libxsmm-skx")
    assert strategy.isa_info.isa == "avx512"
    assert strategy.nano_kernel is get_skx_nano_kernel("libxsmm-skx")

    kdot_strategy = get_xsmm_strategy("zen5-kdot")
    assert isinstance(kdot_strategy.nano_kernel, KdotWithSkxFallbackNanoKernel)


def test_unknown_xsmm_strategy() -> None:
    try:
        get_xsmm_strategy("unknown")
    except ValueError as error:
        assert str(error) == (
            "unknown XSMM strategy 'unknown'; expected one of: "
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst, "
            "zen5-kdot, zen5-kdot-greedy"
        )
    else:
        raise AssertionError("expected an unknown strategy to be rejected")


def test_avx512_kdot_support_and_register_usage() -> None:
    isa_info = AVX512Info()
    kernel = Avx512KdotNanoKernel()
    descriptor = GemmDescriptor(
        m=1,
        n=16,
        k=64,
        lda=1,
        ldb=64,
        ldc=1,
        datatype=builtin.f64,
        aligned_a=False,
        aligned_c=False,
    )

    tile = TileSizes(1, 16, 64)
    assert kernel.supports_tile(descriptor, tile, isa_info)
    assert kernel.register_usage(descriptor, tile, isa_info) == RegisterCount(
        general=5, vector=24, mask=1
    )
    assert not kernel.supports_tile(descriptor, TileSizes(1, 25, 64), isa_info)

    non_kdot_descriptor = _descriptor(m=2, n=16, k=64, datatype=builtin.f64)
    assert not kernel.supports(non_kdot_descriptor, isa_info)


def test_zen5_strategy_keeps_full_k_only_for_kdot_tiles() -> None:
    strategy = get_xsmm_strategy("zen5-kdot")
    kdot_descriptor = GemmDescriptor(
        m=1,
        n=16,
        k=64,
        lda=1,
        ldb=64,
        ldc=1,
        datatype=builtin.f64,
        aligned_a=False,
        aligned_c=False,
    )
    assert (
        strategy.k_tiling.tile_size(
            kdot_descriptor,
            TileSizes(1, 16, 64),
            strategy.isa_info,
            strategy.nano_kernel,
        )
        is None
    )

    fallback_descriptor = _descriptor(m=8, n=16, k=64, datatype=builtin.f64)
    assert (
        strategy.k_tiling.tile_size(
            fallback_descriptor,
            TileSizes(8, 16, 64),
            strategy.isa_info,
            strategy.nano_kernel,
        )
        == 4
    )


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


def test_zen5_kdot_greedy_n_tiling_strategy() -> None:
    named_strategy = get_xsmm_strategy("zen5-kdot-greedy")
    descriptor = _descriptor(m=1, n=25, k=64, datatype=builtin.f64)

    strategy = compute_tiling_strategy(
        descriptor,
        named_strategy.isa_info,
        named_strategy.nano_kernel,
        named_strategy.n_tiling,
    )

    assert strategy == TilingStrategy(
        m_tile_size=1,
        n_ranges=(
            BlockingRange(extent=24, tile_size=8),
            BlockingRange(extent=1, tile_size=1),
        ),
    )


def test_zen5_kdot_greedy_preserves_equalized_fallback() -> None:
    baseline = get_xsmm_strategy("zen5-kdot")
    greedy = get_xsmm_strategy("zen5-kdot-greedy")
    descriptor = _descriptor(m=2, n=25, k=64, datatype=builtin.f64)

    baseline_tiling = compute_tiling_strategy(
        descriptor,
        baseline.isa_info,
        baseline.nano_kernel,
        baseline.n_tiling,
    )
    greedy_tiling = compute_tiling_strategy(
        descriptor,
        greedy.isa_info,
        greedy.nano_kernel,
        greedy.n_tiling,
    )

    assert greedy_tiling == baseline_tiling
