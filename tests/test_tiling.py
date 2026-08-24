from xdsl.dialects import builtin

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.nano_kernel import GemmDescriptor, RegisterCount, TileSizes
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel import SkxNanoKernel, SkxTargetInfo
from autotuner.skx_nofsdbcst_nano_kernel import SkxNofsdbcstNanoKernel
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


def test_skx_target_arch() -> None:
    assert SkxTargetInfo().arch == Arch.LIBXSMM_X86_AVX512_SKX


def test_skx_register_usage() -> None:
    target = SkxTargetInfo()
    kernel = SkxNanoKernel()

    f64 = _descriptor(m=8, n=2, k=2, datatype=builtin.f64)
    assert kernel.register_usage(f64, TileSizes(8, 2, 2), target) == RegisterCount(
        general=5, vector=6, mask=0
    )

    f32 = _descriptor(m=17, n=1, k=2, datatype=builtin.f32)
    assert kernel.register_usage(f32, TileSizes(17, 1, 2), target) == RegisterCount(
        general=5, vector=5, mask=1
    )


def test_skx_fsdbcst_supported_tiles() -> None:
    target = SkxTargetInfo()
    kernel = SkxFsdbcstNanoKernel()
    descriptor = _descriptor(m=8, n=31, k=2, datatype=builtin.f64)

    assert kernel.supports_tile(descriptor, TileSizes(8, 28, 2), target)
    assert kernel.register_usage(
        descriptor, TileSizes(8, 30, 2), target
    ) == RegisterCount(general=5, vector=32, mask=0)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 30, 2), target)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 31, 2), target)
    assert not kernel.supports_tile(descriptor, TileSizes(16, 1, 2), target)


def test_skx_nofsdbcst_supported_tiles() -> None:
    target = SkxTargetInfo()
    kernel = SkxNofsdbcstNanoKernel()
    descriptor = _descriptor(m=40, n=6, k=2, datatype=builtin.f64)

    assert kernel.supports_tile(descriptor, TileSizes(32, 6, 2), target)
    assert kernel.register_usage(
        descriptor, TileSizes(40, 5, 2), target
    ) == RegisterCount(general=5, vector=31, mask=0)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 5, 2), target)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 6, 2), target)
    assert not kernel.supports_tile(descriptor, TileSizes(8, 1, 2), target)


def test_skx_composite_retains_libxsmm_tiling_heuristics() -> None:
    target = SkxTargetInfo()
    kernel = SkxNanoKernel()
    f64 = _descriptor(m=40, n=29, k=2, datatype=builtin.f64)

    assert not SkxFsdbcstNanoKernel().supports_tile(f64, TileSizes(8, 29, 2), target)
    assert not kernel.supports_tile(f64, TileSizes(8, 29, 2), target)
    assert not SkxNofsdbcstNanoKernel().supports_tile(f64, TileSizes(40, 5, 2), target)
    assert not kernel.supports_tile(f64, TileSizes(40, 5, 2), target)


def test_single_n_range_tiling_strategy() -> None:
    strategy = compute_tiling_strategy(
        _descriptor(m=16, n=28, k=16, datatype=builtin.f64),
        SkxTargetInfo(),
        SkxNanoKernel(),
    )
    assert strategy == TilingStrategy(
        m_tile_size=16,
        n_ranges=(BlockingRange(extent=28, tile_size=14),),
    )


def test_two_n_ranges_and_m_remainder_tiling_strategy() -> None:
    strategy = compute_tiling_strategy(
        _descriptor(m=70, n=38, k=128, datatype=builtin.f32),
        SkxTargetInfo(),
        SkxNanoKernel(),
    )
    assert strategy == TilingStrategy(
        m_tile_size=64,
        n_ranges=(
            BlockingRange(extent=18, tile_size=6),
            BlockingRange(extent=20, tile_size=5),
        ),
    )
