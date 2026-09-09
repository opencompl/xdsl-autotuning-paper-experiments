from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512RegisterType,
    SSERegisterType,
)

from autotuner.nano_kernel import (
    GemmDescriptor,
    RegisterCount,
    SupportedTile,
    TileSizes,
)
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel import (
    SKX_NANO_KERNELS,
    AVX512Info,
    SkxNanoKernel,
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
        "libxsmm-skx-fsdbcst",
        "libxsmm-skx-nofsdbcst",
        "llvm-skx-narrow-fsdbcst",
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
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst, "
            "llvm-skx-narrow-fsdbcst"
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
            "libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst, "
            "llvm-skx-narrow-fsdbcst"
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

    supported = kernel.supported_tile_sizes(builtin.f64, isa_info)
    assert len(supported) == 8 * 28
    assert SupportedTile(1, 1) in supported
    assert SupportedTile(8, 28) in supported
    assert SupportedTile(9, 1) not in supported
    assert len(kernel.supported_tile_sizes(builtin.f32, isa_info)) == 16 * 28


def test_skx_nofsdbcst_supported_tiles() -> None:
    isa_info = AVX512Info()
    kernel = SkxNofsdbcstNanoKernel()
    descriptor = _descriptor(m=40, n=6, k=2, datatype=builtin.f64)

    assert kernel.supports_tile(descriptor, TileSizes(32, 6, 2), isa_info)
    assert kernel.supports_tile(descriptor, TileSizes(8, 1, 2), isa_info)
    assert kernel.register_usage(
        descriptor, TileSizes(40, 5, 2), isa_info
    ) == RegisterCount(general=5, vector=31, mask=0)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 5, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(40, 6, 2), isa_info)

    supported = kernel.supported_tile_sizes(builtin.f64, isa_info)
    # num C-registers: (M // 8) * N
    # num A-registers: (M // 8)
    # num B-registers: 1
    # 32 total registers
    #
    # M = 8:   N + 1 + 1 <= 32 ==> N = 30
    # M = 16: 2N + 2 + 1 <= 32 ==> N = 14
    # M = 24: 3N + 3 + 1 <= 32 ==> N =  9
    # M = 32: 4N + 4 + 1 <= 32 ==> N =  6
    assert len(supported) == 8 * (30 + 14 + 9 + 6)
    assert SupportedTile(9, 14) in supported
    assert SupportedTile(17, 9) in supported
    assert SupportedTile(25, 6) in supported
    assert SupportedTile(16, 15) not in supported
    assert SupportedTile(33, 1) not in supported

    # num C-registers: (M // 16) * N
    # num A-registers: (M // 16)
    # num B-registers: 1
    # 32 total registers
    #
    # M = 16:  N + 1 + 1 <= 32 ==> N = 30
    # M = 32: 2N + 2 + 1 <= 32 ==> N = 14
    # M = 48: 3N + 3 + 1 <= 32 ==> N =  9
    # M = 64: 4N + 4 + 1 <= 32 ==> N =  6
    assert len(kernel.supported_tile_sizes(builtin.f32, isa_info)) == 16 * (
        30 + 14 + 9 + 6
    )


def test_vector_register_type_widths() -> None:
    """The lane arithmetic everywhere here rests on these three widths."""
    assert SSERegisterType.bitwidth() == 128
    assert AVX2RegisterType.bitwidth() == 256
    assert AVX512RegisterType.bitwidth() == 512


def test_skx_narrow_fsdbcst_picks_the_narrowest_vector_type() -> None:
    isa_info = AVX512Info()
    kernel = SkxNarrowFsdbcstNanoKernel()

    # An xmm is the narrowest there is, so a one-lane tile still lands in one
    # rather than in a scalar register.
    assert [kernel.vector_type(m, builtin.f64, isa_info) for m in range(1, 9)] == [
        SSERegisterType,
        SSERegisterType,
        AVX2RegisterType,
        AVX2RegisterType,
        AVX512RegisterType,
        AVX512RegisterType,
        AVX512RegisterType,
        AVX512RegisterType,
    ]
    assert [kernel.vector_type(m, builtin.f32, isa_info) for m in range(1, 17)] == (
        [SSERegisterType] * 4 + [AVX2RegisterType] * 4 + [AVX512RegisterType] * 8
    )

    # The nano-kernel the LIBXSMM heuristic reaches for at these M is the
    # full-width one, which masks every lane the tile does not fill.
    assert (
        SkxFsdbcstNanoKernel().vector_type(2, builtin.f64, isa_info)
        is isa_info.vector_type
        is AVX512RegisterType
    )


def test_skx_narrow_fsdbcst_supported_tiles() -> None:
    isa_info = AVX512Info()
    kernel = SkxNarrowFsdbcstNanoKernel()
    descriptor = _descriptor(m=2, n=12, k=64, datatype=builtin.f64)

    # One accumulator per N column, plus the two rotating A columns; no
    # duplicated accumulator sets, and a mask only when M misses its register.
    assert kernel.register_usage(
        descriptor, TileSizes(2, 12, 64), isa_info
    ) == RegisterCount(general=5, vector=14, mask=0)
    assert kernel.register_usage(
        descriptor, TileSizes(1, 12, 64), isa_info
    ) == RegisterCount(general=5, vector=14, mask=1)
    assert kernel.supports_tile(descriptor, TileSizes(2, 30, 2), isa_info)
    assert not kernel.supports_tile(descriptor, TileSizes(2, 31, 2), isa_info)
    # More than one M vector is the wide nano-kernel's business.
    assert not kernel.supports_tile(descriptor, TileSizes(9, 1, 2), isa_info)

    supported = kernel.supported_tile_sizes(builtin.f64, isa_info)
    assert len(supported) == 8 * 30
    assert SupportedTile(1, 1) in supported
    assert SupportedTile(8, 30) in supported
    assert SupportedTile(8, 31) not in supported
    assert SupportedTile(9, 1) not in supported
    assert len(kernel.supported_tile_sizes(builtin.f32, isa_info)) == 16 * 30


def test_skx_narrow_fsdbcst_tiles_like_the_wide_kernel() -> None:
    """From the vector length up, the narrow kernel *is* the wide one.

    Both cover M one vector at a time, so both settle on an M tile of the
    vector length; there the narrow type is the full vector and the two emit
    the same instructions. N of 12 keeps ``fsdbcst`` on a single accumulator
    set, which is all the narrow kernel ever uses.
    """
    isa_info = AVX512Info()
    kernel = SkxNarrowFsdbcstNanoKernel()

    for m in (8, 16, 64):
        descriptor = _descriptor(m=m, n=12, k=64, datatype=builtin.f64)
        strategy = compute_tiling_strategy(descriptor, isa_info, kernel)
        assert strategy.m_tile_size == isa_info.vector_length(builtin.f64)
        assert strategy == compute_tiling_strategy(
            descriptor, isa_info, SkxFsdbcstNanoKernel()
        )


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

    supported = kernel.supported_tile_sizes(builtin.f64, isa_info)
    assert supported == (
        SkxFsdbcstNanoKernel().supported_tile_sizes(builtin.f64, isa_info)
        | SkxNofsdbcstNanoKernel().supported_tile_sizes(builtin.f64, isa_info)
    )


def test_single_n_range_tiling_strategy() -> None:
    strategy = compute_tiling_strategy(
        _descriptor(m=16, n=28, k=16, datatype=builtin.f64),
        AVX512Info(),
        SkxNanoKernel(),
    )
    assert strategy == TilingStrategy(
        m_tile_size=16,
        n_ranges=(BlockingRange(extent=28, tile_size=14),),
        k_tile_size=16,
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
        k_tile_size=4,
    )
