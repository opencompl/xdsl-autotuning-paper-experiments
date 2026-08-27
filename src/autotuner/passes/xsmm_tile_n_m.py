from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects import x86
from xdsl.dialects.x86 import registers
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulMOp, MatmulNOp
from autotuner.libxsmm_gemm.generator_common import (
    LIBXSMM_X86_AVX512_MASK_REG,
    Blocking,
    libxsmm_compute_equalized_blocking,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.passes.xsmm_n_blocking import get_max_n_blocking_for_matmul_n
from autotuner.schedules import matmul_n_to_m, split_n, tile_m, tile_n


def tile_n_m(
    rewriter: PatternRewriter,
    op: MatmulNOp,
    blocking: Blocking,
    *,
    max_m_blocking: int,
    nloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_tmp_reg: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> list[MatmulMOp]:
    if blocking.range_2:
        first, second = split_n(rewriter, op, blocking.range_1)
        n_ranges = (
            (first, blocking.block_1),
            (second, blocking.block_2),
        )
    else:
        n_ranges = ((op, blocking.block_1),)

    res: list[MatmulMOp] = []

    for n_range, n_tile in n_ranges:
        tiled_matmul_n = tile_n(
            rewriter, n_range, n_tile, nloop_register=nloop_register
        )
        matmul_m = matmul_n_to_m(rewriter, tiled_matmul_n)

        tiled, remainder = tile_m(
            rewriter,
            matmul_m,
            m_blocking=min(matmul_m.m_blocking.value.data, max_m_blocking),
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=LIBXSMM_X86_AVX512_MASK_REG,
        )

        res.append(tiled)
        if remainder is not None:
            res.append(remainder)

    return res


@dataclass
class TileMatmulNMPattern(RewritePattern):
    """Choose and materialize the current LIBXSMM N and M tiling."""

    arch: Arch
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        max_n_blocking = get_max_n_blocking_for_matmul_n(op, self.arch)
        blocking = libxsmm_compute_equalized_blocking(
            op.n_blocking.value.data, max_n_blocking
        )
        assert blocking.ret == 0, "Error computing blocking"

        if self.disable_regalloc:
            nloop_register = registers.UNALLOCATED_REG64
            mloop_register = registers.UNALLOCATED_REG64
            mask_tmp_reg = registers.UNALLOCATED_REG64
        else:
            nloop_register = registers.R11
            mloop_register = registers.R10
            mask_tmp_reg = registers.R15

        if isinstance(op.datatype, builtin.Float32Type):
            max_m_blocking = 64
        elif isinstance(op.datatype, builtin.Float64Type):
            max_m_blocking = 32
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_m datatype {op.datatype}"
            )

        tile_n_m(
            rewriter,
            op,
            blocking,
            max_m_blocking=max_m_blocking,
            nloop_register=nloop_register,
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_reg,
        )


@dataclass(frozen=True)
class XsmmTileNMPass(ModulePass):
    """Split and tile N, lower to M, and tile M using the current policy."""

    name = "xsmm-tile-n-m"

    arch: str = "skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        if not (Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT):
            raise PassFailedException(
                "xsmm-tile-n-m currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            TileMatmulNMPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
