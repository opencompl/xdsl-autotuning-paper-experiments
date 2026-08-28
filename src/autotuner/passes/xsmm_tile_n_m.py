from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.dialects.x86 import registers
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp
from autotuner.nano_kernel import GemmDescriptor, NanoKernel, ISAInfo
from autotuner.schedules import matmul_n_to_m, split_n, tile_m, tile_n
from autotuner.strategy import get_xsmm_strategy
from autotuner.tiling import TilingStrategy, compute_tiling_strategy


def tile_n_m(
    rewriter: PatternRewriter,
    op: MatmulOp,
    strategy: TilingStrategy,
    *,
    nloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_tmp_reg: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_reg: x86.registers.AVX512MaskRegisterType = x86.registers.UNALLOCATED_AVX512_MASK,
) -> list[MatmulOp]:
    if len(strategy.n_ranges) == 2:
        first_range, second_range = strategy.n_ranges
        first, second = split_n(rewriter, op, first_range.extent)
        n_ranges = (
            (first, first_range.tile_size),
            (second, second_range.tile_size),
        )
    else:
        assert len(strategy.n_ranges) == 1
        n_ranges = ((op, strategy.n_ranges[0].tile_size),)

    res: list[MatmulOp] = []

    for n_range, n_tile in n_ranges:
        tiled_matmul_n = tile_n(
            rewriter, n_range, n_tile, nloop_register=nloop_register
        )
        matmul_m = matmul_n_to_m(rewriter, tiled_matmul_n)

        tiled, remainder = tile_m(
            rewriter,
            matmul_m,
            m_blocking=strategy.m_tile_size,
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=mask_reg,
        )

        res.append(tiled)
        if remainder is not None:
            res.append(remainder)

    return res


@dataclass
class TileMatmulNMPattern(RewritePattern):
    """Choose and materialize the current LIBXSMM N and M tiling."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.N:
            return
        descriptor = GemmDescriptor(
            m=op.m.value.data,
            n=op.n.value.data,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            ldc=op.ldc.value.data,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
            aligned_c=bool(op.aligned_c),
        )
        try:
            strategy = compute_tiling_strategy(
                descriptor, self.isa_info, self.nano_kernel
            )
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        if self.disable_regalloc:
            nloop_register = registers.UNALLOCATED_REG64
            mloop_register = registers.UNALLOCATED_REG64
            mask_tmp_reg = registers.UNALLOCATED_REG64
        else:
            nloop_register = registers.R11
            mloop_register = registers.R10
            mask_tmp_reg = registers.R15

        tile_n_m(
            rewriter,
            op,
            strategy,
            nloop_register=nloop_register,
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=registers.K1,
        )


@dataclass(frozen=True)
class XsmmTileNMPass(ModulePass):
    """Split and tile N, lower to M, and tile M using the current policy."""

    name = "xsmm-tile-n-m"

    strategy: str = "libxsmm-skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            strategy = get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            TileMatmulNMPattern(
                strategy.isa_info,
                strategy.nano_kernel,
                self.disable_regalloc,
            ),
            apply_recursively=False,
        ).rewrite_module(op)
