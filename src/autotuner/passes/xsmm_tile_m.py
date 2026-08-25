from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.x86 import registers
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulMOp
from autotuner.libxsmm_gemm.generator_common import LIBXSMM_X86_AVX512_MASK_REG
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.schedules import tile_m


@dataclass
class TileMatmulMPattern(RewritePattern):
    """Tile M without changing the pointer results of matmul_m.

    Every generated body advances A and C by its M block and leaves B fixed.
    Composing the loop-carried results therefore advances A and C by the full
    original M extent, regardless of the chosen blocking or remainder mask.
    """

    arch: Arch
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulMOp, rewriter: PatternRewriter, /) -> None:
        if isinstance(op.datatype, builtin.Float32Type):
            max_blocking = 64
        elif isinstance(op.datatype, builtin.Float64Type):
            max_blocking = 32
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_m datatype {op.datatype}"
            )

        m = op.m_blocking.value.data
        m_blocking = min(m, max_blocking)

        mloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R10
        )
        mask_tmp_reg = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R15
        )

        tile_m(
            rewriter,
            op,
            m_blocking=m_blocking,
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=LIBXSMM_X86_AVX512_MASK_REG,
        )


@dataclass(frozen=True)
class XsmmTileMPass(ModulePass):
    """Apply the current AVX-512 F32/F64 M blocking policy to matmul_m.

    The pass materializes M loops, including single-iteration loops, and masks
    partial final vectors while preserving the original pointer results.
    """

    name = "xsmm-tile-m"

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
                "xsmm-tile-m currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            TileMatmulMPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
