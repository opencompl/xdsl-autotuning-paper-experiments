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

from autotuner.dialects.xsmm import MatmulNOp
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.passes.xsmm_n_blocking import get_max_n_blocking_for_matmul_n
from autotuner.schedules import tile_n


@dataclass
class TileMatmulNPattern(RewritePattern):
    """Tile N exactly while preserving the pointer results of matmul_n.

    The tile size is the largest divisor of the N range that does not exceed
    the current register-pressure limit. Every generated body leaves A fixed
    and advances B and C by its N tile, so composing the loop-carried results
    advances B and C by the complete original range without a remainder body.
    """

    arch: Arch
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        n = op.n_blocking.value.data
        max_n_blocking = get_max_n_blocking_for_matmul_n(op, self.arch)
        n_tile = next(
            candidate
            for candidate in range(min(n, max_n_blocking), 0, -1)
            if n % candidate == 0
        )
        nloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R11
        )
        tile_n(rewriter, op, n_tile, nloop_register)


@dataclass(frozen=True)
class XsmmTileNPass(ModulePass):
    """Tile matmul_n into an exact N loop using the current AVX-512 policy."""

    name = "xsmm-tile-n"

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
                "xsmm-tile-n currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            TileMatmulNPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
