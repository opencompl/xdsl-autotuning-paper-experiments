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

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.schedules import tile_k

K_BLOCKING = 4
K_THRESHOLD = 23


@dataclass
class TileMatmulKPattern(RewritePattern):
    """Tile a matmul_k without changing its pointer results.

    Each tiled body advances A and B through its portion of K. The loop-carried
    results therefore already have the same pointer values as the original op;
    the transformation must not reset either pointer after the loop.
    """

    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulKOp, rewriter: PatternRewriter, /) -> None:
        k = op.k_blocking.value.data
        if k <= K_THRESHOLD:
            return

        kloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R12
        )
        tile_k(rewriter, op, K_BLOCKING, kloop_register)


@dataclass(frozen=True)
class XsmmTileKPass(ModulePass):
    """Tile large xsmm.matmul_k operations while preserving pointer results.

    This applies the current LIBXSMM K policy, but the chosen representation does
    not change the operation semantics: in particular, ``b_out`` advances by the
    full K extent for both tiled and untiled operations.
    """

    name = "xsmm-tile-k"

    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TileMatmulKPattern(self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
