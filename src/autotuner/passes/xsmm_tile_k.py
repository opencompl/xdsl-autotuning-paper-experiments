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

from autotuner.dialects.xsmm import MatmulRegOp
from autotuner.schedules import tile_k

K_BLOCKING = 4
K_THRESHOLD = 23


@dataclass
class TileMatmulRegKPattern(RewritePattern):
    """Tile a matmul_reg along K without changing its pointer results.

    Each tiled body uses the K iterator to advance through its reduction slice.
    After the loop, the combined K traversal is normalized once to the original
    operation's iterator.
    """

    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulRegOp, rewriter: PatternRewriter, /) -> None:
        k = op.k.value.data
        if k <= K_THRESHOLD:
            return

        kloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R12
        )
        tile_k(rewriter, op, K_BLOCKING, kloop_register)


@dataclass(frozen=True)
class XsmmTileKPass(ModulePass):
    """Tile large xsmm.matmul_reg operations along K.

    This applies the current LIBXSMM K policy without changing the operation's
    declared iterator contract.
    """

    name = "xsmm-tile-k"

    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TileMatmulRegKPattern(self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
