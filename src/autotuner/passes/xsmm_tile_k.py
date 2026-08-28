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

from autotuner.dialects.xsmm import MatmulRegOp
from autotuner.schedules import tile_k
from autotuner.skx_nano_kernel_utils import descriptor_from_op, tile_sizes_from_op
from autotuner.strategy import XsmmStrategy, get_xsmm_strategy


@dataclass
class TileMatmulRegKPattern(RewritePattern):
    """Tile a matmul_reg along K without changing its pointer results.

    Each tiled body uses the K iterator to advance through its reduction slice.
    After the loop, the combined K traversal is normalized once to the original
    operation's iterator.
    """

    strategy: XsmmStrategy
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulRegOp, rewriter: PatternRewriter, /) -> None:
        k_tile = self.strategy.k_tiling.tile_size(
            descriptor_from_op(op),
            tile_sizes_from_op(op),
            self.strategy.isa_info,
            self.strategy.nano_kernel,
        )
        if k_tile is None:
            return

        kloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R12
        )
        tile_k(rewriter, op, k_tile, kloop_register)


@dataclass(frozen=True)
class XsmmTileKPass(ModulePass):
    """Tile large xsmm.matmul_reg operations along K.

    This applies the current LIBXSMM K policy without changing the operation's
    declared iterator contract.
    """

    name = "xsmm-tile-k"

    strategy: str = "libxsmm-skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            strategy = get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            TileMatmulRegKPattern(strategy, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
