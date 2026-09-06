"""Test-only entry points for the generic XSMM schedule transformations."""

# This module is a pass implementation despite its pytest-style filename.
__test__ = False

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp, MatmulRegOp
from autotuner.schedules import tile_matmul, tile_matmul_reg


def _tile_size(op: MatmulOp | MatmulRegOp) -> int | None:
    attribute = op.attributes.get("test_tile_size")
    if not isinstance(attribute, builtin.IntegerAttr):
        return None
    tile_size = attribute.value.data
    return tile_size if 0 < tile_size else None


class TileMatmulFromAttributePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        tile_size = _tile_size(op)
        iterator = MatmulIterator(op.iterator.data)
        extent = (
            op.m.value.data
            if iterator == MatmulIterator.M
            else op.n.value.data
            if iterator == MatmulIterator.N
            else 0
        )
        if tile_size is None or tile_size > extent:
            return
        tile_matmul(
            rewriter,
            op,
            tile_size,
            x86.registers.UNALLOCATED_REG64,
        )


class TileMatmulRegFromAttributePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulRegOp, rewriter: PatternRewriter, /) -> None:
        tile_size = _tile_size(op)
        if tile_size is None or tile_size > op.k.value.data:
            return
        tile_matmul_reg(
            rewriter,
            op,
            tile_size,
            x86.registers.UNALLOCATED_REG64,
        )


@dataclass(frozen=True)
class TestXsmmTilingPass(ModulePass):
    """Tile XSMM ops using the size in their ``test_tile_size`` attribute."""

    name = "test-xsmm-tiling"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for pattern in (
            TileMatmulFromAttributePattern(),
            TileMatmulRegFromAttributePattern(),
        ):
            PatternRewriteWalker(
                pattern,
                apply_recursively=False,
            ).rewrite_module(op)
