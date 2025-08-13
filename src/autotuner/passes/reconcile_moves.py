"""
Eliminates moves that are the only user of their operand and that only have other uses
before the move.
Note that this pass is currently broken as we use moves for three purposes:

1. change the register to respect the registers designated by ABI for function calls etc.
2. change the registers in for loop body boundaries
3. move into a new register when an operation mutates one of the input registers

3. is currently making us run out of registers but we need 1 & 2 to work.
This pass will break uses 1 and 2 for some inputs, but hopefully not for our libxsmm lowering.

For 1 and 2 I think we need a target-independent n-way move operation as described in Hack's register allocation paper,
as well as a pass that will lower it to moves in the target ISA (always possible at least if you can do register swaps via `a ^= b; b ^= a; a ^= b` ).
"""

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ReconcileMovesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: x86.ops.DS_MovOp, rewriter: PatternRewriter, /):
        if (
            op.destination.has_one_use()
            and op.source.has_one_use()
            and op.source.type == op.destination.type
        ):
            rewriter.replace_matched_op((), (op.source,))


class ReconcileMovesPass(ModulePass):
    """
    Removes unallocated moves that are unique uses and have no user after the move.
    """

    name = "reconcile-moves"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ReconcileMovesPattern(), apply_recursively=False
        ).rewrite_module(op)
