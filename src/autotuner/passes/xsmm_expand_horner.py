"""Expand Horner recurrences locally into fused x86 arithmetic."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from autotuner.dialects.xsmm import HornerOp
from autotuner.schedules import expand_horner


class ExpandHornerPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HornerOp, rewriter: PatternRewriter, /) -> None:
        expand_horner(rewriter, op)


@dataclass(frozen=True)
class XsmmExpandHornerPass(ModulePass):
    """Use ``fully=false`` to expose just one step of each recurrence."""

    name = "xsmm-expand-horner"
    fully: bool = True

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandHornerPattern(), apply_recursively=self.fully
        ).rewrite_module(op)
