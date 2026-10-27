"""Expand polynomial constant setup into inline jump/data/load packets."""

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

from autotuner.dialects.xsmm import PolynomialConstantsOp
from autotuner.schedules import expand_polynomial_constants


class ExpandPolynomialConstantsPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: PolynomialConstantsOp, rewriter: PatternRewriter, /
    ) -> None:
        expand_polynomial_constants(rewriter, op)


@dataclass(frozen=True)
class XsmmExpandPolynomialConstantsPass(ModulePass):
    name = "xsmm-expand-polynomial-constants"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandPolynomialConstantsPattern(), apply_recursively=False
        ).rewrite_module(op)
