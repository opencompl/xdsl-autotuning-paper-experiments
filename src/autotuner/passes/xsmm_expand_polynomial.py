"""Select a fixed polynomial evaluation tree, expanding only the matched op."""

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
from xdsl.utils.exceptions import DiagnosticException

from autotuner.dialects.xsmm import PolynomialOp
from autotuner.schedules import polynomial_to_horner


class ExpandPolynomialPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PolynomialOp, rewriter: PatternRewriter, /) -> None:
        match len(op.coefficients):
            case 4:
                polynomial_to_horner(rewriter, op, seed="argument")
            case 6:
                polynomial_to_horner(rewriter, op, seed="leading-coefficient")
            case _:
                raise DiagnosticException(
                    "libxsmm polynomial schedule requires degree 3 or 5"
                )


@dataclass(frozen=True)
class XsmmExpandPolynomialPass(ModulePass):
    name = "xsmm-expand-polynomial"
    schedule: str = "libxsmm"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        if self.schedule != "libxsmm":
            raise DiagnosticException(f"unknown polynomial schedule: {self.schedule}")
        PatternRewriteWalker(
            ExpandPolynomialPattern(), apply_recursively=False
        ).rewrite_module(op)
