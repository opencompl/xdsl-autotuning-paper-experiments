from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.nano_kernel import NanoKernel, TargetInfo
from autotuner.strategy import get_xsmm_strategy


@dataclass
class ConvertMatmulKPattern(RewritePattern):
    target: TargetInfo
    nano_kernel: NanoKernel
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulKOp, rewriter: PatternRewriter, /) -> None:
        self.nano_kernel.rewrite(
            rewriter,
            op,
            self.target,
            disable_regalloc=self.disable_regalloc,
        )


@dataclass(frozen=True)
class ConvertXsmmToX86Pass(ModulePass):
    """Lower XSMM scheduling operations to x86 instructions.

    ``strategy`` selects a shared scheduling and nano-kernel policy. The default
    preserves the current LIBXSMM-compatible SKX heuristic.
    """

    name = "convert-xsmm-to-x86"

    strategy: str = "libxsmm-skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            strategy = get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ConvertMatmulKPattern(
                strategy.isa_info,
                strategy.nano_kernel,
                self.disable_regalloc,
            ),
            apply_recursively=False,
        ).rewrite_module(op)
