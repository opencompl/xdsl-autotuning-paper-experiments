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
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE
from autotuner.nano_kernel import NanoKernel, TargetInfo
from autotuner.skx_nano_kernel import SkxNanoKernel, SkxTargetInfo


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
    """Lower XSMM scheduling operations to x86 instructions."""

    name = "convert-xsmm-to-x86"

    arch: str = "skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        target = SkxTargetInfo()
        if arch != target.arch:
            raise PassFailedException("convert-xsmm-to-x86 currently supports SKX only")

        PatternRewriteWalker(
            ConvertMatmulKPattern(target, SkxNanoKernel(), self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
