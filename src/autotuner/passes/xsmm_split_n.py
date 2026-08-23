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

from autotuner.dialects.xsmm import MatmulNOp
from autotuner.libxsmm_gemm.generator_common import libxsmm_compute_equalized_blocking
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.passes.xsmm_n_blocking import get_max_n_blocking_for_matmul_n
from autotuner.schedules import split_n


@dataclass
class SplitMatmulNPattern(RewritePattern):
    """Split N into at most two equally blocked ranges.

    The two operations are chained through their pointer results, so together
    they have exactly the pointer semantics of the original matmul_n. Their
    n_start properties partition the original absolute range.
    """

    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        blocking = libxsmm_compute_equalized_blocking(
            op.n_blocking.value.data,
            get_max_n_blocking_for_matmul_n(op, self.arch),
        )
        assert blocking.ret == 0, "Error computing blocking"
        if blocking.range_2:
            split_n(rewriter, op, blocking.range_1)


@dataclass(frozen=True)
class XsmmSplitNPass(ModulePass):
    """Split matmul_n into the equalized N ranges used by LIBXSMM."""

    name = "xsmm-split-n"

    arch: str = "skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        if not (Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT):
            raise PassFailedException(
                "xsmm-split-n currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            SplitMatmulNPattern(arch),
            apply_recursively=False,
        ).rewrite_module(op)
