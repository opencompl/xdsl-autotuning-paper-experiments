"""Expand ``fft.dft`` operations into x86 instructions."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

from autotuner.fft.constants import ConstantPool
from autotuner.fft.expand import (
    DftRegBaseCasePattern,
    DftToRegistersPattern,
    SplitDftPattern,
    SplitDftRegPattern,
    TransposePattern,
    TwiddlePattern,
)
from autotuner.fft.plan import FftStrategy


@dataclass(frozen=True)
class FftApplySchedulePass(ModulePass):
    """Split DFTs with Cooley-Tukey and lower them to register nano-kernels.

    The result still holds ``x86ext`` negations and non-destructive fused
    multiply-adds; ``fft-legalize`` lowers them.
    """

    name = "fft-apply-schedule"

    max_leaf: int = 8
    max_vector_bits: int = 512

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        strategy = FftStrategy(
            max_leaf=self.max_leaf,
            vector_bits=tuple(
                bits for bits in (512, 256, 128) if bits <= self.max_vector_bits
            ),
        )
        pools: dict[x86_func.FuncOp, ConstantPool] = {}
        # Memory first, so the register kernels see their final placement.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [SplitDftPattern(strategy), DftToRegistersPattern(strategy, pools)]
            ),
            apply_recursively=True,
        ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    SplitDftRegPattern(),
                    DftRegBaseCasePattern(pools),
                    TwiddlePattern(pools),
                    TransposePattern(pools),
                ]
            ),
            apply_recursively=True,
        ).rewrite_module(op)
        for pool in pools.values():
            pool.finalize()
