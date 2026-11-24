"""Expand ``fft.dft`` operations into x86 instructions."""

import math
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.fft import DftOp
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

    ``schedule`` gives the size of every pass over memory in execution order
    (see ``FftStrategy.schedule``), and applies to every ``fft.dft`` in the
    module; without it, a cost model picks the passes.
    """

    name = "fft-apply-schedule"

    max_leaf: int = 8
    max_vector_bits: int = 512
    schedule: tuple[int, ...] = ()

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        strategy = FftStrategy(
            max_leaf=self.max_leaf,
            vector_bits=tuple(
                bits for bits in (512, 256, 128) if bits <= self.max_vector_bits
            ),
            schedule=self.schedule,
        )
        if self.schedule:
            for dft in op.walk():
                if isinstance(dft, DftOp) and dft.n.value.data != math.prod(
                    self.schedule
                ):
                    raise PassFailedException(
                        f"the schedule {self.schedule} does not multiply to "
                        f"{dft.n.value.data} points"
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
