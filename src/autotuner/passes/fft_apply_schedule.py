"""Expand ``fft.dft`` operations into x86 instructions by recursive expansion."""

import math
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.fft import DftOp
from autotuner.fft.emit import Labels
from autotuner.fft.plan import FftStrategy
from autotuner.fft.rewrites import (
    DftRegKernel,
    DftToRegisters,
    SplitDft,
    SplitDftReg,
    TileDft,
    Transpose,
    Twiddle,
)


@dataclass(frozen=True)
class FftApplySchedulePass(ModulePass):
    """Split DFTs with Cooley-Tukey and expand them down to x86 instructions.

    Every step is one rewrite of one operation (see ``autotuner.fft.rewrites``),
    applied until no ``fft`` operation is left; the result is ready for
    register allocation.

    ``schedule`` gives the size of every pass over memory in execution order
    (see ``FftStrategy.schedule``), and applies to every ``fft.dft`` in the
    module; without it, a cost model picks the passes.
    """

    name = "fft-apply-schedule"

    max_leaf: int = 16
    max_vector_bits: int = 512
    schedule: tuple[int, ...] = ()
    l1_bytes: int = 32768
    """The L1 data cache size, which decides how twiddle factors are stored
    (see ``compact_twiddles``)."""

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        strategy = FftStrategy(
            max_leaf=self.max_leaf,
            vector_bits=tuple(
                bits for bits in (512, 256, 128) if bits <= self.max_vector_bits
            ),
            schedule=self.schedule,
            l1_bytes=self.l1_bytes,
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
        labels = Labels(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    SplitDft(strategy),
                    TileDft(strategy),
                    DftToRegisters(strategy, labels),
                    SplitDftReg(),
                    DftRegKernel(labels),
                    Twiddle(labels),
                    Transpose(labels),
                ]
            ),
            apply_recursively=True,
        ).rewrite_module(op)
