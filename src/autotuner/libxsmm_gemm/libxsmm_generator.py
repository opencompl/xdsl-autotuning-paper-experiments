from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects import x86_func
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    SSERegisterType,
)
from xdsl.ir import OperationInvT, SSAValue

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch

VectorRegT = SSERegisterType | AVX2RegisterType | AVX512RegisterType


@dataclass
class NLoopVals:
    """SSA values that are live across the N loop.

    Unlike compxsmm (which builds loops with ``x86_scf.ForOp`` and keeps the loop
    counter as the induction variable), ``libxsmm_gemm`` builds loops from raw
    block/branch ops, so the loop counter is an ordinary block argument that must be
    threaded explicitly. ``n_counter`` is the value held in ``gp_reg_nloop``.

    The ``vals`` property defines the block-argument / fallthrough / back-edge operand
    order for the N loop.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    n_counter: SSAValue[GeneralRegisterType]

    @property
    def vals(self) -> tuple[SSAValue, ...]:
        return (self.a, self.b, self.c, self.rbp, self.rsp, self.n_counter)


@dataclass
class MLoopVals:
    """SSA values that are live across the M loop.

    Carries the outer N counter through as well (it is threaded through the inner
    loop's blocks so it remains available at the N footer). ``mask_k1`` is present only
    when C masking is active for the current M block.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    n_counter: SSAValue[GeneralRegisterType]
    m_counter: SSAValue[GeneralRegisterType]
    mask_k1: SSAValue[AVX512MaskRegisterType] | None

    @property
    def vals(self) -> tuple[SSAValue, ...]:
        masks = () if self.mask_k1 is None else (self.mask_k1,)
        return (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            self.n_counter,
            self.m_counter,
            *masks,
        )


@dataclass
class KLoopVals:
    """SSA values that are live across the K loop.

    Carries the outer N and M counters through, plus the accumulator vectors. The K
    counter is inserted last (the K loop opens after ``load_C`` has produced the
    accumulators), so it comes after the accumulators in ``vals``.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    n_counter: SSAValue[GeneralRegisterType]
    m_counter: SSAValue[GeneralRegisterType]
    mask_k1: SSAValue[AVX512MaskRegisterType] | None
    acc_vectors: tuple[SSAValue[VectorRegT], ...]
    # None when the K loop is fully unrolled (no counter register); set by header_kloop
    # when a block K loop is emitted.
    k_counter: SSAValue[GeneralRegisterType] | None = None

    @property
    def vals(self) -> tuple[SSAValue, ...]:
        # Only used to build the block/branch loop's block arguments, which always
        # include the K counter.
        assert self.k_counter is not None
        masks = () if self.mask_k1 is None else (self.mask_k1,)
        return (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            self.n_counter,
            self.m_counter,
            *masks,
            *self.acc_vectors,
            self.k_counter,
        )


@dataclass
class GeneratedCode:
    func_op: x86_func.FuncOp
    builder: Builder
    arch: Arch

    @property
    def current_block(self):
        return self.builder.insertion_point.block

    def insert(self, op: OperationInvT) -> OperationInvT:
        self.builder.insert(op)
        return op
