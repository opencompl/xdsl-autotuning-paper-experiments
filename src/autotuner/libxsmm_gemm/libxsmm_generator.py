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

    Loop induction values are deliberately not part of this payload. Structured loops
    expose them as induction variables, while raw block/branch loops thread them as
    separate block arguments.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]

    @property
    def vals(self) -> tuple[SSAValue, ...]:
        return (self.a, self.b, self.c, self.rbp, self.rsp)


@dataclass
class MLoopVals:
    """SSA values that are live across the M loop.

    ``mask_k1`` is present only when C masking is active for the current M block.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
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
            *masks,
        )


@dataclass
class KLoopVals:
    """SSA values that are live across the K loop.

    Carries the accumulator vectors produced by ``load_C``. Loop induction values are
    handled separately by the control-flow-specific loop helpers.
    """

    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    mask_k1: SSAValue[AVX512MaskRegisterType] | None
    acc_vectors: tuple[SSAValue[VectorRegT], ...]

    @property
    def vals(self) -> tuple[SSAValue, ...]:
        masks = () if self.mask_k1 is None else (self.mask_k1,)
        return (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            *masks,
            *self.acc_vectors,
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
