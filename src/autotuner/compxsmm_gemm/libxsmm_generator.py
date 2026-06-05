from dataclasses import dataclass
from typing import cast

from xdsl.builder import Builder
from xdsl.dialects import x86, x86_func
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    SSERegisterType,
)
from xdsl.ir import Attribute, OperationInvT, SSAValue

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch

VectorRegT = SSERegisterType | AVX2RegisterType | AVX512RegisterType


@dataclass
class NLoopVals:
    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]

    @property
    def vals(self):
        return (self.a, self.b, self.c, self.rbp, self.rsp)


@dataclass
class MLoopVals:
    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    mask_k1: SSAValue[AVX512MaskRegisterType] | None

    @property
    def vals(self):
        masks = () if self.mask_k1 is None else (self.mask_k1,)
        return (self.a, self.b, self.c, self.rbp, self.rsp, *masks)


@dataclass
class KLoopVals:
    a: SSAValue[GeneralRegisterType]
    b: SSAValue[GeneralRegisterType]
    c: SSAValue[GeneralRegisterType]
    rbp: SSAValue[GeneralRegisterType]
    rsp: SSAValue[GeneralRegisterType]
    mask_k1: SSAValue[AVX512MaskRegisterType] | None
    acc_vectors: tuple[SSAValue[VectorRegT], ...]

    @property
    def vals(self):
        masks = () if self.mask_k1 is None else (self.mask_k1,)
        return (self.a, self.b, self.c, self.rbp, self.rsp, *masks, *self.acc_vectors)


@dataclass
class GeneratedCode:
    func_op: x86_func.FuncOp
    builder: Builder
    arch: Arch

    current_val_by_reg: dict[Attribute, SSAValue]

    def get_val(self, reg: x86.ops.R1InvT) -> SSAValue[x86.ops.R1InvT]:
        return cast(SSAValue[x86.ops.R1InvT], self.current_val_by_reg[reg])

    @property
    def current_block(self):
        return self.builder.insertion_point.block

    def insert(self, op: OperationInvT) -> OperationInvT:
        self.builder.insert(op)
        for res in op.results:
            self.current_val_by_reg[res.type] = res
        return op
