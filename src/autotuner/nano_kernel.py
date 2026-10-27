from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.instructions import (
    MaskValue,
    PointerValue,
    VectorValue,
    load_vector,
    store_vector,
)

FloatingPointType: TypeAlias = builtin.Float32Type | builtin.Float64Type
# The element types A and B may have; C accumulates in a `FloatingPointType`.
InputType: TypeAlias = FloatingPointType | builtin.BFloat16Type

# One vector of the C tile: its byte offset from C and, for the last M vector
# of a partial tile, the mask of its lanes.
CAccess: TypeAlias = tuple[int, MaskValue | None]


@dataclass(frozen=True, order=True)
class SupportedTile:
    """An M-by-N tile supported by a nano-kernel."""

    m: int
    n: int


@dataclass(frozen=True)
class TileSizes:
    """The M, N, and K sizes computed by one nano-kernel invocation."""

    m: int
    n: int
    k: int


@dataclass(frozen=True)
class GemmDescriptor:
    """Target-independent information describing a GEMM problem.

    ``datatype`` is C's element type and the accumulation's; A and B have
    ``ab_datatype`` when it is set and ``datatype`` otherwise. ``vnni_a`` says
    A is packed in K pairs, as on ``xsmm.matmul``.
    """

    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    datatype: FloatingPointType
    aligned_a: bool
    aligned_c: bool
    ab_datatype: builtin.BFloat16Type | None = None
    vnni_a: bool = False

    @property
    def input_type(self) -> InputType:
        """The element type of A and B."""
        return self.datatype if self.ab_datatype is None else self.ab_datatype


@dataclass(frozen=True)
class RegisterCount:
    """A register capacity or peak register demand, by register class."""

    general: int = 0
    vector: int = 0
    mask: int = 0

    def fits(self, capacity: RegisterCount) -> bool:
        """Return whether this demand fits within ``capacity``."""
        return (
            self.general <= capacity.general
            and self.vector <= capacity.vector
            and self.mask <= capacity.mask
        )


class ISAInfo(ABC):
    """ISA properties needed by nano-kernel selection and tiling."""

    @property
    @abstractmethod
    def isa(self) -> Literal["avx512"]:
        """Instruction-set identifier used by ISA-specific lowering."""

    @property
    @abstractmethod
    def register_capacity(self) -> RegisterCount:
        """Registers available to the generated kernel."""

    @property
    @abstractmethod
    def vector_type(self) -> type[X86VectorRegisterType]:
        """The widest vector register type generated code may use."""


class NanoKernel(ABC):
    """A parametric register-resident kernel and its planning contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable name used to select and benchmark this nano-kernel."""

    @property
    def k_pack_factor(self) -> int:
        """How many K steps one element of packed A spans.

        LIBXSMM multiplies its K unrolling by this, so a kernel reading A in
        K pairs unrolls as many instructions as one reading it flat.
        """
        return 1

    @abstractmethod
    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        """Return the supported M-by-N tile shapes."""

    def vector_type(
        self,
        m: int,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
        ab_datatype: builtin.BFloat16Type | None = None,
    ) -> type[X86VectorRegisterType]:
        """Return the register type one M vector of an ``m`` tile occupies.

        The ISA's widest by default, so a tile shorter than a full vector leaves
        the surplus lanes masked off. A kernel that lowers a short M tile to a
        narrower register type -- masking fewer lanes, or none -- overrides
        this; the schedule loads and stores the C accumulators in the same type.
        """
        return isa_info.vector_type

    def load_accumulators(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulOp,
        c: PointerValue,
        accesses: Sequence[CAccess],
        registers: Sequence[X86VectorRegisterType],
        *,
        disable_regalloc: bool,
    ) -> tuple[VectorValue, ...]:
        """Return the accumulators of ``op``'s C tile, one per access, n-major.

        Loads C, by default. A kernel whose accumulators hold C in another lane
        order overrides this together with `store_accumulators`.
        """
        return tuple(
            load_vector(
                rewriter,
                insert_point,
                op.datatype,
                c,
                offset,
                destination=register,
                aligned=bool(op.aligned_c),
                mask=mask,
            )
            for (offset, mask), register in zip(accesses, registers, strict=True)
        )

    def store_accumulators(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulOp,
        c: PointerValue,
        accesses: Sequence[CAccess],
        accumulators: Sequence[VectorValue],
        *,
        disable_regalloc: bool,
    ) -> None:
        """Write the accumulators `load_accumulators` set up back to C."""
        for (offset, mask), accumulator in zip(accesses, accumulators, strict=True):
            store_vector(
                rewriter,
                insert_point,
                op.datatype,
                c,
                offset,
                accumulator,
                aligned=bool(op.aligned_c),
                mask=mask,
            )

    @abstractmethod
    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        """Return whether this kernel supports the GEMM configuration."""

    @abstractmethod
    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        """Return whether this kernel can expand the proposed tile."""

    @abstractmethod
    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        """Return the peak register demand of the proposed tile."""

    @abstractmethod
    def attach_mask(
        self,
        rewriter: PatternRewriter,
        op: MatmulOp,
        *,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        """Attach a mask to ``op`` if the kernel needs one, or do nothing."""

    @abstractmethod
    def rewrite(
        self,
        rewriter: PatternRewriter,
        op: MatmulRegOp,
        isa_info: ISAInfo,
        *,
        disable_regalloc: bool,
    ) -> None:
        """Replace ``op`` with this nano-kernel's ISA-specific instructions."""
