from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from xdsl.dialects import builtin, x86
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp

FloatingPointType: TypeAlias = builtin.Float32Type | builtin.Float64Type


@dataclass(frozen=True)
class TileSizes:
    """The M, N, and K sizes computed by one nano-kernel invocation."""

    m: int
    n: int
    k: int


@dataclass(frozen=True)
class VectorLayout:
    """The register bank that holds one M vector of a nano-kernel tile.

    ``lanes`` is the bank's capacity in ``datatype`` elements, not the number of
    elements the tile actually uses: a tile whose M is shorter than ``lanes``
    masks the lanes above it. A kernel is free to pick a bank narrower than the
    ISA's widest one, which costs fewer cycles per operation on cores that split
    512-bit operations over a narrower datapath.
    """

    register_type: type[x86.registers.X86VectorRegisterType]
    lanes: int

    def register(
        self, index: int, *, allocated: bool
    ) -> x86.registers.X86VectorRegisterType:
        """The bank's register at ``index``, or an unallocated one."""
        if not allocated:
            return self.register_type.unallocated()
        return self.register_type.from_index(index)

    @classmethod
    def narrowest(
        cls,
        banks: tuple[type[x86.registers.X86VectorRegisterType], ...],
        datatype: FloatingPointType,
        lanes: int,
    ) -> VectorLayout:
        """The layout using the narrowest of ``banks`` that holds ``lanes``.

        ``banks`` runs narrowest first. This is the one place a lane count is
        rounded up to a register bank, so a kernel that reports a narrowed
        layout and the mask attached to its tiles cannot disagree.
        """
        for bank in banks:
            capacity = bank.bitwidth() // datatype.bitwidth
            if capacity >= lanes:
                return cls(bank, capacity)
        raise ValueError(f"no vector bank holds {lanes} {datatype} elements")


@dataclass(frozen=True)
class GemmDescriptor:
    """Target-independent information describing a GEMM problem."""

    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    datatype: FloatingPointType
    aligned_a: bool
    aligned_c: bool


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
    def vector_banks(self) -> tuple[type[x86.registers.X86VectorRegisterType], ...]:
        """The vector register banks this ISA may use, narrowest first."""

    @abstractmethod
    def vector_length(self, datatype: FloatingPointType) -> int:
        """Number of ``datatype`` elements in one vector register."""

    def widest_vector_layout(self, datatype: FloatingPointType) -> VectorLayout:
        """The layout using the ISA's widest vector bank."""
        return VectorLayout(self.vector_banks[-1], self.vector_length(datatype))

    def narrowest_vector_layout(
        self, datatype: FloatingPointType, lanes: int
    ) -> VectorLayout:
        """The layout using the narrowest bank that holds ``lanes`` elements."""
        return VectorLayout.narrowest(self.vector_banks, datatype, lanes)


class NanoKernel(ABC):
    """A parametric register-resident kernel and its planning contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable name used to select and benchmark this nano-kernel."""

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
    def vector_layout(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> VectorLayout:
        """Return the register bank one M vector of the proposed tile occupies."""

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
        tile_size: int,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        """Attach a mask to ``op`` if the kernel needs one, or do nothing.

        ``tile_size`` is the M extent of one nano-kernel tile, which is what
        decides the mask; ``op`` still spans every tile of the block being
        masked. A kernel that sizes its register bank to the tile has no bank at
        all for that whole extent, so it cannot recover the tile from ``op``.
        """

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
