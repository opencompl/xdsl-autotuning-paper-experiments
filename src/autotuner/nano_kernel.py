from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp

FloatingPointType: TypeAlias = builtin.Float32Type | builtin.Float64Type


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

    @abstractmethod
    def vector_length(self, datatype: FloatingPointType) -> int:
        """Number of ``datatype`` elements in one vector register."""


class NanoKernel(ABC):
    """A parametric register-resident kernel and its planning contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable name used to select and benchmark this nano-kernel."""

    @abstractmethod
    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        """Return the supported M-by-N tile shapes."""

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
