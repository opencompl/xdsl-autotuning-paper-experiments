from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import MatmulKOp

FloatingPointType: TypeAlias = builtin.Float32Type | builtin.Float64Type


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


class TargetInfo(ABC):
    """Target properties needed by nano-kernel selection and tiling."""

    @property
    @abstractmethod
    def arch(self) -> Literal["skx"]:
        """Architecture identifier used by target-specific lowering."""

    @property
    @abstractmethod
    def register_capacity(self) -> RegisterCount:
        """Registers available to the generated kernel."""

    @abstractmethod
    def vector_length(self, datatype: FloatingPointType) -> int:
        """Number of ``datatype`` elements in one vector register."""


class NanoKernel(ABC):
    """A parametric register-resident kernel and its planning contract."""

    @abstractmethod
    def supports(self, descriptor: GemmDescriptor, target: TargetInfo) -> bool:
        """Return whether this kernel supports the GEMM configuration."""

    @abstractmethod
    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> bool:
        """Return whether this kernel can expand the proposed tile."""

    @abstractmethod
    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> RegisterCount:
        """Return the peak register demand of the proposed tile."""

    @abstractmethod
    def rewrite(
        self,
        rewriter: PatternRewriter,
        op: MatmulKOp,
        target: TargetInfo,
        *,
        disable_regalloc: bool,
    ) -> None:
        """Replace ``op`` with the target instructions for this nano-kernel."""
