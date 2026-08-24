from dataclasses import dataclass
from typing import Literal

from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.nano_kernel import (
    FloatingPointType,
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    TargetInfo,
    TileSizes,
)
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel_utils import (
    descriptor_from_op,
    supports_skx,
    tile_sizes_from_op,
)
from autotuner.skx_nofsdbcst_nano_kernel import SkxNofsdbcstNanoKernel


@dataclass(frozen=True)
class SkxTargetInfo(TargetInfo):
    """The register file and vector widths used by the SKX generator."""

    @property
    def arch(self) -> Literal["skx"]:
        return "skx"

    @property
    def register_capacity(self) -> RegisterCount:
        return RegisterCount(general=16, vector=32, mask=8)

    def vector_length(self, datatype: FloatingPointType) -> int:
        match datatype:
            case builtin.Float32Type():
                return 16
            case builtin.Float64Type():
                return 8
        raise ValueError(f"unsupported SKX datatype {datatype}")


class SkxNanoKernel(NanoKernel):
    """The LIBXSMM-compatible SKX nano-kernel selection heuristic."""

    _fsdbcst = SkxFsdbcstNanoKernel()
    _nofsdbcst = SkxNofsdbcstNanoKernel()

    def supports(self, descriptor: GemmDescriptor, target: TargetInfo) -> bool:
        return supports_skx(descriptor, target)

    def _select_nano_kernel(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> NanoKernel:
        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return self._fsdbcst if m_vectors == 1 else self._nofsdbcst

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> bool:
        if not self.supports(descriptor, target):
            return False
        if tile.m <= 0 or tile.n <= 0 or tile.k <= 0:
            return False
        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        if m_vectors > 4 or tile.n > 28:
            return False
        return self._select_nano_kernel(descriptor, tile, target).supports_tile(
            descriptor, tile, target
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> RegisterCount:
        if not self.supports_tile(descriptor, tile, target):
            raise ValueError("unsupported SKX nano-kernel tile")
        return self._select_nano_kernel(descriptor, tile, target).register_usage(
            descriptor,
            tile,
            target,
        )

    def rewrite(
        self,
        rewriter: PatternRewriter,
        op: MatmulKOp,
        target: TargetInfo,
        *,
        disable_regalloc: bool,
    ) -> None:
        descriptor = descriptor_from_op(op)
        tile = tile_sizes_from_op(op)
        if not self.supports_tile(descriptor, tile, target):
            raise PassFailedException("unsupported SKX nano-kernel tile")
        self._select_nano_kernel(descriptor, tile, target).rewrite(
            rewriter,
            op,
            target,
            disable_regalloc=disable_regalloc,
        )
