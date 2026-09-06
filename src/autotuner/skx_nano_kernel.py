from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from typing_extensions import override

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.nano_kernel import (
    FloatingPointType,
    GemmDescriptor,
    ISAInfo,
    NanoKernel,
    RegisterCount,
    SupportedTile,
    TileSizes,
)
from autotuner.schedules import attach_mask
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel_utils import (
    descriptor_from_op,
    tile_sizes_from_op,
)
from autotuner.skx_nofsdbcst_nano_kernel import SkxNofsdbcstNanoKernel


@dataclass(frozen=True)
class AVX512Info(ISAInfo):
    """The architectural register file and vector widths for AVX-512."""

    @property
    def isa(self) -> Literal["avx512"]:
        return "avx512"

    @property
    def register_capacity(self) -> RegisterCount:
        return RegisterCount(general=16, vector=32, mask=8)

    def vector_length(self, datatype: FloatingPointType) -> int:
        match datatype:
            case builtin.Float32Type():
                return 16
            case builtin.Float64Type():
                return 8
        raise ValueError(f"unsupported AVX-512 datatype {datatype}")


class SkxNanoKernel(NanoKernel):
    """The LIBXSMM-compatible SKX nano-kernel selection heuristic."""

    _fsdbcst = SkxFsdbcstNanoKernel()
    _nofsdbcst = SkxNofsdbcstNanoKernel()

    @property
    def name(self) -> str:
        return "libxsmm-skx"

    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        return self._fsdbcst.supported_tile_sizes(
            datatype, isa_info
        ) | self._nofsdbcst.supported_tile_sizes(datatype, isa_info)

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return isa_info.isa == "avx512" and isinstance(
            descriptor.datatype, builtin.Float32Type | builtin.Float64Type
        )

    def _select_nano_kernel(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> NanoKernel:
        vector_length = isa_info.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return self._fsdbcst if m_vectors == 1 else self._nofsdbcst

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self.supports(descriptor, isa_info):
            return False
        if tile.m <= 0 or tile.n <= 0 or tile.k <= 0:
            return False
        vector_length = isa_info.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        if m_vectors > 4 or tile.n > 28:
            return False
        return self._select_nano_kernel(descriptor, tile, isa_info).supports_tile(
            descriptor, tile, isa_info
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self.supports_tile(descriptor, tile, isa_info):
            raise ValueError("unsupported SKX nano-kernel tile")
        return self._select_nano_kernel(descriptor, tile, isa_info).register_usage(
            descriptor,
            tile,
            isa_info,
        )

    @override
    def attach_mask(
        self,
        rewriter: PatternRewriter,
        op: MatmulOp,
        *,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        return attach_mask(
            rewriter,
            op,
            tile_size=op.m.value.data,
            vector_size=512 // op.datatype.bitwidth,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=mask_reg,
        )

    def rewrite(
        self,
        rewriter: PatternRewriter,
        op: MatmulRegOp,
        isa_info: ISAInfo,
        *,
        disable_regalloc: bool,
    ) -> None:
        descriptor = descriptor_from_op(op)
        tile = tile_sizes_from_op(op)
        if not self.supports_tile(descriptor, tile, isa_info):
            raise PassFailedException("unsupported SKX nano-kernel tile")
        self._select_nano_kernel(descriptor, tile, isa_info).rewrite(
            rewriter,
            op,
            isa_info,
            disable_regalloc=disable_regalloc,
        )


SKX_NANO_KERNELS: Mapping[str, NanoKernel] = {
    nano_kernel.name: nano_kernel
    for nano_kernel in (
        SkxNanoKernel(),
        SkxFsdbcstNanoKernel(),
        SkxNofsdbcstNanoKernel(),
    )
}


def get_skx_nano_kernel(name: str) -> NanoKernel:
    """Return the named SKX nano-kernel implementation."""
    try:
        return SKX_NANO_KERNELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(SKX_NANO_KERNELS))
        raise ValueError(
            f"unknown SKX nano-kernel '{name}'; expected one of: {choices}"
        ) from error
