from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, override

from xdsl.dialects import builtin, x86
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
    TileSizes,
    VectorLayout,
)
from autotuner.schedules import attach_mask
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel_utils import (
    descriptor_from_op,
    tile_sizes_from_op,
)
from autotuner.skx_narrow_fsdbcst_nano_kernel import SkxNarrowFsdbcstNanoKernel
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

    @property
    def vector_banks(self) -> tuple[type[x86.registers.X86VectorRegisterType], ...]:
        return (
            x86.registers.SSERegisterType,
            x86.registers.AVX2RegisterType,
            x86.registers.AVX512RegisterType,
        )

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

    def vector_layout(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> VectorLayout:
        return self._select_nano_kernel(descriptor, tile, isa_info).vector_layout(
            descriptor,
            tile,
            isa_info,
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


class SkxNarrowNanoKernel(SkxNanoKernel):
    """The SKX selection heuristic, narrowing a one-M-vector tile's bank.

    Selects between the same two kernels as :class:`SkxNanoKernel`, but a tile
    that fits in a single M vector goes to
    :class:`~autotuner.skx_narrow_fsdbcst_nano_kernel.SkxNarrowFsdbcstNanoKernel`,
    which sizes that vector's register bank to the tile's M. Multi-M-vector
    tiles keep LIBXSMM's full-width register-broadcast kernel.
    """

    _fsdbcst = SkxNarrowFsdbcstNanoKernel()

    @property
    def name(self) -> str:
        return "libxsmm-skx-narrow"


SKX_NANO_KERNELS: Mapping[str, NanoKernel] = {
    nano_kernel.name: nano_kernel
    for nano_kernel in (
        SkxNanoKernel(),
        SkxNarrowNanoKernel(),
        SkxFsdbcstNanoKernel(),
        SkxNarrowFsdbcstNanoKernel(),
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
