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
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel_utils import (
    SKX_VECTOR_BANKS,
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
        return SKX_VECTOR_BANKS

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

    def select_for_tile(self, m: int, vector_length: int) -> NanoKernel:
        """The kernel that lowers an ``m``-row tile, given the full vector length.

        Both the planning entry points and :meth:`attach_mask` route through
        here, so the mask a tile is given always comes from the kernel that goes
        on to lower it.
        """
        return self._fsdbcst if m <= vector_length else self._nofsdbcst

    def _select_nano_kernel(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> NanoKernel:
        return self.select_for_tile(tile.m, isa_info.vector_length(descriptor.datatype))

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
        tile_size: int,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        return self.select_for_tile(tile_size, 512 // op.datatype.bitwidth).attach_mask(
            rewriter,
            op,
            tile_size=tile_size,
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
    """The SKX selection heuristic over all three nano-kernels.

    Dispatches a tile shorter than one full M vector to
    :class:`~autotuner.skx_narrow_fsdbcst_nano_kernel.SkxNarrowFsdbcstNanoKernel`,
    which sizes that vector's register bank to the tile's M. A tile that fills
    the vector exactly has no narrower bank to move to and keeps LIBXSMM's
    full-width memory-broadcast kernel, and a tile spanning several M vectors
    keeps its full-width register-broadcast kernel, so this differs from
    :class:`SkxNanoKernel` only where narrowing is available.
    """

    _narrow_fsdbcst = SkxNarrowFsdbcstNanoKernel()

    @property
    @override
    def name(self) -> str:
        return "libxsmm-skx-narrow"

    @override
    def select_for_tile(self, m: int, vector_length: int) -> NanoKernel:
        if m < vector_length:
            return self._narrow_fsdbcst
        return super().select_for_tile(m, vector_length)

    # @override
    # def attach_mask(self, rewriter: PatternRewriter, op: MatmulOp, *, tile_size: int, mask_tmp_reg: GeneralRegisterType, mask_reg: AVX512MaskRegisterType) -> MatmulOp:
    #     return self.select_for_tile(op.)
    #     return super().attach_mask(rewriter, op, tile_size=tile_size, mask_tmp_reg=mask_tmp_reg, mask_reg=mask_reg)


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
