from typing import override

from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import MatmulOp
from autotuner.nano_kernel import (
    GemmDescriptor,
    ISAInfo,
    TileSizes,
    VectorLayout,
)
from autotuner.schedules import attach_mask
from autotuner.skx_fsdbcst_nano_kernel import SkxFsdbcstNanoKernel
from autotuner.skx_nano_kernel_utils import SKX_VECTOR_BANKS


class SkxNarrowFsdbcstNanoKernel(SkxFsdbcstNanoKernel):
    """The one-M-vector memory-broadcast nano-kernel on the narrowest bank.

    Identical to :class:`SkxFsdbcstNanoKernel` in structure -- one M vector held
    in a register, N accumulators, and one broadcasting FMA per (K, N) step --
    but the M vector occupies the narrowest register bank that covers the tile's
    M rather than always the widest one. LIBXSMM masks a partial M vector inside
    a full-width register, which spends a full-width operation on the few lanes
    that are live; a core that splits 512-bit operations over a narrower datapath
    charges two passes for that, so a tile of four f64 elements runs twice as
    fast in a 256-bit register as in a masked 512-bit one.

    The lanes the bank leaves over are still masked, so a tile whose M is not a
    power of two -- or is below the narrowest bank's capacity -- rounds up to the
    next bank and masks the difference, exactly as the full-width kernel does.
    """

    @property
    @override
    def name(self) -> str:
        return "libxsmm-skx-narrow-fsdbcst"

    @override
    def vector_layout(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> VectorLayout:
        return isa_info.narrowest_vector_layout(descriptor.datatype, tile.m)

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
        # The mask covers what the narrowed bank leaves over, not what a zmm
        # would: a four-element f64 tile fills its ymm and needs no mask at all.
        return attach_mask(
            rewriter,
            op,
            tile_size=tile_size,
            vector_size=VectorLayout.narrowest(
                SKX_VECTOR_BANKS, op.datatype, tile_size
            ).lanes,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=mask_reg,
        )
