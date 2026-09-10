from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from typing_extensions import override

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
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
    def vector_type(self) -> type[X86VectorRegisterType]:
        return AVX512RegisterType


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
        m: int,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> NanoKernel:
        """Return the nano-kernel an M tile of ``m`` lowers to.

        Only M decides, which is what lets the mask and the accumulator
        register type -- neither of which knows the tile's N or K -- ask the
        same question the rewrite does.
        """
        vector_length = isa_info.vector_type.bitwidth() // datatype.bitwidth
        m_vectors = (m + vector_length - 1) // vector_length
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
        vector_length = isa_info.vector_type.bitwidth() // descriptor.datatype.bitwidth
        m_vectors = (tile.m + vector_length - 1) // vector_length
        if m_vectors > 4 or tile.n > 28:
            return False
        return self._select_nano_kernel(
            tile.m, descriptor.datatype, isa_info
        ).supports_tile(descriptor, tile, isa_info)

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self.supports_tile(descriptor, tile, isa_info):
            raise ValueError("unsupported SKX nano-kernel tile")
        return self._select_nano_kernel(
            tile.m, descriptor.datatype, isa_info
        ).register_usage(descriptor, tile, isa_info)

    @override
    def vector_type(
        self,
        m: int,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> type[X86VectorRegisterType]:
        return self._select_nano_kernel(m, datatype, isa_info).vector_type(
            m, datatype, isa_info
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
        # The mask covers the lanes the tile leaves over in the register the
        # rewrite will put it in, so it is that nano-kernel's to attach.  Which
        # ISA to ask is not in question: `supports` refuses anything but
        # AVX-512.
        return self._select_nano_kernel(
            op.m.value.data, op.datatype, AVX512Info()
        ).attach_mask(
            rewriter,
            op,
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
        self._select_nano_kernel(tile.m, descriptor.datatype, isa_info).rewrite(
            rewriter,
            op,
            isa_info,
            disable_regalloc=disable_regalloc,
        )


class SkxPlusNarrowNanoKernel(SkxNanoKernel):
    """The SKX heuristic, with the narrow nano-kernel added to its choices.

    LIBXSMM's heuristic knows two nano-kernels and picks between them on how
    many vectors one M tile spans: ``fsdbcst`` for one, ``nofsdbcst`` for more.
    An M tile shorter than a whole vector still goes to ``fsdbcst``, which
    computes it in a full-width register with the surplus lanes masked off --
    an M of two f64 spends a 512-bit FMA on two useful lanes.  This heuristic
    hands those tiles to ``llvm-skx-narrow-fsdbcst`` instead, which puts them in
    the narrowest register type that covers them, the way LLVM does; on Zen
    that is where ``libxtcmm`` has been beating ``compxsmm``.

    Nothing else moves.  The tile a short M lands in is exactly the one the
    LIBXSMM heuristic would have chosen -- the two kernels agree on which M-by-N
    tiles are legal and on how many registers one costs, so `compute_tiling_strategy`
    returns the same M tile, the same N ranges and the same K blocking as
    ``libxsmm-skx`` does.  Only the instructions inside the tile change, which
    is what makes the two comparable in a figure.

    From half a vector up the narrow kernel *is* the full-width one, so those M
    keep LIBXSMM's choice, duplicated accumulator sets and all.
    """

    _narrow = SkxNarrowFsdbcstNanoKernel()

    @property
    def name(self) -> str:
        return "libxsmm-skx-plusnarrow"

    # `supported_tile_sizes` is inherited: the narrow kernel takes over tiles
    # the wide one already supports rather than adding any of its own.

    @override
    def _select_nano_kernel(
        self,
        m: int,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> NanoKernel:
        """Prefer the narrow kernel when it would use a narrower register."""
        if self._narrow.vector_type(m, datatype, isa_info) is not isa_info.vector_type:
            return self._narrow
        return super()._select_nano_kernel(m, datatype, isa_info)


SKX_NANO_KERNELS: Mapping[str, NanoKernel] = {
    nano_kernel.name: nano_kernel
    for nano_kernel in (
        SkxNanoKernel(),
        SkxPlusNarrowNanoKernel(),
        SkxFsdbcstNanoKernel(),
        SkxNofsdbcstNanoKernel(),
        SkxNarrowFsdbcstNanoKernel(),
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
