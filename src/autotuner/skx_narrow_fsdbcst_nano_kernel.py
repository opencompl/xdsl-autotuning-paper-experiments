from typing_extensions import override
from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.instructions import (
    PointerValue,
    VectorValue,
    advance_pointer,
    load_vector,
    multiply_add_memory,
)
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
from autotuner.skx_nano_kernel_utils import (
    VECTOR_BANK_BITWIDTH,
    MatmulRegValues,
    bank_lanes,
    descriptor_from_op,
    tile_sizes_from_op,
    values_from_op,
    vector_register,
)

# Two A columns are live at a time -- the current K step's and the next one's --
# so the rest of the register file holds accumulators.
_A_VECTORS = 2


def _narrow_bank(
    m: int,
    datatype: FloatingPointType,
    widest: type[X86VectorRegisterType],
) -> type[X86VectorRegisterType]:
    """Return the narrowest bank up to ``widest`` that covers an ``m`` tile.

    There is no bank below 128 bits, so a one-lane tile still lands in an xmm
    with the surplus lanes masked off rather than in the scalar form LLVM uses
    there.
    """
    widest_lanes = bank_lanes(widest, datatype)
    return min(
        (
            bank
            for bank in VECTOR_BANK_BITWIDTH
            if m <= bank_lanes(bank, datatype) <= widest_lanes
        ),
        key=VECTOR_BANK_BITWIDTH.__getitem__,
        default=widest,
    )


class SkxNarrowFsdbcstNanoKernel(NanoKernel):
    """The narrowest-bank one-M-vector memory-broadcast nano-kernel.

    The dataflow is LIBXSMM's ``fsdbcst``: one A column lives in a vector
    register and each of the N accumulators takes an FMA whose second factor is
    an EVEX broadcast of a B scalar straight out of memory. What differs is the
    register bank. LIBXSMM always uses the full ISA vector -- for f64 on
    AVX-512, a zmm with ``{1to8}`` -- and masks off the lanes an M tile shorter
    than that leaves over, so an M-of-2 tile spends a 512-bit FMA to compute two
    useful lanes. This kernel instead picks the narrowest bank that covers the M
    tile: xmm with ``{1to2}`` for two f64 lanes, ymm with ``{1to4}`` for four,
    zmm only from five lanes up. Every lane of the narrow register does useful
    work, and on machines that split a 512-bit vector operation into two
    256-bit halves the FMA also costs half as much.

    This is the inner loop LLVM generates for the same schedule, and it is why
    ``libxtcmm`` beats ``libxsmm`` and ``compxsmm`` on GEMMs whose M is a
    fraction of the vector length -- close to a factor of two on Zen 4 for the
    f64 M-of-1 and M-of-2 shapes. Also matching LLVM, the accumulators are a
    single set: unlike ``fsdbcst``, this kernel never duplicates them to shorten
    the cross-K dependency chain.

    From the vector length up the two coincide: the narrowest bank covering a
    full M vector *is* the full vector, and this kernel then emits exactly the
    instructions ``fsdbcst`` does for the same tile.

    An M tile that does not fill its bank exactly -- three f64 lanes in an
    xmm-and-ymm gap, or the single lane LLVM would give a scalar ``vfmadd231sd``
    -- still masks, but on the narrow bank rather than on a zmm.
    """

    @property
    def name(self) -> str:
        return "llvm-skx-narrow-fsdbcst"

    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        vector_length = isa_info.vector_length(datatype)
        accumulators = isa_info.register_capacity.vector - _A_VECTORS
        return frozenset(
            SupportedTile(m, n)
            for m in range(1, vector_length + 1)
            for n in range(1, accumulators + 1)
        )

    @override
    def vector_bank(
        self,
        m: int,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> type[X86VectorRegisterType]:
        """Return the narrowest bank that covers ``m``."""
        return _narrow_bank(m, datatype, isa_info.vector_bank)

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return isa_info.isa == "avx512" and isinstance(
            descriptor.datatype, builtin.Float32Type | builtin.Float64Type
        )

    def _supports_tile_shape(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self.supports(descriptor, isa_info):
            return False
        if tile.m <= 0 or tile.n <= 0 or tile.k <= 0:
            return False
        # One M vector, in whichever bank covers it.
        return tile.m <= isa_info.vector_length(descriptor.datatype)

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            return False
        return self.register_usage(descriptor, tile, isa_info).fits(
            isa_info.register_capacity,
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            raise ValueError("unsupported SKX narrow fsdbcst nano-kernel tile")

        lanes = bank_lanes(
            self.vector_bank(tile.m, descriptor.datatype, isa_info),
            descriptor.datatype,
        )
        return RegisterCount(
            general=5,
            vector=tile.n + min(tile.k, _A_VECTORS),
            mask=int(tile.m % lanes != 0),
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
            vector_size=bank_lanes(
                _narrow_bank(op.m.value.data, op.datatype, AVX512RegisterType),
                op.datatype,
            ),
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
            raise PassFailedException("unsupported SKX narrow fsdbcst nano-kernel tile")

        insert_point = InsertPoint.before(op)
        bank = self.vector_bank(tile.m, op.datatype, isa_info)
        values = values_from_op(op, bank)
        element_size = op.datatype.size

        accumulators = list(values.accumulators)

        def load_a_column(pointer: PointerValue, k: int) -> VectorValue:
            """Load the A column for K step ``k`` into its rotating register."""
            return load_vector(
                rewriter,
                insert_point,
                op.datatype,
                pointer,
                op.lda.value.data * k * element_size,
                vector_register(
                    k % _A_VECTORS,
                    bank,
                    disable_regalloc=disable_regalloc,
                ),
                aligned=bool(op.aligned_a.value.data),
                mask=values.mask,
            )

        # A is double-buffered: the column for K step k+1 is loaded while the
        # column for step k is still being multiplied into the accumulators.
        a = values.a
        a_vectors: dict[int, VectorValue] = {0: load_a_column(a, 0)}
        if tile.k > 1:
            a_vectors[1] = load_a_column(a, 1)

        for k in range(tile.k):
            if 0 < k < tile.k - 1:
                a_vectors[(k + 1) % _A_VECTORS] = load_a_column(a, k + 1)

            if k == tile.k - 1:
                a = advance_pointer(
                    rewriter,
                    insert_point,
                    a,
                    tile.k * op.lda.value.data * element_size,
                )

            for n in range(tile.n):
                accumulators[n] = multiply_add_memory(
                    rewriter,
                    insert_point,
                    op.datatype,
                    accumulators[n],
                    a_vectors[k % _A_VECTORS],
                    values.b,
                    (k + n * op.ldb.value.data) * element_size,
                )

        b = advance_pointer(
            rewriter,
            insert_point,
            values.b,
            tile.k * element_size,
        )

        result = MatmulRegValues(a, b, values.mask, tuple(accumulators))
        rewriter.replace(op, [], result.vals)
