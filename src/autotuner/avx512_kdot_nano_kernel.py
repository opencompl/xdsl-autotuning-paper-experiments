from typing import cast

from xdsl import ir
from xdsl.dialects import builtin, x86
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulRegOp, matmul_reg_pointer_offsets
from autotuner.nano_kernel import (
    GemmDescriptor,
    ISAInfo,
    NanoKernel,
    RegisterCount,
    TileSizes,
)
from autotuner.skx_nano_kernel_utils import (
    MatmulRegValues,
    VectorValue,
    descriptor_from_op,
    load_vector,
    offset_pointer,
    tile_sizes_from_op,
    values_from_op,
)


class Avx512KdotNanoKernel(NanoKernel):
    """Vectorize a row-major dot product along K.

    CompXSMM swaps row-major GEMMs into its column-major representation.  The
    profitable original ``N=1`` case therefore arrives as ``m=1, lda=1``.  One
    ZMM is kept resident for each eight-element A chunk and reused across every
    output in the N tile.  The C accumulators hold the lane-wise partial sums
    and are horizontally reduced before the existing C-store schedule.
    """

    @property
    def name(self) -> str:
        return "avx512-kdot"

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return (
            isa_info.isa == "avx512"
            and isinstance(descriptor.datatype, builtin.Float64Type)
            and descriptor.m == 1
            and descriptor.lda == 1
            and descriptor.k > 0
            and descriptor.k % isa_info.vector_length(descriptor.datatype) == 0
        )

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self.supports(descriptor, isa_info):
            return False
        if tile.m != 1 or tile.n <= 0 or tile.k != descriptor.k:
            return False
        return self.register_usage(descriptor, tile, isa_info).fits(
            isa_info.register_capacity
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self.supports(descriptor, isa_info):
            raise ValueError("unsupported AVX-512 K-dot GEMM descriptor")
        if tile.m != 1 or tile.n <= 0 or tile.k != descriptor.k:
            raise ValueError("unsupported AVX-512 K-dot tile")

        k_vectors = tile.k // isa_info.vector_length(descriptor.datatype)
        return RegisterCount(
            general=5,
            vector=k_vectors + tile.n,
            mask=1,
        )

    @staticmethod
    def _fixed_vector_alias(
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        value: ir.SSAValue[x86.registers.X86VectorRegisterType],
        register_type: type[x86.registers.X86VectorRegisterType],
    ) -> ir.SSAValue[x86.registers.X86VectorRegisterType]:
        index = value.type.index
        if not isinstance(index, builtin.IntAttr):
            raise PassFailedException(
                "AVX-512 K-dot reduction requires allocated accumulator registers"
            )
        register = register_type.from_index(index.data)
        return rewriter.insert(
            x86.ops.GetAVXRegisterOp(register), insertion_point=insert_point
        ).result

    @staticmethod
    def _horizontal_sum(
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        accumulator: VectorValue,
    ) -> VectorValue:
        """Reduce one ZMM into its low lane, using dead zmm0/zmm1 aliases."""

        high_256 = rewriter.insert(
            x86.ops.DSI_Vextractf64x4Op(
                accumulator,
                1,
                destination=x86.registers.YMM0,
            ),
            insertion_point=insert_point,
        ).destination
        low_256 = Avx512KdotNanoKernel._fixed_vector_alias(
            rewriter,
            insert_point,
            accumulator,
            x86.registers.AVX2RegisterType,
        )
        sum_256 = rewriter.insert(
            x86.ops.DSS_VaddpdOp(
                low_256,
                high_256,
                destination=x86.registers.YMM0,
            ),
            insertion_point=insert_point,
        ).destination

        high_128 = rewriter.insert(
            x86.ops.DSI_Vextractf128Op(
                sum_256,
                1,
                destination=x86.registers.XMM1,
            ),
            insertion_point=insert_point,
        ).destination
        low_128 = rewriter.insert(
            x86.ops.GetAVXRegisterOp(x86.registers.XMM0),
            insertion_point=insert_point,
        ).result
        sum_128 = rewriter.insert(
            x86.ops.DSS_VaddpdOp(
                low_128,
                high_128,
                destination=x86.registers.XMM0,
            ),
            insertion_point=insert_point,
        ).destination
        swapped = rewriter.insert(
            x86.ops.DSSI_VshufpdOp(
                sum_128,
                sum_128,
                1,
                destination=x86.registers.XMM1,
            ),
            insertion_point=insert_point,
        ).destination

        sum_128_sse = ir.SSAValue.get(sum_128, type=x86.registers.SSERegisterType)
        swapped_sse = ir.SSAValue.get(swapped, type=x86.registers.SSERegisterType)
        accumulator_xmm = ir.SSAValue.get(
            Avx512KdotNanoKernel._fixed_vector_alias(
                rewriter,
                insert_point,
                accumulator,
                x86.registers.SSERegisterType,
            ),
            type=x86.registers.SSERegisterType,
        )
        rewriter.insert(
            x86.ops.DSS_VaddsdOp(
                sum_128_sse,
                swapped_sse,
                destination=accumulator_xmm.type,
            ),
            insertion_point=insert_point,
        )

        # The scalar VEX write above defines the low lane of this physical ZMM.
        # The existing masked C store consumes only that lane.
        result = rewriter.insert(
            x86.ops.GetAVXRegisterOp(accumulator.type), insertion_point=insert_point
        ).result
        return cast(VectorValue, result)

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
            raise PassFailedException("unsupported AVX-512 K-dot nano-kernel tile")

        insert_point = InsertPoint.before(op)
        values = values_from_op(op)
        vector_length = isa_info.vector_length(op.datatype)
        element_size = op.datatype.size
        k_vectors = tile.k // vector_length

        a_vectors = tuple(
            load_vector(
                rewriter,
                insert_point,
                op.datatype,
                values.a,
                k_vector * vector_length * element_size,
                x86.registers.AVX512RegisterType.from_index(k_vector),
                aligned=bool(op.aligned_a.value.data),
                mask=None,
            )
            for k_vector in range(k_vectors)
        )

        accumulators = list(values.accumulators)
        for k_vector, a_vector in enumerate(a_vectors):
            k_offset = k_vector * vector_length
            for n in range(tile.n):
                b_offset = (n * op.ldb.value.data + k_offset) * element_size
                result = rewriter.insert(
                    x86.ops.RSM_Vfmadd231pdOp(
                        accumulators[n],
                        a_vector,
                        values.b,
                        b_offset,
                    ),
                    insertion_point=insert_point,
                ).register_out
                accumulators[n] = cast(VectorValue, result)

        accumulators = [
            self._horizontal_sum(rewriter, insert_point, accumulator)
            for accumulator in accumulators
        ]

        desired_a_offset, desired_b_offset = matmul_reg_pointer_offsets(op)
        result = MatmulRegValues(
            offset_pointer(rewriter, insert_point, values.a, desired_a_offset),
            offset_pointer(rewriter, insert_point, values.b, desired_b_offset),
            values.rbp,
            values.rsp,
            values.mask,
            tuple(accumulators),
        )
        rewriter.replace(op, [], result.vals)


class KdotWithSkxFallbackNanoKernel(NanoKernel):
    """Use K-dot when legal and preserve the current SKX kernel elsewhere."""

    def __init__(self, fallback: NanoKernel):
        self.kdot = Avx512KdotNanoKernel()
        self.fallback = fallback

    @property
    def name(self) -> str:
        return "avx512-kdot-with-libxsmm-skx-fallback"

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return self.fallback.supports(descriptor, isa_info)

    def uses_kdot(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        return self.kdot.supports_tile(descriptor, tile, isa_info)

    def _select(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> NanoKernel:
        if self.uses_kdot(descriptor, tile, isa_info):
            return self.kdot
        return self.fallback

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        return self._select(descriptor, tile, isa_info).supports_tile(
            descriptor, tile, isa_info
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        return self._select(descriptor, tile, isa_info).register_usage(
            descriptor, tile, isa_info
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
        self._select(descriptor, tile, isa_info).rewrite(
            rewriter,
            op,
            isa_info,
            disable_regalloc=disable_regalloc,
        )
