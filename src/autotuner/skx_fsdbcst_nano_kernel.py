from typing_extensions import override
from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.instructions import (
    VectorValue,
    add_accumulator_vectors,
    add_vectors,
    advance_pointer,
    load_vector,
    multiply_add_memory,
    zero_vector,
)
from autotuner.nano_kernel import (
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    ISAInfo,
    TileSizes,
)
from autotuner.schedules import attach_mask
from autotuner.skx_nano_kernel_utils import (
    MatmulRegValues,
    descriptor_from_op,
    tile_sizes_from_op,
    values_from_op,
    vector_register,
)


class SkxFsdbcstNanoKernel(NanoKernel):
    """The SKX one-M-vector memory-broadcast nano-kernel."""

    @property
    def name(self) -> str:
        return "libxsmm-skx-fsdbcst"

    @staticmethod
    def _accumulator_sets(tile: TileSizes) -> int:
        if tile.n >= 12:
            accumulator_sets = 1
        elif tile.n >= 6:
            accumulator_sets = 2
        else:
            accumulator_sets = 4
        return min(accumulator_sets, tile.k)

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
        vector_length = isa_info.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return m_vectors == 1

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            return False
        # TODO: The translated LIBXSMM implementation currently asserts n <= 28.
        # Once that implementation restriction is removed, use only register pressure:
        # return self.register_usage(descriptor, tile, isa_info).fits(
        #     isa_info.register_capacity
        # )
        return tile.n <= 28 and self.register_usage(descriptor, tile, isa_info).fits(
            isa_info.register_capacity,
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            raise ValueError("unsupported SKX fsdbcst nano-kernel tile")

        accumulator_sets = self._accumulator_sets(tile)

        return RegisterCount(
            general=5,
            vector=tile.n * accumulator_sets + min(tile.k, 2),
            mask=int(tile.m % isa_info.vector_length(descriptor.datatype) != 0),
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
            raise PassFailedException("unsupported SKX fsdbcst nano-kernel tile")

        insert_point = InsertPoint.before(op)
        values = values_from_op(op)
        vector_reg_count = isa_info.register_capacity.vector
        element_size = op.datatype.size

        accumulator_sets = self._accumulator_sets(tile)

        accumulators_by_index = {
            vector_reg_count - tile.n + n: accumulator
            for n, accumulator in enumerate(values.accumulators)
        }
        for accumulator_set in range(1, accumulator_sets):
            for n in range(tile.n):
                register_index = vector_reg_count - tile.n * (accumulator_set + 1) + n
                register = vector_register(
                    register_index, disable_regalloc=disable_regalloc
                )
                accumulators_by_index[register_index] = zero_vector(
                    rewriter, insert_point, register
                )

        a_vectors: dict[int, VectorValue] = {}
        a = values.a
        for k in range(tile.k):
            if k == 0:
                register_index = 0
                a_vectors[register_index] = load_vector(
                    rewriter,
                    insert_point,
                    op.datatype,
                    a,
                    0,
                    vector_register(register_index, disable_regalloc=disable_regalloc),
                    aligned=bool(op.aligned_a.value.data),
                    mask=values.mask,
                )
                if tile.k > 1:
                    register_index = 1
                    a_vectors[register_index] = load_vector(
                        rewriter,
                        insert_point,
                        op.datatype,
                        a,
                        op.lda.value.data * element_size,
                        vector_register(
                            register_index, disable_regalloc=disable_regalloc
                        ),
                        aligned=bool(op.aligned_a.value.data),
                        mask=values.mask,
                    )
            elif k < tile.k - 1:
                register_index = (k + 1) % 2
                a_vectors[register_index] = load_vector(
                    rewriter,
                    insert_point,
                    op.datatype,
                    a,
                    op.lda.value.data * (k + 1) * element_size,
                    vector_register(register_index, disable_regalloc=disable_regalloc),
                    aligned=bool(op.aligned_a.value.data),
                    mask=values.mask,
                )

            if k == tile.k - 1:
                a = advance_pointer(
                    rewriter,
                    insert_point,
                    a,
                    tile.k * op.lda.value.data * element_size,
                )

            for n in range(tile.n):
                a_register_index = k % 2
                accumulator_index = (
                    vector_reg_count - tile.n * ((k % accumulator_sets) + 1) + n
                )
                accumulators_by_index[accumulator_index] = multiply_add_memory(
                    rewriter,
                    insert_point,
                    op.datatype,
                    accumulators_by_index[accumulator_index],
                    a_vectors[a_register_index],
                    values.b,
                    (k + n * op.ldb.value.data) * element_size,
                )

        b = advance_pointer(
            rewriter,
            insert_point,
            values.b,
            tile.k * element_size,
        )

        for accumulator_set in range(1, accumulator_sets):
            for n in range(tile.n):
                source_index = vector_reg_count - tile.n * (accumulator_set + 1) + n
                main_index = vector_reg_count - tile.n + n
                source = accumulators_by_index[source_index]
                main = accumulators_by_index[main_index]
                add = add_accumulator_vectors if disable_regalloc else add_vectors
                accumulators_by_index[main_index] = add(
                    rewriter,
                    insert_point,
                    op.datatype,
                    source,
                    main,
                    vector_register(main_index, disable_regalloc=disable_regalloc),
                )

        result = MatmulRegValues(
            a,
            b,
            values.mask,
            tuple(
                accumulators_by_index[vector_reg_count - tile.n + n]
                for n in range(tile.n)
            ),
        )
        rewriter.replace(op, [], result.vals)
