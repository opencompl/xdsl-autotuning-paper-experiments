from xdsl.dialects import builtin, x86
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.nano_kernel import (
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    TargetInfo,
    TileSizes,
)
from autotuner.skx_nano_kernel_utils import (
    MatmulKValues,
    VectorValue,
    add_vectors,
    advance_pointer,
    descriptor_from_op,
    load_vector,
    multiply_add_memory,
    tile_sizes_from_op,
    values_from_op,
    vector_register,
    zero_vector,
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

    def supports(self, descriptor: GemmDescriptor, target: TargetInfo) -> bool:
        return target.arch == "skx" and isinstance(
            descriptor.datatype, builtin.Float32Type | builtin.Float64Type
        )

    def _supports_tile_shape(
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
        return m_vectors == 1

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> bool:
        if not self._supports_tile_shape(descriptor, tile, target):
            return False
        # TODO: The translated LIBXSMM implementation currently asserts n <= 28.
        # Once that implementation restriction is removed, use only register pressure:
        # return self.register_usage(descriptor, tile, target).fits(
        #     target.register_capacity
        # )
        return tile.n <= 28 and self.register_usage(descriptor, tile, target).fits(
            target.register_capacity,
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, target):
            raise ValueError("unsupported SKX fsdbcst nano-kernel tile")

        accumulator_sets = self._accumulator_sets(tile)

        return RegisterCount(
            general=5,
            vector=tile.n * accumulator_sets + min(tile.k, 2),
            mask=int(tile.m % target.vector_length(descriptor.datatype) != 0),
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
            raise PassFailedException("unsupported SKX fsdbcst nano-kernel tile")

        insert_point = InsertPoint.before(op)
        values = values_from_op(op)
        vector_reg_count = target.register_capacity.vector
        element_size = op.datatype.size

        accumulator_sets = self._accumulator_sets(tile)

        accumulators_by_index = {
            vector_reg_count - tile.n + n: accumulator
            for n, accumulator in enumerate(values.accumulators)
        }
        for accumulator_set in range(1, accumulator_sets):
            for n in range(tile.n):
                register_index = vector_reg_count - tile.n * (accumulator_set + 1) + n
                register = x86.registers.AVX512RegisterType.from_index(register_index)
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
                accumulators_by_index[main_index] = add_vectors(
                    rewriter,
                    insert_point,
                    op.datatype,
                    accumulators_by_index[source_index],
                    accumulators_by_index[main_index],
                    x86.registers.AVX512RegisterType.from_index(main_index),
                )

        result = MatmulKValues(
            a,
            b,
            values.c,
            values.rbp,
            values.rsp,
            values.mask,
            tuple(
                accumulators_by_index[vector_reg_count - tile.n + n]
                for n in range(tile.n)
            ),
        )
        rewriter.replace(op, [], result.vals)
