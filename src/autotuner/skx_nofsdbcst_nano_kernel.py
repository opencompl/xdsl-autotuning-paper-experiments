from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulRegOp
from autotuner.instructions import (
    advance_pointer,
    broadcast_scalar,
    load_vector,
    multiply_add_registers,
)
from autotuner.nano_kernel import (
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    ISAInfo,
    TileSizes,
)
from autotuner.skx_nano_kernel_utils import (
    MatmulRegValues,
    apply_matmul_reg_pointer_contract,
    descriptor_from_op,
    tile_sizes_from_op,
    values_from_op,
    vector_register,
)


class SkxNofsdbcstNanoKernel(NanoKernel):
    """The SKX multiple-M-vector register-broadcast nano-kernel."""

    @property
    def name(self) -> str:
        return "libxsmm-skx-nofsdbcst"

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
        return m_vectors >= 2

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            return False
        vector_length = isa_info.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        # TODO: The translated LIBXSMM implementation currently asserts at most four
        # M vectors. Once that implementation restriction is removed, use only register
        # pressure:
        # return self.register_usage(descriptor, tile, isa_info).fits(
        #     isa_info.register_capacity
        # )
        return m_vectors <= 4 and self.register_usage(descriptor, tile, isa_info).fits(
            isa_info.register_capacity,
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            raise ValueError("unsupported SKX nofsdbcst nano-kernel tile")

        vector_length = isa_info.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return RegisterCount(
            general=5,
            vector=m_vectors * tile.n + m_vectors + 1,
            mask=int(tile.m % vector_length != 0),
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
            raise PassFailedException("unsupported SKX nofsdbcst nano-kernel tile")

        insert_point = InsertPoint.before(op)
        values = values_from_op(op)
        vector_length = isa_info.vector_length(op.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        element_size = op.datatype.size
        accumulators = list(values.accumulators)
        a = values.a
        b = values.b

        for _ in range(tile.k):
            a_vectors = tuple(
                load_vector(
                    rewriter,
                    insert_point,
                    op.datatype,
                    a,
                    m * vector_length * element_size,
                    vector_register(1 + m, disable_regalloc=disable_regalloc),
                    aligned=bool(op.aligned_a.value.data),
                    mask=values.mask if m == m_vectors - 1 else None,
                )
                for m in range(m_vectors)
            )

            for n in range(tile.n):
                b_vector = broadcast_scalar(
                    rewriter,
                    insert_point,
                    op.datatype,
                    b,
                    op.ldb.value.data * n * element_size,
                    vector_register(0, disable_regalloc=disable_regalloc),
                )

                if n == tile.n - 1:
                    b = advance_pointer(rewriter, insert_point, b, element_size)

                for m in range(m_vectors):
                    if m == 0 and n == tile.n - 1:
                        a = advance_pointer(
                            rewriter,
                            insert_point,
                            a,
                            op.lda.value.data * element_size,
                        )

                    accumulator_index = m + m_vectors * n
                    accumulators[accumulator_index] = multiply_add_registers(
                        rewriter,
                        insert_point,
                        op.datatype,
                        accumulators[accumulator_index],
                        a_vectors[m],
                        b_vector,
                    )

        result = MatmulRegValues(
            a,
            b,
            values.rbp,
            values.rsp,
            values.mask,
            tuple(accumulators),
        )
        result = apply_matmul_reg_pointer_contract(rewriter, insert_point, op, result)
        rewriter.replace(op, [], result.vals)
