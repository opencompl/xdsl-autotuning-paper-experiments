from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.libxsmm_gemm.generator_gemm_avx512_microkernel import (
    libxsmm_generator_gemm_avx512_microkernel_nofsdbcst,
)
from autotuner.nano_kernel import (
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    TargetInfo,
    TileSizes,
)
from autotuner.skx_nano_kernel_utils import (
    create_matmul_k_context,
    descriptor_from_op,
    supports_skx,
    tile_sizes_from_op,
)


class SkxNofsdbcstNanoKernel(NanoKernel):
    """The SKX multiple-M-vector register-broadcast nano-kernel."""

    def supports(self, descriptor: GemmDescriptor, target: TargetInfo) -> bool:
        return supports_skx(descriptor, target)

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
        return m_vectors >= 2

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> bool:
        if not self._supports_tile_shape(descriptor, tile, target):
            return False
        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        # TODO: The translated LIBXSMM implementation currently asserts at most four
        # M vectors. Once that implementation restriction is removed, use only register
        # pressure:
        # return self.register_usage(descriptor, tile, target).fits(
        #     target.register_capacity
        # )
        return m_vectors <= 4 and self.register_usage(descriptor, tile, target).fits(
            target.register_capacity,
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, target):
            raise ValueError("unsupported SKX nofsdbcst nano-kernel tile")

        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return RegisterCount(
            general=5,
            vector=m_vectors * tile.n + m_vectors + 1,
            mask=int(tile.m % vector_length != 0),
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
            raise PassFailedException("unsupported SKX nofsdbcst nano-kernel tile")

        context = create_matmul_k_context(rewriter, op, target)
        values = context.values
        for _ in range(tile.k):
            values = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst(
                context.generated_code,
                context.gp_reg_mapping,
                context.micro_kernel_config,
                context.descriptor,
                tile.m,
                tile.n,
                values,
                disable_regalloc=disable_regalloc,
            )
        rewriter.replace(op, [], values.vals)
