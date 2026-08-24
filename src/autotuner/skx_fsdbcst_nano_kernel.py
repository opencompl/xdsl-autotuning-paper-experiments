from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.libxsmm_gemm.generator_gemm_avx512_microkernel import (
    libxsmm_generator_gemm_avx512_microkernel_fsdbcst,
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


class SkxFsdbcstNanoKernel(NanoKernel):
    """The SKX one-M-vector memory-broadcast nano-kernel."""

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

        if tile.n >= 12:
            accumulator_sets = 1
        elif tile.n >= 6:
            accumulator_sets = 2
        else:
            accumulator_sets = 4
        accumulator_sets = min(accumulator_sets, tile.k)

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

        context = create_matmul_k_context(rewriter, op, target)
        values = libxsmm_generator_gemm_avx512_microkernel_fsdbcst(
            context.generated_code,
            context.gp_reg_mapping,
            context.micro_kernel_config,
            context.descriptor,
            tile.n,
            tile.k,
            context.values,
            disable_regalloc=disable_regalloc,
        )
        rewriter.replace(op, [], values.vals)
