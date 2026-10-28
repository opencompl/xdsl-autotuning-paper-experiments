"""LIBXSMM's BF16 nano-kernels for CPX and later when A is packed in K pairs.

In LIBXSMM's VNNI-2 layout every 32-bit lane of A holds the (k, k+1) pair
``vdpbf16ps`` multiplies, so A loads like an F32 vector of 16 rows of M and
needs no interleave, and the F32 accumulators stay in M order: C is loaded and
stored as it is for F32. B is column-major, so a pair of B is contiguous.

LIBXSMM then picks between the two SKX nano-kernels exactly as it does for F32,
with every K step a pair: an M tile of one vector multiplies by a B pair
broadcast straight from memory (``fsdbcst``), a wider one broadcasts the pair
into a register first (``nofsdbcst``).
"""

from typing_extensions import override
from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.instructions import (
    VectorValue,
    add_accumulator_vectors,
    add_vectors,
    advance_pointer,
    broadcast_dword,
    dot_product_bf16_pairs,
    dot_product_bf16_pairs_memory,
    load_vector,
    zero_vector,
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
    MatmulRegValues,
    descriptor_from_op,
    tile_sizes_from_op,
    values_from_op,
    vector_register,
)

# F32 lanes of C per zmm, which are also the K pairs of A per zmm.
_LANES = 16
# K steps per element of packed A: libxsmm_cpuid_dot_pack_factor(BF16).
_PACK = 2
_MAX_M_VECTORS = 4
_MAX_N = 28


def _m_vectors(m: int) -> int:
    return (m + _LANES - 1) // _LANES


def _accumulator_sets(tile: TileSizes) -> int:
    """How many sets of accumulators ``fsdbcst`` spreads its K pairs over."""
    if tile.n >= 12:
        accumulator_sets = 1
    elif tile.n >= 6:
        accumulator_sets = 2
    else:
        accumulator_sets = 4
    return min(accumulator_sets, tile.k // _PACK)


class CpxBf16VnniNanoKernel(NanoKernel):
    """LIBXSMM's BF16 nano-kernels for A in VNNI-2 layout
    (``LIBXSMM_GEMM_FLAG_VNNI_A``), picked by how many vectors M spans."""

    @property
    def name(self) -> str:
        return "libxsmm-cpx-bf16-vnni"

    @property
    @override
    def k_pack_factor(self) -> int:
        return _PACK

    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        vectors = isa_info.register_capacity.vector
        return frozenset(
            SupportedTile(m, n)
            for m_vectors in range(1, _MAX_M_VECTORS + 1)
            for m in range((m_vectors - 1) * _LANES + 1, m_vectors * _LANES + 1)
            for n in range(
                1,
                _MAX_N + 1
                if m_vectors == 1
                else (vectors - m_vectors - 1) // m_vectors + 1,
            )
        )

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return (
            isa_info.isa == "avx512"
            and isinstance(descriptor.datatype, builtin.Float32Type)
            and isinstance(descriptor.ab_datatype, builtin.BFloat16Type)
            and descriptor.vnni_a
            and descriptor.k % _PACK == 0
            and descriptor.c_datatype is None
        )

    def _supports_tile_shape(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        return (
            self.supports(descriptor, isa_info)
            and tile.m > 0
            and tile.n > 0
            and tile.k > 0
            and tile.k % _PACK == 0
            and _m_vectors(tile.m) <= _MAX_M_VECTORS
            and tile.n <= _MAX_N
        )

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        return self._supports_tile_shape(
            descriptor, tile, isa_info
        ) and self.register_usage(descriptor, tile, isa_info).fits(
            isa_info.register_capacity
        )

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self._supports_tile_shape(descriptor, tile, isa_info):
            raise ValueError("unsupported CPX BF16 VNNI nano-kernel tile")
        m_vectors = _m_vectors(tile.m)
        if m_vectors == 1:
            # fsdbcst: its accumulator sets, and up to two A vectors in flight.
            vector = tile.n * _accumulator_sets(tile) + min(tile.k // _PACK, 2)
        else:
            # nofsdbcst: the accumulators, one A vector per M, the B pair.
            vector = m_vectors * tile.n + m_vectors + 1
        return RegisterCount(general=5, vector=vector, mask=int(tile.m % _LANES != 0))

    @override
    def attach_mask(
        self,
        rewriter: PatternRewriter,
        op: MatmulOp,
        *,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        # A lane holds a pair of A and one F32 of C, so C's mask covers both.
        return attach_mask(
            rewriter,
            op,
            tile_size=op.m.value.data,
            vector_size=_LANES,
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
            raise PassFailedException("unsupported CPX BF16 VNNI nano-kernel tile")
        values = values_from_op(op, AVX512RegisterType)
        insert_point = InsertPoint.before(op)
        rewrite = (
            self._rewrite_fsdbcst
            if _m_vectors(tile.m) == 1
            else self._rewrite_nofsdbcst
        )
        result = rewrite(
            rewriter,
            insert_point,
            op,
            tile,
            values,
            vector_register_count=isa_info.register_capacity.vector,
            disable_regalloc=disable_regalloc,
        )
        rewriter.replace(op, [], result.vals)

    def _rewrite_fsdbcst(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulRegOp,
        tile: TileSizes,
        values: MatmulRegValues,
        *,
        vector_register_count: int,
        disable_regalloc: bool,
    ) -> MatmulRegValues:
        """One M vector: every K pair of A meets each B pair as a broadcast
        memory operand, spread over accumulator sets that are summed at the end."""
        ab_size = op.input_type.size
        lda = op.lda.value.data
        k_pairs = tile.k // _PACK
        accumulator_sets = _accumulator_sets(tile)

        def register(index: int):
            return vector_register(
                index, AVX512RegisterType, disable_regalloc=disable_regalloc
            )

        def load_a_pair(pair: int, index: int) -> VectorValue:
            return load_vector(
                rewriter,
                insert_point,
                op.datatype,
                values.a,
                pair * _PACK * lda * ab_size,
                register(index),
                aligned=bool(op.aligned_a.value.data),
                mask=values.mask,
            )

        accumulators = {
            vector_register_count - tile.n + n: accumulator
            for n, accumulator in enumerate(values.accumulators)
        }
        for accumulator_set in range(1, accumulator_sets):
            for n in range(tile.n):
                index = vector_register_count - tile.n * (accumulator_set + 1) + n
                accumulators[index] = zero_vector(
                    rewriter, insert_point, register(index)
                )

        # Two A registers, the next pair loaded while this one is in use.
        a_vectors: dict[int, VectorValue] = {}
        a = values.a
        for pair in range(k_pairs):
            if pair == 0:
                a_vectors[0] = load_a_pair(0, 0)
                if k_pairs > 1:
                    a_vectors[1] = load_a_pair(1, 1)
            elif pair < k_pairs - 1:
                a_vectors[(pair + 1) % 2] = load_a_pair(pair + 1, (pair + 1) % 2)

            if pair == k_pairs - 1:
                a = advance_pointer(rewriter, insert_point, a, tile.k * lda * ab_size)

            for n in range(tile.n):
                index = (
                    vector_register_count - tile.n * ((pair % accumulator_sets) + 1) + n
                )
                accumulators[index] = dot_product_bf16_pairs_memory(
                    rewriter,
                    insert_point,
                    accumulators[index],
                    a_vectors[pair % 2],
                    values.b,
                    (_PACK * pair + n * op.ldb.value.data) * ab_size,
                )

        b = advance_pointer(rewriter, insert_point, values.b, tile.k * ab_size)

        for accumulator_set in range(1, accumulator_sets):
            for n in range(tile.n):
                source = vector_register_count - tile.n * (accumulator_set + 1) + n
                main = vector_register_count - tile.n + n
                add = add_accumulator_vectors if disable_regalloc else add_vectors
                accumulators[main] = add(
                    rewriter,
                    insert_point,
                    op.datatype,
                    accumulators[source],
                    accumulators[main],
                    register(main),
                )

        return MatmulRegValues(
            a,
            b,
            values.mask,
            tuple(
                accumulators[vector_register_count - tile.n + n] for n in range(tile.n)
            ),
        )

    def _rewrite_nofsdbcst(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulRegOp,
        tile: TileSizes,
        values: MatmulRegValues,
        *,
        vector_register_count: int,
        disable_regalloc: bool,
    ) -> MatmulRegValues:
        """Several M vectors: per K pair, load every M vector of A, then
        broadcast each B pair into a register and multiply it by all of them."""
        ab_size = op.input_type.size
        m_vectors = _m_vectors(tile.m)
        accumulators = list(values.accumulators)
        a = values.a
        b = values.b

        def register(index: int):
            return vector_register(
                index, AVX512RegisterType, disable_regalloc=disable_regalloc
            )

        for _ in range(tile.k // _PACK):
            a_pairs = tuple(
                load_vector(
                    rewriter,
                    insert_point,
                    op.datatype,
                    a,
                    m * _LANES * _PACK * ab_size,
                    register(1 + m),
                    aligned=bool(op.aligned_a.value.data),
                    mask=values.mask if m == m_vectors - 1 else None,
                )
                for m in range(m_vectors)
            )
            for n in range(tile.n):
                b_pair = broadcast_dword(
                    rewriter,
                    insert_point,
                    b,
                    op.ldb.value.data * n * ab_size,
                    register(0),
                )
                if n == tile.n - 1:
                    b = advance_pointer(rewriter, insert_point, b, _PACK * ab_size)
                for m in range(m_vectors):
                    if m == 0 and n == tile.n - 1:
                        a = advance_pointer(
                            rewriter,
                            insert_point,
                            a,
                            _PACK * op.lda.value.data * ab_size,
                        )
                    index = m + m_vectors * n
                    accumulators[index] = dot_product_bf16_pairs(
                        rewriter, insert_point, accumulators[index], a_pairs[m], b_pair
                    )

        return MatmulRegValues(a, b, values.mask, tuple(accumulators))
