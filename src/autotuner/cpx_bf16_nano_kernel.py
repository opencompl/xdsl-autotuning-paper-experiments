"""LIBXSMM's flat BF16 nano-kernel for CPX and later: BF16 A and B, F32 C.

``vdpbf16ps`` multiplies BF16 pairs and adds both products to one F32 lane, so
each instruction consumes two K steps. A is column-major, so the two K rows of a
pair are ``lda`` apart: the kernel loads both, 32 BF16 values of M at a time, and
interleaves them word by word into the (k, k+1) pairs ``vdpbf16ps`` reads. B is
column-major too, so a pair of B is contiguous and one ``vpbroadcastd`` fetches
it.

The interleave works within 128-bit lanes, which leaves the F32 accumulators of
a 32-wide M chunk with their 128-bit lanes out of M order. The accumulators
therefore start at zero, and only when the K reduction is done are they
shuffled back into M order and added to C.
"""

from collections.abc import Sequence

from typing_extensions import override
from xdsl import ir
from xdsl.dialects import builtin, x86
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
    MaskValue,
    PointerValue,
    VectorValue,
    add_vectors,
    advance_pointer,
    broadcast_dword,
    dot_product_bf16_pairs,
    interleave_words,
    load_vector,
    load_words,
    shuffle_128bit_lanes,
    store_vector,
    zero_vector,
)
from autotuner.nano_kernel import (
    CAccess,
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
    descriptor_from_op,
    tile_sizes_from_op,
    vector_register,
)

# F32 lanes of C per zmm, and BF16 lanes of A per zmm.
_C_LANES = 16
_A_LANES = 32
# LIBXSMM keeps eight registers off the accumulators: four A loads, the four
# interleaved A vectors, and the B broadcast reusing a dead load.
_RESERVED_VECTORS = 8
_MAX_M_VECTORS = 4
_MAX_N = 28
# k3, as LIBXSMM hard-codes it, for the mask of A's lanes.
_A_MASK_REGISTER = x86.registers.AVX512MaskRegisterType.from_index(3)


def _m_vectors(m: int) -> int:
    return (m + _C_LANES - 1) // _C_LANES


class CpxBf16NanoKernel(NanoKernel):
    """LIBXSMM's flat BF16 nano-kernel (``l_is_ABF16_BBF16_flat_cpx_gemm``)."""

    @property
    def name(self) -> str:
        return "libxsmm-cpx-bf16"

    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        vectors = isa_info.register_capacity.vector - _RESERVED_VECTORS
        return frozenset(
            SupportedTile(m, n)
            for m_vectors in range(1, _MAX_M_VECTORS + 1)
            for m in range((m_vectors - 1) * _C_LANES + 1, m_vectors * _C_LANES + 1)
            for n in range(1, min(_MAX_N, vectors // m_vectors) + 1)
        )

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        # The K pairs are what make LIBXSMM's odd-K kernels wrong; refuse them.
        return (
            isa_info.isa == "avx512"
            and isinstance(descriptor.datatype, builtin.Float32Type)
            and isinstance(descriptor.ab_datatype, builtin.BFloat16Type)
            and descriptor.k % 2 == 0
        )

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self.supports(descriptor, isa_info):
            return False
        if tile.m <= 0 or tile.n <= 0 or tile.k <= 0 or tile.k % 2:
            return False
        if _m_vectors(tile.m) > _MAX_M_VECTORS or tile.n > _MAX_N:
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
        return RegisterCount(
            general=5,
            vector=_m_vectors(tile.m) * tile.n + _RESERVED_VECTORS,
            mask=int(tile.m % _C_LANES != 0) + int(tile.m % _A_LANES != 0),
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
        # C's tail mask first, in F32 lanes; `xsmm-apply-schedule` expects it
        # there. Then A's, in BF16 lanes: A is loaded 32 values of M at a time.
        op = attach_mask(
            rewriter,
            op,
            tile_size=op.m.value.data,
            vector_size=_C_LANES,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=mask_reg,
        )
        return attach_mask(
            rewriter,
            op,
            tile_size=op.m.value.data,
            vector_size=_A_LANES,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=_A_MASK_REGISTER,
            lane_type=builtin.bf16,
        )

    @override
    def load_accumulators(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulOp,
        c: PointerValue,
        accesses: Sequence[CAccess],
        registers: Sequence[X86VectorRegisterType],
        *,
        disable_regalloc: bool,
    ) -> tuple[VectorValue, ...]:
        # C joins in `store_accumulators`, once the lanes are back in order.
        return tuple(
            zero_vector(rewriter, insert_point, register) for register in registers
        )

    @override
    def store_accumulators(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulOp,
        c: PointerValue,
        accesses: Sequence[CAccess],
        accumulators: Sequence[VectorValue],
        *,
        disable_regalloc: bool,
    ) -> None:
        m_vectors = _m_vectors(op.m.value.data)
        accumulators = list(accumulators)

        def temporary(index: int) -> X86VectorRegisterType:
            return vector_register(
                index, AVX512RegisterType, disable_regalloc=disable_regalloc
            )

        def shuffle(lhs: VectorValue, rhs: VectorValue, selector: int, destination):
            return shuffle_128bit_lanes(
                rewriter, insert_point, lhs, rhs, selector, destination
            )

        # Interleaving a 32-wide M chunk puts M 0-3 in the first 128-bit lane
        # of its low accumulator, M 4-7 in the first of its high one, M 8-11 in
        # the second of the low one, and so on: gather each half back in order.
        for n in range(op.n.value.data):
            for m in range(0, m_vectors - 1, 2):
                lo, hi = n * m_vectors + m, n * m_vectors + m + 1
                low_lanes = shuffle(
                    accumulators[lo], accumulators[hi], 0x44, temporary(0)
                )
                high_lanes = shuffle(
                    accumulators[lo], accumulators[hi], 0xEE, temporary(1)
                )
                accumulators[lo] = shuffle(
                    low_lanes, low_lanes, 0xD8, accumulators[lo].type
                )
                accumulators[hi] = shuffle(
                    high_lanes, high_lanes, 0xD8, accumulators[hi].type
                )
            if m_vectors % 2:
                last = n * m_vectors + m_vectors - 1
                accumulators[last] = shuffle(
                    accumulators[last],
                    accumulators[last],
                    0xD8,
                    accumulators[last].type,
                )

        # The register below the accumulators holds C on its way in.
        c_register = temporary(32 - len(accesses) - 1)
        for index, (offset, mask) in enumerate(accesses):
            c_in = load_vector(
                rewriter,
                insert_point,
                op.datatype,
                c,
                offset,
                c_register,
                aligned=bool(op.aligned_c),
                mask=mask,
            )
            accumulators[index] = add_vectors(
                rewriter,
                insert_point,
                op.datatype,
                accumulators[index],
                c_in,
                accumulators[index].type,
            )
        for (offset, mask), accumulator in zip(accesses, accumulators, strict=True):
            store_vector(
                rewriter,
                insert_point,
                op.datatype,
                c,
                offset,
                accumulator,
                aligned=bool(op.aligned_c),
                mask=mask,
            )

    def _load_a(
        self,
        rewriter: PatternRewriter,
        insert_point: InsertPoint,
        op: MatmulRegOp,
        a: PointerValue,
        a_mask: MaskValue | None,
        *,
        disable_regalloc: bool,
    ) -> list[VectorValue]:
        """Load the next K pair of A and return it as the M vectors of BF16
        pairs ``vdpbf16ps`` reads, in the lane order `store_accumulators` undoes."""
        m_vectors = _m_vectors(op.m.value.data)
        chunks = (m_vectors + 1) // 2
        ab_size = op.input_type.size

        def register(index: int) -> X86VectorRegisterType:
            return vector_register(
                index, AVX512RegisterType, disable_regalloc=disable_regalloc
            )

        # LIBXSMM's registers: K row k of chunk c in (zmm0, zmm6, zmm5, zmm7).
        load_registers = ((0, 6), (5, 7))
        rows = [
            tuple(
                load_words(
                    rewriter,
                    insert_point,
                    a,
                    (chunk * _A_LANES + k * op.lda.value.data) * ab_size,
                    register(load_registers[chunk][k]),
                    # Only the last chunk can be short. LIBXSMM loads a chunk of
                    # at most 16 values as a ymm; the masked zmm reads no more
                    # memory and leaves the same zeros in the upper half.
                    mask=a_mask if chunk == chunks - 1 else None,
                )
                for k in range(2)
            )
            for chunk in range(chunks)
        ]

        pairs = [
            interleave_words(
                rewriter,
                insert_point,
                rows[index // 2][0],
                rows[index // 2][1],
                register(1 + index),
                high=bool(index % 2),
            )
            for index in range(2 * chunks)
        ]
        if m_vectors % 2:
            # The last chunk has at most 16 values of M, all in the low ymm of
            # both halves of its interleave: pack them into one vector, the way
            # LIBXSMM's `vinserti32x8` does.
            low = m_vectors - 1
            pairs[low] = shuffle_128bit_lanes(
                rewriter,
                insert_point,
                pairs[low],
                pairs[low + 1],
                0x44,
                register(1 + low),
            )
        return pairs[:m_vectors]

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
            raise PassFailedException("unsupported CPX BF16 nano-kernel tile")

        m_vectors = _m_vectors(tile.m)
        has_c_mask = tile.m % _C_LANES != 0
        has_a_mask = tile.m % _A_LANES != 0
        if len(op.ins) != has_c_mask + has_a_mask:
            raise PassFailedException(
                "CPX BF16 matmul_reg expects a mask in for each of a partial C "
                "vector and a partial A chunk"
            )
        if len(op.outs) != m_vectors * tile.n:
            raise PassFailedException(
                f"CPX BF16 matmul_reg expected {m_vectors * tile.n} accumulator "
                f"outs, got {len(op.outs)}"
            )
        a_mask = (
            ir.SSAValue.get(op.ins[-1], type=x86.registers.AVX512MaskRegisterType)
            if has_a_mask
            else None
        )

        insert_point = InsertPoint.before(op)
        ab_size = op.input_type.size
        a = ir.SSAValue.get(op.a, type=GeneralRegisterType)
        b = ir.SSAValue.get(op.b, type=GeneralRegisterType)
        accumulators = [
            ir.SSAValue.get(out, type=X86VectorRegisterType) for out in op.outs
        ]

        for _ in range(tile.k // 2):
            a_pairs = self._load_a(
                rewriter,
                insert_point,
                op,
                a,
                a_mask,
                disable_regalloc=disable_regalloc,
            )
            for n in range(tile.n):
                b_pair = broadcast_dword(
                    rewriter,
                    insert_point,
                    b,
                    op.ldb.value.data * n * ab_size,
                    vector_register(
                        0, AVX512RegisterType, disable_regalloc=disable_regalloc
                    ),
                )
                if n == tile.n - 1:
                    b = advance_pointer(rewriter, insert_point, b, 2 * ab_size)
                for m in range(m_vectors):
                    if m == 0 and n == tile.n - 1:
                        a = advance_pointer(
                            rewriter,
                            insert_point,
                            a,
                            2 * op.lda.value.data * ab_size,
                        )
                    index = m + m_vectors * n
                    accumulators[index] = dot_product_bf16_pairs(
                        rewriter,
                        insert_point,
                        accumulators[index],
                        a_pairs[m],
                        b_pair,
                    )

        rewriter.replace(op, [], (a, b, *accumulators))
