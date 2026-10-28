"""LIBXSMM's AMX BF16 nano-kernel for SPR and later, with A packed in K pairs.

An AMX tile is up to 16 rows of up to 64 bytes. LIBXSMM lays column-major C
onto tiles one column per row, so a C tile is up to 16 columns of 16 F32 rows
of M. ``tdpbf16ps c, b, a`` adds to row n of ``c`` the products of row n of
``b`` -- a column of B, up to 32 BF16 of K -- with ``a``, whose rows are K
pairs of A's VNNI-2 layout, 16 M rows of pairs each. So the A tiles load
straight from A packed in K pairs, the B tiles straight from column-major B,
and neither is reshuffled.

A block of C spans at most two tiles in each of M and N -- or, when N fits one
tile, up to four in M -- and so at most four accumulator tiles, which leaves
four tile registers for A and B. The shapes of all eight are fixed by a tile
configuration loaded ahead of the block, so the accumulators and operands live
in the registers LIBXSMM gives them, `tmm0` to `tmm7`, whoever allocates the
general-purpose registers. K is unrolled completely, in steps of up to 32.
"""

from dataclasses import dataclass
from typing import Literal, cast

from typing_extensions import override
from xdsl import ir
from xdsl.dialects import builtin, x86
from xdsl.dialects.x86.registers import (
    AMXTileRegisterType,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.amx import MAX_TILE_ROWS, TILE_REGISTERS, TileConfig, TileShape
from autotuner.dialects.xsmm import MatmulOp, MatmulRegOp
from autotuner.instructions import PointerValue, advance_pointer
from autotuner.nano_kernel import (
    FloatingPointType,
    GemmDescriptor,
    ISAInfo,
    NanoKernel,
    RegisterCount,
    SupportedTile,
    TileSizes,
)
from autotuner.skx_nano_kernel_utils import descriptor_from_op, tile_sizes_from_op

TileValue = ir.SSAValue[AMXTileRegisterType]

# F32 of C in one tile row: the M extent of a tile.
_C_LANES = 16
# K steps per element of packed A: libxsmm_cpuid_dot_pack_factor(BF16).
_PACK = 2
# BF16 of B in one tile row: the most K one `tdpbf16ps` consumes.
_MAX_K_BLOCKING = 32
# The tile registers of A and of B; the accumulators take those below.
_A_REGISTERS = (4, 5)
_B_REGISTERS = (6, 7)
_BF16_SIZE = 2
_F32_SIZE = 4


class AMXInfo(ISAInfo):
    """AVX-512's register file plus the eight AMX tile registers."""

    @property
    def isa(self) -> Literal["avx512"]:
        return "avx512"

    @property
    def register_capacity(self) -> RegisterCount:
        return RegisterCount(general=16, vector=32, mask=8, tile=TILE_REGISTERS)

    @property
    def vector_type(self) -> type[X86VectorRegisterType]:
        return AVX512RegisterType


def k_blocking(k: int) -> int:
    """The K one tile multiplication covers: LIBXSMM's largest even divisor of
    K up to a full tile row of B."""
    blocking = _MAX_K_BLOCKING
    while k % blocking:
        blocking -= _PACK
    return blocking


def _largest_divisor(extent: int, bound: int) -> int:
    return next(d for d in range(min(extent, bound), 0, -1) if extent % d == 0)


@dataclass(frozen=True)
class AmxTileLayout:
    """How a block of C is cut into tiles, and the registers they live in.

    The M tiles are full tiles followed by what is left; the N tiles split the
    block in two halves once it is more than a tile wide. With one N tile the
    accumulators of the M tiles are `tmm0` up, which is what LIBXSMM does for an
    N of at most 16 ("Mx1" blocking); LIBXSMM numbers the two accumulators of a
    one-tile-wide block of a wider N `tmm0` and `tmm2` instead, a renaming.
    """

    m_tiles: tuple[int, ...]
    n_tiles: tuple[int, ...]
    k_blocking: int

    @classmethod
    def for_block(cls, m: int, n: int, k: int) -> "AmxTileLayout":
        if not 0 < n <= 2 * MAX_TILE_ROWS:
            raise ValueError(f"an AMX block spans at most two tiles of N, not {n}")
        n_tiles = (n,) if n <= MAX_TILE_ROWS else ((n + 1) // 2, n // 2)
        max_m_tiles = (TILE_REGISTERS - len(_A_REGISTERS) - len(_B_REGISTERS)) // len(
            n_tiles
        )
        if not 0 < m <= max_m_tiles * _C_LANES:
            raise ValueError(f"an AMX block of N {n} spans at most {max_m_tiles} tiles")
        full, rest = divmod(m, _C_LANES)
        m_tiles = (_C_LANES,) * full + ((rest,) if rest else ())
        if k <= 0 or k % _PACK:
            raise ValueError(f"an AMX BF16 block needs an even K, not {k}")
        return cls(m_tiles, n_tiles, k_blocking(k))

    def m_offset(self, im: int) -> int:
        return im * _C_LANES

    def n_offset(self, in_: int) -> int:
        return sum(self.n_tiles[:in_])

    def c_register(self, im: int, in_: int) -> int:
        return im * len(self.n_tiles) + in_

    def a_register(self, im: int) -> int:
        """The last of several M tiles of A goes to `tmm5`; the others share
        `tmm4`, each loaded just before its tiles of C use it."""
        last = len(self.m_tiles) > 1 and im == len(self.m_tiles) - 1
        return _A_REGISTERS[last]

    def b_register(self, in_: int) -> int:
        return _B_REGISTERS[in_]

    @property
    def c_tiles(self) -> tuple[tuple[int, int], ...]:
        """The (M tile, N tile) of every accumulator, in register order."""
        return tuple(
            (im, in_)
            for im in range(len(self.m_tiles))
            for in_ in range(len(self.n_tiles))
        )

    def tile_config(self) -> TileConfig:
        """LIBXSMM's configuration: every tile 16 rows of 64 bytes unless the
        block uses it, a C tile N rows of M F32, an A tile K/2 rows of M pairs,
        a B tile N rows of K BF16."""
        tiles: list[TileShape | None] = [
            TileShape(MAX_TILE_ROWS, 4 * _C_LANES)
        ] * TILE_REGISTERS
        for im, in_ in self.c_tiles:
            tiles[self.c_register(im, in_)] = TileShape(
                self.n_tiles[in_], _F32_SIZE * self.m_tiles[im]
            )
        for im, m_size in enumerate(self.m_tiles):
            tiles[self.a_register(im)] = TileShape(
                self.k_blocking // _PACK, _PACK * _BF16_SIZE * m_size
            )
        for in_, n_size in enumerate(self.n_tiles):
            tiles[self.b_register(in_)] = TileShape(
                n_size, _BF16_SIZE * self.k_blocking
            )
        return TileConfig(tuple(tiles))


@dataclass(frozen=True)
class AmxCodeBlock:
    """A run of M rows LIBXSMM generates with one tile configuration."""

    m: int
    m_blocking: int
    n_blocking: int
    k_blocking: int

    def layout(self) -> AmxTileLayout:
        return AmxTileLayout.for_block(
            self.m_blocking, self.n_blocking, self.k_blocking
        )


def amx_code_blocks(m: int, n: int, k: int) -> tuple[AmxCodeBlock, ...]:
    """LIBXSMM's AMX blocking of an M-by-N-by-K GEMM.

    A block's tiles must divide it exactly, since their shapes are fixed for the
    whole block, so M and N are blocked by divisors; an M that has no good one
    is cut into a block LIBXSMM's tiles do divide and a remainder block of its
    own, with its own configuration. An N of one tile ("Mx1") makes room for
    four tiles of M instead of two.
    """
    mx1 = n <= MAX_TILE_ROWS
    max_m_blocking = (4 if mx1 else 2) * _C_LANES
    if m % _C_LANES == 0 or m <= max_m_blocking:
        parts = (m,)
    else:
        first = m - m % max_m_blocking
        parts = (first, m - first)
    return tuple(
        AmxCodeBlock(
            part,
            _largest_divisor(part, max_m_blocking),
            _largest_divisor(n, MAX_TILE_ROWS if mx1 else 2 * MAX_TILE_ROWS),
            k_blocking(k),
        )
        for part in parts
    )


def _tile_register(index: int) -> AMXTileRegisterType:
    return AMXTileRegisterType.from_index(index)


def _load_tile(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    pointer: PointerValue,
    stride: PointerValue,
    offset: int,
    scale: int,
    register: int,
) -> TileValue:
    result = rewriter.insert(
        x86.ops.DMX_TileloaddOp(
            pointer,
            stride,
            offset,
            scale=scale,
            destination=_tile_register(register),
        ),
        insertion_point=insert_point,
    ).destination
    return cast(TileValue, result)


class SprAmxBf16NanoKernel(NanoKernel):
    """LIBXSMM's AMX nano-kernel for BF16 A in VNNI-2 layout and flat B,
    accumulating into F32 C in tiles.

    Unlike the vector nano-kernels it is not handed to the generic tiling: the
    AMX schedule blocks the GEMM with `amx_code_blocks`, loads each block's
    `tile_config`, and calls `accumulate_in_tiles` and `rewrite` on its blocks.
    Its operations carry the row strides tile loads need as `ins`: A, B and C's
    on a `xsmm.matmul`, A and B's on a `xsmm.matmul_reg`.
    """

    @property
    def name(self) -> str:
        return "libxsmm-spr-bf16-amx"

    @property
    @override
    def k_pack_factor(self) -> int:
        return _PACK

    def supported_tile_sizes(
        self,
        datatype: FloatingPointType,
        isa_info: ISAInfo,
    ) -> frozenset[SupportedTile]:
        return frozenset(
            SupportedTile(m, n)
            for n in range(1, 2 * MAX_TILE_ROWS + 1)
            for m in range(1, (4 if n <= MAX_TILE_ROWS else 2) * _C_LANES + 1)
        )

    def supports(self, descriptor: GemmDescriptor, isa_info: ISAInfo) -> bool:
        return (
            isa_info.isa == "avx512"
            and isa_info.register_capacity.tile >= TILE_REGISTERS
            and isinstance(descriptor.datatype, builtin.Float32Type)
            and isinstance(descriptor.ab_datatype, builtin.BFloat16Type)
            and descriptor.vnni_a
            and descriptor.k % _PACK == 0
        )

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> bool:
        if not self.supports(descriptor, isa_info):
            return False
        try:
            AmxTileLayout.for_block(tile.m, tile.n, tile.k)
        except ValueError:
            return False
        return True

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
    ) -> RegisterCount:
        if not self.supports_tile(descriptor, tile, isa_info):
            raise ValueError("unsupported SPR AMX BF16 nano-kernel tile")
        # A, B and the strides of A and B; every tile register is configured.
        return RegisterCount(general=4, tile=TILE_REGISTERS)

    @override
    def attach_mask(
        self,
        rewriter: PatternRewriter,
        op: MatmulOp,
        *,
        mask_tmp_reg: GeneralRegisterType,
        mask_reg: AVX512MaskRegisterType,
    ) -> MatmulOp:
        # Tiles have exactly the rows and bytes of their part of C: no mask.
        return op

    def accumulate_in_tiles(
        self, rewriter: PatternRewriter, op: MatmulOp
    ) -> MatmulRegOp:
        """Replace a K-iterating block of C with its accumulator tiles: load
        them, multiply into them with a `xsmm.matmul_reg`, store them."""
        if len(op.ins) != 3 or op.outs:
            raise PassFailedException(
                "the AMX nano-kernel expects the strides of A, B and C as ins"
            )
        lda, ldb, ldc = (
            ir.SSAValue.get(value, type=GeneralRegisterType) for value in op.ins
        )
        c = ir.SSAValue.get(op.c, type=GeneralRegisterType)
        layout = AmxTileLayout.for_block(
            op.m.value.data, op.n.value.data, op.k.value.data
        )
        ldc_bytes = op.ldc.value.data * _F32_SIZE
        insert_point = InsertPoint.before(op)

        def c_offset(im: int, in_: int) -> int:
            return layout.n_offset(in_) * ldc_bytes + layout.m_offset(im) * _F32_SIZE

        accumulators = tuple(
            _load_tile(
                rewriter,
                insert_point,
                c,
                ldc,
                c_offset(im, in_),
                _F32_SIZE,
                layout.c_register(im, in_),
            )
            for im, in_ in layout.c_tiles
        )
        matmul_reg = rewriter.insert(
            MatmulRegOp(
                op.a,
                op.b,
                (lda, ldb),
                accumulators,
                m=op.m.value.data,
                n=op.n.value.data,
                k=op.k.value.data,
                lda=op.lda.value.data,
                ldb=op.ldb.value.data,
                datatype=op.datatype,
                ab_datatype=op.ab_datatype,
                vnni_a=bool(op.vnni_a),
                aligned_a=bool(op.aligned_a),
            ),
            insertion_point=insert_point,
        )
        # LIBXSMM stores the tiles of one N tile after another.
        results = dict(zip(layout.c_tiles, matmul_reg.out_results, strict=True))
        for in_ in range(len(layout.n_tiles)):
            for im in range(len(layout.m_tiles)):
                rewriter.insert(
                    x86.ops.MXS_TilestoredOp(
                        c,
                        ldc,
                        results[im, in_],
                        c_offset(im, in_),
                        scale=_F32_SIZE,
                    ),
                    insertion_point=insert_point,
                )
        rewriter.replace(op, [], (matmul_reg.a_out, matmul_reg.b_out, op.c))
        return matmul_reg

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
            raise PassFailedException("unsupported SPR AMX BF16 nano-kernel tile")
        if len(op.ins) != 2:
            raise PassFailedException(
                "the AMX nano-kernel expects the strides of A and B as ins"
            )
        layout = AmxTileLayout.for_block(tile.m, tile.n, tile.k)
        if len(op.outs) != len(layout.c_tiles):
            raise PassFailedException(
                f"the AMX nano-kernel expects {len(layout.c_tiles)} accumulator "
                f"tiles, got {len(op.outs)}"
            )
        a, b, lda, ldb = (
            ir.SSAValue.get(value, type=GeneralRegisterType)
            for value in (op.a, op.b, *op.ins)
        )
        accumulators: dict[tuple[int, int], TileValue] = {
            position: cast(TileValue, value)
            for position, value in zip(layout.c_tiles, op.outs, strict=True)
        }
        insert_point = InsertPoint.before(op)
        lda_bytes = op.lda.value.data * _BF16_SIZE
        ldb_bytes = op.ldb.value.data * _BF16_SIZE

        for step in range(tile.k // layout.k_blocking):
            k = step * layout.k_blocking
            a_tiles: dict[int, TileValue] = {}
            b_tiles: dict[int, TileValue] = {}
            for in_ in range(len(layout.n_tiles)):
                for im in range(len(layout.m_tiles)):
                    if in_ == 0:
                        # Row p of an A tile is K pair k/2 + p, 2 * lda BF16 on.
                        a_tiles[im] = _load_tile(
                            rewriter,
                            insert_point,
                            a,
                            lda,
                            k * lda_bytes + layout.m_offset(im) * _PACK * _BF16_SIZE,
                            _PACK * _BF16_SIZE,
                            layout.a_register(im),
                        )
                    if im == 0:
                        b_tiles[in_] = _load_tile(
                            rewriter,
                            insert_point,
                            b,
                            ldb,
                            layout.n_offset(in_) * ldb_bytes + k * _BF16_SIZE,
                            _BF16_SIZE,
                            layout.b_register(in_),
                        )
                    accumulators[im, in_] = cast(
                        TileValue,
                        rewriter.insert(
                            x86.ops.RSS_Tdpbf16psOp(
                                accumulators[im, in_], b_tiles[in_], a_tiles[im]
                            ),
                            insertion_point=insert_point,
                        ).register_out,
                    )

        a = advance_pointer(rewriter, insert_point, a, tile.k * lda_bytes)
        b = advance_pointer(rewriter, insert_point, b, tile.k * _BF16_SIZE)
        rewriter.replace(
            op, [], (a, b, *(accumulators[position] for position in layout.c_tiles))
        )
