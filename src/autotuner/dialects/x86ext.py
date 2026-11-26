"""x86 instructions the FFT expansions need that xDSL's x86 dialect lacks.

Instructions with a memory operand address either ``[register + offset]`` or,
without a register, ``[rip + symbol + offset]``: the expansions keep their
constants next to the code that uses them (see ``DataOp``), so they need no
register to address them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from xdsl.backend.assembly_printer import AssemblyPrintable, AssemblyPrinter, reg
from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import (
    I64,
    IntAttr,
    IntegerAttr,
    StringAttr,
    UnitAttr,
    i64,
)
from xdsl.dialects.x86.assembly import memory_access_str
from xdsl.dialects.x86.ops import (
    SI64,
    DM_Operation,
    DSI_Operation,
    DSS_Operation,
    DSSI_Operation,
    MS_Operation,
    X86Instruction,
    si64,
)
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
    SSERegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect
from xdsl.utils.exceptions import VerifyException

Vector = X86VectorRegisterType


class FmaKind(StrEnum):
    """The sign pattern of a fused multiply-add ``±(a * b) ± c``.

    The alternating kinds give the addend opposite signs in even and odd
    lanes, which on interleaved complex numbers are the real and imaginary
    parts: with ``c`` the swap of a complex ``z``, ``a * b + i * z`` is
    ``MADDSUB`` and ``a * b - i * z`` is ``MSUBADD``.
    """

    FMADD = "madd"  # a * b + c
    FMSUB = "msub"  # a * b - c
    FNMADD = "nmadd"  # -(a * b) + c
    FNMSUB = "nmsub"  # -(a * b) - c
    MADDSUB = "maddsub"  # a * b - c in even lanes, a * b + c in odd lanes
    MSUBADD = "msubadd"  # a * b + c in even lanes, a * b - c in odd lanes

    @property
    def alternating(self) -> bool:
        return self in (FmaKind.MADDSUB, FmaKind.MSUBADD)

    @property
    def negates_product(self) -> bool:
        return self in (FmaKind.FNMADD, FmaKind.FNMSUB)

    @property
    def negates_addend(self) -> bool:
        return self in (FmaKind.FMSUB, FmaKind.FNMSUB)

    def negating(self, addend: bool) -> tuple["FmaKind", bool]:
        """The kind computing this FMA with its addend (or else a factor)
        negated, and whether its result must then be negated as well."""
        if self.alternating:
            flipped = FmaKind.MSUBADD if self is FmaKind.MADDSUB else FmaKind.MADDSUB
            # -(a * b) -/+ c = -(a * b +/- c): no FMA negates an alternating product.
            return flipped, not addend
        product, negated_addend = self.negates_product, self.negates_addend
        if addend:
            negated_addend = not negated_addend
        else:
            product = not product
        return FmaKind.of(product, negated_addend), False

    @staticmethod
    def of(negate_product: bool, negate_addend: bool) -> "FmaKind":
        return {
            (False, False): FmaKind.FMADD,
            (False, True): FmaKind.FMSUB,
            (True, False): FmaKind.FNMADD,
            (True, True): FmaKind.FNMSUB,
        }[(negate_product, negate_addend)]


def _lane_suffix(lane_bitwidth: int) -> str:
    return {32: "ps", 64: "pd"}[lane_bitwidth]


@dataclass(frozen=True)
class Address:
    """A memory operand: ``[base + offset]``, or ``[rip + symbol + offset]``
    without a base register."""

    base: SSAValue | None
    offset: int = 0
    symbol: str | None = None

    def __post_init__(self):
        assert (self.base is None) != (self.symbol is None)

    def __add__(self, offset: int) -> "Address":
        return Address(self.base, self.offset + offset, self.symbol)

    @property
    def operands(self) -> list[SSAValue]:
        return [] if self.base is None else [self.base]

    @property
    def attributes(self) -> dict[str, Attribute | None]:
        return {
            "memory_offset": IntegerAttr(self.offset, si64),
            "symbol": None if self.symbol is None else StringAttr(self.symbol),
        }


def _address_str(op: Operation) -> str:
    """The memory operand of ``op``, from its ``memory``, ``symbol``, and
    ``memory_offset``."""
    offset = op.memory_offset.value.data  # pyright: ignore
    memory = op.memory  # pyright: ignore
    if memory is None:
        symbol = op.symbol.data  # pyright: ignore
        return f"[rip + {symbol}{offset:+d}]" if offset else f"[rip + {symbol}]"
    return memory_access_str(memory, op.memory_offset)  # pyright: ignore


def _memory_operand(op: Operation, register: SSAValue, lane_bitwidth: int) -> str:
    address = _address_str(op)
    if op.broadcast is None:  # pyright: ignore
        return address
    register_type = register.type
    assert isinstance(register_type, X86VectorRegisterType)
    return f"{address}{{1to{register_type.bitwidth() // lane_bitwidth}}}"


# region: concrete register-register instructions


class _VectorDSS(DSS_Operation[Vector, Vector, Vector]):
    pass


@irdl_op_definition
class VsubpdOp(_VectorDSS):
    name = "x86ext.dss.vsubpd"


@irdl_op_definition
class VsubpsOp(_VectorDSS):
    name = "x86ext.dss.vsubps"


@irdl_op_definition
class VmulpdOp(_VectorDSS):
    name = "x86ext.dss.vmulpd"


@irdl_op_definition
class VmulpsOp(_VectorDSS):
    name = "x86ext.dss.vmulps"


@irdl_op_definition
class VunpcklpdOp(_VectorDSS):
    """Interleave the even 64-bit lanes of each 128-bit chunk of both sources."""

    name = "x86ext.dss.vunpcklpd"


@irdl_op_definition
class VunpckhpdOp(_VectorDSS):
    """Interleave the odd 64-bit lanes of each 128-bit chunk of both sources."""

    name = "x86ext.dss.vunpckhpd"


@irdl_op_definition
class VunpcklpsOp(_VectorDSS):
    """Interleave the low two 32-bit lanes of each 128-bit chunk of both sources."""

    name = "x86ext.dss.vunpcklps"


@irdl_op_definition
class VunpckhpsOp(_VectorDSS):
    """Interleave the high two 32-bit lanes of each 128-bit chunk of both sources."""

    name = "x86ext.dss.vunpckhps"


class _VectorDSSI(DSSI_Operation[Vector, Vector, Vector]):
    pass


@irdl_op_definition
class VshufpsOp(_VectorDSSI):
    """Per 128-bit chunk, two 32-bit lanes of source0 then two of source1."""

    name = "x86ext.dssi.vshufps"


@irdl_op_definition
class Vshuff64x2Op(_VectorDSSI):
    """Select 128-bit chunks: the lower half of d from source0, the upper from source1."""

    name = "x86ext.dssi.vshuff64x2"


class _VectorDSI(DSI_Operation[Vector, Vector]):
    pass


@irdl_op_definition
class VpermilpdOp(_VectorDSI):
    """Within each 128-bit chunk, lane ``i`` takes the lane bit ``i`` selects."""

    name = "x86ext.dsi.vpermilpd"


@irdl_op_definition
class VpermilpsOp(_VectorDSI):
    """Within each 128-bit chunk, lane ``i`` takes the lane bits ``2i:2i+2`` select."""

    name = "x86ext.dsi.vpermilps"


# endregion

# region: concrete instructions with a memory operand


class _DSMB_Operation(X86Instruction):
    """``d = op(s, [m + offset])``, the memory operand optionally broadcast."""

    destination = result_def(Vector)
    source = operand_def(Vector)
    memory = opt_operand_def(GeneralRegisterType)
    symbol = opt_attr_def(StringAttr)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    broadcast = opt_attr_def(UnitAttr)

    traits = traits_def(MemoryReadEffect())

    @classmethod
    def lane_bitwidth(cls) -> int:
        raise NotImplementedError()

    def __init__(
        self,
        source: SSAValue,
        memory: Address,
        *,
        broadcast: bool = False,
        destination: X86VectorRegisterType,
    ):
        super().__init__(
            operands=[source, memory.operands],
            attributes={
                **memory.attributes,
                "broadcast": UnitAttr() if broadcast else None,
            },
            result_types=[destination],
        )

    def assembly_line_args(self):
        return (
            reg(self.destination),
            reg(self.source),
            _memory_operand(self, self.destination, self.lane_bitwidth()),
        )


@irdl_op_definition
class VaddpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vaddpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VaddpsMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vaddps"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 32


@irdl_op_definition
class VsubpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vsubpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VsubpsMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vsubps"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 32


@irdl_op_definition
class VmulpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vmulpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VmulpsMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vmulps"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 32


@irdl_op_definition
class VxorpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vxorpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VxorpsMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vxorps"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 32


@irdl_op_definition
class VunpcklpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vunpcklpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VunpckhpdMemOp(_DSMB_Operation):
    name = "x86ext.dsm.vunpckhpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


@irdl_op_definition
class VshufpsMemOp(X86Instruction):
    """``vshufps d, s, [m + offset], imm``."""

    name = "x86ext.dsmi.vshufps"

    destination = result_def(Vector)
    source = operand_def(Vector)
    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    immediate = attr_def(IntegerAttr[I64])

    traits = traits_def(MemoryReadEffect())

    assembly_format = (
        "$source `,` `[` $memory (`+` $memory_offset^)? `]` `,` $immediate attr-dict "
        "`:` `(` type($source) `,` type($memory) `)` `->` type($destination)"
    )

    def __init__(
        self,
        source: SSAValue,
        memory: SSAValue,
        memory_offset: int,
        immediate: int,
        *,
        destination: X86VectorRegisterType,
    ):
        super().__init__(
            operands=[source, memory],
            attributes={
                "memory_offset": IntegerAttr(memory_offset, si64),
                "immediate": IntegerAttr(immediate, i64),
            },
            result_types=[destination],
        )

    def assembly_line_args(self):
        return (
            reg(self.destination),
            reg(self.source),
            memory_access_str(self.memory, self.memory_offset),
            self.immediate,
        )


class _PermilMemOp(X86Instruction):
    """``vpermil d, [m], imm``: load and permute within 128-bit chunks."""

    destination = result_def(Vector)
    memory = opt_operand_def(GeneralRegisterType)
    symbol = opt_attr_def(StringAttr)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    immediate = attr_def(IntegerAttr[I64])

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self, memory: Address, immediate: int, *, destination: X86VectorRegisterType
    ):
        super().__init__(
            operands=[memory.operands],
            attributes={**memory.attributes, "immediate": IntegerAttr(immediate, i64)},
            result_types=[destination],
        )

    def assembly_line_args(self):
        return reg(self.destination), _address_str(self), self.immediate


@irdl_op_definition
class DMI_VpermilpdOp(_PermilMemOp):
    name = "x86ext.dmi.vpermilpd"


@irdl_op_definition
class DMI_VpermilpsOp(_PermilMemOp):
    name = "x86ext.dmi.vpermilps"


@irdl_op_definition
class DM_VmovsdOp(DM_Operation[Vector, GeneralRegisterType]):
    """Load one double into the low lane, zeroing the others."""

    name = "x86ext.dm.vmovsd"


@irdl_op_definition
class DM_VmovssOp(DM_Operation[Vector, GeneralRegisterType]):
    """Load one float into the low lane, zeroing the others."""

    name = "x86ext.dm.vmovss"


@irdl_op_definition
class MS_VmovssOp(MS_Operation[GeneralRegisterType, Vector]):
    """Store the low float of a vector."""

    name = "x86ext.ms.vmovss"


# endregion

# region: moving the 128-bit chunks of wider registers
#
# These name a register at another width than its SSA type, as the
# instructions do: xmm0, ymm0, and zmm0 are one register.


def _at_width(value: SSAValue, bits: int) -> str:
    """The name of the register holding ``value`` at width ``bits``."""
    register_type = value.type
    assert isinstance(register_type, X86VectorRegisterType)
    assert isinstance(register_type.index, IntAttr), "registers must be allocated"
    prefix = {128: "x", 256: "y", 512: "z"}[bits]
    return f"{prefix}mm{register_type.index.data}"


def _chunk_mnemonic(prefix: str, value: SSAValue, *operands: SSAValue) -> str:
    """The AVX form for ymm registers that VEX can encode, else AVX-512's."""
    register_type = value.type
    assert isinstance(register_type, X86VectorRegisterType)
    vex = register_type.bitwidth() == 256 and all(
        isinstance(v.type, X86VectorRegisterType)
        and isinstance(v.type.index, IntAttr)
        and v.type.index.data < 16
        for v in (value, *operands)
    )
    return f"{prefix}f128" if vex else f"{prefix}f64x2"


@irdl_op_definition
class InsertChunkOp(X86Instruction):
    """``d = s`` with 128-bit chunk ``index`` replaced by the xmm ``chunk``;
    ``s`` may be narrower than ``d``, its upper chunks being replaced later."""

    name = "x86ext.insert_chunk"

    destination = result_def(Vector)
    source = operand_def(Vector)
    chunk = operand_def(SSERegisterType)
    index = attr_def(IntegerAttr[I64])

    assembly_format = (
        "$source `,` $chunk `,` $index attr-dict `:` "
        "`(` type($source) `,` type($chunk) `)` `->` type($destination)"
    )

    def __init__(
        self,
        source: SSAValue,
        chunk: SSAValue,
        index: int,
        *,
        destination: X86VectorRegisterType,
    ):
        super().__init__(
            operands=[source, chunk],
            attributes={"index": IntegerAttr(index, i64)},
            result_types=[destination],
        )

    def assembly_instruction_name(self) -> str:
        return _chunk_mnemonic("vinsert", self.destination, self.source, self.chunk)

    def assembly_line_args(self):
        bits = self.destination.type.bitwidth()
        return (
            reg(self.destination),
            _at_width(self.source, bits),
            reg(self.chunk),
            self.index,
        )


@irdl_op_definition
class InsertChunkMemOp(X86Instruction):
    """``d = s`` with 128-bit chunk ``index`` loaded from ``[m + offset]``."""

    name = "x86ext.insert_chunk_mem"

    destination = result_def(Vector)
    source = operand_def(Vector)
    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    index = attr_def(IntegerAttr[I64])

    traits = traits_def(MemoryReadEffect())

    assembly_format = (
        "$source `,` `[` $memory (`+` $memory_offset^)? `]` `,` $index attr-dict "
        "`:` `(` type($source) `,` type($memory) `)` `->` type($destination)"
    )

    def __init__(
        self,
        source: SSAValue,
        memory: SSAValue,
        memory_offset: int,
        index: int,
        *,
        destination: X86VectorRegisterType,
    ):
        super().__init__(
            operands=[source, memory],
            attributes={
                "memory_offset": IntegerAttr(memory_offset, si64),
                "index": IntegerAttr(index, i64),
            },
            result_types=[destination],
        )

    def assembly_instruction_name(self) -> str:
        return _chunk_mnemonic("vinsert", self.destination, self.source)

    def assembly_line_args(self):
        bits = self.destination.type.bitwidth()
        return (
            reg(self.destination),
            _at_width(self.source, bits),
            memory_access_str(self.memory, self.memory_offset),
            self.index,
        )


@irdl_op_definition
class ExtractChunkOp(X86Instruction):
    """``d = `` 128-bit chunk ``index`` of ``s``, as an xmm register."""

    name = "x86ext.extract_chunk"

    destination = result_def(SSERegisterType)
    source = operand_def(Vector)
    index = attr_def(IntegerAttr[I64])

    assembly_format = (
        "$source `,` $index attr-dict `:` `(` type($source) `)` `->` type($destination)"
    )

    def __init__(self, source: SSAValue, index: int):
        super().__init__(
            operands=[source],
            attributes={"index": IntegerAttr(index, i64)},
            result_types=[SSERegisterType.unallocated()],
        )

    def assembly_instruction_name(self) -> str:
        return _chunk_mnemonic("vextract", self.source, self.destination)

    def assembly_line_args(self):
        return reg(self.destination), reg(self.source), self.index


@irdl_op_definition
class StoreChunkOp(X86Instruction):
    """Store 128-bit chunk ``index`` of ``s``, or the 64-bit ``half`` of it,
    to ``[m + offset]``."""

    name = "x86ext.store_chunk"

    source = operand_def(Vector)
    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    index = attr_def(IntegerAttr[I64])
    half = opt_attr_def(IntegerAttr[I64])

    traits = traits_def(MemoryWriteEffect())

    assembly_format = (
        "$source `,` `[` $memory (`+` $memory_offset^)? `]` `,` $index attr-dict "
        "`:` `(` type($source) `,` type($memory) `)` `->` `(` `)`"
    )

    def __init__(
        self,
        source: SSAValue,
        memory: SSAValue,
        memory_offset: int,
        index: int,
        half: int | None = None,
    ):
        assert half is None or index == 0, "only the low chunk stores a half"
        super().__init__(
            operands=[source, memory],
            attributes={
                "memory_offset": IntegerAttr(memory_offset, si64),
                "index": IntegerAttr(index, i64),
                "half": None if half is None else IntegerAttr(half, i64),
            },
        )

    def assembly_instruction_name(self) -> str:
        if self.index.value.data:
            return _chunk_mnemonic("vextract", self.source)
        if self.half is None:
            return "vmovupd"
        return "vmovhpd" if self.half.value.data else "vmovsd"

    def assembly_line_args(self):
        memory = memory_access_str(self.memory, self.memory_offset)
        if self.index.value.data:
            return memory, reg(self.source), self.index
        return memory, _at_width(self.source, 128)


@irdl_op_definition
class VmovhpdMemOp(_DSMB_Operation):
    """``d = s`` with its upper 64 bits loaded from memory."""

    name = "x86ext.dsm.vmovhpd"

    @classmethod
    def lane_bitwidth(cls) -> int:
        return 64


# endregion

# region: other instructions


@irdl_op_definition
class LeaRipOp(X86Instruction):
    """``lea d, [rip + symbol]``: the address of a label in this object."""

    name = "x86ext.lea_rip"

    destination = result_def(GeneralRegisterType)
    symbol = attr_def(StringAttr)

    assembly_format = "$symbol attr-dict `:` `(` `)` `->` type($destination)"

    def __init__(self, symbol: str, *, destination: GeneralRegisterType):
        super().__init__(
            attributes={"symbol": StringAttr(symbol)}, result_types=[destination]
        )

    def assembly_instruction_name(self) -> str:
        return "lea"

    def assembly_line_args(self):
        return reg(self.destination), f"[rip + {self.symbol.data}]"


# endregion

# region: fused multiply-add


class _FmaBase(X86Instruction):
    irdl_options = (ParsePropInAttrDict(),)

    kind = prop_def(StringAttr)
    form = prop_def(IntegerAttr[I64])
    lane_bitwidth = prop_def(IntegerAttr[I64])

    def assembly_instruction_name(self) -> str:
        return (
            f"vf{self.kind.data}{self.form.value.data}"
            f"{_lane_suffix(self.lane_bitwidth.value.data)}"
        )

    def verify_(self) -> None:
        FmaKind(self.kind.data)
        if self.form.value.data not in (132, 213, 231):
            raise VerifyException("form must be 132, 213, or 231")


@irdl_op_definition
class FmaOp(_FmaBase):
    """A destructive FMA ``vf<kind><form><pd|ps> r, s1, s2``.

    By form, with ``r`` both an input and the result:

    - 132: ``r = ±(r * s2) ± s1``
    - 213: ``r = ±(s1 * r) ± s2``
    - 231: ``r = ±(s1 * s2) ± r``
    """

    name = "x86ext.fma"

    register_in = operand_def(Vector)
    register_out = result_def(Vector)
    source1 = operand_def(Vector)
    source2 = operand_def(Vector)

    assembly_format = (
        "$register_in `,` $source1 `,` $source2 attr-dict `:` "
        "`(` type($register_in) `,` type($source1) `,` type($source2) `)` "
        "`->` type($register_out)"
    )

    def __init__(
        self,
        register_in: SSAValue,
        source1: SSAValue,
        source2: SSAValue,
        *,
        kind: FmaKind,
        form: int,
        lane_bitwidth: int,
    ):
        super().__init__(
            operands=[register_in, source1, source2],
            properties={
                "kind": StringAttr(kind),
                "form": IntegerAttr(form, i64),
                "lane_bitwidth": IntegerAttr(lane_bitwidth, i64),
            },
            result_types=[register_in.type],
        )

    def assembly_line_args(self):
        return reg(self.register_in), reg(self.source1), reg(self.source2)

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.source2), (), ((self.register_in, self.register_out),)
        )


@irdl_op_definition
class FmaMemOp(_FmaBase):
    """A destructive FMA whose second source is in memory, optionally broadcast.

    By form, with ``r`` both an input and the result:

    - 132: ``r = ±(r * [m]) ± s1``
    - 231: ``r = ±(s1 * [m]) ± r``
    """

    name = "x86ext.fma_mem"

    register_in = operand_def(Vector)
    register_out = result_def(Vector)
    source1 = operand_def(Vector)
    memory = opt_operand_def(GeneralRegisterType)
    symbol = opt_attr_def(StringAttr)
    memory_offset = attr_def(IntegerAttr[SI64], default_value=IntegerAttr(0, si64))
    broadcast = opt_attr_def(UnitAttr)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        register_in: SSAValue,
        source1: SSAValue,
        memory: Address,
        *,
        broadcast: bool,
        kind: FmaKind,
        form: int,
        lane_bitwidth: int,
    ):
        super().__init__(
            operands=[register_in, source1, memory.operands],
            properties={
                "kind": StringAttr(kind),
                "form": IntegerAttr(form, i64),
                "lane_bitwidth": IntegerAttr(lane_bitwidth, i64),
            },
            attributes={
                **memory.attributes,
                "broadcast": UnitAttr() if broadcast else None,
            },
            result_types=[register_in.type],
        )

    def assembly_line_args(self):
        return (
            reg(self.register_in),
            reg(self.source1),
            _memory_operand(self, self.register_in, self.lane_bitwidth.value.data),
        )

    def get_register_constraints(self) -> RegisterConstraints:
        ins = (self.source1,) if self.memory is None else (self.source1, self.memory)
        return RegisterConstraints(ins, (), ((self.register_in, self.register_out),))


# endregion


@irdl_op_definition
class DataOp(IRDLOperation, AssemblyPrintable):
    """A labelled, aligned block of constants in a read-only data section.

    It may sit anywhere, including inside a function: it switches to its
    section and back. A section named ``.rodata.cst<n>`` is mergeable, so the
    linker keeps one copy of each distinct ``n``-byte constant.
    """

    name = "x86ext.data"

    label = attr_def(StringAttr)
    section = attr_def(StringAttr)
    alignment = attr_def(IntegerAttr[I64])
    word_bits = attr_def(IntegerAttr[I64])
    words = attr_def(StringAttr)
    """Space-separated hexadecimal words."""

    assembly_format = "attr-dict"

    def __init__(
        self,
        label: str,
        words: Sequence[int],
        *,
        word_bits: int,
        alignment: int,
        section: str = ".rodata",
    ):
        super().__init__(
            attributes={
                "label": StringAttr(label),
                "section": StringAttr(section),
                "alignment": IntegerAttr(alignment, i64),
                "word_bits": IntegerAttr(word_bits, i64),
                "words": StringAttr(" ".join(f"{word:x}" for word in words)),
            }
        )

    def print_assembly(self, printer: AssemblyPrinter) -> None:
        directive = {32: ".long", 64: ".quad"}[self.word_bits.value.data]
        section = self.section.data
        if section.startswith(".rodata.cst"):
            size = section.removeprefix(".rodata.cst")
            section = f'{section},"aM",@progbits,{size}'
        printer.print_string(f".pushsection {section}\n", indent=0)
        printer.print_string(f".p2align {self.alignment.value.data}\n", indent=0)
        printer.print_string(f"{self.label.data}:\n", indent=0)
        for word in self.words.data.split():
            printer.print_string(f"    {directive} 0x{word}\n", indent=0)
        printer.print_string(".popsection\n", indent=0)


X86EXT = Dialect(
    "x86ext",
    [
        VsubpdOp,
        VsubpsOp,
        VmulpdOp,
        VmulpsOp,
        VunpcklpdOp,
        VunpckhpdOp,
        VunpcklpsOp,
        VunpckhpsOp,
        VshufpsOp,
        Vshuff64x2Op,
        VpermilpdOp,
        VpermilpsOp,
        VaddpdMemOp,
        VaddpsMemOp,
        VsubpdMemOp,
        VsubpsMemOp,
        VmulpdMemOp,
        VmulpsMemOp,
        VxorpdMemOp,
        VxorpsMemOp,
        VunpcklpdMemOp,
        VunpckhpdMemOp,
        VshufpsMemOp,
        DMI_VpermilpdOp,
        DMI_VpermilpsOp,
        InsertChunkOp,
        InsertChunkMemOp,
        ExtractChunkOp,
        StoreChunkOp,
        VmovhpdMemOp,
        DM_VmovsdOp,
        DM_VmovssOp,
        MS_VmovssOp,
        LeaRipOp,
        FmaOp,
        FmaMemOp,
        DataOp,
    ],
    [],
)
