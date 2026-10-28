from collections.abc import Sequence
from enum import StrEnum

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import (
    BFloat16Type,
    BoolAttr,
    Float32Type,
    Float64Type,
    IntegerAttr,
    StringAttr,
    UnitAttr,
    i64,
)
from xdsl.dialects.x86.ops import DSS_Operation
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
    X86RegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect
from xdsl.utils.exceptions import VerifyException


class MatmulIterator(StrEnum):
    NONE = "none"
    M = "m"
    N = "n"
    K = "k"


def _parse_variadic_operands(parser: Parser, keyword: str) -> tuple[SSAValue, ...]:
    if not parser.parse_optional_keyword(keyword):
        return ()

    operands_pos = parser.pos
    parser.parse_punctuation("(")
    unresolved_operands = parser.parse_comma_separated_list(
        Parser.Delimiter.NONE, parser.parse_unresolved_operand
    )
    parser.parse_punctuation(":")
    operand_types = parser.parse_comma_separated_list(
        Parser.Delimiter.NONE, parser.parse_type
    )
    parser.parse_punctuation(")")
    return tuple(
        parser.resolve_operands(unresolved_operands, operand_types, operands_pos)
    )


def _print_variadic_operands(
    printer: Printer, keyword: str, operands: Sequence[SSAValue]
) -> None:
    if not operands:
        return

    printer.print_string(f" {keyword}(")
    printer.print_list(operands, printer.print_ssa_value)
    printer.print_string(" : ")
    printer.print_list(operands, lambda operand: printer.print_attribute(operand.type))
    printer.print_string(")")


class AccumulatorAddOp(
    DSS_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """A vector add whose destination reuses the accumulator source register."""

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1,), (), ((self.source2, self.destination),)
        )


@irdl_op_definition
class AccumulatorAddpdOp(AccumulatorAddOp):
    name = "xsmm.accumulator.vaddpd"

    def assembly_instruction_name(self) -> str:
        return "vaddpd"


@irdl_op_definition
class AccumulatorAddpsOp(AccumulatorAddOp):
    name = "xsmm.accumulator.vaddps"

    def assembly_instruction_name(self) -> str:
        return "vaddps"


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """A blocked matrix multiplication over complete M, N, and K extents.

    ``iterator`` describes the pointer-result contract of the operation for the
    currently supported non-transposed inputs:

    - ``n`` leaves A unchanged and advances B by ``n * ldb`` and C by
      ``n * ldc`` elements;
    - ``m`` advances A and C by ``m`` elements and leaves B unchanged;
    - ``k`` advances A by ``k * lda`` and B by ``k`` elements and leaves C
      unchanged;
    - ``none`` passes all three matrix pointers through unchanged.

    ``ins`` are additional read-only registers. ``outs`` are additional
    loop-carried registers and have pairwise matching ``out_results``. The
    current schedule uses ``ins`` for optional AVX-512 masks.

    ``datatype`` is the element type of the accumulation, and of C unless
    ``c_datatype`` is set (a BF16 C the F32 accumulation is rounded back to); A
    and B have ``ab_datatype`` when it is set (BF16 inputs accumulating in F32),
    and ``datatype`` otherwise. Pointer advances count elements of each
    matrix's own type.

    ``vnni_a`` says A is packed in LIBXSMM's VNNI-2 layout: each K pair of A is
    stored as ``lda`` adjacent (k, k+1) element pairs, so element (m, k) is at
    ``(k // 2) * 2 * lda + 2 * m + k % 2``. K steps still count elements, so
    advancing K by ``k`` still moves A by ``k * lda``; advancing M by ``m``
    moves it by ``2 * m``.
    """

    name = "xsmm.matmul"

    a = operand_def(GeneralRegisterType)
    b = operand_def(GeneralRegisterType)
    c = operand_def(GeneralRegisterType)
    ins = var_operand_def(X86RegisterType)
    outs = var_operand_def(X86RegisterType)

    a_out = result_def(GeneralRegisterType)
    b_out = result_def(GeneralRegisterType)
    c_out = result_def(GeneralRegisterType)
    out_results = var_result_def(X86RegisterType)

    m = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    lda = prop_def(IntegerAttr)
    ldb = prop_def(IntegerAttr)
    ldc = prop_def(IntegerAttr)
    datatype = prop_def(Float32Type | Float64Type)
    ab_datatype = opt_prop_def(BFloat16Type)
    c_datatype = opt_prop_def(BFloat16Type)
    vnni_a = opt_prop_def(UnitAttr)
    aligned_a = prop_def(BoolAttr)
    aligned_c = prop_def(BoolAttr)
    iterator = prop_def(StringAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

    @property
    def input_type(self) -> Float32Type | Float64Type | BFloat16Type:
        """The element type of A and B: ``ab_datatype``, else ``datatype``."""
        return self.datatype if self.ab_datatype is None else self.ab_datatype

    @property
    def output_type(self) -> Float32Type | Float64Type | BFloat16Type:
        """The element type of C: ``c_datatype``, else ``datatype``."""
        return self.datatype if self.c_datatype is None else self.c_datatype

    @property
    def a_vector_length(self) -> int:
        """How many M rows of A one 512-bit vector holds: a K pair of each
        when A is in VNNI layout, one element otherwise."""
        return 512 // ((1 + bool(self.vnni_a)) * self.input_type.bitwidth)

    def __init__(
        self,
        a: SSAValue,
        b: SSAValue,
        c: SSAValue,
        ins: Sequence[SSAValue] = (),
        outs: Sequence[SSAValue] = (),
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        ldc: int,
        datatype: Float32Type | Float64Type,
        ab_datatype: BFloat16Type | None = None,
        c_datatype: BFloat16Type | None = None,
        vnni_a: bool = False,
        aligned_a: bool,
        aligned_c: bool,
        iterator: MatmulIterator,
    ):
        super().__init__(
            operands=(a, b, c, ins, outs),
            result_types=(
                a.type,
                b.type,
                c.type,
                tuple(out.type for out in outs),
            ),
            properties={
                "m": IntegerAttr(m, i64),
                "n": IntegerAttr(n, i64),
                "k": IntegerAttr(k, i64),
                "lda": IntegerAttr(lda, i64),
                "ldb": IntegerAttr(ldb, i64),
                "ldc": IntegerAttr(ldc, i64),
                "datatype": datatype,
                "aligned_a": BoolAttr.from_bool(aligned_a),
                "aligned_c": BoolAttr.from_bool(aligned_c),
                "iterator": StringAttr(iterator),
            }
            | ({} if ab_datatype is None else {"ab_datatype": ab_datatype})
            | ({} if c_datatype is None else {"c_datatype": c_datatype})
            | ({"vnni_a": UnitAttr()} if vnni_a else {}),
        )

    @classmethod
    def parse(cls, parser: Parser) -> "MatmulOp":
        operands_pos = parser.pos
        unresolved_a = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        unresolved_b = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        unresolved_c = parser.parse_unresolved_operand()

        attributes = parser.parse_optional_attr_dict()
        ins = _parse_variadic_operands(parser, "ins")
        outs = _parse_variadic_operands(parser, "outs")

        parser.parse_punctuation(":")
        pointer_types = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parser.parse_type
        )
        if len(pointer_types) != 3:
            parser.raise_error("expected exactly three pointer types")

        a, b, c = parser.resolve_operands(
            (unresolved_a, unresolved_b, unresolved_c), pointer_types, operands_pos
        )
        property_names = cls.get_irdl_definition().properties
        properties = {
            name: value for name, value in attributes.items() if name in property_names
        }
        attributes = {
            name: value
            for name, value in attributes.items()
            if name not in property_names
        }

        return cls.build(
            operands=(a, b, c, ins, outs),
            result_types=(a.type, b.type, c.type, tuple(out.type for out in outs)),
            properties=properties,
            attributes=attributes,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.a)
        printer.print_string(", ")
        printer.print_ssa_value(self.b)
        printer.print_string(", ")
        printer.print_ssa_value(self.c)

        attributes = {**self.properties, **self.attributes}
        attributes.pop("operandSegmentSizes", None)
        attributes.pop("resultSegmentSizes", None)
        if attributes:
            printer.print_op_attributes(attributes)

        _print_variadic_operands(printer, "ins", self.ins)
        _print_variadic_operands(printer, "outs", self.outs)

        printer.print_string(" : (")
        printer.print_list(
            (self.a, self.b, self.c),
            lambda operand: printer.print_attribute(operand.type),
        )
        printer.print_string(")")

    def verify_(self) -> None:
        try:
            MatmulIterator(self.iterator.data)
        except ValueError as error:
            allowed = ", ".join(iterator.value for iterator in MatmulIterator)
            raise VerifyException(
                f"iterator must be one of {allowed}, got {self.iterator.data}"
            ) from error

        integer_properties = {
            "m": self.m.value.data,
            "n": self.n.value.data,
            "k": self.k.value.data,
            "lda": self.lda.value.data,
            "ldb": self.ldb.value.data,
            "ldc": self.ldc.value.data,
        }
        for name, value in integer_properties.items():
            if value <= 0:
                raise VerifyException(f"{name} must be positive, got {value}")

        if self.vnni_a and self.ab_datatype is None:
            raise VerifyException("VNNI A requires BF16 inputs")
        if self.c_datatype is not None and not isinstance(self.datatype, Float32Type):
            raise VerifyException("a BF16 C requires F32 accumulation")
        if self.aligned_a.value.data and self.lda.value.data % self.a_vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )
        # In elements of the accumulation: a BF16 C moves 16 elements, a ymm,
        # at a time, as an F32 one does a zmm.
        vector_length = 512 // self.datatype.bitwidth
        if self.aligned_c.value.data and self.ldc.value.data % vector_length:
            raise VerifyException(
                "aligned C requires ldc to be a multiple of the vector length"
            )

        inputs = (
            self.a,
            self.b,
            self.c,
            *self.outs,
        )
        outputs = (
            self.a_out,
            self.b_out,
            self.c_out,
            *self.out_results,
        )
        if tuple(value.type for value in inputs) != tuple(
            value.type for value in outputs
        ):
            raise VerifyException("operand and result types must match pairwise")


@irdl_op_definition
class MatmulRegOp(IRDLOperation):
    """A blocked, register-resident matrix multiplication body.

    The operation performs ``k`` K steps, advancing A by ``k * lda`` elements
    and B by ``k`` elements for the currently supported non-transposed inputs.

    C is register-resident and represented by ``outs``, so this operation does
    not carry a C pointer.

    ``ins`` are additional read-only registers. ``outs`` are additional
    loop-carried registers and have pairwise matching ``out_results``. The
    current schedule uses ``ins`` for optional masks and ``outs`` for C
    accumulators. ``datatype``, ``ab_datatype`` and ``vnni_a`` are as on
    ``xsmm.matmul``.
    """

    name = "xsmm.matmul_reg"

    a = operand_def(GeneralRegisterType)
    b = operand_def(GeneralRegisterType)
    ins = var_operand_def(X86RegisterType)
    outs = var_operand_def(X86RegisterType)

    a_out = result_def(GeneralRegisterType)
    b_out = result_def(GeneralRegisterType)
    out_results = var_result_def(X86RegisterType)

    m = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    lda = prop_def(IntegerAttr)
    ldb = prop_def(IntegerAttr)
    datatype = prop_def(Float32Type | Float64Type)
    ab_datatype = opt_prop_def(BFloat16Type)
    vnni_a = opt_prop_def(UnitAttr)
    aligned_a = prop_def(BoolAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect())

    @property
    def input_type(self) -> Float32Type | Float64Type | BFloat16Type:
        """The element type of A and B: ``ab_datatype``, else ``datatype``."""
        return self.datatype if self.ab_datatype is None else self.ab_datatype

    @property
    def a_vector_length(self) -> int:
        """How many M rows of A one 512-bit vector holds: a K pair of each
        when A is in VNNI layout, one element otherwise."""
        return 512 // ((1 + bool(self.vnni_a)) * self.input_type.bitwidth)

    def __init__(
        self,
        a: SSAValue,
        b: SSAValue,
        ins: Sequence[SSAValue] = (),
        outs: Sequence[SSAValue] = (),
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        datatype: Float32Type | Float64Type,
        ab_datatype: BFloat16Type | None = None,
        vnni_a: bool = False,
        aligned_a: bool,
    ):
        super().__init__(
            operands=(a, b, ins, outs),
            result_types=(
                a.type,
                b.type,
                tuple(out.type for out in outs),
            ),
            properties={
                "m": IntegerAttr(m, i64),
                "n": IntegerAttr(n, i64),
                "k": IntegerAttr(k, i64),
                "lda": IntegerAttr(lda, i64),
                "ldb": IntegerAttr(ldb, i64),
                "datatype": datatype,
                "aligned_a": BoolAttr.from_bool(aligned_a),
            }
            | ({} if ab_datatype is None else {"ab_datatype": ab_datatype})
            | ({"vnni_a": UnitAttr()} if vnni_a else {}),
        )

    @classmethod
    def parse(cls, parser: Parser) -> "MatmulRegOp":
        operands_pos = parser.pos
        unresolved_a = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        unresolved_b = parser.parse_unresolved_operand()

        attributes = parser.parse_optional_attr_dict()
        ins = _parse_variadic_operands(parser, "ins")
        outs = _parse_variadic_operands(parser, "outs")

        parser.parse_punctuation(":")
        pointer_types = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parser.parse_type
        )
        if len(pointer_types) != 2:
            parser.raise_error("expected exactly two pointer types")

        a, b = parser.resolve_operands(
            (unresolved_a, unresolved_b), pointer_types, operands_pos
        )
        property_names = cls.get_irdl_definition().properties
        properties = {
            name: value for name, value in attributes.items() if name in property_names
        }
        attributes = {
            name: value
            for name, value in attributes.items()
            if name not in property_names
        }

        return cls.build(
            operands=(a, b, ins, outs),
            result_types=(a.type, b.type, tuple(out.type for out in outs)),
            properties=properties,
            attributes=attributes,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.a)
        printer.print_string(", ")
        printer.print_ssa_value(self.b)

        attributes = {**self.properties, **self.attributes}
        attributes.pop("operandSegmentSizes", None)
        attributes.pop("resultSegmentSizes", None)
        if attributes:
            printer.print_op_attributes(attributes)

        _print_variadic_operands(printer, "ins", self.ins)
        _print_variadic_operands(printer, "outs", self.outs)

        printer.print_string(" : (")
        printer.print_list(
            (self.a, self.b),
            lambda operand: printer.print_attribute(operand.type),
        )
        printer.print_string(")")

    def verify_(self) -> None:
        integer_properties = {
            "m": self.m.value.data,
            "n": self.n.value.data,
            "k": self.k.value.data,
            "lda": self.lda.value.data,
            "ldb": self.ldb.value.data,
        }
        for name, value in integer_properties.items():
            if value <= 0:
                raise VerifyException(f"{name} must be positive, got {value}")

        if self.vnni_a and self.ab_datatype is None:
            raise VerifyException("VNNI A requires BF16 inputs")
        if self.aligned_a.value.data and self.lda.value.data % self.a_vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )

        inputs = (
            self.a,
            self.b,
            *self.outs,
        )
        outputs = (
            self.a_out,
            self.b_out,
            *self.out_results,
        )
        if tuple(value.type for value in inputs) != tuple(
            value.type for value in outputs
        ):
            raise VerifyException("operand and result types must match pairwise")


XSMM = Dialect(
    "xsmm",
    [AccumulatorAddpdOp, AccumulatorAddpsOp, MatmulOp, MatmulRegOp],
    [],
)
