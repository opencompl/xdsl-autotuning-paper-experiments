from collections.abc import Sequence
from enum import StrEnum

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import (
    BoolAttr,
    Float32Type,
    Float64Type,
    IntegerAttr,
    StringAttr,
    i64,
)
from xdsl.dialects.x86.ops import DSS_Operation
from xdsl.dialects.x86.registers import (
    AVX512RegisterType,
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
    opt_operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect, Pure
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
    current schedule uses an ``in`` for an optional AVX-512 mask.
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
    aligned_a = prop_def(BoolAttr)
    aligned_c = prop_def(BoolAttr)
    iterator = prop_def(StringAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

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
            },
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

        vector_length = 512 // self.datatype.bitwidth
        if self.aligned_a.value.data and self.lda.value.data % vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )
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
    current schedule uses ``ins`` for an optional mask and ``outs`` for C
    accumulators.
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
    aligned_a = prop_def(BoolAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect())

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
            },
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

        vector_length = 512 // self.datatype.bitwidth
        if self.aligned_a.value.data and self.lda.value.data % vector_length:
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


@irdl_op_definition
class HornerOp(IRDLOperation):
    """Evaluate or continue an FP32 Horner recurrence in each AVX-512 lane.

    Coefficients are listed in ascending power order. Starting with
    ``accumulator``, consume them in reverse order using ``fma(acc, x, c)``.
    Each step is fused; unlike a polynomial, this operation fixes evaluation
    order. With no remaining coefficients the result is the accumulator.

    Without an accumulator, initialize with ``fma(x, c[-1], c[-2])``, then
    continue with the remaining coefficients. A single coefficient is a
    constant result. Initialization preserves x and the coefficient registers.
    """

    name = "xsmm.horner"
    argument = operand_def(AVX512RegisterType)
    accumulator = opt_operand_def(AVX512RegisterType)
    coefficients = var_operand_def(AVX512RegisterType)
    result = result_def(AVX512RegisterType)
    datatype = prop_def(Float32Type)
    traits = traits_def(Pure())
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def __init__(
        self,
        argument: SSAValue,
        accumulator: SSAValue | None,
        coefficients: Sequence[SSAValue],
        *,
        destination: AVX512RegisterType | None = None,
    ):
        super().__init__(
            operands=(argument, accumulator, coefficients),
            result_types=(
                destination
                if destination is not None
                else accumulator.type
                if accumulator is not None
                else AVX512RegisterType.unallocated(),
            ),
            properties={"datatype": Float32Type()},
        )

    @classmethod
    def parse(cls, parser: Parser) -> "HornerOp":
        pos = parser.pos
        argument = parser.parse_unresolved_operand()
        accumulator = None
        if parser.parse_optional_keyword("accumulator") is not None:
            parser.parse_punctuation("(")
            accumulator = parser.parse_unresolved_operand()
            parser.parse_punctuation(")")
        parser.parse_keyword("coefficients")
        coefficients = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        attributes = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        function_type = parser.parse_function_type()
        has_accumulator = accumulator is not None
        operands = parser.resolve_operands(
            (argument, *((accumulator,) if has_accumulator else ()), *coefficients),
            function_type.inputs.data,
            pos,
        )
        return cls.build(
            operands=(
                operands[0],
                operands[1] if has_accumulator else None,
                operands[2 if has_accumulator else 1 :],
            ),
            result_types=function_type.outputs.data,
            properties={"datatype": attributes.pop("datatype", Float32Type())},
            attributes=attributes,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.argument)
        if self.accumulator is not None:
            printer.print_string(" accumulator(")
            printer.print_ssa_value(self.accumulator)
            printer.print_string(")")
        printer.print_string(" coefficients(")
        printer.print_list(self.coefficients, printer.print_ssa_value)
        printer.print_string(")")
        printer.print_op_attributes({**self.attributes, "datatype": self.datatype})
        printer.print_string(" : (")
        printer.print_list(self.operands, lambda v: printer.print_attribute(v.type))
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)

    def verify_(self) -> None:
        if self.accumulator is None:
            if not self.coefficients:
                raise VerifyException("uninitialized Horner requires coefficients")
        elif self.result.type != self.accumulator.type:
            raise VerifyException("Horner result must reuse the accumulator register")


@irdl_op_definition
class ExpOp(IRDLOperation):
    """Out-of-place approximate FP32 exp over a packed, full-vector column.

    Degree selects LIBXSMM's cubic or quintic numerical contract. Input and
    output must not overlap; neither pointer is advanced by the operation.
    """

    name = "xsmm.exp"
    input = operand_def(GeneralRegisterType)
    output = operand_def(GeneralRegisterType)
    length = prop_def(IntegerAttr)
    degree = prop_def(IntegerAttr)
    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())
    assembly_format = "$input `,` $output `length` $length `degree` $degree attr-dict `:` type(operands)"

    def __init__(self, input: SSAValue, output: SSAValue, length: int, degree: int):
        super().__init__(
            operands=[input, output],
            properties={
                "length": IntegerAttr(length, i64),
                "degree": IntegerAttr(degree, i64),
            },
        )

    def verify_(self) -> None:
        if self.length.value.data <= 0 or self.length.value.data % 16:
            raise VerifyException("exp length must be positive and divisible by 16")
        if self.degree.value.data not in (3, 5):
            raise VerifyException("exp degree must be 3 or 5")


XSMM = Dialect(
    "xsmm",
    [AccumulatorAddpdOp, AccumulatorAddpsOp, MatmulOp, MatmulRegOp, HornerOp, ExpOp],
    [],
)
