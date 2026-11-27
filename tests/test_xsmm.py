import pytest
from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, f32
from xdsl.dialects.x86.registers import AVX512RegisterType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from autotuner.dialects.xsmm import (
    AccumulatorAddpdOp,
    AccumulatorAddpsOp,
    HornerOp,
    PolynomialConstantsOp,
    PolynomialOp,
)


@pytest.mark.parametrize(
    ("op_type", "mnemonic"),
    ((AccumulatorAddpdOp, "vaddpd"), (AccumulatorAddpsOp, "vaddps")),
)
def test_accumulator_add_register_constraints(
    op_type: type[AccumulatorAddpdOp] | type[AccumulatorAddpsOp], mnemonic: str
) -> None:
    register_type = AVX512RegisterType.unallocated()
    source = create_ssa_value(register_type)
    accumulator = create_ssa_value(register_type)

    op = op_type(source, accumulator, destination=register_type)
    ins, outs, inouts = op.get_register_constraints()

    assert tuple(ins) == (source,)
    assert tuple(outs) == ()
    assert tuple(inouts) == ((accumulator, op.destination),)
    assert op.assembly_instruction_name() == mnemonic


def test_horner_requires_a_seed_and_preserves_accumulator_assignment() -> None:
    x = create_ssa_value(AVX512RegisterType.from_index(0))
    accumulator = create_ssa_value(AVX512RegisterType.from_index(1))
    with pytest.raises(
        VerifyException, match="uninitialized Horner requires coefficients"
    ):
        HornerOp(x, None, ()).verify()
    with pytest.raises(VerifyException, match="reuse the accumulator register"):
        HornerOp(x, accumulator, (x,), destination=x.type).verify()


def test_polynomial_constants_require_matching_nonempty_results() -> None:
    with pytest.raises(VerifyException, match="must not be empty"):
        PolynomialConstantsOp((), ".Lempty", ()).verify()
    with pytest.raises(
        VerifyException, match="one register per polynomial coefficient"
    ):
        PolynomialConstantsOp((FloatAttr(1.0, f32),), ".Lmissing", ()).verify()


def test_polynomial_requires_constant_bindings_and_explicit_permissions() -> None:
    register = AVX512RegisterType.unallocated()
    x = create_ssa_value(register)
    constants = PolynomialConstantsOp(
        (FloatAttr(0.0, f32), FloatAttr(2.0, f32)), ".Lcoeff", (register,) * 2
    )
    polynomial = PolynomialOp(x, constants)
    polynomial.verify()

    polynomial.properties["coefficients"] = ArrayAttr(
        (FloatAttr(-0.0, f32), FloatAttr(2.0, f32))
    )
    with pytest.raises(VerifyException, match="match coefficient bits"):
        polynomial.verify()
    polynomial.properties["coefficients"] = constants.coefficients

    polynomial.operands = (x, x, constants.registers[1])
    with pytest.raises(VerifyException, match="come from polynomial_constants"):
        polynomial.verify()
    polynomial.operands = (x, *constants.registers)

    polynomial.properties["fastmath"] = FastMathFlagsAttr("none")
    with pytest.raises(VerifyException, match="requires reassoc and contract"):
        polynomial.verify()
