import pytest
from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.backend.register_allocatable import (
    HasRegisterConstraints,
    HasRegisterConstraintsTrait,
    RegisterConstraints,
)
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import test, x86
from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, ModuleOp, f32
from xdsl.dialects.x86.registers import AVX512RegisterType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from autotuner.dialects.xsmm import (
    AccumulatorAddpdOp,
    AccumulatorAddpsOp,
    ExpOp,
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


@pytest.mark.parametrize("assigned", (False, True))
@pytest.mark.parametrize("initialized", (False, True))
def test_horner_declares_register_webs(assigned: bool, initialized: bool) -> None:
    registers = tuple(
        AVX512RegisterType.from_index(i)
        if assigned
        else AVX512RegisterType.unallocated()
        for i in range(4)
    )
    x, acc, c = (create_ssa_value(r) for r in registers[:3])
    op = HornerOp(x, acc if initialized else None, (c,), destination=registers[1])
    assert isinstance(op, HasRegisterConstraints)
    assert op.has_trait(HasRegisterConstraintsTrait)
    assert op.get_register_constraints() == RegisterConstraints(
        (x, c),
        () if initialized else (op.result,),
        ((acc, op.result),) if initialized else (),
    )
    assert set(RegisterAllocatedMemoryEffect.iter_used_registers(op)) == (
        set(registers[:3]) if assigned else set()
    )
    op.verify()


def test_horner_rejects_a_later_use_of_the_consumed_accumulator() -> None:
    inputs = test.TestOp(result_types=[AVX512RegisterType.unallocated()] * 3)
    x, acc, c = inputs.results
    op = HornerOp(x, acc, (c,))
    after = test.TestOp(operands=(op.result, x, c, acc))
    module = ModuleOp([inputs, op, after])
    module.verify()
    with pytest.raises(VerifyException, match="should not be read after in/out"):
        VerifyLivenessContext(set()).process_region(module.body)


@pytest.mark.parametrize("assigned", (False, True))
def test_exp_declares_consumed_pointer_pairs(assigned: bool) -> None:
    registers = (
        (x86.registers.R8, x86.registers.R9)
        if assigned
        else (
            x86.registers.UNALLOCATED_REG64,
            x86.registers.UNALLOCATED_REG64,
        )
    )
    inputs = test.TestOp(result_types=registers)
    op = ExpOp(*inputs.results, length=528, degree=5)
    assert isinstance(op, HasRegisterConstraints)
    assert op.has_trait(HasRegisterConstraintsTrait)
    assert op.get_register_constraints() == RegisterConstraints(
        (), (), ((inputs.results[0], op.input_out), (inputs.results[1], op.output_out))
    )
    assert tuple(result.type for result in op.results) == registers
    after = test.TestOp(operands=op.results)
    module = ModuleOp([inputs, op, after])
    module.verify()
    VerifyLivenessContext(set()).process_region(module.body)


@pytest.mark.parametrize("pointer", (0, 1))
def test_exp_rejects_a_later_use_of_a_consumed_pointer(pointer: int) -> None:
    inputs = test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64] * 2)
    op = ExpOp(*inputs.results, length=528, degree=5)
    after = test.TestOp(operands=(inputs.results[pointer],))
    module = ModuleOp([inputs, op, after])
    module.verify()
    with pytest.raises(VerifyException, match="should not be read after in/out"):
        VerifyLivenessContext(set()).process_region(module.body)


@pytest.mark.parametrize("alias_argument", (False, True))
def test_horner_multistep_accumulator_cannot_alias_read_only_inputs(
    alias_argument: bool,
) -> None:
    x, acc, c = (create_ssa_value(AVX512RegisterType.unallocated()) for _ in range(3))
    argument = acc if alias_argument else x
    coefficients = (c, c) if alias_argument else (acc, c)
    # A single fused step reads all inputs before overwriting its accumulator.
    HornerOp(argument, acc, coefficients[:1]).verify()
    with pytest.raises(VerifyException, match="multi-step Horner accumulator"):
        HornerOp(argument, acc, coefficients).verify()


@pytest.mark.parametrize("alias_argument", (False, True))
@pytest.mark.parametrize("polynomial", (False, True))
def test_seeded_expansion_rejects_clobbering_preassignments(
    alias_argument: bool, polynomial: bool
) -> None:
    x = create_ssa_value(AVX512RegisterType.from_index(0))
    constants = PolynomialConstantsOp(
        tuple(FloatAttr(float(i), f32) for i in range(4)),
        ".Lcoeff",
        tuple(AVX512RegisterType.from_index(i) for i in range(1, 5)),
    )
    destination = x.type if alias_argument else constants.registers[0].type
    op = (
        PolynomialOp(x, constants, destination=destination)
        if polynomial
        else HornerOp(x, None, constants.registers, destination=destination)
    )
    # Even if the source dies at the high-level operation, it is needed after
    # the seed inside its expansion. Type equality is a physical alias here.
    with pytest.raises(VerifyException, match="preassigned result must be distinct"):
        op.verify()


@pytest.mark.parametrize("assigned", (False, True))
def test_polynomial_interfaces_preserve_read_only_bindings(assigned: bool) -> None:
    registers = tuple(
        AVX512RegisterType.from_index(i)
        if assigned
        else AVX512RegisterType.unallocated()
        for i in range(4)
    )
    constants = PolynomialConstantsOp(
        (FloatAttr(1.0, f32), FloatAttr(2.0, f32)), ".Lc", registers[1:3]
    )
    x = create_ssa_value(registers[0])
    op = PolynomialOp(x, constants, destination=registers[3])
    for operation, expected in (
        (constants, RegisterConstraints((), constants.registers, ())),
        (op, RegisterConstraints((x, *constants.registers), (op.result,), ())),
    ):
        assert isinstance(operation, HasRegisterConstraints)
        assert operation.has_trait(HasRegisterConstraintsTrait)
        assert tuple(map(tuple, operation.get_register_constraints())) == expected
        operation.verify()
    assert set(RegisterAllocatedMemoryEffect.iter_used_registers(op)) == (
        set(registers) if assigned else set()
    )


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
