import pytest
from xdsl.dialects.x86.registers import AVX512RegisterType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from autotuner.dialects.xsmm import AccumulatorAddpdOp, AccumulatorAddpsOp, HornerOp


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
