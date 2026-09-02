import pytest
from xdsl.dialects.x86.registers import AVX512RegisterType
from xdsl.utils.test_value import create_ssa_value

from autotuner.dialects.xsmm import AccumulatorAddpdOp, AccumulatorAddpsOp


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
