import pytest
from xdsl.dialects import test, x86, x86_scf
from xdsl.dialects.builtin import Float64Type, ModuleOp
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp, MatmulRegOp
from autotuner.schedules import (
    loop_matmul,
    loop_matmul_reg,
    set_matmul_iterator,
    tile_matmul,
    tile_matmul_reg,
)


def make_matmul(
    iterator: MatmulIterator,
) -> tuple[ModuleOp, MatmulOp, test.TestOp, SSAValue, SSAValue]:
    pointer_ops = [
        test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64]) for _ in range(5)
    ]
    input_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64])
    output_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_AVX512_MASK])
    a, b, c, rbp, rsp = (pointer_op.results[0] for pointer_op in pointer_ops)
    input_value = input_op.results[0]
    output_value = output_op.results[0]
    matmul = MatmulOp(
        a,
        b,
        c,
        rbp,
        rsp,
        (input_value,),
        (output_value,),
        m=8,
        n=3,
        k=4,
        lda=8,
        ldb=4,
        ldc=8,
        datatype=Float64Type(),
        aligned_a=True,
        aligned_c=True,
        iterator=iterator,
    )
    consumer = test.TestOp(operands=matmul.results)
    module = ModuleOp([*pointer_ops, input_op, output_op, matmul, consumer])
    return module, matmul, consumer, input_value, output_value


def make_matmul_reg() -> tuple[ModuleOp, MatmulRegOp, test.TestOp]:
    pointer_ops = [
        test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64]) for _ in range(4)
    ]
    output_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_AVX512])
    a, b, rbp, rsp = (pointer_op.results[0] for pointer_op in pointer_ops)
    matmul = MatmulRegOp(
        a,
        b,
        rbp,
        rsp,
        outs=(output_op.results[0],),
        m=8,
        n=3,
        k=4,
        lda=8,
        ldb=4,
        datatype=Float64Type(),
        aligned_a=True,
    )
    consumer = test.TestOp(operands=matmul.results)
    module = ModuleOp([*pointer_ops, output_op, matmul, consumer])
    return module, matmul, consumer


@pytest.mark.parametrize(
    ("source", "target", "expected_adjustments"),
    (
        (
            MatmulIterator.N,
            MatmulIterator.M,
            (("x86.ri.sub", 64), ("x86.ri.add", 96), ("x86.ri.add", 128)),
        ),
        (
            MatmulIterator.M,
            MatmulIterator.N,
            (("x86.ri.add", 64), ("x86.ri.sub", 96), ("x86.ri.sub", 128)),
        ),
        (
            MatmulIterator.N,
            MatmulIterator.NONE,
            (("x86.ri.add", 0), ("x86.ri.add", 96), ("x86.ri.add", 192)),
        ),
        (
            MatmulIterator.NONE,
            MatmulIterator.N,
            (("x86.ri.add", 0), ("x86.ri.sub", 96), ("x86.ri.sub", 192)),
        ),
        (
            MatmulIterator.M,
            MatmulIterator.NONE,
            (("x86.ri.add", 64), ("x86.ri.add", 0), ("x86.ri.add", 64)),
        ),
        (
            MatmulIterator.NONE,
            MatmulIterator.M,
            (("x86.ri.sub", 64), ("x86.ri.add", 0), ("x86.ri.sub", 64)),
        ),
    ),
)
def test_set_matmul_iterator(
    source: MatmulIterator,
    target: MatmulIterator,
    expected_adjustments: tuple[tuple[str, int] | None, ...],
) -> None:
    module, matmul, consumer, input_value, output_value = make_matmul(source)

    replacement = set_matmul_iterator(PatternRewriter(matmul), matmul, target)

    adjustments = tuple(
        op
        for op in module.body.block.ops
        if isinstance(op, (x86.ops.RI_AddOp, x86.ops.RI_SubOp))
    )
    assert tuple(
        (adjustment.name, adjustment.immediate.value.data) for adjustment in adjustments
    ) == tuple(adjustment for adjustment in expected_adjustments if adjustment)
    assert replacement.iterator.data == target
    assert replacement.ins[0] is input_value
    assert replacement.outs[0] is output_value
    adjustment_iter = iter(adjustments)
    assert tuple(operand.owner for operand in consumer.operands[:3]) == tuple(
        replacement if expected is None else next(adjustment_iter)
        for expected in expected_adjustments
    )
    assert consumer.operands[3] is replacement.rbp_out
    assert consumer.operands[4] is replacement.rsp_out
    assert consumer.operands[5] is replacement.out_results[0]
    module.verify()


def test_set_matmul_iterator_is_noop_for_current_iterator() -> None:
    module, matmul, consumer, _, _ = make_matmul(MatmulIterator.N)
    original_ops = tuple(module.body.block.ops)

    replacement = set_matmul_iterator(PatternRewriter(matmul), matmul, MatmulIterator.N)

    assert replacement is matmul
    assert tuple(module.body.block.ops) == original_ops
    assert tuple(consumer.operands) == tuple(matmul.results)


@pytest.mark.parametrize(
    ("iterator", "tile_size"),
    [
        (MatmulIterator.M, 8),
        (MatmulIterator.N, 3),
    ],
)
def test_loop_matmul_elides_single_iteration_loop(
    iterator: MatmulIterator, tile_size: int
) -> None:
    module, matmul, consumer, _, _ = make_matmul(iterator)
    original_ops = tuple(module.body.block.ops)

    tiled = loop_matmul(PatternRewriter(matmul), matmul, tile_size)

    assert tiled is matmul
    assert tuple(module.body.block.ops) == original_ops
    assert tuple(consumer.operands) == tuple(matmul.results)
    assert not any(isinstance(op, x86_scf.ForOp) for op in module.walk())


def test_tile_matmul_elides_single_iteration_loop() -> None:
    module, matmul, _, _, _ = make_matmul(MatmulIterator.M)
    original_ops = tuple(module.body.block.ops)

    tiled, remainder = tile_matmul(PatternRewriter(matmul), matmul, 8)

    assert tiled is matmul
    assert remainder is None
    assert tuple(module.body.block.ops) == original_ops


def test_loop_matmul_reg_elides_single_iteration_loop() -> None:
    module, matmul, consumer = make_matmul_reg()
    original_ops = tuple(module.body.block.ops)

    tiled = loop_matmul_reg(PatternRewriter(matmul), matmul, 4)

    assert tiled is matmul
    assert tuple(module.body.block.ops) == original_ops
    assert tuple(consumer.operands) == tuple(matmul.results)
    assert not any(isinstance(op, x86_scf.ForOp) for op in module.walk())


def test_tile_matmul_reg_elides_single_iteration_loop() -> None:
    module, matmul, _ = make_matmul_reg()
    original_ops = tuple(module.body.block.ops)

    tiled, remainder = tile_matmul_reg(PatternRewriter(matmul), matmul, 4)

    assert tiled is matmul
    assert remainder is None
    assert tuple(module.body.block.ops) == original_ops
