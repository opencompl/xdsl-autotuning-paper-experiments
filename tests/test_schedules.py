import ctypes
import platform
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.backend.register_allocatable import HasRegisterConstraints
from xdsl.context import Context
from xdsl.dialects import test, x86, x86_scf
from xdsl.dialects.builtin import Float64Type, FloatAttr, ModuleOp, f32
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from autotuner.dialects.xsmm import (
    HornerOp,
    MatmulIterator,
    MatmulOp,
    MatmulRegOp,
    PolynomialConstantsOp,
    PolynomialOp,
)
from autotuner.passes.xsmm_expand_polynomial import XsmmExpandPolynomialPass
from autotuner.schedules import (
    expand_horner,
    expand_polynomial_constants,
    loop_matmul,
    loop_matmul_reg,
    set_matmul_iterator,
    tile_matmul,
    tile_matmul_reg,
)


def register_webs(module: ModuleOp) -> dict[SSAValue, set[SSAValue]]:
    """Connect declared inout pairs and all four members of loop tuples."""
    webs = {
        value: {value} for op in module.walk() for value in (*op.operands, *op.results)
    }
    groups: list[tuple[SSAValue, ...]] = []
    for op in module.walk():
        if isinstance(op, HasRegisterConstraints):
            groups.extend(op.get_register_constraints().inouts)
        if isinstance(op, x86_scf.ForOp):
            groups.append((op.lb, op.body.block.args[0], op.lb_end))
            terminator = op.body.block.last_op
            assert isinstance(terminator, x86_scf.YieldOp)
            groups.extend(
                zip(
                    op.iter_args,
                    op.body.block.args[1:],
                    terminator.arguments,
                    op.res,
                    strict=True,
                )
            )
    for group in groups:
        merged = set[SSAValue]().union(*(webs.get(value, {value}) for value in group))
        for value in merged:
            webs[value] = merged
    return webs


def make_matmul(
    iterator: MatmulIterator,
) -> tuple[ModuleOp, MatmulOp, test.TestOp, SSAValue, SSAValue]:
    pointer_ops = [
        test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64]) for _ in range(3)
    ]
    input_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64])
    output_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_AVX512_MASK])
    a, b, c = (pointer_op.results[0] for pointer_op in pointer_ops)
    input_value = input_op.results[0]
    output_value = output_op.results[0]
    matmul = MatmulOp(
        a,
        b,
        c,
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
        test.TestOp(result_types=[x86.registers.UNALLOCATED_REG64]) for _ in range(2)
    ]
    output_op = test.TestOp(result_types=[x86.registers.UNALLOCATED_AVX512])
    a, b = (pointer_op.results[0] for pointer_op in pointer_ops)
    matmul = MatmulRegOp(
        a,
        b,
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
    assert consumer.operands[3] is replacement.out_results[0]
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


@pytest.mark.parametrize(
    ("coefficient_count", "initialized"),
    ((0, True), (1, True), (3, True), (1, False), (2, False), (4, False)),
)
@pytest.mark.parametrize("assigned", (False, True))
def test_horner_expansion_preserves_surrounding_operations(
    coefficient_count: int,
    initialized: bool,
    assigned: bool,
) -> None:
    inputs = test.TestOp(
        result_types=[
            x86.registers.AVX512RegisterType.from_index(i)
            if assigned
            else x86.registers.UNALLOCATED_AVX512
            for i in range(coefficient_count + 2)
        ]
    )
    x, accumulator, *coefficients = inputs.results
    horner = HornerOp(
        x,
        accumulator if initialized else None,
        coefficients,
        destination=SSAValue.get(
            accumulator, type=x86.registers.AVX512RegisterType
        ).type,
    )
    before = test.TestOp(operands=[x, accumulator])
    consumer = test.TestOp(operands=[horner.result, x])
    after = test.TestOp(operands=coefficients)
    module = ModuleOp([inputs, before, horner, consumer, after])

    module.verify()
    VerifyLivenessContext(set()).process_region(module.body)
    while horner is not None:
        horner = expand_horner(PatternRewriter(horner), horner)
        ops = tuple(module.body.block.ops)
        assert ops[:2] == (inputs, before)
        assert ops[-2:] == (consumer, after)
        assert tuple(before.operands) == (x, accumulator)
        assert tuple(after.operands) == tuple(coefficients)
        assert consumer.operands[1] is x
        module.verify()
        VerifyLivenessContext(set()).process_region(module.body)
        webs = register_webs(module)
        result_web = webs[consumer.operands[0]]
        assert (accumulator in result_web) == initialized
        assert result_web.isdisjoint((x, *coefficients))
        assert {value.type for value in result_web} == {accumulator.type}
        # Distinct external read-only values remain distinct webs, even when
        # their unallocated types all denote the same register class.
        for value in (x, *coefficients):
            assert webs[value] == {value}

    if coefficient_count == 0:
        assert consumer.operands[0] is accumulator


def test_polynomial_constant_expansion_preserves_its_position() -> None:
    constants = PolynomialConstantsOp(
        (FloatAttr(1.5, f32), FloatAttr(-0.5, f32)),
        ".Lcoefficient",
        (x86.registers.UNALLOCATED_AVX512,) * 2,
    )
    before = test.TestOp()
    between = test.TestOp()
    consumer = test.TestOp(operands=constants.results)
    after = test.TestOp()
    module = ModuleOp([before, constants, between, consumer, after])

    expand_polynomial_constants(PatternRewriter(constants), constants)

    ops = tuple(module.body.block.ops)
    assert ops[0] is before
    assert ops[-3:] == (between, consumer, after)
    assert all(operand.owner in ops[1:-3] for operand in consumer.operands)
    module.verify()


@pytest.mark.parametrize("schedule", ("libxsmm", "estrin5"))
def test_polynomial_expansion_is_local(schedule: str) -> None:
    register = x86.registers.UNALLOCATED_AVX512
    constants = PolynomialConstantsOp(
        tuple(FloatAttr(value, f32) for value in (1, -2, 3, -4, 5, -6)),
        ".Lcoeff",
        (register,) * 6,
    )
    argument = test.TestOp(result_types=[register])
    polynomial = PolynomialOp(argument.results[0], constants)
    before = test.TestOp(operands=constants.results)
    after = test.TestOp(
        operands=[polynomial.result, *argument.results, *constants.results]
    )
    module = ModuleOp([constants, argument, before, polynomial, after])
    module.verify()
    VerifyLivenessContext(set()).process_region(module.body)
    XsmmExpandPolynomialPass(schedule).apply(Context(), module)
    ops = tuple(module.body.block.ops)
    assert ops[:3] == (constants, argument, before)
    assert ops[-1] is after
    assert tuple(before.operands) == tuple(constants.results)
    assert tuple(after.operands[1:]) == (*argument.results, *constants.results)
    module.verify()
    VerifyLivenessContext(set()).process_region(module.body)
    webs = register_webs(module)
    assert webs[after.operands[0]].isdisjoint(after.operands[1:])
    for value in after.operands[1:]:
        assert webs[value] == {value}


@pytest.mark.parametrize(
    "coefficients",
    ((1.25, -2.5, 0.75, 0.125, -0.5, 0.0625), (-1.0, 1.0 - 2**-23, 0.0, 0.0, 0.0, 0.0)),
)
def test_allocated_estrin_matches_fused_reference(
    tmp_path: Path, coefficients: tuple[float, ...]
) -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or "avx512f" not in Path("/proc/cpuinfo").read_text()
        or not shutil.which("clang")
    ):
        pytest.skip("requires AVX-512 and clang")
    constants = ", ".join(f"{c:.17e} : f32" for c in coefficients)
    source = tmp_path / "polynomial.mlir"
    source.write_text(f"""
!v = !x86.avx512reg
x86_func.func public @polynomial(%in: !x86.reg64<rdi>, %out: !x86.reg64<rsi>) {{
  %c0, %c1, %c2, %c3, %c4, %c5 = xsmm.polynomial_constants [{constants}]
      label_prefix ".Lcoeff" : () -> (!v, !v, !v, !v, !v, !v)
  %r = x86.dm.vmovups [%in] : (!x86.reg64<rdi>) -> !v
  %p = xsmm.polynomial %r coefficients [{constants}]
      bindings(%c0, %c1, %c2, %c3, %c4, %c5)
      {{datatype = f32, fastmath = #arith.fastmath<reassoc, contract>}}
      : (!v, !v, !v, !v, !v, !v, !v) -> !v
  x86.ms.vmovups [%out], %p : (!x86.reg64<rsi>, !v) -> ()
  x86.ms.vmovups [%out + 64], %r : (!x86.reg64<rsi>, !v) -> ()
  x86.ms.vmovups [%out + 128], %c0 : (!x86.reg64<rsi>, !v) -> ()
  x86_func.ret
}}
""")
    assembly = tmp_path / "polynomial.S"
    library = tmp_path / "polynomial.so"
    pipeline = (
        "xsmm-expand-polynomial{schedule=estrin5},xsmm-expand-horner,"
        "xsmm-expand-polynomial-constants,x86-regalloc-verify-liveness,"
        "x86-allocate-registers-with-masks,x86-regalloc-verify-liveness"
    )
    subprocess.run(
        ["xdsl-opt", str(source), "-p", pipeline, "-t", "x86-asm", "-o", str(assembly)],
        check=True,
    )
    subprocess.run(
        ["clang", "-shared", "-Wl,-z,noexecstack", "-o", str(library), str(assembly)],
        check=True,
    )
    kernel = ctypes.CDLL(str(library)).polynomial
    kernel.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
    kernel.restype = None
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = (ctypes.c_float,) * 3
    fma.restype = ctypes.c_float
    inputs = (ctypes.c_float * 16)(
        1 + 2**-23,
        -0.3125,
        0.1,
        -0.1,
        0,
        -0.0,
        0.25,
        0.345,
        -0.345,
        0.9,
        -1.125,
        1.5,
        -1.5,
        1 - 2**-23,
        0.0001,
        -0.0001,
    )
    saved = bytes(inputs)
    outputs = (ctypes.c_float * 48)()
    kernel(inputs, outputs)
    c0, c1, c2, c3, c4, c5 = coefficients
    for index, r in enumerate(inputs):
        # Independent scalar libm fmaf reference, with separately rounded powers.
        r2 = ctypes.c_float(r * r).value
        r4 = ctypes.c_float(r2 * r2).value
        expected = fma(fma(r, c5, c4), r4, fma(fma(r, c3, c2), r2, fma(r, c1, c0)))
        assert struct.pack("f", outputs[index]) == struct.pack("f", expected)
    if coefficients[0] == -1.0:
        # Separate multiply/add would incorrectly round this cancellation to zero.
        assert outputs[0] == -(2**-46)
    assert bytes(inputs) == saved
    assert list(outputs[16:32]) == list(inputs)
    assert list(outputs[32:48]) == [c0] * 16
