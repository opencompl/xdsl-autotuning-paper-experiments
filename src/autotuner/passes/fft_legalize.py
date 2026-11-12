"""Turn the FFT expansions' pseudo-instructions into x86 instructions.

- Negations fold into the operations that consume them: an addition of a
  negated value becomes a subtraction, and a fused multiply-add flips the sign
  of its product or addend. A negation no consumer absorbs becomes an XOR with
  the sign bit.
- A non-destructive fused multiply-add overwrites whichever input dies there:
  the addend (form 231) or a factor (213, or 132 when the other factor is in
  memory). When none dies, it overwrites a copy of the addend.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86, x86_func
from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.dialects.x86.registers import X86VectorRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from autotuner.dialects import x86ext
from autotuner.dialects.x86ext import FmaKind
from autotuner.fft.constants import ConstantPool
from autotuner.fft.values import as_vector

ADD = (x86.ops.DSS_VaddpdOp, x86.ops.DSS_VaddpsOp)
SUB = (x86ext.VsubpdOp, x86ext.VsubpsOp)
MUL = (x86ext.VmulpdOp, x86ext.VmulpsOp)
MUL_MEM = (x86ext.VmulpdMemOp, x86ext.VmulpsMemOp)


def _unallocated(value: SSAValue) -> X86VectorRegisterType:
    value_type = value.type
    assert isinstance(value_type, X86VectorRegisterType)
    return type(value_type).unallocated()


def _pick(classes: tuple[type, type], double: bool) -> type:
    return classes[0] if double else classes[1]


class _NegationFolder:
    def __init__(self, func: x86_func.FuncOp):
        self.func = func
        self._pool: ConstantPool | None = None

    def pool(self, lane_bitwidth: int) -> ConstantPool:
        if self._pool is None:
            datatype = Float64Type() if lane_bitwidth == 64 else Float32Type()
            self._pool = ConstantPool(self.func, datatype)
        return self._pool

    def run(self) -> None:
        worklist = [
            op for op in self.func.body.block.ops if isinstance(op, x86ext.VnegOp)
        ]
        while worklist:
            neg = worklist.pop()
            if neg.parent is None:
                continue
            xor: SSAValue | None = None
            # Each fold replaces the user, which drops at least one use.
            while neg.result.uses:
                use = next(iter(neg.result.uses))
                folded = self._fold(neg, use.operation, use.index)
                if folded is None:
                    if xor is None:
                        xor = self._xor(neg)
                    use.operation.operands[use.index] = xor
                else:
                    worklist.extend(folded)
            Rewriter.erase_op(neg)

    def _xor(self, neg: x86ext.VnegOp) -> SSAValue:
        bits = neg.lane_bitwidth.value.data
        pool = self.pool(bits)
        op_type = x86ext.VxorpdMemOp if bits == 64 else x86ext.VxorpsMemOp
        xor = op_type(
            neg.source,
            pool.pointer,
            pool.sign_mask_offset(),
            broadcast=True,
            destination=_unallocated(neg.source),
        )
        Rewriter.insert_op(xor, InsertPoint.before(neg))
        return xor.destination

    def _replace(
        self,
        user: Operation,
        new_ops: list[Operation],
        result: SSAValue,
        negated: bool,
        lane_bitwidth: int,
    ) -> list[x86ext.VnegOp]:
        created: list[x86ext.VnegOp] = []
        if negated:
            neg = x86ext.VnegOp(result, lane_bitwidth=lane_bitwidth)
            new_ops.append(neg)
            created.append(neg)
            result = neg.result
        Rewriter.replace_op(user, new_ops, [result])
        return created

    def _fold(
        self, neg: x86ext.VnegOp, user: Operation, index: int
    ) -> list[x86ext.VnegOp] | None:
        """Absorb the negation into ``user``'s operand ``index``, returning the
        negations that creates, or ``None`` if ``user`` cannot absorb it."""
        x = neg.source
        bits = neg.lane_bitwidth.value.data
        double = bits == 64
        add = _pick(ADD, double)
        sub = _pick(SUB, double)
        dest = _unallocated(x)
        operands = list(user.operands)

        if isinstance(user, x86ext.VnegOp):
            Rewriter.replace_op(user, [], [x])
            return []
        if isinstance(user, ADD):
            other = operands[1 - index]
            op = sub(other, x, destination=dest)
            return self._replace(user, [op], op.destination, False, bits)
        if isinstance(user, SUB):
            if index == 1:
                op = add(operands[0], x, destination=dest)
                return self._replace(user, [op], op.destination, False, bits)
            # -x - b = -(x + b)
            op = add(x, operands[1], destination=dest)
            return self._replace(user, [op], op.destination, True, bits)
        if isinstance(user, MUL):
            operands[index] = x
            op = type(user)(
                as_vector(operands[0]), as_vector(operands[1]), destination=dest
            )
            return self._replace(user, [op], op.destination, True, bits)
        if isinstance(user, MUL_MEM) and index == 0:
            assert isinstance(user, x86ext._DSMB_Operation)
            op = type(user)(
                x,
                user.memory,
                user.memory_offset.value.data,
                broadcast=user.broadcast is not None,
                destination=dest,
            )
            return self._replace(user, [op], op.destination, True, bits)
        if isinstance(user, x86ext.VirtualFmaOp):
            kind = FmaKind(user.kind.data)
            product, addend = kind.negates_product, kind.negates_addend
            if index == 2:
                addend = not addend
            else:
                product = not product
            operands[index] = x
            op = x86ext.VirtualFmaOp(
                *operands, kind=FmaKind.of(product, addend), lane_bitwidth=bits
            )
            return self._replace(user, [op], op.result, False, bits)
        if isinstance(user, x86ext.VirtualFmaMemOp) and index in (0, 1):
            kind = FmaKind(user.kind.data)
            product, addend = kind.negates_product, kind.negates_addend
            if index == 1:
                addend = not addend
            else:
                product = not product
            operands[index] = x
            op = x86ext.VirtualFmaMemOp(
                operands[0],
                operands[1],
                user.memory,
                user.memory_offset.value.data,
                broadcast=bool(user.broadcast.value.data),
                kind=FmaKind.of(product, addend),
                lane_bitwidth=bits,
            )
            return self._replace(user, [op], op.result, False, bits)
        return None


def _legalize_fmas(func: x86_func.FuncOp) -> None:
    block = func.body.block
    position: dict[Operation, int] = {op: i for i, op in enumerate(block.ops)}

    def dies_at(value: SSAValue, op: Operation) -> bool:
        here = position[op]
        return all(position[use.operation] <= here for use in value.uses)

    for op in list(block.ops):
        if not isinstance(op, x86ext.VirtualFmaOp | x86ext.VirtualFmaMemOp):
            continue
        kind = FmaKind(op.kind.data)
        bits = op.lane_bitwidth.value.data
        new_ops: list[Operation] = []

        # Overwrite the addend (231) or a factor (213, or 132 with the other
        # factor in memory), whichever dies here; else a copy of the addend.
        if isinstance(op, x86ext.VirtualFmaOp):
            a, b, c = op.a, op.b, op.c
            if dies_at(c, op):
                register_in, s1, s2, form = c, a, b, 231
            elif dies_at(a, op):
                register_in, s1, s2, form = a, b, c, 213
            elif dies_at(b, op):
                register_in, s1, s2, form = b, a, c, 213
            else:
                copy = _copy(c, bits)
                new_ops.append(copy)
                register_in, s1, s2, form = copy.results[0], a, b, 231
            new = x86ext.FmaOp(
                register_in, s1, s2, kind=kind, form=form, lane_bitwidth=bits
            )
        else:
            a, c = op.a, op.c
            if dies_at(c, op):
                register_in, s1, form = c, a, 231
            elif dies_at(a, op):
                register_in, s1, form = a, c, 132
            else:
                copy = _copy(c, bits)
                new_ops.append(copy)
                register_in, s1, form = copy.results[0], a, 231
            new = x86ext.FmaMemOp(
                register_in,
                s1,
                op.memory,
                op.memory_offset.value.data,
                broadcast=bool(op.broadcast.value.data),
                kind=kind,
                form=form,
                lane_bitwidth=bits,
            )
        new_ops.append(new)
        for new_op in new_ops:
            position[new_op] = position[op]
        Rewriter.replace_op(op, new_ops, [new.register_out])


def _copy(value: SSAValue, bits: int) -> Operation:
    op_type = x86.ops.DS_VmovapdOp if bits == 64 else x86.ops.DS_VmovapsOp
    return op_type(value, destination=_unallocated(value))


@dataclass(frozen=True)
class FftLegalizePass(ModulePass):
    """Lower the FFT expansions' negations and fused multiply-adds."""

    name = "fft-legalize"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in list(op.body.block.ops):
            if isinstance(func, x86_func.FuncOp) and func.body.blocks:
                folder = _NegationFolder(func)
                folder.run()
                _legalize_fmas(func)
                if folder._pool is not None:
                    folder._pool.finalize()
