"""A per-function pool of floating-point constants in read-only data.

The pool is addressed through one register, loaded with ``lea`` at the start
of the function, so the expansions can use constants as memory operands. Its
state lives entirely in the IR -- the ``lea`` and the ``x86ext.rodata`` after
the function -- so a later pass can append to it.
"""

import struct
from collections.abc import Sequence

from xdsl.dialects import x86, x86_func
from xdsl.dialects.builtin import Float32Type, Float64Type, ModuleOp
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint, Rewriter

from autotuner.dialects.x86ext import LeaRipOp, RodataOp

ALIGNMENT = 64


def _bits(value: float, element_bits: int) -> int:
    if element_bits == 64:
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0]


class ConstantPool:
    """The constants of one function, all of one floating-point type."""

    def __init__(self, func: x86_func.FuncOp, datatype: Float32Type | Float64Type):
        self.func = func
        self.element_bits = datatype.bitwidth
        self.label = f".L{func.sym_name.data}_constants"
        self.words: list[int] = []
        self.offsets: dict[tuple[int, ...], int] = {}
        self._pointer: SSAValue | None = None
        self._rodata: RodataOp | None = None

        for op in func.body.block.ops:
            if isinstance(op, LeaRipOp) and op.symbol.data == self.label:
                self._pointer = op.destination
        module = func.parent_op()
        assert isinstance(module, ModuleOp)
        for op in module.body.block.ops:
            if isinstance(op, RodataOp) and op.label.data == self.label:
                if op.word_bits.value.data != self.element_bits:
                    raise ValueError("a function's constants must share one type")
                self._rodata = op
                self.words = [int(word, 16) for word in op.words.data.split()]

    @property
    def element_size(self) -> int:
        return self.element_bits // 8

    @property
    def pointer(self) -> SSAValue:
        """The register holding the pool's address."""
        if self._pointer is None:
            lea = LeaRipOp(self.label, destination=x86.registers.UNALLOCATED_REG64)
            Rewriter.insert_op(lea, InsertPoint.at_start(self.func.body.block))
            self._pointer = lea.destination
        return self._pointer

    def offset_of_bits(self, words: Sequence[int]) -> int:
        """The byte offset of these words, appended if not yet in the pool."""
        key = tuple(words)
        if key not in self.offsets:
            # Search before appending, so that constants a previous pass added
            # are found again.
            alignment_words = min(len(key) * self.element_size, ALIGNMENT) // (
                self.element_size
            )
            for start in range(0, len(self.words) - len(key) + 1, alignment_words):
                if tuple(self.words[start : start + len(key)]) == key:
                    self.offsets[key] = start * self.element_size
                    break
            else:
                while len(self.words) % alignment_words:
                    self.words.append(0)
                self.offsets[key] = len(self.words) * self.element_size
                self.words.extend(key)
        return self.offsets[key]

    def offset(self, values: Sequence[float]) -> int:
        """The byte offset of a vector of constants (or of one, to broadcast)."""
        return self.offset_of_bits([_bits(v, self.element_bits) for v in values])

    def sign_mask_offset(self) -> int:
        """The byte offset of a scalar with only the sign bit set."""
        return self.offset_of_bits([1 << (self.element_bits - 1)])

    def finalize(self) -> None:
        """Write the pool to the read-only data section after the function."""
        if not self.words:
            return
        rodata = RodataOp(self.label, 6, self.element_bits, self.words)
        if self._rodata is None:
            Rewriter.insert_op(rodata, InsertPoint.after(self.func))
        else:
            Rewriter.replace_op(self._rodata, rodata)
        self._rodata = rodata
