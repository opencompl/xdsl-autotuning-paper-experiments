"""Use xDSL's allocator with the mask bank missing from its pinned AVX-512 pool."""

from dataclasses import dataclass

from xdsl.backend.x86.arch import AVX512
from xdsl.backend.x86.register_allocation import X86RegisterAllocator
from xdsl.backend.x86.register_stack import X86RegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, x86_func
from xdsl.dialects.x86.registers import K
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class X86AllocateRegistersWithMasksPass(ModulePass):
    name = "x86-allocate-registers-with-masks"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # k0 encodes "no mask" in writemask operands: only k1-k7 are usable.
        # Keep the GP/vector pool, order, and allocation algorithm unchanged.
        registers = (*AVX512.default_allocatable_registers(), *K[1:])
        for func in op.walk():
            if isinstance(func, x86_func.FuncOp):
                stack = X86RegisterStack.get(reversed(registers))
                X86RegisterAllocator(stack).allocate_func(func)
