from typing import Sequence
from xdsl.ir import Block

from xdsl.traits import IsTerminator, MemoryEffectKind, get_effects
from xdsl.dialects.x86.ops import LabelOp
from xdsl.ir.core import Operation

from autotuner.graph import IntAdjacency


def permute(block: Block, old_indices: Sequence[int]) -> None:
    """
    Reorders the operations in `block` according to the given permutation.
    After permuting, the operation at index `i` in the block used to be at
    `old_indices[i]`.
    """
    if sorted(old_indices) != list(range(len(old_indices))):
        raise ValueError(f"Indices {old_indices} are not a valid permutation")
    tuple_ops = tuple(block.ops)
    if len(old_indices) != len(tuple_ops):
        raise ValueError(
            f"Invalid permutation length {len(old_indices)}, expected {len(tuple_ops)}"
        )

    for op in block.ops:
        op.detach()
    for old_index in old_indices:
        block.add_op(tuple_ops[old_index])


def has_effect(op: Operation, effect: MemoryEffectKind) -> bool:
    """
    Returns if the operation has the given side effects, and possibly others.
    """
    effects = get_effects(op)
    return effects is not None and any(e.kind == effect for e in effects)


def generate_adjacency(block: Block) -> IntAdjacency:
    """
    Generates an adjacency from a basic block.
    Nodes are integer-valued, based on enumeration of the instructions in the input block.
    Edges represent the dependencies between them.

    Rules:
    - Any memory write depends on all earlier memory reads and writes.
    - Any memory read depends on all earlier memory writes.
    - If the op is a label or a terminator, it must remain in place. Do not add to adjacency.
    - Any op with an allocated register effect depends on all earlier ops and all later ops depend on it

    """
    adjacency = IntAdjacency()
    tuple_ops = tuple(block.ops)
    num_ops = len(tuple_ops)
    for i, insn in enumerate(block.ops):
        if insn.has_trait(IsTerminator) or isinstance(insn, LabelOp):
            continue
        if has_effect(insn, MemoryEffectKind.WRITE):
            for j in range(i + 1, num_ops - 1):
                if has_effect(tuple_ops[j], MemoryEffectKind.READ) or has_effect(
                    tuple_ops[j], MemoryEffectKind.WRITE
                ):
                    print(f"insn {i} has memory write. insn {j} depends on it")
                    adjacency.insert_edge(i, j)
        elif has_effect(insn, MemoryEffectKind.READ):
            for j in range(i + 1, num_ops - 1):
                if has_effect(tuple_ops[j], MemoryEffectKind.WRITE):
                    print(f"insn {i} has memory read. insn {j} depends on it")
                    adjacency.insert_edge(i, j)
    return adjacency
