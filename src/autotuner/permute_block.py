from collections.abc import Iterator
from typing import Sequence
from xdsl.ir import Block

from xdsl.traits import MemoryEffectKind, get_effects
from xdsl.dialects.x86.ops import LabelOp
from xdsl.ir.core import Operation

from autotuner.graph import IntAdjacency, iter_topological_sort


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
    - Any memory write depends on all earlier memory reads and the previous write.
    - Any memory read depends on the previous memory write.
    - If the op is a label or a terminator, it must remain in place. Do not add to adjacency.
    - Any op with an allocated register effect depends on all earlier ops and all later ops depend on it

    """
    adjacency = IntAdjacency()

    for i, insn in enumerate(block.ops):
        last_write: int | None = None
        prev_reads: list[int] = []
        for i, insn in enumerate(block.ops):
            if has_effect(insn, MemoryEffectKind.WRITE):
                if last_write is not None:
                    adjacency.insert_edge(last_write, i)
                    for r in prev_reads:
                        adjacency.insert_edge(r, i)
                last_write = i
            elif last_write is not None and has_effect(insn, MemoryEffectKind.READ):
                adjacency.insert_edge(last_write, i)
                prev_reads.append(i)
    return adjacency


def iter_permutations(block: Block) -> Iterator[tuple[int, ...]]:
    adjacency = generate_adjacency(block)
    last_index = len(block.ops) - 1
    if isinstance(block.first_op, LabelOp):
        for p in iter_topological_sort(adjacency):
            yield (0, *p, last_index)
    else:
        for p in iter_topological_sort(adjacency):
            yield (*p, last_index)
