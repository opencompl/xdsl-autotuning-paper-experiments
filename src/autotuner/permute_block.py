from typing import Sequence
from xdsl.ir import Block


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
