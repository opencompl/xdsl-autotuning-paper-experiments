from typing import Sequence
from xdsl.ir import Block


def permute(block: Block, orderings: Sequence[int]) -> None:
    tuple_ops = tuple(block.ops)
    assert sorted(orderings) == list(range(len(tuple_ops))), (
        "Invalid permutation provided"
    )
    for op in block.ops:
        op.detach()
    for i in orderings:
        block.add_op(tuple_ops[i])
