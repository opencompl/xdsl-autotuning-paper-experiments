import re
import pytest

from xdsl.dialects import test
from xdsl.ir import Block

from autotuner.permute_block import permute


def test_permute_block_ops():
    # Operations on these constants
    a = test.TestOp()
    b = test.TestOp()
    c = test.TestOp()
    d = test.TestOp()
    e = test.TestOp()
    f = test.TestOp()

    # Create Block from operations
    block = Block([a, b, c, d, e, f])
    orderings = [2, 5, 0, 4, 1, 3]
    permute(block, orderings)

    assert tuple(block.ops) == (c, f, a, e, b, d)

    with pytest.raises(
        ValueError, match=re.escape("Indices [1, 1] are not a valid permutation")
    ):
        permute(block, [1, 1])

    with pytest.raises(ValueError, match="Invalid permutation length 2, expected 6"):
        permute(block, [0, 1])
