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
    block0 = Block([a, b, c, d, e, f])
    orderings = [2, 5, 0, 4, 1, 3]
    permute(block0, orderings)

    assert tuple(block0.ops) == (c, f, a, e, b, d)

    with pytest.raises(AssertionError, match="Invalid permutation provided"):
        a = test.TestOp()
        b = test.TestOp()
        c = test.TestOp()
        block1 = Block([a, b, c])
        orderings_invalid = [2, 5, 6]
        permute(block1, orderings_invalid)