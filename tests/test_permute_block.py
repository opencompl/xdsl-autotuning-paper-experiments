import re
import pytest

from xdsl.dialects import test
from xdsl.dialects.x86.ops import LabelOp
from xdsl.ir import Block
from xdsl.builder import ImplicitBuilder

from autotuner.permute_block import permute, generate_adjacency


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


def test_generate_adjacency():
    block = Block()

    with ImplicitBuilder(block):
        LabelOp("start")
        test.TestWriteOp()
        test.TestReadOp()
        test.TestWriteOp()
        test.TestReadOp()
        test.TestReadOp()
        test.TestWriteOp()
        test.TestTermOp()

        adj = generate_adjacency(block)
        expected_edges = {
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 3),
            (2, 6),
            (3, 4),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 6),
            (5, 6),
        }

        # Check terminators and labels do not move and have all other ops dependent on them
        num_ops = len(block.ops)
        for fixed_idx in [0, 7]:
            for i in range(num_ops):
                if i != fixed_idx:
                    expected_edges.add((fixed_idx, i))

        actual_edges = set()
        for src in adj.sources():
            for tgt in adj.targets_for_source(src):
                actual_edges.add((src, tgt))
        assert expected_edges == actual_edges
