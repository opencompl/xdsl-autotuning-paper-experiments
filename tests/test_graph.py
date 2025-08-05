import pytest

from autotuner.graph import (
    CycleException,
    IntAdjacency,
    iter_topological_sort,
)
from xdsl.transforms.memref_stream_interleave import factors


def test_int_adjacency():
    empty = IntAdjacency()
    assert not empty.sources()
    assert not empty.targets()
    assert not empty.targets_for_source(1)
    assert not empty.targets_for_sources({1, 2})
    assert not empty.sources_for_target(1)
    assert not empty.sources_for_targets({1, 2})
    assert not empty.nodes()
    assert not empty.roots()
    assert not empty.has_edge(1, 2)
    assert not empty.has_cycles()
    empty.assert_acyclic()

    #  1      5
    # 2 3    6 7
    #    4
    trees = IntAdjacency.from_tuples(((1, 2), (1, 3), (3, 4), (5, 6), (5, 7)))
    assert set(trees.sources()) == {1, 3, 5}
    assert set(trees.targets()) == {2, 3, 4, 6, 7}
    assert set(trees.targets_for_source(1)) == {2, 3}
    assert set(trees.targets_for_sources({1, 5})) == {2, 3, 6, 7}
    assert set(trees.sources_for_target(3)) == {1}
    assert set(trees.sources_for_targets({1, 2})) == {1}
    assert set(trees.nodes()) == set(range(1, 8))
    assert set(trees.roots()) == {1, 5}
    assert trees.has_edge(1, 2)
    assert not trees.has_edge(2, 1)
    assert not trees.has_edge(1, 5)
    assert not trees.has_cycles()
    trees.assert_acyclic()

    # a circle
    o = IntAdjacency.from_tuples(((1, 2), (2, 3), (3, 1)))
    assert set(o.sources()) == {1, 2, 3}
    assert set(o.targets()) == {1, 2, 3}
    assert set(o.targets_for_source(1)) == {2}
    assert set(o.targets_for_sources({1, 2})) == {2, 3}
    assert set(o.sources_for_target(1)) == {3}
    assert set(o.sources_for_targets({1, 2})) == {1, 3}
    assert set(o.nodes()) == {1, 2, 3}
    assert not o.roots()
    assert o.has_edge(1, 2)
    assert not o.has_edge(2, 1)
    assert o.has_cycles()
    with pytest.raises(CycleException):
        o.assert_acyclic()

    # a circle and a bar
    oi = IntAdjacency.from_tuples(((1, 2), (2, 3), (3, 1), (4, 5), (5, 6)))
    assert set(oi.sources()) == {1, 2, 3, 4, 5}
    assert set(oi.targets()) == {1, 2, 3, 5, 6}
    assert set(oi.targets_for_source(1)) == {2}
    assert set(oi.targets_for_sources({1, 2, 5})) == {2, 3, 6}
    assert set(oi.sources_for_target(1)) == {3}
    assert set(oi.sources_for_targets({1, 2, 5})) == {1, 3, 4}
    assert set(oi.nodes()) == {1, 2, 3, 4, 5, 6}
    assert set(oi.roots()) == {4}
    assert oi.has_edge(1, 2)
    assert not oi.has_edge(2, 1)
    assert oi.has_cycles()
    with pytest.raises(CycleException):
        oi.assert_acyclic()


def test_permutations():
    # A graph from numbers to their multiples up to 7
    oi = IntAdjacency.from_tuples(
        ((factor, i) for i in range(7) for factor in factors(i) if factor != i)
    )
    assert oi.has_edge(2, 4)
    assert not oi.has_edge(2, 5)

    permutations = set(
        "".join(str(i) for i in permutation)
        for permutation in iter_topological_sort(oi)
    )
    # 2 is never after 4 or 6
    # 3 is never after 6
    assert permutations == {
        "123456",
        "123465",
        "123546",
        "123564",
        "123645",
        "123654",
        "124356",
        "124365",
        "124536",
        "125346",
        "125364",
        "125436",
        "132456",
        "132465",
        "132546",
        "132564",
        "132645",
        "132654",
        "135246",
        "135264",
        "152346",
        "152364",
        "152436",
        "153246",
        "153264",
    }
