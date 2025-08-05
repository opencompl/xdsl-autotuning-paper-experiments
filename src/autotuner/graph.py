from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet, MutableSet
from typing import Any, Generic, NamedTuple, Self, TypeAlias

from typing_extensions import TypeVar

Node = TypeVar("Node", bound=Hashable)


class CycleException(ValueError):
    def __str__(self) -> str:
        return "Graph is not acyclic."


class AbstractAdjacency(Generic[Node], ABC):
    @classmethod
    @abstractmethod
    def empty(cls) -> MutableSet[Node]: ...

    @abstractmethod
    def sources(self) -> AbstractSet[Node]: ...

    @abstractmethod
    def targets(self) -> AbstractSet[Node]: ...

    @abstractmethod
    def targets_for_source(self, source: Node) -> AbstractSet[Node]: ...

    @abstractmethod
    def targets_for_sources(self, sources: AbstractSet[Node]) -> AbstractSet[Node]: ...

    @abstractmethod
    def sources_for_target(self, target: Node) -> AbstractSet[Node]: ...

    @abstractmethod
    def sources_for_targets(self, targets: AbstractSet[Node]) -> AbstractSet[Node]: ...

    def nodes(self) -> AbstractSet[Node]:
        return self.sources() | self.targets()

    def roots(self) -> AbstractSet[Node]:
        return self.sources() - self.targets()

    def has_edge(self, source: Node, target: Node) -> bool:
        return target in self.targets_for_source(source)

    def has_cycles(self) -> bool:
        if not (targets := self.targets()):
            return False

        if not (roots := self.sources() - targets):
            return True

        # Invariants:
        # `seen` is superset of `roots`
        # all sources of `roots` are in `seen`
        seen = roots

        while roots:
            # All targets for current roots
            root_targets = self.targets_for_sources(roots)
            # All sources for the targets, these may not be the same as roots
            root_targets_sources = self.sources_for_targets(root_targets)
            # Remove all those sources that we've seen before
            unseen_root_targets_sources = root_targets_sources - seen
            # The next roots are nodes that are pointed to only by `seen` or `roots`
            next_roots = root_targets - self.targets_for_sources(
                unseen_root_targets_sources
            )
            if not seen.isdisjoint(next_roots):
                return True
            seen |= next_roots
            roots = next_roots

        # If there are any nodes that are not in seen then they are a part of a cycle
        return seen < self.nodes()

    def assert_acyclic(self):
        if self.has_cycles():
            raise CycleException()


EdgeValue = TypeVar("EdgeValue", bound=Hashable)


class Edge(Generic[Node, EdgeValue], NamedTuple):
    source: Node
    target: Node
    value: EdgeValue


class AbstractGraph(Generic[Node, EdgeValue], AbstractAdjacency[Node], ABC):
    @abstractmethod
    def edges(self) -> AbstractSet[Edge[Node, EdgeValue]]: ...

    @abstractmethod
    def edges_from(self, source: Node) -> AbstractSet[Edge[Node, EdgeValue]]: ...

    @abstractmethod
    def edges_to(self, target: Node) -> AbstractSet[Edge[Node, EdgeValue]]: ...

    @abstractmethod
    def edge_values_between(
        self, source: Node, target: Node
    ) -> AbstractSet[EdgeValue]: ...

    def has_node(self, node: Node) -> bool:
        return node in self.nodes()

    def has_edge(
        self, source: Node, target: Node, value: EdgeValue | None = None
    ) -> bool:
        values = self.edge_values_between(source, target)
        return bool(values) if value is None else value in values


class AbstractMutableGraph(AbstractGraph[Node, EdgeValue], ABC):
    @abstractmethod
    def insert_node(self, node: Node) -> bool: ...

    @abstractmethod
    def delete_node(self, node: Node) -> bool: ...

    @abstractmethod
    def insert_edge(self, source: Node, target: Node, value: EdgeValue) -> bool: ...

    @abstractmethod
    def delete_edge(self, source: Node, target: Node, value: EdgeValue) -> bool: ...


Index: TypeAlias = int

EMPTY_INDEX_SET = frozenset[Index]()


class IntAdjacency(AbstractAdjacency[Index]):
    _targets_by_source: dict[Index, set[Index]]
    _sources_by_target: dict[Index, set[Index]]

    def __init__(
        self,
        _targets_by_source: dict[Index, set[Index]] | None = None,
        _sources_by_target: dict[Index, set[Index]] | None = None,
    ):
        self._targets_by_source = (
            {} if _targets_by_source is None else _targets_by_source
        )
        self._sources_by_target = (
            {} if _sources_by_target is None else _sources_by_target
        )

    # AbstractAdjacency

    @classmethod
    def empty(cls) -> MutableSet[Index]:
        return set()

    def sources(self) -> AbstractSet[Index]:
        return self._targets_by_source.keys()

    def targets(self) -> AbstractSet[Index]:
        return self._sources_by_target.keys()

    def targets_for_source(self, source: Index) -> AbstractSet[Index]:
        return self._targets_by_source.get(source, EMPTY_INDEX_SET)

    def targets_for_sources(self, sources: AbstractSet[Index]) -> AbstractSet[Index]:
        res = EMPTY_INDEX_SET.copy()
        for source in sources:
            res |= self.targets_for_source(source)
        return res

    def sources_for_target(self, target: Index) -> AbstractSet[Index]:
        return self._sources_by_target.get(target, EMPTY_INDEX_SET)

    def sources_for_targets(self, targets: AbstractSet[Index]) -> AbstractSet[Index]:
        res = EMPTY_INDEX_SET.copy()
        for target in targets:
            res |= self.sources_for_target(target)
        return res

    def insert_edge(self, source: Index, target: Index, value: None = None) -> bool:
        if self.has_edge(source, target):
            return False

        if source in self._targets_by_source:
            self._targets_by_source[source].add(target)
        else:
            self._targets_by_source[source] = {target}
        if target in self._sources_by_target:
            self._sources_by_target[target].add(source)
        else:
            self._sources_by_target[target] = {source}
        return True

    # IntAdjacency

    @classmethod
    def from_tuples(cls, edges: Iterable[tuple[Index, Index]]) -> Self:
        res = cls()
        for source, target in edges:
            res.insert_edge(source, target)
        return res


class IntGraph(IntAdjacency, AbstractMutableGraph[Index, None]):
    _nodes: set[Index]

    def __init__(
        self,
        _nodes: set[Index] | None = None,
        _targets_by_source: dict[Index, set[Index]] | None = None,
        _sources_by_target: dict[Index, set[Index]] | None = None,
    ):
        self._nodes = set() if _nodes is None else _nodes
        super().__init__(_targets_by_source, _sources_by_target)

    # AbstractGraph

    def nodes(self) -> AbstractSet[Index]:
        return self._nodes

    # AbstractmutableGraph

    def insert_node(self, node: Index) -> bool:
        if node in self._nodes:
            return False
        self._nodes.add(node)
        return True

    def delete_node(self, node: Index) -> bool:
        if node not in self.nodes():
            return False
        self._nodes.remove(node)
        del self._sources_by_target[node]
        del self._targets_by_source[node]
        return True

    def insert_edge(self, source: Index, target: Index, value: None = None) -> bool:
        if self.has_edge(source, target):
            return False

        if not self.insert_node(source):
            self._targets_by_source[source].add(target)
        else:
            self._targets_by_source[source] = {target}
        if self.insert_node(target):
            self._sources_by_target[target] = {source}
        else:
            self._sources_by_target[target].add(source)
        return True

    def delete_edge(self, source: Index, target: Index, value: None = None) -> bool:
        if not self.has_edge(source, target):
            return False

        self._targets_by_source[source].remove(target)
        self._sources_by_target[target].remove(source)
        return True


def _iter_topological_sort(
    g: AbstractAdjacency[Node],
    root_nodes: Sequence[Node],
    explored_nodes: MutableSet[Node],
) -> Iterator[tuple[Node, ...]]:
    """
    Assumes that the input is acyclic.
    """
    if not root_nodes:
        yield ()
        return

    for i, root in enumerate(root_nodes):
        # Add root to explored nodes
        explored_nodes.add(root)
        # Remove root from root nodes
        # Check for new roots (nodes whose sources are in the explored nodes)
        new_root_nodes = [*root_nodes[:i], *root_nodes[i + 1 :]]
        for target in g.targets_for_source(root):
            if g.sources_for_target(target) <= explored_nodes:
                new_root_nodes.append(target)
        # The new root nodes are the remaining roots plus any new candidates
        for permutation in _iter_topological_sort(g, new_root_nodes, explored_nodes):
            yield (root, *permutation)
        explored_nodes.remove(root)


def iter_topological_sort(g: AbstractAdjacency[Node]) -> Iterator[Sequence[Node]]:
    g.assert_acyclic()
    root_nodes = g.nodes() - g.targets()
    empty = g.empty()
    yield from _iter_topological_sort(g, tuple(root_nodes), empty)


def topological_sort(g: AbstractGraph[Node, Any]) -> Sequence[Node]:
    return next(iter_topological_sort(g))
