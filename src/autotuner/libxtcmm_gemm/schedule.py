"""Apply a LIBXSMM schedule (`XtcGemmPlan`) to a matmul via the XTC
scheduler and write the resulting XTC source IR (the linalg payload plus the
schedule as the transform dialect, before it is applied).

Dimension mapping. LIBXSMM is column-major; XTC's ``linalg.matmul``
``C[I,J] += A[I,K] * B[K,J]`` is row-major, so J (the last dim) is contiguous and
is the axis we vectorize. Hence:

    LIBXSMM M (vectorized)      -> XTC J (contiguous, named "M")
    LIBXSMM N (split / blocked) -> XTC I (rows, named "N")
    K                           -> K

so the tensors are ``A=(n, k)``, ``B=(k, m)`` -> ``C=(n, m)`` and we vectorize M.

Loop nest: N, M and K are each split into their tiers, nested outer-to-inner,
matching LIBXSMM's ``for n: for m: for k:`` structure -- N into equalized
column tiers, M into whole-register-block tiers plus an optional remainder, K
into a rolled loop plus an optional unrolled remainder. Each (N tier, M tier,
K tier) leaf is the register-blocked microkernel.
"""

from __future__ import annotations

from pathlib import Path

import xtc.graphs.xtc.op as O
from xdsl.dialects.linalg.ops import FillOp
from xtc.backends.mlir import Backend
from xtc.itf.schd.scheduler import Scheduler

from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxtcmm_gemm.plan import XtcGemmPlan
from autotuner.tiling import BlockingRange

# LIBXSMM element datatype -> XTC dtype name.
_XTC_DTYPE = {Datatype.F32: "float32", Datatype.F64: "float64"}

# The scheduled dimensions, outer to inner (N rows, M contiguous/vectorized, K).
_DIMS = ("N", "M", "K")


def _split_tiers(
    scheduler: Scheduler,
    parent: str,
    dim: str,
    ranges: tuple[BlockingRange, ...],
) -> list[tuple[str, BlockingRange]]:
    """Split ``dim`` into its 1-or-2 tiers under ``parent``.

    Returns the ``(root, range)`` pair per tier. A single tier is not split; a
    two-tier split keeps the loops outer to ``dim`` (the already-split dims) in
    place and adds the equal-length ``_top``/``_bot`` segments after them.
    """
    if len(ranges) == 1:
        return [(parent, ranges[0])]
    outer_loops = _DIMS[: _DIMS.index(dim)]
    top, bottom = ranges
    scheduler.split(dim, {f"{dim}_top": 0, f"{dim}_bot": top.extent}, root=parent)
    scheduler.interchange([*outer_loops, f"{dim}_top", f"{dim}_bot"], root=parent)
    return [(f"{parent}/{dim}_top", top), (f"{parent}/{dim}_bot", bottom)]


def _schedule_leaf(
    scheduler: Scheduler,
    root: str,
    n_range: BlockingRange,
    m_range: BlockingRange,
    k_range: BlockingRange,
    plan: XtcGemmPlan,
) -> None:
    """Schedule one (N tier, M tier, K tier) leaf as the register-blocked microkernel."""
    scheduler.tile("N", {"n_col": n_range.tile_size}, root=root)  # N columns
    # M loops over register blocks of m_range.tile_size, each block holding
    # ceil(m_range.tile_size / vector_length) zmm registers of one vector lane.
    # A single-block tier folds its (single-trip) M loop away.
    scheduler.tile(
        "M",
        {
            "m_reg": m_range.tile_size,
            "m_lane": min(plan.vector_length, m_range.tile_size),
        },
        root=root,
    )
    # K reduces in a rolled loop over k_range.tile_size-wide blocks, each block
    # a fully-unrolled body of k_range.tile_size steps. A single-block tier
    # (K below the unroll threshold) folds its K loop away -> K fully unrolled.
    scheduler.tile("K", {"k_step": k_range.tile_size}, root=root)
    scheduler.interchange(
        ["N", "M", "K", "n_col", "m_reg", "k_step", "m_lane"], root=root
    )
    # When the block width is not a multiple of the vector length its tail lane
    # is a masked (partial) vector, so vectorize to a fixed width and let XTC
    # mask the remainder -- exactly what LIBXSMM does with its mask register.
    # A dict pins the vector width (masked); a bare list vectorizes to the tile.
    m_lane: list[str] | dict[str, int | None] = (
        {"m_lane": plan.vector_length}
        if m_range.tile_size % plan.vector_length
        else ["m_lane"]
    )
    scheduler.vectorize(m_lane, root=root)
    # Unroll the K block body, the N columns, and the M vector registers within
    # a block (rounding up: the tail masked lane still counts as one register).
    m_regs = (m_range.tile_size + plan.vector_length - 1) // plan.vector_length
    scheduler.unroll(
        {"k_step": k_range.tile_size, "n_col": n_range.tile_size, "m_reg": m_regs},
        root=root,
    )


def emit_mlir(
    plan: XtcGemmPlan, desc: GEMMDescriptor, output: Path, routine_name: str = "matmul"
) -> None:
    """Build the matmul, apply ``plan``, and write the XTC source IR.

    The source IR is the ``linalg.matmul`` payload plus the schedule expressed
    as the transform dialect, before it is applied: a declarative record of the
    LIBXSMM schedule (tiling, splitting, vectorization, unrolling). The emitted
    function is named ``routine_name`` and has the direct row-major ABI
    ``void routine_name(A[M*K], B[K*N], C[M*N])`` computing ``C += A*B``, so the
    benchmark harness links it as ``matmul`` with no wrapper.
    """
    dtype = _XTC_DTYPE[desc.datatype.a]
    # Row-major swap: XTC I = LIBXSMM N (rows), XTC J = LIBXSMM M (contiguous).
    a = O.tensor((desc.n, desc.k), dtype, name="A")
    b = O.tensor((desc.k, desc.m), dtype, name="B")
    with O.graph(name=routine_name) as graph_builder:
        O.matmul(a, b, name="C")
    backend = Backend(graph_builder.graph)

    # LIBXSMM's ABI is beta=1 (C += A*B); drop the graph matmul's zero-fill of C.
    for op in tuple(backend.xdsl_func.walk()):
        if isinstance(op, FillOp):
            op.detach()

    scheduler = backend.get_scheduler()
    scheduler.set_dims(list(_DIMS))
    for n_root, n_range in _split_tiers(scheduler, ".", "N", plan.n_ranges):
        for m_root, m_range in _split_tiers(scheduler, n_root, "M", plan.m_ranges):
            for k_root, k_range in _split_tiers(scheduler, m_root, "K", plan.k_ranges):
                _schedule_leaf(scheduler, k_root, n_range, m_range, k_range, plan)
    scheduler.get_loop_nest().check()

    source_ir = backend.get_compiler().get_source_ir(scheduler.schedule())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source_ir)
