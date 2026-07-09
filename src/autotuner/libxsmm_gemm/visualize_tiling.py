"""Visualize the tiling/splitting tree of the libxsmm SSE/AVX/AVX2/AVX512 GEMM
generator as a minimal HTML diagram over the A, B and C matrices.

The generator's kernel builder splits the problem along N (outer), then M (middle),
then blocks along K (inner). Those decisions live as inline control flow inside
``libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel`` / ``..._kloop`` and are made by a
handful of *pure* helper functions. This module reuses those helper functions verbatim
and only mirrors the surrounding loop skeletons so that, instead of emitting x86 code,
it records the resulting tiles and renders them.

Only 64-bit float (F64) dense GEMM on the skylake (``LIBXSMM_X86_AVX512_SKX``)
architecture is currently supported by the generator, so that is what we visualize.

Run as::

    python -m autotuner.libxsmm_gemm.visualize_tiling --m 64 --n 16 --k 128 -o tiling.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from autotuner.libxsmm_gemm.generator_common import (
    MicroKernelConfig,
    libxsmm_compute_equalized_blocking,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking,
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype

# The only combination the generator currently supports.
ARCH = Arch.LIBXSMM_X86_AVX512_SKX


# --------------------------------------------------------------------------------------
# Tile data model
# --------------------------------------------------------------------------------------


@dataclass
class Block:
    """One tile along a single axis.

    ``start``/``size`` describe the covered range. ``loop_id`` groups tiles emitted by
    the same hardware loop (so the renderer can bracket them with a ``xN`` label).
    ``kind`` is a short tag used only for optional highlighting/labelling.
    """

    start: int
    size: int
    loop_id: int
    kind: str = ""  # "", "masked", "remainder", "unrolled"


@dataclass
class Tiling:
    m: int
    n: int
    k: int
    n_blocks: list[Block]
    m_blocks: list[Block]
    k_blocks: list[Block]
    # parameters worth surfacing in the caption
    max_n_blocking: int
    vector_length: int
    vector_reg_count: int
    k_blocking: int
    k_threshold: int


# --------------------------------------------------------------------------------------
# Split computation (reuses the generator's own decision functions)
# --------------------------------------------------------------------------------------


def _make_descriptor(m: int, n: int, k: int) -> GEMMDescriptor:
    f64 = Datatype.F64
    dt = DescDatatype(f64, f64, f64, f64)
    # Leading dimensions do not influence the tiling for a plain dense F64 GEMM; use the
    # natural column-major minimums (mirrors tests/libxsmm/test_generator_gemm.py).
    return GEMMDescriptor(m, n, k, m, k, m, dt, GEMMFlag(0), GEMMPrefetchType.NONE)


def _compute_max_n_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor
) -> int:
    """Mirror of generator_gemm_sse_avx_avx2_avx512.py:326-368.

    Start from the architecture cap, then shrink until the N accumulators plus the M
    block registers fit into the vector register file.
    """
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        config, desc, ARCH
    )
    if max_n_blocking > 3:
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, ARCH, 0
        )
        init_m_blocks = (
            init_m_blocking + config.vector_length - 1
        ) // config.vector_length
        # F64/SKX takes the else-branch (`init_m_blocks * max_n + init_m_blocks + 1`).
        while (
            init_m_blocks * max_n_blocking + init_m_blocks + 1
        ) > config.vector_reg_count:
            max_n_blocking -= 1
    assert max_n_blocking, "max_n_blocking collapsed to 0"
    return max_n_blocking


def _compute_n_blocks(
    config: MicroKernelConfig, desc: GEMMDescriptor, max_n_blocking: int
) -> list[Block]:
    """Mirror of the two-tier equalized split + the `while n_done` walk (:371, :450)."""
    blocking = libxsmm_compute_equalized_blocking(desc.n, max_n_blocking)
    n_N = [blocking.range_1, blocking.range_2]
    n_n = [blocking.block_1, blocking.block_2]
    assert n_N[0], "equalized blocking produced an empty first tier"

    blocks: list[Block] = []
    n_done = 0
    n_count = 0
    loop_id = 0
    while n_done != desc.n:
        n_blocking = n_n[n_count]
        tier_range = n_N[n_count]
        # Each tier is one hardware N-loop repeating tier_range // n_blocking tiles.
        start = n_done
        end = n_done + tier_range
        while start < end:
            blocks.append(Block(start, n_blocking, loop_id))
            start += n_blocking
        n_done += tier_range
        n_count += 1
        loop_id += 1
    return blocks


def _compute_m_blocks(config: MicroKernelConfig, desc: GEMMDescriptor) -> list[Block]:
    """Mirror of the `while m_done` walk (:474-740).

    The M split depends only on ``m`` (and arch/datatype) so it is identical for every
    N tile; we compute it once.
    """
    blocks: list[Block] = []
    m_done = 0
    loop_id = 0
    m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
        config, desc, ARCH, 0
    )
    while m_done != desc.m:
        assert m_blocking, "m_blocking collapsed to 0"
        m_done_old = m_done
        # Consume all full m_blocking chunks at once (one hardware M-loop).
        n_full = (desc.m - m_done_old) // m_blocking
        m_done = m_done_old + n_full * m_blocking
        if m_done != m_done_old:
            # get_m_blocking stores whether this block width needs masking.
            masked = "masked" if config.use_masking_a_c else ""
            start = m_done_old
            for _ in range(n_full):
                blocks.append(Block(start, m_blocking, loop_id, masked))
                start += m_blocking
            loop_id += 1
        # Recompute to obtain the (smaller) remainder block width.
        m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, ARCH, m_blocking
        )
    return blocks


def _compute_k_blocks(desc: GEMMDescriptor) -> tuple[list[Block], int, int]:
    """Mirror of libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop (:762-949).

    For F64/SKX the hard-coded parameters are k_blocking=4, k_threshold=23. Returns the
    blocks together with (k_blocking, k_threshold) for the caption.
    """
    k = desc.k
    k_blocking = 4
    k_threshold = 23

    blocks: list[Block] = []
    loop_id = 0

    if k % k_blocking == 0 and k_threshold < k:
        # Strategy 1: pure blocked loop.
        start = 0
        while start < k:
            blocks.append(Block(start, k_blocking, loop_id, "looped"))
            start += k_blocking
    elif k <= k_threshold:
        # Strategy 2: fully unrolled, single tile.
        blocks.append(Block(0, k, loop_id, "unrolled"))
    else:
        # Strategy 3: largest blocked part + remainder.
        max_blocked_k = (k // k_blocking) * k_blocking
        start = 0
        while start < max_blocked_k:
            blocks.append(Block(start, k_blocking, loop_id, "looped"))
            start += k_blocking
        loop_id += 1
        remainder = k - max_blocked_k
        if remainder:
            blocks.append(Block(max_blocked_k, remainder, loop_id, "remainder"))

    return blocks, k_blocking, k_threshold


def compute_tiling(m: int, n: int, k: int) -> Tiling:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n and k must all be positive")

    desc = _make_descriptor(m, n, k)
    config = MicroKernelConfig()
    libxsmm_generator_gemm_init_micro_kernel_config(config, ARCH, desc, False)
    assert config.vector_length, (
        "micro kernel config was not populated for F64/skylake"
    )

    max_n_blocking = _compute_max_n_blocking(config, desc)
    n_blocks = _compute_n_blocks(config, desc, max_n_blocking)
    m_blocks = _compute_m_blocks(config, desc)
    k_blocks, k_blocking, k_threshold = _compute_k_blocks(desc)

    # Each axis partition must exactly cover its dimension.
    assert sum(b.size for b in n_blocks) == n, "N blocks do not cover n"
    assert sum(b.size for b in m_blocks) == m, "M blocks do not cover m"
    assert sum(b.size for b in k_blocks) == k, "K blocks do not cover k"

    return Tiling(
        m=m,
        n=n,
        k=k,
        n_blocks=n_blocks,
        m_blocks=m_blocks,
        k_blocks=k_blocks,
        max_n_blocking=max_n_blocking,
        vector_length=config.vector_length,
        vector_reg_count=config.vector_reg_count,
        k_blocking=k_blocking,
        k_threshold=k_threshold,
    )


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------

# A single faint tint per special tile kind; everything else stays grayscale.
_KIND_FILL = {
    "": "transparent",
    "looped": "transparent",
    "unrolled": "transparent",
    "masked": "rgba(180,120,0,0.16)",  # M remainder that needs masking
    "remainder": "rgba(0,110,160,0.16)",  # K remainder
}


def _loop_groups(blocks: list[Block]) -> list[tuple[int, int, int]]:
    """Return (loop_id, count, size) for each contiguous run of one loop_id."""
    groups: list[tuple[int, int, int]] = []
    for b in blocks:
        if groups and groups[-1][0] == b.loop_id:
            lid, count, size = groups[-1]
            groups[-1] = (lid, count + 1, size)
        else:
            groups.append((b.loop_id, 1, b.size))
    return groups


def _tiles_html(
    blocks: list[Block],
    *,
    axis: str,  # "x" (blocks laid horizontally) or "y" (vertically)
    scale: float,
    min_px: float,
) -> str:
    """Render blocks as a flex row/column of tile <div>s, with per-loop labels."""
    direction = "row" if axis == "x" else "column"
    cells: list[str] = []
    prev_loop = None
    for b in blocks:
        extent = max(b.size * scale, min_px)
        size_css = (
            f"width:{extent:.2f}px" if axis == "x" else f"height:{extent:.2f}px"
        )
        fill = _KIND_FILL.get(b.kind, "transparent")
        new_group = "grp" if b.loop_id != prev_loop else ""
        prev_loop = b.loop_id
        title = f"{b.kind or 'tile'} @{b.start} size {b.size}"
        cells.append(
            f'<div class="tile {new_group}" style="{size_css};background:{fill}" '
            f'title="{title}"><span>{b.size}</span></div>'
        )
    return f'<div class="tiles {direction}">{"".join(cells)}</div>'


def _loop_labels_html(blocks: list[Block], *, axis: str, scale: float, min_px: float) -> str:
    """A row/column of `xN` labels, one per hardware loop, aligned to the tiles."""
    spans: list[str] = []
    for _lid, count, size in _loop_groups(blocks):
        extent = max(size * scale, min_px) * count
        dim = "width" if axis == "x" else "height"
        label = f"&times;{count}" if count > 1 else "&times;1"
        spans.append(
            f'<div class="looplbl" style="{dim}:{extent:.2f}px">{label}</div>'
        )
    direction = "row" if axis == "x" else "column"
    return f'<div class="loops {direction}">{"".join(spans)}</div>'


def _auto_scale(t: Tiling) -> float:
    """Pixels per matrix element that keeps the whole diagram within a sane canvas.

    Layout width spans A (k) + C (n); height spans B (k) + C (m). Pick the largest
    scale (capped at 14) that fits both, so large K/M problems shrink to fit while
    small ones stay legible.
    """
    max_w, max_h = 1100.0, 900.0
    scale = min(14.0, max_w / (t.k + t.n + 2), max_h / (t.k + t.m + 2))
    return max(scale, 1.5)


def render_html(
    tiling: Tiling, *, scale: float | None = None, min_px: float = 6.0
) -> str:
    t = tiling
    if scale is None:
        scale = _auto_scale(t)

    # Matrix bodies. Each matrix is a grid of tiles: rows x cols.
    #   A: rows = M split, cols = K split
    #   B: rows = K split, cols = N split
    #   C: rows = M split, cols = N split
    def matrix_grid(rows: list[Block], cols: list[Block]) -> str:
        # Build one flex column of rows; each row is a flex row of cells. Using nested
        # flex keeps row/column boundaries aligned across the three matrices.
        row_html: list[str] = []
        prev_row_loop = None
        for r in rows:
            h = max(r.size * scale, min_px)
            cell_html: list[str] = []
            prev_col_loop = None
            for c in cols:
                w = max(c.size * scale, min_px)
                # A cell's "kind" for tinting: prefer the row (M) kind, else col kind.
                kind = r.kind or c.kind
                fill = _KIND_FILL.get(kind, "transparent")
                col_grp = "cgrp" if c.loop_id != prev_col_loop else ""
                prev_col_loop = c.loop_id
                cell_html.append(
                    f'<div class="cell {col_grp}" '
                    f'style="width:{w:.2f}px;height:{h:.2f}px;background:{fill}"></div>'
                )
            row_grp = "rgrp" if r.loop_id != prev_row_loop else ""
            prev_row_loop = r.loop_id
            row_html.append(
                f'<div class="mrow {row_grp}">{"".join(cell_html)}</div>'
            )
        return f'<div class="matrix">{"".join(row_html)}</div>'

    a_grid = matrix_grid(t.m_blocks, t.k_blocks)
    b_grid = matrix_grid(t.k_blocks, t.n_blocks)
    c_grid = matrix_grid(t.m_blocks, t.n_blocks)

    n_axis = _loop_labels_html(t.n_blocks, axis="x", scale=scale, min_px=min_px)
    k_axis_x = _loop_labels_html(t.k_blocks, axis="x", scale=scale, min_px=min_px)
    m_axis_y = _loop_labels_html(t.m_blocks, axis="y", scale=scale, min_px=min_px)

    def split_summary(blocks: list[Block]) -> str:
        parts = [f"{count}&times;{size}" for _l, count, size in _loop_groups(blocks)]
        return " + ".join(parts)

    css = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 13px/1.4 -apple-system, system-ui, sans-serif; margin: 24px;
       color: #1a1a1a; background: #fff; }
h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
.caption { color: #555; margin: 0 0 20px; }
.caption code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }
.layout { display: grid;
          grid-template-columns: max-content max-content max-content;
          grid-template-rows: max-content max-content max-content;
          gap: 6px 10px; align-items: end; }
/* rows/cols:  [m-axis] [A/C label] [matrix]                                    */
.cell-B { grid-column: 3; grid-row: 1; }
.cell-A { grid-column: 1; grid-row: 3; }
.cell-C { grid-column: 3; grid-row: 3; }
.axis-mC { grid-column: 2; grid-row: 3; }
.axis-nC { grid-column: 3; grid-row: 4; }
.title { font-weight: 600; margin-bottom: 3px; }
.dim { color: #777; font-weight: 400; }
.matrix { display: flex; flex-direction: column;
          border: 2px solid #333; width: max-content; }
.mrow { display: flex; flex-direction: row; }
/* thin internal tile borders, heavier borders between hardware-loop groups */
.cell { border-right: 1px solid #ccc; border-bottom: 1px solid #ccc; }
.mrow .cell:last-child { border-right: none; }
.mrow:last-child .cell { border-bottom: none; }
.cell.cgrp { border-left: 1.5px solid #333; }
.mrow .cell.cgrp:first-child { border-left: none; }
.mrow.rgrp { border-top: 1.5px solid #333; }
.matrix .mrow.rgrp:first-child { border-top: none; }
/* loop-count labels */
.loops { display: flex; }
.loops.row { flex-direction: row; }
.loops.column { flex-direction: column; }
.looplbl { color: #333; font-size: 11px; text-align: center;
           border-top: 1px solid #999; padding: 2px 0; }
.loops.column .looplbl { border-top: none; border-left: 1px solid #999;
                         writing-mode: vertical-rl; padding: 0 2px; text-align: center; }
.axis-mC .loops.column { align-items: stretch; height: 100%; }
.legend { margin-top: 22px; color: #555; }
.legend .sw { display: inline-block; width: 12px; height: 12px; vertical-align: -1px;
              border: 1px solid #999; margin: 0 4px 0 14px; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181c; }
  .caption { color: #aaa; } .caption code { background: #2a2d33; }
  .matrix { border-color: #ccc; }
  .cell { border-right-color: #3a3d44; border-bottom-color: #3a3d44; }
  .cell.cgrp { border-left-color: #ccc; } .mrow.rgrp { border-top-color: #ccc; }
  .dim, .legend { color: #aaa; } .looplbl { color: #ccc; border-top-color: #666; }
  .loops.column .looplbl { border-left-color: #666; }
}
"""

    html = f"""<meta charset="utf-8">
<title>libxsmm GEMM tiling {t.m}x{t.n}x{t.k}</title>
<style>{css}</style>
<h1>libxsmm GEMM tiling &mdash; M={t.m}, N={t.n}, K={t.k} (F64, skylake)</h1>
<p class="caption">
N &rarr; M &rarr; K split tree. Heavy borders separate hardware-loop groups;
<code>&times;N</code> labels give each loop's repeat count. Tint marks
tiles that differ in kind.<br>
N: {split_summary(t.n_blocks)} &nbsp;&middot;&nbsp;
M: {split_summary(t.m_blocks)} &nbsp;&middot;&nbsp;
K: {split_summary(t.k_blocks)} &nbsp;&middot;&nbsp;
<code>max_n_blocking={t.max_n_blocking}</code>
<code>vector_length={t.vector_length}</code>
<code>k_blocking={t.k_blocking}</code>
<code>k_threshold={t.k_threshold}</code>
</p>

<div class="layout">
  <div class="cell-B">
    <div class="title">B <span class="dim">(K&times;N)</span></div>
    {b_grid}
  </div>

  <div class="cell-A">
    <div class="title">A <span class="dim">(M&times;K)</span></div>
    {a_grid}
    <div style="margin-top:4px">{k_axis_x}</div>
  </div>

  <div class="axis-mC">{m_axis_y}</div>
  <div class="cell-C">
    <div class="title">C <span class="dim">(M&times;N)</span></div>
    {c_grid}
  </div>
  <div class="axis-nC">{n_axis}</div>
</div>

<p class="legend">
<span class="sw" style="background:{_KIND_FILL['masked']}"></span>masked M remainder
<span class="sw" style="background:{_KIND_FILL['remainder']}"></span>K remainder
</p>
"""
    return html


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _text_summary(t: Tiling) -> str:
    def fmt(blocks: list[Block]) -> str:
        return " + ".join(
            f"{count}x{size}" for _l, count, size in _loop_groups(blocks)
        )

    return (
        f"M={t.m} N={t.n} K={t.k}  (F64, skylake)\n"
        f"  N split : {fmt(t.n_blocks)}   (max_n_blocking={t.max_n_blocking})\n"
        f"  M split : {fmt(t.m_blocks)}   (vector_length={t.vector_length})\n"
        f"  K split : {fmt(t.k_blocks)}   "
        f"(k_blocking={t.k_blocking}, k_threshold={t.k_threshold})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the libxsmm GEMM tiling tree (F64, skylake) as HTML."
    )
    parser.add_argument("--m", type=int, required=True, help="rows of C / A")
    parser.add_argument("--n", type=int, required=True, help="cols of C / B")
    parser.add_argument("--k", type=int, required=True, help="contraction dimension")
    parser.add_argument(
        "-o", "--out", default="tiling.html", help="output HTML file (default tiling.html)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="pixels per matrix element (default: auto-fit to canvas)",
    )
    args = parser.parse_args()

    tiling = compute_tiling(args.m, args.n, args.k)
    html = render_html(tiling, scale=args.scale)
    with open(args.out, "w") as f:
        f.write(html)

    print(_text_summary(tiling))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
