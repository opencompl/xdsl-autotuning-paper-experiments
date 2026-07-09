"""Visualize the tiling/splitting tree of the libxsmm SSE/AVX/AVX2/AVX512 GEMM
generator as a minimal matplotlib SVG diagram over the A, B and C matrices.

The generator's kernel builder splits the problem along N (outer), then M (middle),
then blocks along K (inner). Those decisions live as inline control flow inside
``libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel`` / ``..._kloop`` and are made by a
handful of *pure* helper functions. This module reuses those helper functions verbatim
and only mirrors the surrounding loop skeletons so that, instead of emitting x86 code,
it records the resulting tiles and renders them.

Only 64-bit float (F64) dense GEMM on the skylake (``LIBXSMM_X86_AVX512_SKX``)
architecture is currently supported by the generator, so that is what we visualize.

Run as::

    # write a static SVG
    python -m autotuner.libxsmm_gemm.visualize_tiling --m 64 --n 16 --k 128 -o tiling.svg

    # scrub M/N/K live with sliders
    python -m autotuner.libxsmm_gemm.visualize_tiling --interactive --m 64 --n 16 --k 128
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


def _compute_max_n_blocking(config: MicroKernelConfig, desc: GEMMDescriptor) -> int:
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
    assert config.vector_length, "micro kernel config was not populated for F64/skylake"

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
# Rendering (matplotlib -> SVG)
# --------------------------------------------------------------------------------------

# A single faint tint per special tile kind; everything else stays uncolored.
_FILL_MASKED = "#f0e2c4"  # M remainder that needs masking
_FILL_REMAINDER = "#cfe3ef"  # K remainder
_GRID_THIN = "#c8c8c8"
_GRID_HEAVY = "#333333"
_EDGE = "#222222"


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


def _group_spans(blocks: list[Block]) -> list[tuple[int, int, int, int]]:
    """(start_offset, extent, count, size) per hardware-loop group, in element units."""
    spans: list[tuple[int, int, int, int]] = []
    pos = 0
    for _lid, count, size in _loop_groups(blocks):
        extent = count * size
        spans.append((pos, extent, count, size))
        pos += extent
    return spans


def _internal_boundaries(blocks: list[Block]) -> list[tuple[int, bool]]:
    """(position, heavy) for each internal tile boundary; heavy at loop-group change."""
    bounds: list[tuple[int, bool]] = []
    pos = 0
    for i, b in enumerate(blocks):
        if i > 0:
            bounds.append((pos, blocks[i - 1].loop_id != b.loop_id))
        pos += b.size
    return bounds


def _split_summary(blocks: list[Block]) -> str:
    return " + ".join(f"{count}×{size}" for _l, count, size in _loop_groups(blocks))


def _draw(ax, tiling: Tiling) -> None:
    """Draw one tiling onto ``ax`` (clears it first).

    All three matrices share the one Axes with equal aspect, so tile boundaries line
    up exactly:

        B  (K x N)            top-right   -> N columns align with C
        A  (M x K)  C (M x N) bottom row  -> M rows of A align with C

    Coordinates are in matrix-element units and y is measured downward (the axis is
    inverted at the end) so element (0,0) sits at each matrix's top-left corner. This
    is the single source of truth shared by the SVG export and the interactive window.
    """
    from matplotlib.patches import Patch, Rectangle

    t = tiling
    m, n, k = t.m, t.n, t.k

    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    gap = max(2.0, 0.05 * max(k + n, k + m))  # blank space between the matrices
    x0 = k + gap  # left edge of B and C
    y0 = k + gap  # top edge of A and C
    width = x0 + n
    height = y0 + m
    pad = max(1.0, 0.03 * max(width, height))

    # --- whole-band tints (drawn first, under the grid) ------------------------------
    # K remainder first, masked M second, so the shared corner in A reads as "masked"
    # (row priority), matching how the kernel loads/stores the masked M tail.
    for kb in t.k_blocks:
        if kb.kind == "remainder":
            ax.add_patch(
                Rectangle(
                    (kb.start, y0),
                    kb.size,
                    m,
                    facecolor=_FILL_REMAINDER,
                    edgecolor="none",
                    zorder=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x0, kb.start),
                    n,
                    kb.size,
                    facecolor=_FILL_REMAINDER,
                    edgecolor="none",
                    zorder=0,
                )
            )
    # A masked M block masks only its *tail* lanes: the last (size % vector_length)
    # rows, i.e. the partial vector left over after the full vector registers. Tint
    # just those rows (not the whole block) and mark where the full-vector part ends.
    for mb in t.m_blocks:
        if mb.kind == "masked":
            tail = mb.size % t.vector_length or mb.size
            ts = mb.start + mb.size - tail
            ax.add_patch(
                Rectangle(
                    (0, y0 + ts),
                    k,
                    tail,
                    facecolor=_FILL_MASKED,
                    edgecolor="none",
                    zorder=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x0, y0 + ts),
                    n,
                    tail,
                    facecolor=_FILL_MASKED,
                    edgecolor="none",
                    zorder=0,
                )
            )
            if tail < mb.size:
                for xlo, xhi in ((0, k), (x0, x0 + n)):
                    ax.plot(
                        [xlo, xhi],
                        [y0 + ts, y0 + ts],
                        color="#b58a2e",
                        lw=0.6,
                        ls=(0, (4, 2)),
                        zorder=2,
                    )

    # --- internal grid lines (thin per-tile, heavy between hardware-loop groups) ------
    def vlines(bounds, x_base, y_lo, y_hi):
        for pos, heavy in bounds:
            ax.plot(
                [x_base + pos, x_base + pos],
                [y_lo, y_hi],
                color=_GRID_HEAVY if heavy else _GRID_THIN,
                lw=1.6 if heavy else 0.5,
                zorder=2,
            )

    def hlines(bounds, y_base, x_lo, x_hi):
        for pos, heavy in bounds:
            ax.plot(
                [x_lo, x_hi],
                [y_base + pos, y_base + pos],
                color=_GRID_HEAVY if heavy else _GRID_THIN,
                lw=1.6 if heavy else 0.5,
                zorder=2,
            )

    m_b = _internal_boundaries(t.m_blocks)
    n_b = _internal_boundaries(t.n_blocks)
    k_b = _internal_boundaries(t.k_blocks)

    hlines(m_b, y0, 0, k)  # A: M rows
    vlines(k_b, 0, y0, y0 + m)  # A: K cols
    hlines(k_b, 0, x0, x0 + n)  # B: K rows
    vlines(n_b, x0, 0, k)  # B: N cols
    hlines(m_b, y0, x0, x0 + n)  # C: M rows
    vlines(n_b, x0, y0, y0 + m)  # C: N cols

    # --- outer borders ---------------------------------------------------------------
    for x, y, w, h in ((0, y0, k, m), (x0, 0, n, k), (x0, y0, n, m)):
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=_EDGE, lw=2, zorder=3)
        )

    # --- matrix titles (just above each matrix's top-left corner) --------------------
    tkw = dict(fontsize=11, fontweight="bold", va="bottom")
    ax.text(0, y0 - pad * 0.4, "A  (M×K)", ha="left", **tkw)
    ax.text(x0, 0 - pad * 0.4, "B  (K×N)", ha="left", **tkw)
    ax.text(x0, y0 - pad * 0.4, "C  (M×N)", ha="left", **tkw)

    # --- per-loop "count x size" labels + axis titles --------------------------------
    lab_y = y0 + m + pad * 0.6
    for x_base, blocks in ((x0, t.n_blocks), (0, t.k_blocks)):
        for start, extent, count, size in _group_spans(blocks):
            ax.text(
                x_base + start + extent / 2,
                lab_y,
                f"{count}×{size}",
                ha="center",
                va="top",
                fontsize=8,
            )
    ax.text(
        x0 + n / 2,
        lab_y + pad * 1.8,
        "N",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )
    ax.text(
        k / 2,
        lab_y + pad * 1.8,
        "K",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )

    lab_x = -pad * 0.6
    for start, extent, count, size in _group_spans(t.m_blocks):
        ax.text(
            lab_x,
            y0 + start + extent / 2,
            f"{count}×{size}",
            ha="right",
            va="center",
            fontsize=8,
            rotation=90,
        )
    ax.text(
        lab_x - pad * 1.8,
        y0 + m / 2,
        "M",
        ha="right",
        va="center",
        fontsize=10,
        fontstyle="italic",
        rotation=90,
    )

    # --- legend (only for tile kinds actually present) -------------------------------
    handles = []
    if any(b.kind == "masked" for b in t.m_blocks):
        handles.append(
            Patch(
                facecolor=_FILL_MASKED,
                edgecolor="#999",
                label="masked M tail (M % vector_length)",
            )
        )
    if any(b.kind == "remainder" for b in t.k_blocks):
        handles.append(
            Patch(facecolor=_FILL_REMAINDER, edgecolor="#999", label="K remainder")
        )
    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            fontsize=8,
        )

    ax.set_xlim(lab_x - pad * 2.6, width + pad)
    ax.set_ylim(0 - pad * 0.6, lab_y + pad * 3.0)
    ax.invert_yaxis()


def _title_and_caption(t: Tiling) -> tuple[str, str]:
    title = f"libxsmm GEMM tiling — M={t.m}, N={t.n}, K={t.k}  (F64, skylake)"
    caption = (
        f"N → M → K split.   "
        f"N: {_split_summary(t.n_blocks)}    "
        f"M: {_split_summary(t.m_blocks)}    "
        f"K: {_split_summary(t.k_blocks)}\n"
        f"max_n_blocking={t.max_n_blocking}   vector_length={t.vector_length}   "
        f"k_blocking={t.k_blocking}   k_threshold={t.k_threshold}"
    )
    return title, caption


def render_svg(tiling: Tiling, out_path: str) -> None:
    """Draw the tiling and write it to ``out_path`` as a standalone SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tiling
    gap = max(2.0, 0.05 * max(t.k + t.n, t.k + t.m))
    width = t.k + gap + t.n
    height = t.k + gap + t.m
    longest = max(width, height)
    fig, ax = plt.subplots(
        figsize=(8.0 * width / longest + 2.4, 8.0 * height / longest + 2.4)
    )
    _draw(ax, t)
    title, caption = _title_and_caption(t)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93, caption, ha="center", va="top", fontsize=9, color="#444")
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def run_interactive(m0: int, n0: int, k0: int, vmax: int = 128) -> None:
    """Open a window with M/N/K sliders that redraw the tiling live as you scrub."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if "agg" in plt.get_backend().lower():
        print(
            "warning: matplotlib is using a non-interactive backend "
            f"({plt.get_backend()!r}); no window will appear. Try installing a GUI "
            "backend (e.g. PyQt) or set MPLBACKEND (e.g. 'MacOSX')."
        )

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.2)
    # Persistent title/caption artists, updated in place so they never accumulate.
    sup = fig.suptitle("", fontsize=13, fontweight="bold")
    cap = fig.text(0.5, 0.9, "", ha="center", va="top", fontsize=9, color="#444")

    sliders = []
    for i, (label, init) in enumerate((("M", m0), ("N", n0), ("K", k0))):
        s_ax = fig.add_axes((0.15, 0.10 - i * 0.04, 0.7, 0.025))
        sliders.append(Slider(s_ax, label, 1, vmax, valinit=min(init, vmax), valstep=1))
    s_m, s_n, s_k = sliders

    def update(_event=None):
        t = compute_tiling(int(s_m.val), int(s_n.val), int(s_k.val))
        _draw(ax, t)
        title, caption = _title_and_caption(t)
        sup.set_text(title)
        cap.set_text(caption)
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    update()
    plt.show()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _text_summary(t: Tiling) -> str:
    def fmt(blocks: list[Block]) -> str:
        return " + ".join(f"{count}x{size}" for _l, count, size in _loop_groups(blocks))

    return (
        f"M={t.m} N={t.n} K={t.k}  (F64, skylake)\n"
        f"  N split : {fmt(t.n_blocks)}   (max_n_blocking={t.max_n_blocking})\n"
        f"  M split : {fmt(t.m_blocks)}   (vector_length={t.vector_length})\n"
        f"  K split : {fmt(t.k_blocks)}   "
        f"(k_blocking={t.k_blocking}, k_threshold={t.k_threshold})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the libxsmm GEMM tiling tree (F64, skylake)."
    )
    parser.add_argument("--m", type=int, default=64, help="rows of C / A (default 64)")
    parser.add_argument("--n", type=int, default=16, help="cols of C / B (default 16)")
    parser.add_argument(
        "--k", type=int, default=128, help="contraction dimension (default 128)"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="open a window with M/N/K sliders and redraw live (--m/--n/--k seed it)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=128,
        help="slider max in interactive mode (default 128)",
    )
    parser.add_argument(
        "-o", "--out", default="tiling.svg", help="output SVG file (default tiling.svg)"
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.m, args.n, args.k, vmax=args.max)
        return

    tiling = compute_tiling(args.m, args.n, args.k)
    render_svg(tiling, args.out)

    print(_text_summary(tiling))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
