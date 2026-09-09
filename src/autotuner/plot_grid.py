"""A grid of K sweeps over the nano-kernels, one small panel per (M, N) shape.

    uv run plot-grid data/rapper/f64.nanokernel_grid.jsonl \\
        --output plots/f64.nanokernel_grid.rapper.pdf

The figure fits one column of the paper template: M runs down the rows, N runs
across the columns, and inside every panel the x axis is K.  All panels share
their limits, so only the bottom-left one is ticked and the rest are read off
the M and N headers around the grid.  M is the swept dimension and keeps the y
axis, so its sixteen values take the rows and N's seven take the columns: the
figure comes out a column wide and tall rather than a page wide and squat.

Each curve is one nano-kernel, pinned rather than picked by the heuristic, so
which of them wins at a given tile shape can be read off the grid.  fsdbcst and
narrow fsdbcst both support one f64 vector of tile M, which is the matrix's M,
so they appear only in the top rows; the rest of the grid is nofsdbcst on its
own.  The two one-vector kernels run the same dataflow on different register
banks, so the top rows are where the narrow bank is paid for or is not: from a
full M vector up the narrowest bank covering the tile *is* the full vector and
the two differ only in fsdbcst's duplicated accumulators, while below it narrow
fsdbcst does useful work in every lane where fsdbcst masks lanes off.

A panel's points are tens of cycles apiece, so the dataset holds several passes
over every sample and this figure draws the fastest of them -- see
`best_of_repeats`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from autotuner.datasets import NANOKERNEL_VARIANTS
from autotuner.plot_style import (
    COLUMN_WIDTH,
    GRID,
    INK,
    INK_MUTED,
    save,
    use_paper_style,
    variant_style,
)

# The nano-kernels this figure puts side by side, in legend order.
VARIANTS = NANOKERNEL_VARIANTS

# What identifies one measurement, so what the repeated passes of the dataset
# have in common and `best_of_repeats` groups by.
SAMPLE_KEY = ("variant", "M", "N", "K")

# Top of the % of peak axis, and the ticks drawn below it.
Y_TOP = 112.0
Y_TICKS = (0, 50, 100)

# Panel height as a fraction of its width, and the gap between neighbouring
# panels as a fraction of one panel.
PANEL_ASPECT = 0.8
PANEL_GAP = 0.18

# Margins in inches: the M header and the ticked panel's labels on the left,
# the N header on top, the K label and the legend underneath.
MARGIN_LEFT = 0.50
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.34
MARGIN_BOTTOM = 0.52

# The index headers are smaller than the body text: there are 32 of them, and
# they are furniture rather than data.
HEADER_SIZE = 5.0
INDEX_SIZE = 8.0

# Curves are thinner than in a single-panel figure, where a panel is ten times
# this wide, but not so thin that two of them stop being separable.
LINE_WIDTH = 0.65


def percent_of_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``percent`` column holding throughput as a share of machine peak."""
    measured = df[df["time"] > 0].copy()
    assert isinstance(measured, pd.DataFrame)

    peaks = measured["peak"].dropna().unique()
    if len(peaks) != 1:
        raise ValueError(f"expected one peak in the dataset, found {sorted(peaks)}")
    peak = float(peaks[0])
    if peak == 0.0:
        raise ValueError("the dataset has no peak, so % of peak is undefined")

    measured["percent"] = (measured["flops"] / measured["time"]) / peak * 100
    return measured


def best_of_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each sample's fastest pass, dropping the rest.

    The grid is swept several times over -- `datasets.DATASET_REPEATS` -- because
    a single nano-kernel invocation is short enough that anything else the
    machine happens to be doing shows up in the number.  That noise is
    one-sided: it can only ever make a kernel look slower than it is, never
    faster, so the fastest pass is the least disturbed estimate of the kernel
    rather than a lucky outlier.  A dataset with one pass per sample comes
    through unchanged.
    """
    # Sort and deduplicate rather than group and take the minimum: this keeps
    # the whole row of the pass that won, and it does not care whether the
    # frame's index labels are unique the way a read straight from jsonl is.
    kept = df.sort_values("time").drop_duplicates(list(SAMPLE_KEY))
    assert isinstance(kept, pd.DataFrame)
    return kept


def grid_figure(
    nrows: int, ncols: int, *, width: float = COLUMN_WIDTH
) -> tuple[Figure, Any]:
    """A column-wide grid of panels, with the margins sized in inches.

    ``tight_layout`` is deliberately not used: the M and N headers are figure
    text placed against these margins, so they have to stay put.
    """
    use_paper_style()

    axes_width = width - MARGIN_LEFT - MARGIN_RIGHT
    panel_width = axes_width / (ncols + (ncols - 1) * PANEL_GAP)
    panel_height = panel_width * PANEL_ASPECT
    axes_height = panel_height * (nrows + (nrows - 1) * PANEL_GAP)
    height = axes_height + MARGIN_TOP + MARGIN_BOTTOM

    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(
        left=MARGIN_LEFT / width,
        right=1 - MARGIN_RIGHT / width,
        bottom=MARGIN_BOTTOM / height,
        top=1 - MARGIN_TOP / height,
        wspace=PANEL_GAP,
        hspace=PANEL_GAP,
    )
    return fig, axs


def draw_panel(
    ax: Axes,
    panel: pd.DataFrame,
    *,
    variants: Sequence[str],
    ks: Sequence[int],
    ticked: bool,
) -> None:
    """Draw one K sweep, ticked only if this is the panel that carries the key."""
    ax.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
    ax.set_ylim(0, Y_TOP)

    # Peak, as a hairline rather than the dashes a single-panel figure uses:
    # at this size a dash pattern reads as another curve.
    ax.axhline(100, linewidth=0.3, color=GRID, zorder=1)

    for variant in variants:
        group = panel[panel["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        if group.empty:
            continue
        group = group.sort_values("K")
        style = variant_style(variant) | {
            "marker": "none",
            "linewidth": LINE_WIDTH,
        }
        ax.plot(group["K"], group["percent"], zorder=2, **style)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.3)

    ax.set_xticks([min(ks), (min(ks) + max(ks)) // 2, max(ks)])
    ax.set_yticks(Y_TICKS)
    if not ticked:
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        return

    # The one key panel: a percent sign on the top tick names the y unit, so
    # no axis label has to compete with the M header for the left margin.
    ax.set_yticklabels([f"{t}%" if t == max(Y_TICKS) else str(t) for t in Y_TICKS])
    ax.tick_params(labelsize=HEADER_SIZE, length=1.2, width=0.3, pad=1.0)
    ax.set_xlabel("K", fontsize=HEADER_SIZE, labelpad=1.0)


def label_grid(fig: Figure, axs: Any, ms: Sequence[int], ns: Sequence[int]) -> None:
    """Write the M and N indices in the margins around the grid.

    ``ms`` indexes the rows and ``ns`` the columns, the way the panels are laid
    out: M is the swept dimension and keeps the y axis.
    """
    width, height = fig.get_size_inches()

    # Column headers, sitting on top of the first row of panels.
    top = axs[0, 0].get_position().y1
    for column, n in enumerate(ns):
        box = axs[0, column].get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            top + 0.03 / height,
            str(n),
            ha="center",
            va="bottom",
            fontsize=HEADER_SIZE,
            color=INK_MUTED,
        )
    fig.text(
        (axs[0, 0].get_position().x0 + axs[0, -1].get_position().x1) / 2,
        1 - 0.03 / height,
        "N",
        ha="center",
        va="top",
        fontsize=INDEX_SIZE,
        color=INK,
    )

    # Row labels, just left of the column of y tick labels on the key panel.
    row_label_x = (MARGIN_LEFT - 0.20) / width
    for row, m in enumerate(ms):
        box = axs[row, 0].get_position()
        fig.text(
            row_label_x,
            (box.y0 + box.y1) / 2,
            str(m),
            ha="right",
            va="center",
            fontsize=HEADER_SIZE,
            color=INK_MUTED,
        )
    fig.text(
        0.04 / width,
        (axs[-1, 0].get_position().y0 + axs[0, 0].get_position().y1) / 2,
        "M",
        ha="left",
        va="center",
        rotation=90,
        fontsize=INDEX_SIZE,
        color=INK,
    )


def legend_below_grid(fig: Figure, variants: Sequence[str]) -> None:
    """One legend for the whole grid, in the bottom margin.

    The keys are built from the palette rather than harvested from a panel, so
    the order is the caller's and an empty panel cannot leave a curve out.
    """
    handles: list[Line2D] = []
    labels: list[str] = []
    for variant in variants:
        style = variant_style(variant)
        handles.append(
            Line2D(
                [],
                [],
                color=str(style["color"]),
                linestyle=style["linestyle"],
                # A touch heavier than the panels: a hairline dash pattern is
                # not readable in a legend key this short.
                linewidth=LINE_WIDTH * 1.5,
            )
        )
        labels.append(str(style["label"]))

    height = fig.get_size_inches()[1]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03 / height),
        ncol=len(variants),
    )


def plot_grid(
    df: pd.DataFrame,
    *,
    variants: Sequence[str] = VARIANTS,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Plot % of peak against K for every (M, N) in the dataset."""
    # The minimum is taken after the measured rows are picked out, so an
    # unmeasured 0 cannot win a sample's minimum.
    df = best_of_repeats(percent_of_peak(df))
    missing = [v for v in variants if v not in set(df["variant"])]
    if missing:
        raise ValueError(f"the dataset has no samples for {missing}")

    ms = sorted(df["M"].unique())
    ns = sorted(df["N"].unique())
    ks = sorted(df["K"].unique())

    fig, axs = grid_figure(len(ms), len(ns), width=width)
    panels = dict(tuple(df.groupby(["M", "N"])))
    for row, m in enumerate(ms):
        for column, n in enumerate(ns):
            panel = panels.get((m, n), df.iloc[:0])
            assert isinstance(panel, pd.DataFrame)
            draw_panel(
                axs[row, column],
                panel,
                variants=variants,
                ks=ks,
                ticked=(row, column) == (len(ms) - 1, 0),
            )

    label_grid(fig, axs, ms, ns)
    legend_below_grid(fig, variants)

    # No tight crop: the margins above were sized to the paper's text width.
    save(fig, output_path, tight=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a grid of K sweeps, one panel per (M, N) shape."
    )
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help=f"Variant to draw, repeatable (default: {' '.join(VARIANTS)})",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=COLUMN_WIDTH,
        help="Figure width in inches (default: one column of the template)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_grid(
        df,
        variants=args.variant or VARIANTS,
        width=args.width,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
