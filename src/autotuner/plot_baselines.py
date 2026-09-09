"""A sweep over the blocked dimension of one large tile, per data type.

    uv run plot-baselines data/rapper/f32.ttile.jsonl data/rapper/f64.ttile.jsonl \\
        --output plots/baselines.rapper.pdf

The figure is one column wide, with one panel per input file -- f32 and f64 of
the same machine.  Inside a panel the x axis is N, the dimension the kernel
blocks, with M = K fixed at the tile the dataset measured, and the y axis is
throughput as a share of machine peak.  All of the machine's implementations
share the panel, so which of them the generated kernels land on can be read off
directly.

One machine per figure: the machine's name goes in the file name, and its
display name in the LaTeX caption, so nothing here has to label it.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from autotuner.plot_style import (
    COLUMN_WIDTH,
    GRID,
    save,
    use_paper_style,
    variant_style,
)
from autotuner.plot_throughput import result_machine_label

# The implementations this figure puts side by side, in legend order: the
# LIBXSMM family first -- the baseline and the three things we generate from
# its schedule -- then the vendor libraries and the unscheduled C.
VARIANTS = (
    "libxsmm",
    "xdsl_libxsmm",
    "compxsmm",
    "libxtcmm",
    "mkl",
    "aocl",
    "naive_c",
)

# Top of the % of peak axis: 100 is the top gridline, with just enough room
# above it that the curves touching peak are not clipped by the frame.
Y_TOP = 104.0
Y_TICKS = (0, 25, 50, 75, 100)

# Panel height as a fraction of its width, and the gap between the panels as a
# fraction of one panel.
PANEL_ASPECT = 0.85
PANEL_GAP = 0.12

# Margins in inches: the y label and its ticks on the left, the panel titles on
# top, the N label and the legend underneath.
MARGIN_LEFT = 0.40
MARGIN_RIGHT = 0.12
MARGIN_TOP = 0.16
MARGIN_BOTTOM = 0.58

# Stroke widths, from the first variant to the last.  The LIBXSMM family -- the
# baseline and the three kernels generated from its schedule -- agrees almost
# everywhere, so each of those curves is drawn thinner than the one it lands on:
# where they coincide the earlier curves stay visible as a halo around the later
# ones instead of being painted over.  The narrowing bottoms out at ``THINNEST``,
# so the variants past that family are all drawn at one hairline width.
WIDEST = 1.8
NARROWING = 0.35
THINNEST = 0.7

# Ticks on the N axis: where the sweep starts, then every ``X_TICK_STEP`` up to
# where it ends.  The two panels sweep different ranges, so the ticks are per
# panel and the shared y axis is what ties them together.
X_TICK_STEP = 20


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


def panel_title(df: pd.DataFrame) -> str:
    """The data type and the tile the panel holds fixed, e.g. ``f64, M = K = 64``."""
    dtypes = set(df["dtype"])
    ms = set(df["M"])
    ks = set(df["K"])
    if not len(dtypes) == len(ms) == len(ks) == 1:
        raise ValueError("a panel needs one dtype and one M = K tile per file")
    (dtype,) = dtypes
    (m,) = ms
    (k,) = ks
    if m != k:
        raise ValueError(f"this figure needs M = K, got M = {m}, K = {k}")
    return f"{dtype}, M = K = {m}"


def sweep_figure(ncols: int, *, width: float = COLUMN_WIDTH) -> tuple[Figure, Any]:
    """A column-wide row of panels, with the margins sized in inches.

    ``tight_layout`` is deliberately not used: the bottom margin is what holds
    the legend, so it has to keep the height reserved for it here.
    """
    use_paper_style()

    axes_width = width - MARGIN_LEFT - MARGIN_RIGHT
    panel_width = axes_width / (ncols + (ncols - 1) * PANEL_GAP)
    panel_height = panel_width * PANEL_ASPECT
    height = panel_height + MARGIN_TOP + MARGIN_BOTTOM

    fig, axs = plt.subplots(
        1, ncols, figsize=(width, height), squeeze=False, sharey=True
    )
    fig.subplots_adjust(
        left=MARGIN_LEFT / width,
        right=1 - MARGIN_RIGHT / width,
        bottom=MARGIN_BOTTOM / height,
        top=1 - MARGIN_TOP / height,
        wspace=PANEL_GAP,
    )
    return fig, axs[0]


def draw_panel(
    ax: Axes,
    df: pd.DataFrame,
    *,
    variants: Sequence[str],
    ticked: bool,
) -> None:
    """Draw one N sweep, y-labelled only on the panel that carries the axis."""
    ns = sorted(df["N"].unique())
    ax.set_xlim(ns[0], ns[-1])
    ax.set_ylim(0, Y_TOP)

    # Peak, as a hairline rather than a dashed key of its own: at this size a
    # dash pattern reads as another curve, and the top tick already names it.
    ax.axhline(100, linewidth=0.3, color=GRID, zorder=1)

    for index, variant in enumerate(variants):
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        if group.empty:
            continue
        group = group.sort_values("N")
        # No markers: twenty of them per curve would cover the curve itself, so
        # color, dash pattern and stroke width carry the distinction.
        style = variant_style(variant) | {
            "marker": "none",
            "linewidth": max(WIDEST - NARROWING * index, THINNEST),
        }
        ax.plot(group["N"], group["percent"], zorder=2, **style)

    ax.set_xticks([ns[0], *range(X_TICK_STEP, ns[-1] + 1, X_TICK_STEP)])
    ax.set_yticks(list(Y_TICKS))
    ax.set_xlabel("N")
    if ticked:
        ax.set_ylabel("% of peak")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def legend_below_panels(fig: Figure, variants: Sequence[str], *, ncol: int = 4) -> None:
    """One legend for both panels, in the bottom margin.

    The keys are built from the palette rather than harvested from a panel, so
    the order is the caller's and a variant missing from one panel still shows
    the style it has in the other.
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
                linewidth=WIDEST * 0.7,
            )
        )
        labels.append(str(style["label"]))

    height = fig.get_size_inches()[1]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02 / height),
        ncol=ncol,
    )


def plot_baselines(
    dfs: Sequence[pd.DataFrame],
    *,
    variants: Sequence[str] = VARIANTS,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Plot % of peak against N, one panel per dataset and one curve per variant."""
    if not dfs:
        raise ValueError("this figure needs at least one dataset")

    machines = {result_machine_label(df)[0] for df in dfs}
    if len(machines) != 1:
        raise ValueError(
            f"one machine per figure, got {sorted(machines)}; "
            "plot each machine into its own file"
        )

    panels = [percent_of_peak(df) for df in dfs]
    # Only the variants this machine actually measured, in the caller's order.
    measured = {variant for panel in panels for variant in panel["variant"]}
    drawn = [variant for variant in variants if variant in measured]
    if not drawn:
        raise ValueError(f"the datasets have no samples for {list(variants)}")

    fig, axs = sweep_figure(len(panels), width=width)
    for index, (ax, panel) in enumerate(zip(axs, panels)):
        draw_panel(ax, panel, variants=drawn, ticked=index == 0)
        ax.set_title(panel_title(panel))

    legend_below_panels(fig, drawn)

    # No tight crop: the margins above were sized to the paper's column width.
    save(fig, output_path, tight=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot % of peak against N, one panel per input file."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input JSONL data files, one panel each, all of the same machine",
    )
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
        help="Figure width in inches (default: one column of the paper template)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    dfs = [pd.read_json(path, lines=True) for path in args.inputs]
    plot_baselines(
        dfs,
        variants=args.variant or VARIANTS,
        width=args.width,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
