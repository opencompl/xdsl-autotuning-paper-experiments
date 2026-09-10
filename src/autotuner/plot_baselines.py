"""A sweep over square matmuls, per data type.

    uv run plot-baselines data/rapper/f32.squares.jsonl data/rapper/f64.squares.jsonl \\
        --output plots/baselines.rapper.pdf

The figure is one column wide, with one panel per input file -- f32 and f64 of
the same machine.  Inside a panel the x axis is the problem size, with
M = N = K set to each of the sizes the dataset measured, and the y axis is
throughput as a share of machine peak.  All of the machine's implementations
share the panel, so which of them the generated kernels land on can be read off
directly.

One machine per figure: the machine's name goes in the file name, and its
display name in the LaTeX caption, so nothing here has to label it.

The small end of a sweep is tens of cycles per point, short enough that
anything else the machine is doing lands in the number, so the datasets hold
several passes over every sample and this figure draws the fastest of them --
see `plot_data.best_of_repeats`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from autotuner.plot_data import best_of_repeats, percent_of_peak
from autotuner.plot_style import (
    COLUMN_WIDTH,
    GRID,
    save,
    use_paper_style,
    variant_style,
)
from autotuner.plot_throughput import result_machine_label

# The implementations this figure puts side by side, in legend order: LIBXSMM
# first, then the two compilers we generate from its schedule, then the vendor
# libraries.  The x86 dialect kernel is not here: the squares figure is where
# it is priced against LIBXSMM, and on this axis it lands on LIBXSMM.
VARIANTS = (
    "libxsmm",
    "compxsmm",
    "libxtcmm",
    "mkl",
    "aocl",
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
# top, the size label and the legend underneath.
MARGIN_LEFT = 0.40
MARGIN_RIGHT = 0.12
MARGIN_TOP = 0.16
MARGIN_BOTTOM = 0.58

# Stroke widths, from the first variant to the last.  LIBXSMM and the kernels
# generated from its schedule agree almost everywhere, so each of those curves
# is drawn thinner than the one it lands on: where they coincide the earlier
# curves stay visible as a halo around the later ones instead of being painted
# over.  The narrowing bottoms out at ``THINNEST``, so the vendor libraries
# past that family are all drawn at one hairline width.
WIDEST = 1.8
NARROWING = 0.35
THINNEST = 0.7

# Ticks on the size axis: the ends, and every sixteenth size between them.  A
# panel that swept a shorter range simply shows the ticks that fall inside it.
X_TICKS = (1, 16, 32, 48, 64)


def panel_title(df: pd.DataFrame) -> str:
    """The data type the panel holds fixed, e.g. ``f64``."""
    dtypes = set(df["dtype"])
    if len(dtypes) != 1:
        raise ValueError("a panel needs one dtype per file")
    (dtype,) = dtypes
    return str(dtype)


def sizes(df: pd.DataFrame) -> list[int]:
    """The sizes the dataset measured, refusing one that is not square."""
    if not (df["M"].eq(df["N"]) & df["M"].eq(df["K"])).all():
        raise ValueError("this figure needs a dataset whose shapes all have M = N = K")
    return sorted(df["M"].unique())


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
    """Draw one square sweep, y-labelled only on the panel that carries the axis."""
    measured = sizes(df)
    ax.set_xlim(measured[0], measured[-1])
    ax.set_ylim(0, Y_TOP)

    # Peak, as a hairline rather than a dashed key of its own: at this size a
    # dash pattern reads as another curve, and the top tick already names it.
    ax.axhline(100, linewidth=0.3, color=GRID, zorder=1)

    for index, variant in enumerate(variants):
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        if group.empty:
            continue
        group = group.sort_values("M")
        # No markers: sixty-four of them per curve would cover the curve itself,
        # so color, dash pattern and stroke width carry the distinction.
        style = variant_style(variant) | {
            "marker": "none",
            "linewidth": max(WIDEST - NARROWING * index, THINNEST),
        }
        ax.plot(group["M"], group["percent"], zorder=2, **style)

    ax.set_xticks(list(X_TICKS))
    ax.set_yticks(list(Y_TICKS))
    ax.set_xlabel("M = N = K")
    if ticked:
        ax.set_ylabel("% of peak")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def legend_below_panels(fig: Figure, variants: Sequence[str], *, ncol: int = 3) -> None:
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
    """Plot % of peak against square problem size, one panel per dataset."""
    if not dfs:
        raise ValueError("this figure needs at least one dataset")

    machines = {result_machine_label(df)[0] for df in dfs}
    if len(machines) != 1:
        raise ValueError(
            f"one machine per figure, got {sorted(machines)}; "
            "plot each machine into its own file"
        )

    # The minimum is taken after the measured rows are picked out, so an
    # unmeasured 0 cannot win a sample's minimum.
    panels = [best_of_repeats(percent_of_peak(df)) for df in dfs]
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
        description="Plot % of peak against square problem size, one panel per file."
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
