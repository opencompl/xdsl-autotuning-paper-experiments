"""A sweep over square matmuls, our kernels and the vendor libraries together.

    uv run plot-squares-baselines data/rapper/f64.squares.jsonl \\
        --output plots/f64.squares_baselines.rapper.pdf

This is the squares figure (``plot_squares``) with MKL and AOCL drawn alongside
the kernels we generate, so one panel answers both questions at once: what the
register allocator costs against LIBXSMM, and where all of that lands against
the vendor libraries.

The paper puts four of these -- f32 and f64 of two machines -- in a 2x2 grid
spanning the page width, so a panel is half the page wide and the four are
sized identically: the margins are fixed in inches and the axes sit at the same
place in every panel, whether or not it carries an axis label, so the four tile
without drifting out of alignment.  Which labels a panel carries is the
caller's to say (``--no-ylabel`` for the right column, ``--no-xlabel`` for the
top row), and the legend is a figure of its own (``--legend-only``) to be set
underneath all four.

The panel names the machine it measured, since four of them share a caption:
the microarchitecture and the data type sit in the bottom right corner of the
axes, on a translucent patch so a curve passing under them still reads.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from autotuner.machines import MACHINES
from autotuner.plot_data import best_of_repeats
from autotuner.plot_style import (
    GRID,
    PAGE_WIDTH,
    save,
    use_paper_style,
    variant_style,
)
from autotuner.plot_throughput import result_machine_label

# The implementations this figure puts side by side, in legend order: LIBXSMM
# first, then everything we generate from its schedule, then the two vendor
# libraries.  It is ``plot_squares``' list with MKL and AOCL appended, and the
# legend that goes under the grid is built from exactly this order.
VARIANTS = (
    "libxsmm",
    "xdsl_libxsmm",
    "compxsmm",
    "compxsmm_manual",
    "compxsmm_plusnarrow",
    "libxtcmm",
    "mkl",
    "aocl",
)

# Half of the page width: two of these sit side by side in the paper's grid.
PANEL_WIDTH = PAGE_WIDTH / 2

# Top of the % of peak axis: 100 is the top gridline, with just enough room
# above it that the curves touching peak are not clipped by the frame.
Y_TOP = 104.0
Y_TICKS = (0, 25, 50, 75, 100)

# Ticks on the size axis: the ends, and every sixteenth size between them.
X_TICKS = (1, 16, 32, 48, 64)

# Panel height as a fraction of the width of its axes.
PANEL_ASPECT = 0.60

# Margins in inches: the y label and its ticks on the left, the size label and
# its ticks underneath, and on top only the half of the ``100`` tick label that
# rises above the frame -- the panel's name is inside the axes, not over them.
# The margins do not depend on which labels the panel was asked for: a panel
# without a y label keeps the room for one, because the four panels of the grid
# have to put their axes in the same place as each other or the columns stop
# lining up.
MARGIN_LEFT = 0.44
MARGIN_RIGHT = 0.10
MARGIN_TOP = 0.07
MARGIN_BOTTOM = 0.34

# Stroke widths, from the first variant to the last.  LIBXSMM and the kernels
# generated from its schedule agree almost everywhere, so each of those curves
# is drawn thinner than the one it lands on: where they coincide the earlier
# curves stay visible as a halo around the later ones instead of being painted
# over.  The narrowing bottoms out at ``THINNEST``, so the vendor libraries
# past that family are all drawn at one hairline width.
WIDEST = 1.8
NARROWING = 0.28
THINNEST = 0.6

# Where the panel's name sits, in axes coordinates, and how much room its
# patch leaves around the text.  The bottom right corner is the one the curves
# leave alone: they climb away from the origin and stay high, and the slowest
# of them is still above this by the time it reaches the right hand edge.
LABEL_POSITION = (0.97, 0.04)
LABEL_PAD = 0.28

# The patch under the name is translucent rather than opaque: where a curve
# does pass beneath it, it should show through rather than be cut in two.
LABEL_ALPHA = 0.78

# The paper's name for each machine family, so a panel's name says what the
# hardware is rather than which host it was measured on.  A family that is not
# named here falls back to its own key.
MICROARCHITECTURES = {
    "zen4": "Zen 4",
    "zen5": "Zen 5",
    "cascadelake": "Cascade Lake",
    "emeraldrapids": "Emerald Rapids",
    "icelake-server": "Ice Lake",
    "skylake-avx512": "Skylake-SP",
    "apple-m2-max": "Apple M2 Max",
    "generic-x86-64": "x86-64",
}


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


def sizes(df: pd.DataFrame) -> list[int]:
    """The sizes the dataset measured, refusing one that is not square."""
    if not (df["M"].eq(df["N"]) & df["M"].eq(df["K"])).all():
        raise ValueError("this figure needs a dataset whose shapes all have M = N = K")
    return sorted(df["M"].unique())


def microarchitecture(df: pd.DataFrame) -> str:
    """The display name of the microarchitecture the dataset was measured on.

    The family is read from the machine configuration rather than from the
    dataset's own ``family`` column, so the name a panel shows is the one the
    rest of the harness agrees on.
    """
    machine, _ = result_machine_label(df)
    family = MACHINES[machine].family
    return MICROARCHITECTURES.get(family, family)


def dtype(df: pd.DataFrame) -> str:
    """The data type the dataset holds fixed, e.g. ``f64``."""
    dtypes = set(df["dtype"])
    if len(dtypes) != 1:
        raise ValueError(f"one dtype per figure, got {sorted(dtypes)}")
    (value,) = dtypes
    return str(value)


def panel_name(df: pd.DataFrame) -> str:
    """What the panel measured: the microarchitecture and the data type."""
    return f"{microarchitecture(df)} · {dtype(df)}"


def label_panel(ax: Axes, text: str) -> None:
    """Name the panel inside its own axes, in the corner the curves leave free.

    Four of these share one caption, so each has to say which machine and data
    type it is.  It goes in the axes rather than above them: a title over the
    frame costs the grid a band of height per row, and with the panels this
    small that band is worth more to the curves.
    """
    x, y = LABEL_POSITION
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        # Above the curves, so the patch covers them rather than the reverse.
        zorder=3,
        bbox={
            "facecolor": "white",
            "alpha": LABEL_ALPHA,
            "edgecolor": GRID,
            "linewidth": 0.4,
            "boxstyle": f"round,pad={LABEL_PAD}",
        },
    )


def panel_figure(*, width: float = PANEL_WIDTH) -> tuple[Figure, Any]:
    """One panel, with its margins sized in inches.

    ``tight_layout`` is deliberately not used, and neither is a tight crop on
    the way out: the point of sizing the margins here is that every panel of
    the grid comes out the same size, which a crop to the ink would undo.
    """
    use_paper_style()

    axes_width = width - MARGIN_LEFT - MARGIN_RIGHT
    height = axes_width * PANEL_ASPECT + MARGIN_TOP + MARGIN_BOTTOM

    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(
        left=MARGIN_LEFT / width,
        right=1 - MARGIN_RIGHT / width,
        bottom=MARGIN_BOTTOM / height,
        top=1 - MARGIN_TOP / height,
    )
    return fig, ax


def line_width(index: int) -> float:
    """The stroke width of the ``index``-th curve, in legend order."""
    return max(WIDEST - NARROWING * index, THINNEST)


def draw(
    ax: Axes,
    df: pd.DataFrame,
    variants: Sequence[str],
    *,
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
    """Draw one curve per variant, over the sizes the dataset measured."""
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
        # No markers: sixty-four of them per curve would cover the curve
        # itself, so color, dash pattern and stroke width carry the
        # distinction.
        style = variant_style(variant) | {
            "marker": "none",
            "linewidth": line_width(index),
        }
        ax.plot(group["M"], group["percent"], zorder=2, **style)

    ax.set_xticks(list(X_TICKS))
    ax.set_yticks(list(Y_TICKS))
    # The ticks stay on every panel; only the axis names are the caller's to
    # drop, since in the grid one name per row and per column is enough.
    if xlabel:
        ax.set_xlabel("problem size (M = N = K)")
    if ylabel:
        ax.set_ylabel("% of peak")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def plot_squares_baselines(
    df: pd.DataFrame,
    *,
    variants: Sequence[str] = VARIANTS,
    width: float = PANEL_WIDTH,
    xlabel: bool = True,
    ylabel: bool = True,
    title: bool = True,
    output_path: Path | None = None,
) -> None:
    """Plot % of peak against square problem size, one curve per variant."""
    # The minimum is taken after the measured rows are picked out, so an
    # unmeasured 0 cannot win a sample's minimum.
    df = best_of_repeats(percent_of_peak(df))
    missing = [v for v in variants if v not in set(df["variant"])]
    if missing:
        raise ValueError(f"the dataset has no samples for {missing}")

    fig, ax = panel_figure(width=width)
    draw(ax, df, variants, xlabel=xlabel, ylabel=ylabel)
    if title:
        label_panel(ax, panel_name(df))

    # No tight crop: the margins above were sized so the four panels match.
    save(fig, output_path, tight=False)


def plot_legend(
    *,
    variants: Sequence[str] = VARIANTS,
    width: float = PAGE_WIDTH,
    ncol: int = 4,
    output_path: Path | None = None,
) -> None:
    """The legend on its own, to be set underneath the grid of panels.

    The keys are built from the palette rather than harvested from a panel, so
    the order is the caller's and the strip does not depend on which panel it
    is placed under.  There is no frame: the strip sits on the page rather than
    inside an axes, and a box around it would read as a figure of its own.
    """
    use_paper_style()

    handles: list[Line2D] = []
    labels: list[str] = []
    for index, variant in enumerate(variants):
        style = variant_style(variant)
        handles.append(
            Line2D(
                [],
                [],
                color=str(style["color"]),
                linestyle=style["linestyle"],
                # A touch heavier than the panels: a hairline dash pattern is
                # not readable in a legend key this short.
                linewidth=max(line_width(index), 1.0),
            )
        )
        labels.append(str(style["label"]))

    rows = -(-len(variants) // ncol)
    fig = plt.figure(figsize=(width, 0.22 * rows))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False)

    # A tight crop here, unlike the panels: the strip should be exactly as wide
    # as its keys so it can be centred under the grid.
    save(fig, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plot % of peak against square problem size, our kernels and the "
            "vendor libraries in one panel."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input JSONL data file (not needed with --legend-only)",
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
        default=None,
        help=(
            "Figure width in inches "
            f"(default: {PANEL_WIDTH:.3g} for a panel, {PAGE_WIDTH:.3g} for the legend)"
        ),
    )
    parser.add_argument(
        "--xlabel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Name the size axis (drop it on the top row of the grid)",
    )
    parser.add_argument(
        "--ylabel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Name the % of peak axis (drop it on the right column of the grid)",
    )
    parser.add_argument(
        "--title",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Name the panel with its microarchitecture and data type",
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Draw the legend alone, page wide, instead of a panel",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=4,
        help="Keys per row in the legend (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    variants = args.variant or VARIANTS

    if args.legend_only:
        plot_legend(
            variants=variants,
            width=args.width if args.width is not None else PAGE_WIDTH,
            ncol=args.legend_columns,
            output_path=args.output,
        )
        return

    if args.input is None:
        parser.error("an input dataset is required unless --legend-only is given")

    df = pd.read_json(args.input, lines=True)
    plot_squares_baselines(
        df,
        variants=variants,
        width=args.width if args.width is not None else PANEL_WIDTH,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
