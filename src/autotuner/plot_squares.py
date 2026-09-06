"""A sweep over square matmuls, one curve per implementation.

    uv run plot-squares data/rapper/f64.squares.jsonl --output plots/f64.squares.rapper.pdf

The figure is one column wide: the x axis is the problem size, with M = N = K
set to each of 1..64, and the y axis is throughput as a share of machine peak.
It puts LIBXSMM next to the two things we generate from it -- the x86 dialect
kernel, and CompXSMM with and without xDSL's register allocator -- so the price
of allocating registers rather than assigning them by hand is visible.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes

from autotuner.plot_style import (
    COLUMN_WIDTH,
    column_figure,
    legend_inside,
    save,
    tidy_axes,
    variant_style,
)

# The implementations this figure puts side by side, in legend order.
VARIANTS = ("libxsmm", "xdsl_libxsmm", "compxsmm", "compxsmm_manual")

# Top of the % of peak axis: 100 is the top gridline, with just enough room
# above it that the curves touching peak are not clipped by the frame.
Y_TOP = 104.0
Y_TICKS = (0, 25, 50, 75, 100)

# Ticks on the size axis: the ends, and every sixteenth size between them.
X_TICKS = (1, 16, 32, 48, 64)

# Stroke widths, from the first variant to the last.  The four curves agree
# almost everywhere, so each one is drawn thinner than the one it lands on:
# where they coincide the earlier curves stay visible as a halo around the
# later ones instead of being painted over.
WIDEST = 1.8
NARROWING = 0.4


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


def draw(ax: Axes, df: pd.DataFrame, variants: Sequence[str]) -> None:
    """Draw one curve per variant, over the sizes the dataset measured."""
    measured = sizes(df)
    ax.set_xlim(measured[0], measured[-1])
    ax.set_ylim(0, Y_TOP)

    for index, variant in enumerate(variants):
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        if group.empty:
            continue
        group = group.sort_values("M")
        # No markers: 64 of them per curve would cover the curve itself, so
        # color, dash pattern and stroke width carry the distinction.
        style = variant_style(variant) | {
            "marker": "none",
            "linewidth": WIDEST - NARROWING * index,
        }
        ax.plot(group["M"], group["percent"], **style)

    ax.set_xticks(list(X_TICKS))
    ax.set_yticks(list(Y_TICKS))
    ax.set_xlabel("problem size (M = N = K)")
    ax.set_ylabel("% of peak")
    tidy_axes(ax)


def plot_squares(
    df: pd.DataFrame,
    *,
    variants: Sequence[str] = VARIANTS,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Plot % of peak against square problem size, one curve per variant."""
    df = percent_of_peak(df)
    missing = [v for v in variants if v not in set(df["variant"])]
    if missing:
        raise ValueError(f"the dataset has no samples for {missing}")

    fig, ax = column_figure(width=width)
    draw(ax, df, variants)
    # The curves climb to the left and stay high, so the bottom right corner is
    # empty and the legend costs the figure no height.
    legend_inside(fig, ax, loc="lower right")
    save(fig, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot % of peak against square problem size, one curve per variant."
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
        help="Figure width in inches (default: one column of the paper template)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_squares(
        df,
        variants=args.variant or VARIANTS,
        width=args.width,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
