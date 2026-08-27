"""Line plots of throughput along one matrix dimension.

    uv run plot-sweep data/tower/f64.mn_sweep.jsonl --x mn --output plot.pdf

Handles the paper's two sweep figures (square M = N kernels at fixed K, and a
sweep over N at fixed M and K) as well as any other dataset in which a single
dimension varies.
"""

import pandas as pd
from matplotlib.axes import Axes

from pathlib import Path

from autotuner.plot_ttile import Y_TOP, plot_axis_throughput
from autotuner.plot_style import column_figure, legend_inside, save

AXES = ("M", "N", "K")

# Panel aspect when the y axis spans the whole 0..Y_TOP range.  A raised floor
# scales it down from here, so one inch of panel always holds the same number
# of percentage points and two sweeps stay comparable side by side.
FULL_ASPECT = 0.52


def sweep_column(df: pd.DataFrame, axis: str) -> tuple[pd.DataFrame, str]:
    """Return the frame to plot and the name of its x column.

    ``axis`` is one of ``M``, ``N``, ``K``, ``mn`` (the square M = N diagonal)
    or ``auto`` to pick the single dimension that varies.
    """
    if axis == "auto":
        varying = [a for a in AXES if df[a].nunique() > 1]
        match varying:
            case [only]:
                axis = only
            case ["M", "N"] if (df["M"] == df["N"]).any():
                axis = "mn"
            case _:
                raise ValueError(
                    f"cannot pick a sweep axis, these vary: {varying or 'none'}"
                )

    if axis != "mn":
        assert axis in AXES
        return df, axis

    square = df[df["M"] == df["N"]].copy()
    assert isinstance(square, pd.DataFrame)
    if square.empty:
        raise ValueError("no square (M = N) samples in the dataset")
    square = square.rename(columns={"M": "M,N"})
    del square["N"]
    assert isinstance(square, pd.DataFrame)
    return square, "M,N"


def plot_sweep(
    df: pd.DataFrame,
    *,
    axis: str = "auto",
    ymin: float = 0.0,
    output_path: Path | None = None,
) -> None:
    """Plot one throughput sweep, sized for a single paper column."""
    df, x_row = sweep_column(df, axis)

    # Shorter than a square panel: the curves live in the top half, so the
    # extra height was empty anyway.  Raising ``ymin`` crops the empty band off
    # the bottom of the axis and takes the panel height with it.
    fig, ax = column_figure(aspect=FULL_ASPECT * (Y_TOP - ymin) / Y_TOP)
    assert isinstance(ax, Axes)

    plot_axis_throughput(df, ax, x_row=x_row, ymin=ymin)

    # No title: the caption belongs in the LaTeX figure, not in the image.
    legend_inside(fig, ax)

    save(fig, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot kernel throughput along one matrix dimension."
    )
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--x",
        default="auto",
        choices=("auto", "M", "N", "K", "mn"),
        help="Dimension on the x axis ('mn' plots the square M = N diagonal)",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Bottom of the %% of peak axis; the panel is shortened to match",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_sweep(df, axis=args.x, ymin=args.ymin, output_path=args.output)


if __name__ == "__main__":
    main()
