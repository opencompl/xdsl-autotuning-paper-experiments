"""Line plots of throughput along one matrix dimension.

    uv run plot-sweep data/tower/f64.mn_sweep.jsonl --x mn --output plot.pdf

Handles the paper's two sweep figures (square M = N kernels at fixed K, and a
sweep over N at fixed M and K) as well as any other dataset in which a single
dimension varies.
"""

import pandas as pd
from matplotlib.axes import Axes

from pathlib import Path

from autotuner.plot_ttile import plot_axis_throughput
from autotuner.plot_style import column_figure, legend_below, save

AXES = ("M", "N", "K")


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
    output_path: Path | None = None,
) -> None:
    """Plot one throughput sweep, sized for a single paper column."""
    df, x_row = sweep_column(df, axis)

    fig, ax = column_figure()
    assert isinstance(ax, Axes)

    plot_axis_throughput(df, ax, x_row=x_row)

    # No title: the caption belongs in the LaTeX figure, not in the image.
    legend_below(fig, ax, ncol=min(3, df["variant"].nunique() + 1))

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
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_sweep(df, axis=args.x, output_path=args.output)


if __name__ == "__main__":
    main()
