# uv run plot-heatmap data/tower/f64.mk_sweep.jsonl --output plot.pdf

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pathlib import Path

from autotuner.plot_style import (
    BASELINE,
    DIVERGING,
    INK,
    OURS,
    SEQUENTIAL,
    column_figure,
    contrasting_ink,
    save,
    sized_figure,
    sorted_variants,
    variant_label,
)


# Which end of the color bar means what.  Spelled out beside the bar so the
# sign of a cell reads without the caption; short, because they flank the bar.
BASELINE_BETTER = "LIBXSMM\nbetter"
OURS_BETTER = "x86 dialect\nbetter"

# Geometry of the difference figure, in inches.  Placed by hand because its
# parts want different widths: the cell grid is deliberately small, while the
# color bar row also has to hold the two captions above.
_GRID_WIDTH = 1.90  # cell grid
_GRID_HEIGHT = 1.45  # shorter than it is wide: the cells only hold one number
_Y_LABELS = 0.30  # M tick labels and the axis label, left of the grid
_X_LABELS = 0.24  # K tick labels and the axis label, under the grid
_BAR_WIDTH = 1.65
_BAR_HEIGHT = 0.085
_BAR_LABELS = 0.24  # color bar tick labels and its label
_SIDE_TEXT = 0.55  # room beside the bar for one of the captions
_GAP = 0.06  # between the grid's x label and the bar
_MARGIN = 0.03


def _column_dim(valid_data: pd.DataFrame) -> str:
    """Dimension on the x axis: N when it varies, K otherwise."""
    if valid_data["N"].nunique() > 1:
        return "N"
    return "K"


def plot_axis_heatmap(
    valid_data: pd.DataFrame,
    ax: Axes,
    title: str,
    *,
    vmin: float = 0.0,
    show_xlabel: bool = True,
    annotate: bool = True,
):
    """Draw one % of peak heatmap over M and the varying column dimension."""
    if valid_data.empty:
        ax.set_title(f"{title} (no data)", fontsize=7, pad=2)
        ax.axis("off")
        return None

    peak_series = pd.Series(valid_data["peak"], dtype="float64")
    peaks = peak_series.dropna().unique()
    peak = float(peaks[0])

    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]
    valid_data["perf"] = (valid_data["throughput"] / peak) * 100

    col_dim = _column_dim(valid_data)
    heatmap_data = valid_data.pivot(index="M", columns=col_dim, values="perf")

    im = ax.imshow(heatmap_data, cmap=SEQUENTIAL, aspect="auto", vmin=vmin, vmax=100)

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color=contrasting_ink(SEQUENTIAL((val - vmin) / (100 - vmin))),
                        fontsize=5,
                    )

    if show_xlabel:
        ax.set_xlabel(col_dim)
    ax.set_ylabel("M")
    ax.set_title(title, fontsize=7, pad=2)

    return im


def plot_heatmap_throughput_over_peak(
    df: pd.DataFrame, output_path: Path | None = None
):
    """Stack one % of peak heatmap per variant in a single-column figure."""

    variants = sorted_variants(df["variant"])
    if not variants:
        raise ValueError("No variants in dataframe")

    # One panel per row: at column width a row of panels would be illegible.
    ncols = df[_column_dim(df)].nunique()
    fig, axes = column_figure(
        aspect=0.62, nrows=len(variants), squeeze=False, layout="constrained"
    )
    assert isinstance(fig, Figure)

    # Share one scale across panels, starting at the floor of the measured data
    # so the differences between variants stay visible.
    measured = df[df["time"] > 0]
    perf = (measured["flops"] / measured["time"]) / measured["peak"] * 100
    vmin = float(np.floor(perf.min() / 10) * 10) if len(perf) else 0.0

    ims = []
    for idx, variant in enumerate(variants):
        ax = axes.flat[idx]
        valid_data = df[(df["variant"] == variant) & (df["time"] > 0)].copy()
        assert isinstance(valid_data, pd.DataFrame)

        ims.append(
            plot_axis_heatmap(
                valid_data,
                ax,
                variant_label(variant),
                vmin=vmin,
                show_xlabel=idx == len(variants) - 1,
                # Per-cell numbers only fit while the cells are large enough.
                annotate=ncols <= 10,
            )
        )

    # No suptitle: the caption belongs in the LaTeX figure, not in the image.
    ims = [im for im in ims if im is not None]
    if ims:
        cbar = fig.colorbar(
            ims[0],
            ax=axes.ravel().tolist(),
            fraction=0.04,
            pad=0.02,
        )
        cbar.set_label("% of peak")
        cbar.ax.tick_params(length=1.5)

    save(fig, output_path)


def _percent_of_peak(df: pd.DataFrame, variant: str, col_dim: str) -> pd.DataFrame:
    """``% of peak`` for one variant, as an M by ``col_dim`` grid."""
    rows = df[(df["variant"] == variant) & (df["time"] > 0)]
    assert isinstance(rows, pd.DataFrame)
    if rows.empty:
        raise ValueError(f"no valid samples for variant {variant!r}")

    peak = float(pd.Series(rows["peak"], dtype="float64").dropna().iloc[0])
    perf = (rows["flops"] / rows["time"]) / peak * 100
    return rows.assign(perf=perf).pivot(index="M", columns=col_dim, values="perf")


def plot_difference_heatmap(df: pd.DataFrame, output_path: Path | None = None):
    """One heatmap of ours minus the baseline, in points of peak."""
    col_dim = _column_dim(df)
    ours = _percent_of_peak(df, OURS, col_dim)
    baseline = _percent_of_peak(df, BASELINE, col_dim)
    delta = ours - baseline

    limit = float(np.ceil(np.nanmax(np.abs(delta.to_numpy()))))

    width = max(_GRID_WIDTH + 2 * _Y_LABELS, _BAR_WIDTH + 2 * _SIDE_TEXT)
    height = (
        _MARGIN + _GRID_HEIGHT + _X_LABELS + _GAP + _BAR_HEIGHT + _BAR_LABELS + _MARGIN
    )
    fig = sized_figure(width, height)

    bar_bottom = (_MARGIN + _BAR_LABELS) / height
    ax = fig.add_axes(
        (
            (width - _GRID_WIDTH) / 2 / width,
            bar_bottom + (_BAR_HEIGHT + _GAP + _X_LABELS) / height,
            _GRID_WIDTH / width,
            _GRID_HEIGHT / height,
        )
    )
    cbar_ax = fig.add_axes(
        (
            (width - _BAR_WIDTH) / 2 / width,
            bar_bottom,
            _BAR_WIDTH / width,
            _BAR_HEIGHT / height,
        )
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    im = ax.imshow(delta, cmap=DIVERGING, aspect="auto", vmin=-limit, vmax=limit)

    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_yticks(np.arange(len(delta.index)))
    ax.set_xticklabels(delta.columns)
    ax.set_yticklabels(delta.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(delta.index)):
        for j in range(len(delta.columns)):
            val = delta.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}",
                    ha="center",
                    va="center",
                    color=contrasting_ink(im.cmap(im.norm(val))),
                    fontsize=5,
                )

    ax.set_xlabel(col_dim)
    ax.set_ylabel("M")

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("point difference between % of peak")
    cbar.ax.tick_params(length=1.5)

    pad = 0.05 / width
    for x, caption, align in (
        ((width - _BAR_WIDTH) / 2 / width - pad, BASELINE_BETTER, "right"),
        ((width + _BAR_WIDTH) / 2 / width + pad, OURS_BETTER, "left"),
    ):
        fig.text(
            x,
            bar_bottom + _BAR_HEIGHT / height / 2,
            caption,
            ha=align,
            va="center",
            fontsize=6,
            color=INK,
            linespacing=1.1,
        )

    save(fig, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot matmul performance heatmaps.")
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--mode",
        default="panels",
        choices=("panels", "diff"),
        help="'panels' draws one %% of peak heatmap per variant, 'diff' a single "
        "heatmap of ours minus LIBXSMM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    if args.mode == "diff":
        plot_difference_heatmap(df=df, output_path=args.output)
    else:
        plot_heatmap_throughput_over_peak(df=df, output_path=args.output)


if __name__ == "__main__":
    main()
