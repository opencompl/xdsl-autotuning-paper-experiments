# uv run plot-heatmap data/tower/f64.mk_sweep.jsonl --output plot.pdf

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pathlib import Path

from autotuner.plot_style import (
    INK,
    SEQUENTIAL,
    column_figure,
    save,
    sorted_variants,
    variant_label,
)


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
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        color="white" if val >= vmin + 0.6 * (100 - vmin) else INK,
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot matmul performance heatmaps, one panel per variant."
    )
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    plot_heatmap_throughput_over_peak(df=df, output_path=args.output)


if __name__ == "__main__":
    main()
