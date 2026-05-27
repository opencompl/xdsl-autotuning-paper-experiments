# uv run src/plot_heatmap.py data/small_matrices.f64.neon.jsonl

import os
from matplotlib.axes import Subplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_axis_heatmap(valid_data: pd.DataFrame, ax: Subplot, title: str):
    if valid_data.empty:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return

    peak_series = pd.Series(valid_data["peak"], dtype="float64")
    peaks = peak_series.dropna().unique()
    peak = float(peaks[0])

    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]

    valid_data["perf"] = (valid_data["throughput"] / peak) * 100

    # Determine columns dynamically: use 'N' if it varies, otherwise look for 'K'
    col_dim = "N"
    if "N" in valid_data.columns and valid_data["N"].nunique() <= 1 and "K" in valid_data.columns:
        col_dim = "K"

    heatmap_data = valid_data.pivot(index="M", columns=col_dim, values="perf")

    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=50, vmax=100)

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)

    # Add high-contrast text color inside cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val >= 75.0 else "black"
                
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                    fontweight="bold"
                )

    ax.set_xlabel("K" if col_dim == "K" else "N")
    ax.set_ylabel("M")
    ax.set_title(title)

    return im


def plot_heatmap_throughput_over_peak(
    df: pd.DataFrame, output_path: Path | None = None
):
    """Plot heatmaps of throughput over peak perf for small matrices, one subplot per variant."""

    variants = sorted(df["variant"].unique())
    if not variants:
        raise ValueError("No variants in dataframe")

    n = len(variants)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    ims = []

    for idx, variant in enumerate(variants):
        ax = axes.flat[idx]
        valid_data = df[(df["variant"] == variant) & (df["time"] > 0)].copy()
        assert isinstance(valid_data, pd.DataFrame)

        ims.append(plot_axis_heatmap(valid_data, ax, variant))

    for j in range(len(variants), len(axes.flat)):
        axes.flat[j].axis("off")

    if "N" in df.columns and df["N"].nunique() == 1 and df["N"].iloc[0] == 16:
        config_title = "Performance of matrix multiplication kernels, for 4 ≤ M ≤ 64, N = 16, 4 ≤ K ≤ 64"
    else:
        config_title = "Performance of small square matrix multiplication kernels, for 1 ≤ M ≤ 16, 1 ≤ N ≤ 16, K = 64"

    fig.suptitle(config_title, y=1.02)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if ims:
        cbar = fig.colorbar(
            ims[0],
            ax=axes.ravel().tolist(),
            fraction=0.035,
            pad=0.04,
        )
        cbar.set_label("% of Peak Performance", rotation=270, labelpad=20)

    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot matmul performance data for small matrices."
    )
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot dir (optional, if not set the plots are only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    plot_heatmap_throughput_over_peak(df=df, output_path=args.output)
