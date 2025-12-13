# uv run src/plot_ttile.py data/ttile.neon.jsonl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from autotuner.plot_ttile import plot_axis_throughput


def plot_flops_per_time(
    df: pd.DataFrame, title: str, xlabel: str, output_path: Path | None = None
):
    """Plot FLOPs per time for each kernel variant."""

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_axis_throughput(df, ax, x_row=xlabel)

    ax.set_title(title)
    ax.legend(title="Variant")
    plt.tight_layout()

    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_heatmap_throughput_over_peak(
    df: pd.DataFrame, output_path: Path | None = None
):
    """Plot a heatmap of throughput over peak perf for small matrices."""

    peaks = df["peak"].dropna().unique()
    peak = float(peaks[0])

    # Filter out invalid time values (negative or zero)
    valid_data = df[df["time"] > 0].copy()

    # Calculate FLOPs per time (throughput)
    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]

    valid_data["perf"] = (valid_data["throughput"] / peak) * 100

    heatmap_data = valid_data.pivot(index="M", columns="N", values="perf")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% of Peak Performance", rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)

    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            ax.text(
                j,
                i,
                f"{heatmap_data.iloc[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
            )

    ax.set_xlabel("N")
    ax.set_ylabel("M")

    plt.tight_layout()

    ax.set_title(
        "Performance of small square matrix multiplication kernels, for 1 ≤ M ≤ 16, 1 ≤ N ≤ 16, K = 64"
    )

    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main(heatmap: bool):
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

    if heatmap:
        heatmap_df = df[df["variant"] == "libxsmm"]
        assert isinstance(heatmap_df, pd.DataFrame)
        plot_heatmap_throughput_over_peak(df=heatmap_df, output_path=args.output)
        return

    targets = set(df["target"])
    dtypes = set(df["dtype"])
    assert len(targets) == len(dtypes) == 1
    (target,) = targets
    (dtype,) = dtypes

    for m, group in df.groupby("M"):
        assert isinstance(m, int)
        if args.output is not None:
            tiny_path = (
                args.output.parent / "small_matrices" / f"{target}.{dtype}.{m}xNx64.png"
            )
            os.makedirs(tiny_path.parent, exist_ok=True)
        else:
            tiny_path = None
        plot_flops_per_time(
            group,
            title=f"M = {m}, K = 64 and 1 ≤ N ≤ 16",
            xlabel="N",
            output_path=tiny_path,
        )

    square_matrices = df[df["M"] == df["N"]].copy()
    assert isinstance(square_matrices, pd.DataFrame)
    square_matrices = square_matrices.rename(columns={"M": "M,N"})
    del square_matrices["N"]
    assert isinstance(square_matrices, pd.DataFrame)
    plot_flops_per_time(
        square_matrices,
        title="Performance of small square matrix multiplication kernels, for M = N, K = 64 and 1 ≤ M,N ≤ 16",
        xlabel="M,N",
        output_path=args.output,
    )


def main_heatmap():
    main(heatmap=True)


def main_line():
    main(heatmap=False)
