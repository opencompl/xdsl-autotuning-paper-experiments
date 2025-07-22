# uv run src/plot_ttile.py data/ttile.neon.jsonl

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_flops_per_time(df: pd.DataFrame, output_file: Path | None = None):
    """Plot FLOPs per time with various visualizations."""

    # Filter out invalid time values (negative or zero)
    valid_data = df[df["time"] > 0].copy()
    # Note: invalid_data contains data points with negative or zero time values
    # invalid_data = df[df["time"] <= 0]

    # Calculate FLOPs per time (throughput) and plot only throughput vs M
    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(valid_data["M"], valid_data["throughput"], "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("M")
    ax.set_ylabel("Throughput (FLOPs per Time)")
    ax.set_title(
        "Performance of small matrix multiplication kernels, for N = K = 128 and 8 ≤ M ≤ 50"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, valid_data["M"].max() + 2)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot ttile performance data.")
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_flops_per_time(df, output_file=args.output)


if __name__ == "__main__":
    main()
