# uv run plot-cube data/cube.neon.jsonl

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_cube_bar_chart(df: pd.DataFrame, output_file: Path | None = None):
    """Plot a bar chart of throughput for each variant in the cube dataset."""

    # Filter out invalid time values (negative or zero)
    valid_data = df[df["time"] > 0].copy()

    # Calculate throughput (FLOPs per time)
    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]

    # Sort by throughput descending for better visualization
    # valid_data = valid_data.sort_values("throughput", ascending=False)

    # Bar chart: x = variant, y = throughput
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        valid_data["variant"],
        valid_data["throughput"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(valid_data)],
    )

    # Annotate bars with throughput values
    for bar, throughput in zip(bars, valid_data["throughput"]):
        ax.annotate(
            f"{throughput:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Variant")
    ax.set_ylabel("Throughput (FLOPs per Time)")
    ax.set_title("4x4x4 Matrix Multiplication: Throughput by Variant")
    ax.grid(axis="y", alpha=0.3)
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

    parser = argparse.ArgumentParser(description="Plot 4x4x4 performance bar chart.")
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    plot_cube_bar_chart(df, output_file=args.output)


if __name__ == "__main__":
    main()
