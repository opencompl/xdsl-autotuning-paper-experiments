# uv run src/plot_ttile.py data/ttile.neon.jsonl

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_flops_per_time(df: pd.DataFrame, output_file: Path | None = None):
    """Plot FLOPs per time for each kernel variant."""

    # Filter out invalid time values (negative or zero)
    valid_data = df[df["time"] > 0].copy()

    # Calculate FLOPs per time (throughput)
    valid_data["throughput"] = valid_data["flops"] / valid_data["time"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Assign a color and marker for each variant
    import itertools

    colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
    markers = itertools.cycle(["o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "x"])

    for (variant, group), color, marker in zip(
        valid_data.groupby("variant"), colors, markers
    ):
        group = group.sort_values("M")
        ax.plot(
            group["M"],
            group["throughput"],
            label=variant,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("M")
    ax.set_ylabel("Throughput (FLOPs per Time)")
    ax.set_title(
        "Performance of small matrix multiplication kernels, for N = K = 128 and 8 ≤ M ≤ 50"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, valid_data["M"].max() + 2)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)  # Avoid log(0); adjust as needed for your data
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Variant")
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
