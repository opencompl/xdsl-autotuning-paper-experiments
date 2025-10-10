# uv run src/plot_ttile.py data/ttile.neon.jsonl

import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_flops_per_time(
    df: pd.DataFrame, m: int | None, output_dir: Path | None = None
):
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
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values("N")
        ax.plot(
            group["N"],
            group["throughput"],
            label=variant,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    ax.set_ylabel("Throughput (FLOPs per Time)")
    if m is not None:
        ax.set_title(
            f"Performance of small matrix multiplication kernels, for M = {m}, K = 64 and 1 ≤ N ≤ 16"
        )
        ax.set_xlabel("N")
    else:
        ax.set_title(
            "Performance of small square matrix multiplication kernels, for M = N, K = 64 and 1 ≤ M,N ≤ 16"
        )
        ax.set_xlabel("M, N")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, valid_data["N"].max() + 2)
    ax.set_ylim(bottom=1e-2)  # Avoid log(0); adjust as needed for your data
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Variant")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if m is not None:
            output_path = os.path.join(output_dir, f"plot_{m}xNx64.png")
        else:
            output_path = os.path.join(output_dir, "plot_MxNx64.png")
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

    for m, group in df.groupby("M"):
        assert isinstance(m, int)
        plot_flops_per_time(group, m, output_dir=args.output)

    square_matrices = df[df["M"] == df["N"]]
    plot_flops_per_time(square_matrices, m=None, output_dir=args.output)


if __name__ == "__main__":
    main()
