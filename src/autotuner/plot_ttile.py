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

    ns = set(df.N)
    ks = set(df.K)
    dtypes = set(df["dtype"])
    assert len(ns) == len(ks) == len(dtypes) == 1
    (n,) = ns
    (k,) = ks
    (dtype,) = dtypes

    # Get peak performance if available

    peak = None
    if "peak" in df.columns:
        peaks = df["peak"].dropna().unique()
        if len(peaks) > 0:
            peak = float(peaks[0])
            if peak == 0.0:
                # Peak is not set
                peak = None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Peak perf horizontal line at 100%
    if peak is not None:
        # Convert throughput to percentage of peak
        valid_data["throughput_percent"] = (valid_data["throughput"] / peak) * 100
        ax.axhline(100, linestyle="--", linewidth=1, label="Peak perf (100%)")
        y_col = "throughput_percent"
    else:
        y_col = "throughput"

    # Assign a color and marker for each variant
    import itertools

    colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
    markers = itertools.cycle(["o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "x"])

    for (variant, group), color, marker in zip(
        valid_data.groupby("variant"), colors, markers
    ):
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values("M")
        ax.plot(
            group["M"],
            group[y_col],
            label=variant,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("M")
    if peak is not None:
        ax.set_ylabel("% of Peak Performance")
        ax.set_ylim(0, 110.0)
    else:
        ax.set_ylabel("Throughput (FLOPs per Time)")
        ax.set_ylim(bottom=1e-2)  # Avoid log(0); adjust as needed for your data
    ax.set_title(f"Performance of small matrix multiplication kernels, for N = K = {n}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, valid_data["M"].max() + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Variant")
    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
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
