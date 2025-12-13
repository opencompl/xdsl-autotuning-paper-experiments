# uv run src/plot_ttile.py data/ttile.neon.jsonl

from matplotlib.axes import Axes
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_axis_throughput(
    df: pd.DataFrame,
    ax: Axes,
    *,
    x_row: str,
):
    # Get peak performance if available

    # Filter out invalid time values (negative or zero)
    valid_data = df[df["time"] > 0].copy()
    assert isinstance(valid_data, pd.DataFrame)
    df = valid_data

    # Calculate FLOPs per time (throughput)
    df["throughput"] = df["flops"] / df["time"]

    peak = None
    if "peak" in df.columns:
        peaks = df["peak"].dropna().unique()
        if len(peaks) > 0:
            peak = float(peaks[0])
            if peak == 0.0:
                # Peak is not set
                peak = None

    # Peak perf horizontal line at 100%
    if peak is not None:
        # Convert throughput to percentage of peak
        df["throughput_percent"] = (df["throughput"] / peak) * 100
        ax.axhline(100, linestyle="--", linewidth=1, label="Peak perf (100%)")
        y_col = "throughput_percent"
    else:
        y_col = "throughput"

    # Assign a color and marker for each variant
    import itertools

    colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
    markers = itertools.cycle(["o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "x"])

    for (variant, group), color, marker in zip(df.groupby("variant"), colors, markers):
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values(x_row)
        ax.plot(
            group[x_row],
            group[y_col],
            label=variant,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    if peak is not None:
        ax.set_ylabel("% of Peak Performance")
        ax.set_ylim(0, 110.0)
    else:
        ax.set_ylabel("Throughput (FLOPs per Time)")
        ax.set_ylim(bottom=1e-2)  # Avoid log(0); adjust as needed for your data

    ax.set_xlabel(x_row)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df[x_row].max() + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_flops_per_time(df: pd.DataFrame, output_file: Path | None = None):
    """Plot FLOPs per time for each kernel variant."""

    ns = set(df.N)
    ks = set(df.K)
    dtypes = set(df["dtype"])
    assert len(ns) == len(ks) == len(dtypes) == 1
    (n,) = ns
    (k,) = ks
    assert n == k
    (dtype,) = dtypes

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_axis_throughput(df, ax, x_row="M")

    ax.set_title(f"N = K = {n}, {dtype}")
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
