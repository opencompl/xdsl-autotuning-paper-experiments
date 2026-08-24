# uv run plot-ttile data/neon/f32.ttile.jsonl

from collections.abc import Collection, Iterable
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import pandas as pd

from pathlib import Path

from autotuner.plot_style import (
    INK_MUTED,
    column_figure,
    legend_below,
    integer_ticks,
    save,
    sorted_variants,
    tidy_axes,
    variant_style,
)

TARGET_NAME = {
    "tower": "AMD Zen 5",
    "pinocchio": "Intel Skylake",
    "neon": "Apple M2 Max",
}

AXIS_LABEL = {
    "M": "M",
    "N": "N",
    "K": "K",
    "M,N": "M = N",
}


def plot_axis_throughput(
    df: pd.DataFrame,
    ax: Axes,
    *,
    x_row: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """Draw one throughput-versus-size axis, styled for a single column."""

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
        ax.axhline(
            100,
            linestyle=(0, (4, 2)),
            linewidth=0.6,
            color=INK_MUTED,
            label="peak",
            zorder=1,
        )
        y_col = "throughput_percent"
    else:
        y_col = "throughput"

    # A dense sweep gets fewer markers, otherwise they merge into a band.
    points = df[x_row].nunique()
    markevery = max(1, -(-points // 32))

    for variant in sorted_variants(df["variant"]):
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values(x_row)
        ax.plot(
            group[x_row],
            group[y_col],
            markevery=markevery,
            zorder=2,
            **variant_style(variant),
        )

    if peak is not None:
        if show_ylabel:
            ax.set_ylabel("% of peak")
        ax.set_ylim(0, 112.0)
        ax.set_yticks([0, 25, 50, 75, 100])
    else:
        if show_ylabel:
            ax.set_ylabel("throughput (FLOP/cycle)")
        ax.set_ylim(bottom=0.0)

    if show_xlabel:
        ax.set_xlabel(AXIS_LABEL.get(x_row, x_row))
    ax.set_xticks(integer_ticks(df[x_row]))
    ax.set_xlim(0, df[x_row].max() * 1.04)
    tidy_axes(ax)


def plot_flops_per_time(df: pd.DataFrame, output_file: Path | None = None):
    """Plot FLOPs per time for each kernel variant."""

    ns = set(df.N)
    ks = set(df.K)
    dtypes = set(df["dtype"])
    assert len(ns) == len(ks) == len(dtypes) == 1
    (n,) = ns
    (k,) = ks
    assert n == k

    fig, ax = column_figure()
    assert isinstance(ax, Axes)

    plot_axis_throughput(df, ax, x_row="M")

    # No title: the caption belongs in the LaTeX figure, not in the image.
    legend_below(fig, ax)

    save(fig, output_file)


def plot_combined(output_file: Path | None):
    """Combined 2x2 subplot of the four ttile datasets, sharing axes and legend."""
    # Hardcode input files
    input_files = [
        ("data/tower/f32.ttile.jsonl", "(a)"),
        ("data/tower/f64.ttile.jsonl", "(b)"),
        ("data/pinocchio/f32.ttile.jsonl", "(c)"),
        ("data/pinocchio/f64.ttile.jsonl", "(d)"),
    ]

    dfs = []
    for path, _ in input_files:
        df = pd.read_json(path, lines=True)
        dfs.append(df)

    # Short panel labels; the full description belongs in the LaTeX caption.
    labels = []
    for df, (_, prefix) in zip(dfs, input_files):
        ns = set(df.N)
        ks = set(df.K)
        dts = set(df["dtype"])
        tgs = set(df["target"])
        assert len(ns) == len(ks) == len(dts) == len(tgs) == 1
        (n,) = ns
        (dtype,) = dts
        (target,) = tgs
        labels.append(f"{prefix} {dtype}, N = K = {n}, {TARGET_NAME[target]}")

    fig, axs = column_figure(aspect=0.78, nrows=2, ncols=2, sharex=True, sharey=True)
    axs = axs.flatten()

    # For legend
    handles_labels: tuple[Iterable[Artist], Collection[str]] | None = None
    for idx, (df, ax, label) in enumerate(zip(dfs, axs, labels)):
        plot_axis_throughput(
            df,
            ax,
            x_row="M",
            show_xlabel=idx >= 2,
            show_ylabel=idx % 2 == 0,
        )
        ax.set_title(label, fontsize=6, pad=2)
        # Only gather legend once
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    # One shared legend below the grid
    assert handles_labels is not None
    handles, labels_ = handles_labels
    fig.legend(
        handles,
        labels_,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
    )
    fig.tight_layout(pad=0.1, rect=(0, 0.04, 1, 1))

    save(fig, output_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot ttile performance data.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input JSONL data file (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    if args.input is None:
        plot_combined(args.output)
        return

    df = pd.read_json(args.input, lines=True)
    plot_flops_per_time(df, output_file=args.output)


if __name__ == "__main__":
    main()
