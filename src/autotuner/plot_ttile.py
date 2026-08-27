# uv run plot-ttile data/neon/f32.ttile.jsonl

from collections.abc import Collection, Iterable
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import pandas as pd

from pathlib import Path

from autotuner.plot_style import (
    BASELINE,
    INK_MUTED,
    OURS,
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
    "rapper": "AMD Zen 4",
    "pinocchio": "Intel Skylake",
    "neon": "Apple M2 Max",
}

# Top of the % of peak axis
Y_TOP = 112.0

# Markers on our curve: their size, and the gap between the two head-to-head
# curves that earns one, both in typographic points.
MARKER_SIZE = 2.0
MARKER_MIN_GAP = 2.5

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
    ymin: float = 0.0,
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
            zorder=1,
        )
        y_col = "throughput_percent"
    else:
        y_col = "throughput"

    # Axes and ticks before the curves: the divergence test below measures a
    # distance in points, which needs the final data-to-display transform.
    if peak is not None:
        if show_ylabel:
            ax.set_ylabel("% of peak")
        ax.set_ylim(ymin, Y_TOP)
        # The round quarter-peak ticks above the floor, plus the floor itself so
        # the bottom of the axis is labelled rather than left to be inferred.
        ax.set_yticks([ymin, *(t for t in (25, 50, 75, 100) if t > ymin)])
    else:
        if show_ylabel:
            ax.set_ylabel("throughput (FLOP/cycle)")
        ax.set_ylim(bottom=ymin)

    if show_xlabel:
        ax.set_xlabel(AXIS_LABEL.get(x_row, x_row))
    ax.set_xticks(integer_ticks(df[x_row]))
    ax.set_xlim(0, df[x_row].max() * 1.04)
    tidy_axes(ax)

    # A dense sweep gets fewer markers, otherwise they merge into a band.
    points = df[x_row].nunique()
    markevery = max(1, -(-points // 32))
    apart = divergent_points(ax, df, x_row=x_row, y_col=y_col)

    for variant in sorted_variants(df["variant"]):
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values(x_row)
        style = variant_style(variant)
        if variant == OURS and apart:
            # On the labelled line rather than on an artist of their own, so
            # the legend preview shows the marker along with the line.
            style |= {"marker": "o", "markersize": MARKER_SIZE}
            every = [i for i, x in enumerate(group[x_row]) if x in apart]
        else:
            every = markevery
        ax.plot(group[x_row], group[y_col], markevery=every, zorder=2, **style)


def divergent_points(
    ax: Axes, df: pd.DataFrame, *, x_row: str, y_col: str
) -> set[float]:
    """The x positions where our curve visibly parts from the baseline."""
    if not {OURS, BASELINE} <= set(df["variant"]):
        return set()

    # pivot_table aligns the two curves on x and tolerates repeated samples.
    curves = df.pivot_table(index=x_row, columns="variant", values=y_col)
    gap = (curves[OURS] - curves[BASELINE]).abs()

    figure = ax.get_figure()
    assert figure is not None
    to_data = ax.transData.inverted()
    pixels = MARKER_MIN_GAP * figure.dpi / 72
    threshold = abs(
        to_data.transform((0.0, pixels))[1] - to_data.transform((0.0, 0.0))[1]
    )

    return set(gap[gap >= threshold].index)


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
