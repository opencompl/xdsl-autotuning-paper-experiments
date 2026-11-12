"""A sweep over DFT sizes, per data type.

    uv run plot-dft data/rapper/f32.dft.jsonl data/rapper/f64.dft.jsonl \\
        --output plots/dft.rapper.pdf

The DFT counterpart of `plot_baselines`, laid out the same way: one column
wide, one panel per input file -- f32 and f64 of the same machine -- with the
transform size N on the x axis and throughput as a share of machine peak on the
y axis.  Throughput counts 5 N log2 N operations per transform, FFTW's
convention (see `evaluate_dft`).  That count is exact for a radix-2 transform
only, and overstates other sizes by an amount that varies from size to size, so
the powers of two are joined into a curve and the other sizes are dots.

N is drawn on a log axis: the sweep is every supported size up to 128 and then
the powers of two, so a linear axis would crowd the small sizes into its left
edge.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes

from autotuner.datasets import DFT_VARIANTS
from autotuner.plot_baselines import (
    legend_below_panels,
    panel_title,
    percent_of_peak,
    sweep_figure,
)
from autotuner.plot_data import best_of_repeats
from autotuner.plot_style import COLUMN_WIDTH, GRID, save, tidy_axes, variant_style
from autotuner.plot_throughput import result_machine_label

# Top of the % of peak axis, with room above 100 for a curve touching peak.
Y_TOP = 104.0
Y_TICKS = (0, 25, 50, 75, 100)

# Powers of four, so the labels on the log axis stay apart at column width.
X_TICKS = (2, 8, 32, 128, 512)

# Each curve is drawn thinner than the one before it, as in `plot_baselines`.
WIDEST = 1.4
NARROWING = 0.5


def draw_panel(
    ax: Axes, df: pd.DataFrame, *, variants: Sequence[str], ticked: bool
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(df["N"].min(), df["N"].max())
    ax.set_ylim(0, Y_TOP)
    ax.axhline(100, linewidth=0.3, color=GRID, zorder=1)

    for index, variant in enumerate(variants):
        group = df[df["variant"] == variant].sort_values("N")
        if group.empty:
            continue
        style = variant_style(variant)
        # Only the powers of two are joined, as benchFFT draws them: the
        # operation count overstates the other sizes by a factor that changes
        # from one size to the next, so a line through them would zigzag
        # between neighbours rather than show a trend.  The other sizes are
        # drawn as dots in the same color.
        powers = group[(group["N"] & (group["N"] - 1)) == 0]
        others = group[(group["N"] & (group["N"] - 1)) != 0]
        ax.plot(
            powers["N"],
            powers["percent"],
            zorder=3,
            **style | {"markersize": 2.2, "linewidth": WIDEST - NARROWING * index},
        )
        ax.plot(
            others["N"],
            others["percent"],
            zorder=2,
            linestyle="none",
            marker=style["marker"],
            markersize=1.6,
            color=style["color"],
            alpha=0.55,
        )

    ax.set_xticks(list(X_TICKS), [str(t) for t in X_TICKS])
    ax.minorticks_off()
    ax.set_yticks(list(Y_TICKS))
    ax.set_xlabel("N")
    if ticked:
        ax.set_ylabel("% of peak")
    tidy_axes(ax)


def plot_dft(
    dfs: Sequence[pd.DataFrame],
    *,
    variants: Sequence[str] = DFT_VARIANTS,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Plot % of peak against DFT size, one panel per dataset."""
    if not dfs:
        raise ValueError("this figure needs at least one dataset")
    machines = {result_machine_label(df)[0] for df in dfs}
    if len(machines) != 1:
        raise ValueError(f"one machine per figure, got {sorted(machines)}")

    panels = [best_of_repeats(percent_of_peak(df), ("variant", "N")) for df in dfs]
    for panel in panels:
        missing = [v for v in variants if v not in set(panel["variant"])]
        if missing:
            raise ValueError(f"the dataset has no samples for {missing}")

    fig, axs = sweep_figure(len(panels), width=width)
    for index, (ax, panel) in enumerate(zip(axs, panels)):
        draw_panel(ax, panel, variants=variants, ticked=index == 0)
        ax.set_title(panel_title(panel))

    legend_below_panels(fig, variants)
    save(fig, output_path, tight=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot % of peak against DFT size, one panel per file."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input JSONL data files, one panel each, all of the same machine",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=COLUMN_WIDTH,
        help="Figure width in inches (default: one column of the paper template)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    plot_dft(
        [pd.read_json(path, lines=True) for path in args.inputs],
        width=args.width,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
