"""A sweep over DFT sizes, per data type, as speedup over FFTW.

    uv run plot-dft data/rapper/f32.dft.jsonl data/rapper/f64.dft.jsonl \\
        --output plots/dft.rapper.pdf

The DFT counterpart of `plot_baselines`, laid out the same way: one column
wide, one panel per input file -- f32 and f64 of the same machine -- with the
transform size N on the x axis and each variant's speedup over FFTW on the y
axis: FFTW's time for that N divided by the variant's.  FFTW itself is the flat
line at 1.  A ratio of times needs no operation count, so it holds for every
size, not only the powers of two that FFTW's 5 N log2 N convention counts
exactly.  The powers of two are still joined into a curve and the other sizes
drawn as dots, as benchFFT draws them: the speedup jumps between neighbouring
sizes with different factorisations, so a line through them all would zigzag
rather than show a trend.

N is drawn on a log axis: the sweep is every supported size up to 128 and then
the powers of two, so a linear axis would crowd the small sizes into its left
edge.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from autotuner.datasets import DFT_VARIANTS
from autotuner.plot_baselines import legend_below_panels, panel_title, sweep_figure
from autotuner.plot_data import best_of_repeats
from autotuner.plot_style import COLUMN_WIDTH, GRID, save, tidy_axes, variant_style
from autotuner.plot_throughput import result_machine_label

# The variant every other is compared against.
BASELINE = "fftw"

# Powers of four, so the labels on the log axis stay apart at column width.
X_TICKS = (2, 8, 32, 128, 512)

# Each curve is drawn thinner than the one before it, as in `plot_baselines`.
WIDEST = 1.4
NARROWING = 0.5


def speedup_over_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``speedup`` column: the baseline's time at that N over the row's."""
    measured = best_of_repeats(df[df["time"] > 0], ("variant", "N"))
    baseline = measured[measured["variant"] == BASELINE].set_index("N")["time"]
    if baseline.empty:
        raise ValueError(f"the dataset has no samples for {BASELINE}")
    measured = measured.copy()
    measured["speedup"] = measured["N"].map(baseline) / measured["time"]
    return measured.dropna(subset=["speedup"])


def draw_panel(
    ax: Axes,
    df: pd.DataFrame,
    *,
    variants: Sequence[str],
    y_top: float,
    ticked: bool,
) -> None:
    ax.set_xscale("log", base=2)
    measured = sorted(df["N"].unique())
    ax.set_xlim(measured[0], measured[-1])
    ax.set_ylim(0, y_top)
    for tick in range(1, math.floor(y_top) + 1):
        ax.axhline(tick, linewidth=0.3, color=GRID, zorder=1)

    for index, variant in enumerate(variants):
        style = variant_style(variant)
        width = WIDEST - NARROWING * index
        if variant == BASELINE:
            ax.axhline(
                1,
                zorder=3,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=width,
            )
            continue
        group = df[df["variant"] == variant]
        assert isinstance(group, pd.DataFrame)
        if group.empty:
            continue
        group = group.sort_values("N")
        powers = group[(group["N"] & (group["N"] - 1)) == 0]
        others = group[(group["N"] & (group["N"] - 1)) != 0]
        curve_style: dict[str, Any] = style | {"markersize": 2.2, "linewidth": width}
        ax.plot(powers["N"], powers["speedup"], zorder=4, **curve_style)
        ax.plot(
            others["N"],
            others["speedup"],
            zorder=2,
            linestyle="none",
            marker=style["marker"],
            markersize=1.6,
            color=style["color"],
            alpha=0.55,
        )

    ax.set_xticks(list(X_TICKS), [str(t) for t in X_TICKS])
    ax.minorticks_off()
    ax.set_yticks(range(math.floor(y_top) + 1))
    ax.set_xlabel("N")
    if ticked:
        ax.set_ylabel("Speedup over FFTW")
    tidy_axes(ax)


def plot_dft(
    dfs: Sequence[pd.DataFrame],
    *,
    variants: Sequence[str] = DFT_VARIANTS,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Plot speedup over FFTW against DFT size, one panel per dataset."""
    if not dfs:
        raise ValueError("this figure needs at least one dataset")
    machines = {result_machine_label(df)[0] for df in dfs}
    if len(machines) != 1:
        raise ValueError(f"one machine per figure, got {sorted(machines)}")

    panels = [speedup_over_baseline(df) for df in dfs]
    for panel in panels:
        missing = [v for v in variants if v not in set(panel["variant"])]
        if missing:
            raise ValueError(f"the dataset has no samples for {missing}")

    # One y range for every panel, with headroom above the fastest point,
    # rounded up to a whole speedup so the top tick has a label.
    y_top = math.ceil(max(float(p["speedup"].max()) for p in panels) * 1.04)

    fig, axs = sweep_figure(len(panels), width=width)
    for index, (ax, panel) in enumerate(zip(axs, panels)):
        draw_panel(ax, panel, variants=variants, y_top=y_top, ticked=index == 0)
        ax.set_title(panel_title(panel))

    legend_below_panels(fig, variants)
    save(fig, output_path, tight=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot speedup over FFTW against DFT size, one panel per file."
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
