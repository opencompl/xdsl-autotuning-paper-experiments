"""Plot XTC's static spill stores and reloads in the paper's shared style.

    uv run plot-spills data/tower/f64.squares.spills.jsonl --output plots/f64.squares.spills.tower.pdf

Square and tile datasets are line sweeps. The fixed-K small-matrix dataset is
two heatmaps with a common scale, one for stores and one for reloads. Counts
include folded accesses and are static sites, not operations per invocation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from autotuner.plot_style import (
    COLUMN_WIDTH,
    GRID,
    INK_MUTED,
    column_figure,
    save,
    tidy_axes,
    variant_style,
)


def plot_spills(
    df: pd.DataFrame,
    *,
    width: float = COLUMN_WIDTH,
    output_path: Path | None = None,
) -> None:
    """Draw one machine/dtype's XTC counts without treating missing data as zero."""
    if df.empty:
        raise ValueError("no XTC spill data")
    for field in ("machine", "dtype", "llvm_version"):
        if df[field].nunique(dropna=False) != 1:
            raise ValueError(f"a figure needs one {field}")
    if set(df["variant"]) != {"libxtcmm"} or set(df["method"]) != {
        "llvm-regalloc-static"
    }:
        raise ValueError("this figure needs LLVM static spill counts for XTC")
    if df.duplicated(["M", "N", "K"]).any():
        raise ValueError("duplicate shapes: compiler counts are not timing repeats")
    counts = df[["spills", "reloads"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError(
            "spill counts must be nonnegative integers, not missing measurements"
        )

    color = variant_style("libxtcmm")["color"]
    maximum = max(1, int(counts.max()))
    if (df["M"].eq(df["N"]) & df["M"].eq(df["K"])).all():
        x, xlabel = "M", "problem size (M = N = K)"
    elif df["M"].nunique() == 1 and df["K"].nunique() == 1:
        x, xlabel = "N", f"N (M = {df['M'].iloc[0]}, K = {df['K'].iloc[0]})"
    else:
        if df["K"].nunique() != 1:
            raise ValueError("heatmaps need a single K")
        fig, axes = column_figure(width=width, aspect=1.6, ncols=2)
        cmap = LinearSegmentedColormap.from_list("xtc_spills", ["white", color])
        cmap.set_bad(GRID)
        ms, ns = sorted(df["M"].unique()), sorted(df["N"].unique())
        for ax, metric, label in zip(
            axes, ("spills", "reloads"), ("spill stores", "reloads")
        ):
            values = df.pivot(index="N", columns="M", values=metric).reindex(
                index=ns, columns=ms
            )
            ax.imshow(
                values,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=maximum,
                aspect="equal",
                interpolation="nearest",
            )
            for sizes, set_ticks in ((ms, ax.set_xticks), (ns, ax.set_yticks)):
                positions = [
                    i
                    for i, size in enumerate(sizes)
                    if i in (0, len(sizes) - 1) or size % 4 == 0
                ]
                set_ticks(positions, [sizes[i] for i in positions])
            ax.set_xlabel("M")
            ax.set_title(label)
        axes[0].set_ylabel("N")
        axes[1].tick_params(labelleft=False)
        fig.subplots_adjust(left=0.10, right=0.96, bottom=0.29, top=0.90, wspace=0.10)
        bar = fig.colorbar(
            axes[0].images[0],
            cax=fig.add_axes((0.10, 0.14, 0.86, 0.035)),
            orientation="horizontal",
        )
        bar.set_label(f"XTC static sites (K = {df['K'].iloc[0]})")
        bar.locator = MaxNLocator(nbins=4, integer=True)
        bar.update_ticks()
        save(fig, output_path)
        return

    fig, ax = column_figure(width=width)
    ordered = df.sort_values(x)
    for metric, label, linestyle, linewidth, line_color in (
        ("spills", "XTC spill stores", "-", 1.4, color),
        ("reloads", "XTC reloads", "--", 0.8, INK_MUTED),
    ):
        ax.plot(
            ordered[x],
            ordered[metric],
            label=label,
            color=line_color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    sizes = ordered[x].to_numpy()
    lo, hi = int(sizes.min()), int(sizes.max())
    ax.set_xlim(lo - 0.5 if lo == hi else lo, hi + 0.5 if lo == hi else hi)
    ax.set_xticks(sorted({lo, hi, *(v for v in (16, 32, 48, 64, 128) if lo < v < hi)}))
    ax.set_ylim(0, maximum * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("static spill / reload sites")
    tidy_axes(ax)
    # The spill peaks can occur anywhere in a sweep. Reserve space above the
    # axes instead of putting a legend over the largest counts.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    fig.tight_layout(pad=0.1)
    save(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--width", type=float, default=COLUMN_WIDTH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_spills(
        pd.read_json(args.input, lines=True), width=args.width, output_path=args.output
    )


if __name__ == "__main__":
    main()
