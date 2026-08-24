# uv run src/plot_ttile_combined.py data/small_matrices.f64.tower.jsonl

import pandas as pd

from pathlib import Path

from autotuner.plot_ttile import plot_axis_throughput
from autotuner.plot_style import column_figure, save


# Two panels per row keeps each panel legible in one paper column.
NCOLS = 2


def plot_ttile_combined(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """One panel per M in a single-column grid: all variants, M = 1..16 vs N."""
    if df.empty:
        return

    targets = set(df["target"])
    dtypes = set(df["dtype"])
    assert len(targets) == len(dtypes) == 1

    nrows = -(-16 // NCOLS)
    fig, axes = column_figure(
        aspect=0.5, nrows=nrows, ncols=NCOLS, sharex=True, sharey=True
    )

    legend_handles: list | None = None
    legend_labels: list[str] | None = None

    for m in range(1, 17):
        row, col = (m - 1) // NCOLS, (m - 1) % NCOLS
        ax = axes[row, col]
        sub = df[df["M"] == m]
        assert isinstance(sub, pd.DataFrame)
        show_xlabel = row == nrows - 1
        show_ylabel = col == 0
        if sub.empty or not (sub["time"] > 0).any():
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
            )
        else:
            plot_axis_throughput(
                sub,
                ax,
                x_row="N",
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
            )
            if legend_handles is None:
                h, lbls = ax.get_legend_handles_labels()
                if h:
                    legend_handles, legend_labels = list(h), list(lbls)
        ax.set_title(f"M = {m}", fontsize=6, pad=1.5)

    # No suptitle: the caption belongs in the LaTeX figure, not in the image.
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(3, len(legend_labels)),
        )

    fig.tight_layout(pad=0.1, rect=(0, 0.025, 1, 1))

    save(fig, output_path)


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

    plot_ttile_combined(df, output_path=args.output)
