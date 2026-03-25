# uv run src/plot_ttile_combined.py data/small_matrices.f64.tower.jsonl

import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from autotuner.plot_ttile import TARGET_NAME, plot_axis_throughput


def plot_ttile_combined(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """One 4×4 figure per target: all variants on each subplot, M = 1..16 vs N."""
    if df.empty:
        return

    targets = set(df["target"])
    dtypes = set(df["dtype"])
    assert len(targets) == len(dtypes) == 1
    (target,) = targets
    (dtype,) = dtypes
    target_pretty = TARGET_NAME.get(target, target)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14), sharex=True, sharey=True)
    fig.suptitle(
        f"{dtype}, {target_pretty} — K = 64, 1 ≤ N ≤ 16 (all variants)",
        fontsize=14,
    )

    legend_handles: list | None = None
    legend_labels: list[str] | None = None

    for m in range(1, 17):
        row, col = (m - 1) // 4, (m - 1) % 4
        ax = axes[row, col]
        sub = df[df["M"] == m]
        assert isinstance(sub, pd.DataFrame)
        show_xlabel = row == 3
        show_ylabel = col == 0
        if sub.empty or not (sub["time"] > 0).any():
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
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
        ax.set_title(f"M = {m}", fontsize=9)

    if legend_handles is not None and legend_labels is not None:
        ncol = min(4, len(legend_labels))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            fontsize=8,
            title="Variant / ref.",
        )

    plt.tight_layout(rect=(0, 0.06, 1, 0.96))

    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
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

    plot_ttile_combined(df, output_path=args.output)
