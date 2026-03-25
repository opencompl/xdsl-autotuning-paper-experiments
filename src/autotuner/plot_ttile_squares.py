# uv run src/plot_ttile.py data/ttile.neon.jsonl

import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from autotuner.plot_ttile import plot_axis_throughput


def plot_flops_per_time(
    df: pd.DataFrame, title: str, xlabel: str, output_path: Path | None = None
):
    """Plot FLOPs per time for each kernel variant."""

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_axis_throughput(df, ax, x_row=xlabel)

    ax.set_title(title)
    ax.legend(title="Variant")
    plt.tight_layout()

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

    square_matrices = df[df["M"] == df["N"]].copy()
    assert isinstance(square_matrices, pd.DataFrame)
    square_matrices = square_matrices.rename(columns={"M": "M,N"})
    del square_matrices["N"]
    assert isinstance(square_matrices, pd.DataFrame)
    plot_flops_per_time(
        square_matrices,
        title="Performance of small square matrix multiplication kernels, for M = N, K = 64 and 1 ≤ M,N ≤ 16",
        xlabel="M,N",
        output_path=args.output,
    )
