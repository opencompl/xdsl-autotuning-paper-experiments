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

    # Extract the true dataset format directly
    dtype_str = str(df['dtype'].iloc[0]).upper() if 'dtype' in df.columns else "F64"
    k_val = df['K'].iloc[0] if 'K' in df.columns else 16
    
    # Generate an upscaled, clear header 
    ax.set_title(f"Performance of Square {dtype_str} Kernels (M = N, K = {k_val})", fontsize=14, pad=12)
    
    # Upscale labels and ticks for pristine PDF scaling
    ax.set_xlabel("Matrix Dimensions (M, N)", fontsize=12, labelpad=8)
    ax.set_ylabel("Performance", fontsize=12, labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Clean, larger legend bounding frame
    ax.legend(title="Variant", fontsize=11, title_fontsize=11, loc="best")
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

    df = pd.read_json(args.input, lines=True)

    if df["M"].nunique() == 1 and df["N"].nunique() > 1:
        return plot_flops_per_time(df, f"Performance of matrix multiplication kernels, for M = {df['M'].iloc[0]}, K = {df['K'].iloc[0]}", "N", args.output)

    square_matrices = df[df["M"] == df["N"]].copy()
    assert isinstance(square_matrices, pd.DataFrame)
    square_matrices = square_matrices.rename(columns={"M": "M,N"})
    del square_matrices["N"]
    assert isinstance(square_matrices, pd.DataFrame)

    # Extract actual parameters dynamically from your data frame
    k_val = square_matrices['K'].iloc[0] if 'K' in square_matrices.columns else 16
    max_mn = square_matrices['M'].max() if 'M' in square_matrices.columns else 64

    plot_flops_per_time(
        square_matrices,
        title=f"Performance of square matrix multiplication kernels, for M = N, K = {k_val} and 1 ≤ M,N ≤ {max_mn}",
        xlabel="M,N",
        output_path=args.output,
    )
