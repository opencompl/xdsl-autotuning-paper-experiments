# uv run plot-microkernels data/tower/f64.microkernels.jsonl --output plots/tower/f64.microkernels.png

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from autotuner.plot_heatmap import plot_axis_heatmap

# The two AVX512 nanokernels the F64/skylake generator can emit for a register tile.
# The auto-dispatch picks one from the register (vectorized) tile width -- under the
# row-major A/B swap the N dimension of a data point: N <= vector_length -> fsdbcst,
# otherwise nofsdbcst. The sweep additionally forces nofsdbcst on the N <= vector_length
# tiles it also works on, so the nofsdbcst heatmap spans the full N range while fsdbcst
# stays a ragged N <= vector_length slice; the two overlap on the small-N region.
NANOKERNELS = ["fsdbcst", "nofsdbcst"]

# Invalid (M, N) combinations (register budget exceeded) have no measurement; render
# those cells in a neutral gray so the valid region stands out.
_INVALID_FILL = "#e6e6e6"


def plot_microkernel_heatmaps(df: pd.DataFrame, output_path: Path | None = None):
    """Plot one perf heatmap per nanokernel (fsdbcst, nofsdbcst).

    Mirrors ``plot_heatmap`` (throughput as a percentage of peak, ``YlOrRd``), but the
    subplots are split by nanokernel instead of variant and cover only the register
    tiles each nanokernel actually supports.
    """
    if "nanokernel" not in df.columns:
        raise ValueError(
            "input has no 'nanokernel' column; regenerate the microkernel dataset "
            "(the json rule now emits it)"
        )

    fig, axes = plt.subplots(
        1, len(NANOKERNELS), figsize=(5 * len(NANOKERNELS), 5), squeeze=False
    )
    ims = []

    for idx, nanokernel in enumerate(NANOKERNELS):
        ax = axes.flat[idx]
        valid_data = df[(df["nanokernel"] == nanokernel) & (df["time"] > 0)].copy()
        assert isinstance(valid_data, pd.DataFrame)

        n_tiles = len(valid_data)
        title = f"{nanokernel}  ({n_tiles} tiles)"
        im = plot_axis_heatmap(valid_data, ax, title)
        if im is not None:
            ims.append(im)
            # Unsupported (M, N) cells are NaN in the pivot -> transparent in imshow;
            # a gray axes background makes them read as "invalid" rather than "0%".
            ax.set_facecolor(_INVALID_FILL)

    fig.suptitle(
        "Microkernel performance by nanokernel (F64, skylake, K = 64)\n"
        "gray = configuration not supported (register budget exceeded)",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if ims:
        cbar = fig.colorbar(
            ims[0],
            ax=axes.ravel().tolist(),
            fraction=0.035,
            pad=0.04,
        )
        cbar.set_label("% of Peak Performance", rotation=270, labelpad=20)

    if output_path:
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot microkernel performance heatmaps, one per nanokernel."
    )
    parser.add_argument("input", type=Path, help="Input JSONL data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (optional, if not set the plot is only shown)",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    plot_microkernel_heatmaps(df=df, output_path=args.output)
