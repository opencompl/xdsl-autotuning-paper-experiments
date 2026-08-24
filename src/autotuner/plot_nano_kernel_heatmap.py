import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _tick_positions(length: int, maximum_ticks: int) -> np.ndarray:
    """Return readable, evenly-spaced integer tick positions."""
    stride = max(1, int(np.ceil(length / maximum_ticks)))
    positions = np.arange(0, length, stride)
    if positions[-1] != length - 1:
        positions = np.append(positions, length - 1)
    return positions


def plot_nano_kernel_heatmap(
    data: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plot one nano-kernel's supported M/N domain as percent of peak."""
    required_columns = {
        "M",
        "N",
        "K",
        "peak",
        "flops",
        "time",
        "variant",
        "target",
        "dtype",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing input columns: {missing}")

    valid_data = data[data["time"] > 0].copy()
    if valid_data.empty:
        raise ValueError("input contains no positive timing measurements")

    k_values = set(valid_data["K"])
    if len(k_values) != 1:
        raise ValueError(f"expected one K, got {sorted(k_values)}")
    k = int(next(iter(k_values)))

    metadata_columns = ("variant", "target", "dtype", "peak")
    metadata: dict[str, Any] = {}
    for column in metadata_columns:
        values = pd.Series(valid_data[column]).dropna().unique()
        if len(values) != 1:
            raise ValueError(f"expected one {column}, got {list(values)}")
        metadata[column] = values[0]

    peak = float(metadata["peak"])
    if peak <= 0:
        raise ValueError(f"peak performance must be positive, got {peak}")

    valid_data["percent_peak"] = valid_data["flops"] / valid_data["time"] / peak * 100
    # CompXSMM's row-major entry point swaps A/B and M/N before invoking the
    # column-major generator. Undo that representation detail so these axes are
    # the M/N dimensions seen by MatmulK and the selected nano-kernel.
    valid_data["nano_kernel_m"] = valid_data["N"]
    valid_data["nano_kernel_n"] = valid_data["M"]
    heatmap = valid_data.pivot(
        index="nano_kernel_m",
        columns="nano_kernel_n",
        values="percent_peak",
    )

    measured_m = np.asarray(heatmap.index, dtype=np.int64)
    measured_n = np.asarray(heatmap.columns, dtype=np.int64)
    m_values = np.arange(int(measured_m.min()), int(measured_m.max()) + 1)
    n_values = np.arange(int(measured_n.min()), int(measured_n.max()) + 1)
    heatmap = heatmap.reindex(index=m_values, columns=n_values)

    values = np.ma.masked_invalid(heatmap.to_numpy(dtype=float))
    colormap = plt.colormaps["YlOrRd"].copy()
    colormap.set_bad("#e5e7eb")

    width = max(8.0, len(n_values) * 0.36)
    height = max(5.0, len(m_values) * 0.18)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(
        values,
        cmap=colormap,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=100,
    )

    x_ticks = _tick_positions(len(n_values), maximum_ticks=28)
    y_ticks = _tick_positions(len(m_values), maximum_ticks=20)
    axis.set_xticks(x_ticks, n_values[x_ticks])
    axis.set_yticks(y_ticks, m_values[y_ticks])
    axis.set_xlabel("N tile size")
    axis.set_ylabel("M tile size")

    variant = str(metadata["variant"])
    nano_kernel = variant.removeprefix("compxsmm-")
    axis.set_title(
        f"{nano_kernel} on {metadata['target']} ({metadata['dtype']}, K = {k})"
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("% of peak performance")

    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot performance over the supported M/N nano-kernel tiles."
    )
    parser.add_argument("input", type=Path, help="Input JSONL dataset")
    parser.add_argument("--output", type=Path, help="Output image path")
    arguments = parser.parse_args()

    plot_nano_kernel_heatmap(
        pd.read_json(arguments.input, lines=True),
        arguments.output,
    )


if __name__ == "__main__":
    main()
