"""Shared pieces of the exploratory throughput plots.

The machine-metadata helpers and the one throughput axis that the small-matrix
figures draw.  There is no figure of its own here: the paper's N sweep lives in
``plot_baselines``, and these are what the PNG plots around it share.
"""

from collections.abc import Mapping

from matplotlib.axes import Axes
import pandas as pd

from autotuner.machines import MACHINES, Machine
from autotuner.plot_style import variant_label


def machine_display_names(
    machine_configs: Mapping[str, Machine] = MACHINES,
) -> dict[str, str]:
    """Return display names from the shared machine configuration."""
    return {
        machine_name: machine.display_name
        for machine_name, machine in machine_configs.items()
    }


def result_machine_label(
    df: pd.DataFrame, machine_configs: Mapping[str, Machine] = MACHINES
) -> tuple[str, str]:
    """Return the single machine identifier and display label in a result set."""
    machines = set(df["machine"])
    assert len(machines) == 1
    machine = str(next(iter(machines)))
    try:
        display_name = machine_configs[machine].display_name
    except KeyError as error:
        raise ValueError(f"machine '{machine}' is not configured") from error
    return machine, display_name


def plot_axis_throughput(
    df: pd.DataFrame,
    ax: Axes,
    *,
    x_row: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    # Get peak performance if available

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
        ax.axhline(100, linestyle="--", linewidth=1, label="Peak perf (100%)")
        y_col = "throughput_percent"
    else:
        y_col = "throughput"

    # Assign a color and marker for each variant
    import itertools

    colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
    markers = itertools.cycle(["o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "x"])

    for (variant, group), color, marker in zip(df.groupby("variant"), colors, markers):
        assert isinstance(group, pd.DataFrame)
        group = group.sort_values(x_row)
        ax.plot(
            group[x_row],
            group[y_col],
            label=variant_label(str(variant)),
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

    if peak is not None:
        if show_ylabel:
            ax.set_ylabel("% of Peak Performance")
        ax.set_ylim(0, 110.0)
    else:
        if show_ylabel:
            ax.set_ylabel("Throughput (FLOPs per Time)")
        ax.set_ylim(bottom=1e-2)  # Avoid log(0); adjust as needed for your data

    if show_xlabel:
        ax.set_xlabel(x_row)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df[x_row].max() + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
