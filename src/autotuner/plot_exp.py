"""The initial FP32 exp sweep: cycles per element and speedup over LIBXSMM."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from autotuner.datasets import EXP_DEGREES, EXP_LENGTHS, EXP_PLACEMENTS
from autotuner.plot_style import (
    PAGE_WIDTH,
    column_figure,
    save,
    tidy_axes,
    variant_style,
)


def exp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compare medians at matched placements, giving every placement equal weight."""
    if df.empty or df["machine"].nunique() != 1:
        raise ValueError("exp needs measurements from exactly one machine")
    if not (
        df["kernel"].eq("exp")
        & df["dtype"].eq("f32")
        & df["approximation"].eq(
            df["degree"].map({3: "libxsmm-cubic", 5: "libxsmm-quintic"})
        )
    ).all():
        raise ValueError("exp needs FP32 with matching degree-3 or degree-5 metadata")
    if not (np.isfinite(df["time"]) & df["time"].gt(0)).all():
        raise ValueError("exp timings must be finite and positive")

    if "measurement" not in df or not all(df["measurement"].eq("paired-placement-v1")):
        raise ValueError("exp needs paired-placement measurements; rerun the dataset")
    key = ["variant", "degree", "elements"]
    for _, group in df.groupby(key):
        if set(group["placement_offset"]) != set(EXP_PLACEMENTS):
            raise ValueError("exp needs all four matched placements")
    pairs = ["degree", "elements", "repeat", "placement_offset", "order"]
    grouped = df.groupby(pairs)
    if (
        df.duplicated([*pairs, "variant"]).any()
        or not all(grouped["variant"].nunique().eq(df["variant"].nunique()))
        or not all(grouped["code_address"].nunique().eq(1))
        or not all(
            df.groupby([*key, "repeat", "placement_offset"])["order"].agg(
                lambda values: set(values) == {0, 1}
            )
        )
    ):
        raise ValueError("exp needs paired variants at the same address in both orders")

    # Median across passes and forward/reverse orders at EACH placement, then an equal
    # average across placements. Never pick a different best alignment per variant.
    placed = df.groupby([*key, "placement_offset"], as_index=False).agg(
        time=("time", "median")
    )
    measured = cast(
        pd.DataFrame, placed.groupby(key, as_index=False).agg(time=("time", "mean"))
    )
    expected = {(degree, length) for degree in EXP_DEGREES for length in EXP_LENGTHS}
    for variant, group in measured.groupby("variant"):
        if set(zip(group["degree"], group["elements"], strict=True)) != expected:
            raise ValueError(f"{variant} needs all seven exp lengths for both degrees")
    baseline = measured.query("variant == 'libxsmm'").set_index(["degree", "elements"])
    if baseline.empty:
        raise ValueError("exp needs a LIBXSMM baseline")
    baseline["baseline_time"] = baseline.pop("time")
    measured = measured.join(
        baseline[["baseline_time"]],
        on=["degree", "elements"],
    )
    measured["cycles_per_element"] = measured["time"] / measured["elements"]
    measured["speedup"] = measured.pop("baseline_time") / measured["time"]
    return measured.sort_values(["degree", "variant", "elements"])


def plot_exp(df: pd.DataFrame, *, output_path: Path | None = None) -> None:
    measured = exp_metrics(df)
    fig, axes = column_figure(
        nrows=2, ncols=2, width=PAGE_WIDTH, aspect=0.5, sharex=True, sharey="row"
    )
    variants = ["libxsmm", *sorted(set(measured["variant"]) - {"libxsmm"})]
    for column, degree in enumerate(EXP_DEGREES):
        cycles, speedup = axes[:, column]
        for variant in variants:
            group = measured[
                (measured["variant"] == variant) & (measured["degree"] == degree)
            ]
            style = variant_style(variant) | {"marker": "o"}
            if variant == "xdsl_libxsmm":
                # Keep the baseline visible when the literal port coincides.
                style |= {
                    "linestyle": "--",
                    "markerfacecolor": "none",
                    "markeredgewidth": 0.8,
                    "markersize": 4,
                }
            cycles.plot(group["elements"], group["cycles_per_element"], **style)
            speedup.plot(group["elements"], group["speedup"], **style)

        cycles.set_title(f"Degree {degree}")
        cycles.legend(loc="upper right")
        speedup.set_xlabel("FP32 elements (N = 1)")
        speedup.set_xscale("log", base=2)
        speedup.set_xticks(EXP_LENGTHS, [str(n) for n in EXP_LENGTHS])
        for ax in (cycles, speedup):
            tidy_axes(ax)
    # Fix the lower limits only after both degrees have contributed to the
    # shared autoscale; doing it inside the loop clips the second column.
    for ax in axes[:, 0]:
        ax.set_ylim(bottom=0)
    # Nearly coincident curves still need room for their markers above 1x.
    axes[1, 0].set_ylim(top=cast(float, measured["speedup"].max()) * 1.05)
    axes[0, 0].set_ylabel("cycles / element\n(mean over placements)")
    axes[1, 0].set_ylabel("speedup vs LIBXSMM\n(same degree)")
    fig.tight_layout(pad=0.4)
    save(fig, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FP32 exp JSONL measurements")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_exp(pd.read_json(args.input, lines=True), output_path=args.output)


if __name__ == "__main__":
    main()
