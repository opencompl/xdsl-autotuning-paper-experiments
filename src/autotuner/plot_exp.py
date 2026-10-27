"""The initial FP32 exp sweep: cycles per element and speedup over LIBXSMM."""

from pathlib import Path

import numpy as np
import pandas as pd

from autotuner.datasets import EXP_DEGREES, EXP_LENGTHS
from autotuner.plot_data import best_of_repeats
from autotuner.plot_style import (
    PAGE_WIDTH,
    column_figure,
    save,
    tidy_axes,
    variant_style,
)


def exp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compare implementations at the same length and polynomial degree."""
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

    measured = best_of_repeats(df, ("variant", "degree", "elements"))
    expected = {(degree, length) for degree in EXP_DEGREES for length in EXP_LENGTHS}
    for variant, group in measured.groupby("variant"):
        if set(zip(group["degree"], group["elements"], strict=True)) != expected:
            raise ValueError(f"{variant} needs all seven exp lengths for both degrees")
    baseline = measured[measured["variant"] == "libxsmm"].set_index(
        ["degree", "elements"]
    )
    if baseline.empty:
        raise ValueError("exp needs a LIBXSMM baseline")
    measured = measured.join(
        baseline["time"].rename("baseline_time"), on=["degree", "elements"]
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
    axes[0, 0].set_ylabel("cycles / element")
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
