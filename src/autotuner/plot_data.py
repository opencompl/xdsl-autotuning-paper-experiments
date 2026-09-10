"""Turning a measured jsonl into the numbers a figure draws.

The figures read their datasets the same way -- throughput as a share of the
machine's peak, and, for a dataset swept more than once, the fastest pass of
each sample -- so both live here rather than once per figure.
"""

import pandas as pd

# What identifies one measurement, so what the repeated passes of a dataset
# have in common and `best_of_repeats` groups by.
SAMPLE_KEY = ("variant", "M", "N", "K")


def percent_of_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``percent`` column holding throughput as a share of machine peak."""
    measured = df[df["time"] > 0].copy()
    assert isinstance(measured, pd.DataFrame)

    peaks = measured["peak"].dropna().unique()
    if len(peaks) != 1:
        raise ValueError(f"expected one peak in the dataset, found {sorted(peaks)}")
    peak = float(peaks[0])
    if peak == 0.0:
        raise ValueError("the dataset has no peak, so % of peak is undefined")

    measured["percent"] = (measured["flops"] / measured["time"]) / peak * 100
    return measured


def best_of_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each sample's fastest pass, dropping the rest.

    A dataset whose kernels run for too little time to be quiet is swept
    several times over -- `datasets.DATASET_REPEATS` -- because whatever else
    the machine happens to be doing then shows up in the number.  That noise is
    one-sided: it can only ever make a kernel look slower than it is, never
    faster, so the fastest pass is the least disturbed estimate of the kernel
    rather than a lucky outlier.  A dataset with one pass per sample comes
    through unchanged.
    """
    # Sort and deduplicate rather than group and take the minimum: this keeps
    # the whole row of the pass that won, and it does not care whether the
    # frame's index labels are unique the way a read straight from jsonl is.
    kept = df.sort_values("time").drop_duplicates(list(SAMPLE_KEY))
    assert isinstance(kept, pd.DataFrame)
    return kept
