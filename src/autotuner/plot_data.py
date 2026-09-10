"""Turning a measured jsonl into the numbers a figure draws."""

import pandas as pd

# What identifies one measurement, so what the repeated passes of a dataset
# have in common and `best_of_repeats` groups by.
SAMPLE_KEY = ("variant", "M", "N", "K")


def best_of_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each sample's fastest pass, dropping the rest.

    A dataset too short-running to be quiet is swept several times over --
    `datasets.DATASET_REPEATS` -- and that noise is one-sided: it can only make
    a kernel look slower, never faster, so the fastest pass is the least
    disturbed estimate rather than a lucky outlier.  A dataset with one pass
    per sample comes through unchanged.
    """
    # Sort and deduplicate rather than group and take the minimum: this keeps
    # the whole row of the pass that won, and it does not care whether the
    # frame's index labels are unique the way a read straight from jsonl is.
    kept = df.sort_values("time").drop_duplicates(list(SAMPLE_KEY))
    assert isinstance(kept, pd.DataFrame)
    return kept
