import pandas as pd

from autotuner.plot_data import best_of_repeats

VARIANT = "libxsmm-skx-fsdbcst"


def passes(*times: float, m: int = 2, n: int = 1, k: int = 1) -> pd.DataFrame:
    """One sample measured once per given time, the way the jsonl records it."""
    return pd.DataFrame(
        [
            {
                "M": m,
                "N": n,
                "K": k,
                "time": time,
                "variant": VARIANT,
                "repeat": index + 1,
            }
            for index, time in enumerate(times)
        ]
    )


def test_the_fastest_pass_of_a_sample_is_the_one_kept() -> None:
    kept = best_of_repeats(passes(30.0, 25.0, 41.0))

    assert list(kept["time"]) == [25.0]
    assert list(kept["repeat"]) == [2]


def test_a_sample_measured_once_comes_through_unchanged() -> None:
    single = passes(30.0)

    assert best_of_repeats(single).to_dict("records") == single.to_dict("records")


def test_every_sample_keeps_its_own_fastest_pass() -> None:
    df = pd.concat([passes(30.0, 25.0), passes(51.0, 60.0, k=2)])

    kept = best_of_repeats(df).sort_values("K")

    assert list(zip(kept["K"], kept["time"])) == [(1, 25.0), (2, 51.0)]
