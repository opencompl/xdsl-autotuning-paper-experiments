import matplotlib.pyplot as plt
import pandas as pd
import pytest

from autotuner.plot_exp import exp_metrics, plot_exp


def measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine": "tower",
            "kernel": "exp",
            "dtype": "f32",
            "degree": degree,
            "approximation": "libxsmm-cubic" if degree == 3 else "libxsmm-quintic",
            "elements": length,
            "variant": variant,
            "time": length * cost,
        }
        for degree, costs in (
            (3, (("libxsmm", 3), ("libxsmm", 2), ("compxsmm", 4))),
            (5, (("libxsmm", 8), ("libxsmm", 6), ("compxsmm", 3))),
        )
        for length in (16, 32, 64, 128, 256, 512, 1024)
        for variant, cost in costs
    )


def test_exp_compares_best_cycles_at_each_length_and_degree() -> None:
    result = exp_metrics(measurements())
    assert len(result) == 28
    baseline = result[result["variant"] == "libxsmm"]
    ours = result[result["variant"] == "compxsmm"]
    assert baseline["cycles_per_element"].tolist() == [2] * 7 + [6] * 7
    assert baseline["speedup"].tolist() == [1] * 14
    assert ours["cycles_per_element"].tolist() == [4] * 7 + [3] * 7
    assert ours["speedup"].tolist() == [0.5] * 7 + [2] * 7


def test_exp_rejects_incomplete_or_invalid_measurements() -> None:
    df = measurements()
    with pytest.raises(ValueError, match="seven exp lengths"):
        exp_metrics(df[df["elements"] != 32])
    with pytest.raises(ValueError, match="both degrees"):
        exp_metrics(df[df["degree"] == 3])
    with pytest.raises(ValueError, match="both degrees"):
        exp_metrics(df[(df["variant"] != "libxsmm") | (df["degree"] == 3)])
    with pytest.raises(ValueError, match="LIBXSMM baseline"):
        exp_metrics(df[df["variant"] == "compxsmm"])
    with pytest.raises(ValueError, match="matching degree"):
        exp_metrics(df.assign(approximation="libxsmm-cubic"))
    df.loc[0, "time"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        exp_metrics(df)


def test_shared_axes_include_both_degrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_exp(measurements())
    fig = plt.gcf()
    try:
        for ax in fig.axes:
            lower, upper = ax.get_ylim()
            for line in ax.lines:
                assert lower <= min(line.get_ydata())
                assert max(line.get_ydata()) < upper
    finally:
        plt.close(fig)
