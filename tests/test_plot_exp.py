import matplotlib.pyplot as plt
import numpy as np
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
            "time": length * (cost + noise),
            "measurement": "paired-placement-v1",
            "placement_offset": offset,
            "repeat": repeat,
            "order": order,
            "code_address": hex(0x1000 * repeat + offset),
        }
        for degree, variants in (
            (
                3,
                (
                    ("libxsmm", (1, 3, 5, 7), (0, 1, 40)),
                    ("compxsmm", (4, 8, 2, 10), (0, 2, 90)),
                ),
            ),
            (
                5,
                (
                    ("libxsmm", (8, 12, 10, 14), (0, 2, 70)),
                    ("compxsmm", (2, 4, 6, 8), (0, 1, 120)),
                ),
            ),
        )
        for length in (16, 32, 64, 128, 256, 512, 1024)
        for variant, costs, noises in variants
        for offset, cost in zip((0, 16, 32, 48), costs, strict=True)
        for repeat, noise in enumerate(noises, 1)
        for order in (0, 1)
    )


def test_exp_uses_medians_then_equal_weight_for_every_placement() -> None:
    result = exp_metrics(measurements())
    assert len(result) == 28
    baseline = result[result["variant"] == "libxsmm"]
    ours = result[result["variant"] == "compxsmm"]
    assert baseline["cycles_per_element"].tolist() == [5] * 7 + [13] * 7
    assert baseline["speedup"].tolist() == [1] * 14
    assert ours["cycles_per_element"].tolist() == [8] * 7 + [6] * 7
    assert ours["speedup"].tolist() == [5 / 8] * 7 + [13 / 6] * 7


def test_exp_rejects_incomplete_or_invalid_measurements() -> None:
    df = measurements()
    with pytest.raises(ValueError, match="seven exp lengths"):
        exp_metrics(df[df["elements"] != 32])
    with pytest.raises(ValueError, match="both degrees"):
        exp_metrics(df[df["degree"] == 3])
    with pytest.raises(ValueError, match="paired variants"):
        exp_metrics(df[(df["variant"] != "libxsmm") | (df["degree"] == 3)])
    with pytest.raises(ValueError, match="LIBXSMM baseline"):
        exp_metrics(df[df["variant"] == "compxsmm"])
    with pytest.raises(ValueError, match="matching degree"):
        exp_metrics(df.assign(approximation="libxsmm-cubic"))
    with pytest.raises(ValueError, match="paired-placement"):
        exp_metrics(df.drop(columns="measurement"))
    with pytest.raises(ValueError, match="four matched placements"):
        exp_metrics(df.query("placement_offset != 16"))
    with pytest.raises(ValueError, match="both orders"):
        exp_metrics(df.query("order == 0"))
    bad_address = df.copy()
    bad_address.loc[0, "code_address"] = "0xdeadbeef"
    with pytest.raises(ValueError, match="same address"):
        exp_metrics(bad_address)
    df.loc[0, "time"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        exp_metrics(df)


def test_shared_axes_include_both_degrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_exp(measurements())
    fig = plt.gcf()
    try:
        assert len(fig.axes) == 4
        for ax in fig.axes:
            lower, upper = ax.get_ylim()
            for line in ax.lines:
                assert lower <= min(line.get_ydata())
                assert max(line.get_ydata()) < upper
    finally:
        plt.close(fig)


def test_schedule_plot_requires_and_compares_the_three_quintic_curves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    df = measurements().query("degree == 5")
    with pytest.raises(ValueError, match="Horner and Estrin"):
        plot_exp(df, schedules=True)
    estrin = df.query("variant == 'compxsmm'").assign(variant="compxsmm_estrin")
    estrin["time"] *= 0.75
    # Retain a control in the data, but not in the three-curve figure.
    control = df.query("variant == 'libxsmm'").assign(variant="xdsl_libxsmm")
    df = pd.concat([df, estrin, control], ignore_index=True)
    with pytest.raises(ValueError, match="both degrees"):
        exp_metrics(df)
    result = exp_metrics(df, degrees=(5,))
    assert (
        result.query("variant == 'compxsmm_estrin'")["cycles_per_element"].tolist()
        == [4.5] * 7
    )
    plot_exp(df, schedules=True)
    fig = plt.gcf()
    try:
        assert len(fig.axes) == 1
        assert [list(line.get_ydata()) for line in fig.axes[0].lines] == [
            [13] * 7,
            [6] * 7,
            [4.5] * 7,
        ]
        for ax in fig.axes:
            assert [line.get_label() for line in ax.lines] == [
                "LIBXSMM",
                "CompXSMM / Horner",
                "CompXSMM / Estrin",
            ]
            lower, upper = ax.get_ylim()
            for line in ax.lines:
                values = np.asarray(line.get_ydata())
                assert lower <= values.min()
                assert values.max() < upper
    finally:
        plt.close(fig)
