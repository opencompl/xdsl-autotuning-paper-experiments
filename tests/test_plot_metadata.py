from dataclasses import replace

import pandas as pd
import pytest

from autotuner.machines import MACHINES
from autotuner.plot_ttile import machine_display_names, result_machine_label


def test_machine_display_names_come_from_machine_config() -> None:
    names = machine_display_names()

    assert names["pinocchio"] == "Intel Cascade Lake"
    assert result_machine_label(pd.DataFrame([{"machine": "pinocchio"}])) == (
        "pinocchio",
        "Intel Cascade Lake",
    )


def test_explicit_machine_config_is_authoritative() -> None:
    machine_configs = {
        **MACHINES,
        "tower": replace(MACHINES["tower"], display_name="Configured Name"),
    }
    results = pd.DataFrame([{"machine": "tower", "display_name": "Stale Name"}])

    assert result_machine_label(results, machine_configs) == (
        "tower",
        "Configured Name",
    )


def test_unknown_machine_is_rejected() -> None:
    with pytest.raises(ValueError, match="is not configured"):
        result_machine_label(pd.DataFrame([{"machine": "unknown"}]))
