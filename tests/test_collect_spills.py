from pathlib import Path

import pytest
import yaml

from autotuner.collect_spills import read_spills


def test_function_totals_include_folded_accesses_but_not_loop_or_helper_totals(
    tmp_path: Path,
) -> None:
    report = tmp_path / "spills.yaml"
    report.write_text("""--- !Missed
Pass: regalloc
Name: LoopSpillReloadCopies
Function: matmul
Args:
  - NumSpills: '2'
  - NumReloads: '3'
--- !Missed
Pass: regalloc
Name: SpillReloadCopies
Function: helper
Args:
  - NumSpills: '99'
--- !Missed
Pass: regalloc
Name: SpillReloadCopies
Function: matmul
Args:
  - NumSpills: '3'
  - NumFoldedSpills: '2'
  - NumReloads: '5'
  - NumFoldedReloads: '7'
  - TotalSpillsCost: '123.0'
  - NumVRCopies: '42'
...
""")
    assert read_spills(report) == {
        "spills": 5,
        "reloads": 12,
        "folded_spills": 2,
        "folded_reloads": 7,
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        """--- !Missed
Pass: regalloc
Name: SpillReloadCopies
Function: matmul
Args:
  - NumVRCopies: '4'
...
""",
    ],
)
def test_a_successful_report_with_no_spills_is_zero(tmp_path: Path, text: str) -> None:
    report = tmp_path / "spills.yaml"
    report.write_text(text)
    assert read_spills(report) == {
        "spills": 0,
        "reloads": 0,
        "folded_spills": 0,
        "folded_reloads": 0,
    }


def test_missing_or_broken_reports_are_not_zero(tmp_path: Path) -> None:
    report = tmp_path / "spills.yaml"
    with pytest.raises(FileNotFoundError):
        read_spills(report)
    report.write_text("--- !Missed\nArgs: [")
    with pytest.raises(yaml.YAMLError):
        read_spills(report)
