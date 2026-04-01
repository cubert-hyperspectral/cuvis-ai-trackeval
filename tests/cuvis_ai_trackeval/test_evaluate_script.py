from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cuvis_ai_trackeval.cli.evaluate import main


def _extract_table_value(markdown: str, label: str) -> float:
    pattern = rf"\|\s*{re.escape(label)}\s*\|\s*([-+]?\d*\.?\d+)\s*\|"
    match = re.search(pattern, markdown)
    if match is None:
        raise AssertionError(f"Could not find table value for label '{label}'")
    return float(match.group(1))


@pytest.mark.unit
def test_evaluate_script_generates_markdown(
    gt_coco_path: Path,
    pred_coco_path: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "report"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--gt",
            str(gt_coco_path),
            "--pred",
            str(pred_coco_path),
            "--output-dir",
            str(out_dir),
            "--tracker-name",
            "test_tracker",
            "--metrics",
            "hota",
            "--metrics",
            "clear",
            "--metrics",
            "identity",
        ],
    )

    assert result.exit_code == 0, result.output

    report_path = out_dir / "test_tracker_metrics_report.md"
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")
    assert "## HOTA Details" in content
    assert "## CLEAR Details" in content
    assert "## Identity Details" in content

    assert "nan" not in content.lower()
    assert "none" not in content.lower()


@pytest.mark.unit
def test_evaluate_script_same_json_is_perfect(same_coco_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "perfect"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--gt",
            str(same_coco_path),
            "--pred",
            str(same_coco_path),
            "--output-dir",
            str(out_dir),
            "--tracker-name",
            "same_tracker",
            "--metrics",
            "hota",
            "--metrics",
            "clear",
            "--metrics",
            "identity",
        ],
    )

    assert result.exit_code == 0, result.output

    report_path = out_dir / "same_tracker_metrics_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    for metric in ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR"]:
        value = _extract_table_value(content, metric)
        assert math.isclose(value, 1.0, rel_tol=1e-6, abs_tol=1e-6), f"{metric}={value}"

    assert np.isclose(_extract_table_value(content, "FP"), 0.0)
    assert np.isclose(_extract_table_value(content, "FN"), 0.0)
    assert np.isclose(_extract_table_value(content, "IDSW"), 0.0)
