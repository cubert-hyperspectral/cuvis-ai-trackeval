"""CLI for evaluating tracking results with TrackEval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from tabulate import tabulate

from cuvis_ai_trackeval._dataset import (
    DEFAULT_METRICS,
    CuvisCOCODataset,
    extract_combined_results,
    run_trackeval_evaluation,
)


def _fmt_float(value: Any) -> str:
    return f"{float(value):.6f}"


def _fmt_int(value: Any) -> str:
    return str(int(round(float(value))))


def _mean(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _render_markdown_report(
    *,
    tracker_name: str,
    gt_path: Path,
    pred_path: Path,
    metrics: list[str],
    match_threshold: float,
    combined: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Tracking Evaluation Report",
        "",
        f"**Tracker:** {tracker_name}",
        f"**Ground truth:** {gt_path}",
        f"**Predictions:** {pred_path}",
        f"**Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Match threshold (CLEAR/Identity):** {match_threshold}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    summary_rows: list[list[str]] = []
    hota_res = combined.get("HOTA")
    clear_res = combined.get("CLEAR")
    identity_res = combined.get("Identity")

    if hota_res is not None:
        summary_rows.extend(
            [
                ["HOTA", _fmt_float(_mean(hota_res["HOTA"]))],
                ["DetA", _fmt_float(_mean(hota_res["DetA"]))],
                ["AssA", _fmt_float(_mean(hota_res["AssA"]))],
                ["LocA", _fmt_float(_mean(hota_res["LocA"]))],
            ]
        )
    if clear_res is not None:
        summary_rows.extend(
            [
                ["MOTA", _fmt_float(clear_res["MOTA"])],
                ["MOTP", _fmt_float(clear_res["MOTP"])],
            ]
        )
    if identity_res is not None:
        summary_rows.extend(
            [
                ["IDF1", _fmt_float(identity_res["IDF1"])],
                ["IDP", _fmt_float(identity_res["IDP"])],
                ["IDR", _fmt_float(identity_res["IDR"])],
            ]
        )

    lines.append(
        tabulate(
            summary_rows,
            headers=["Metric", "Value"],
            tablefmt="github",
            disable_numparse=True,
        )
    )

    if hota_res is not None:
        lines.extend(
            [
                "",
                "---",
                "",
                "## HOTA Details",
                "",
            ]
        )
        alphas = np.arange(0.05, 0.99, 0.05)
        hota_rows: list[list[str]] = []
        for idx, alpha in enumerate(alphas):
            hota_rows.append(
                [
                    f"{alpha:.2f}",
                    _fmt_float(hota_res["HOTA"][idx]),
                    _fmt_float(hota_res["DetA"][idx]),
                    _fmt_float(hota_res["AssA"][idx]),
                    _fmt_float(hota_res["LocA"][idx]),
                ]
            )
        hota_rows.append(
            [
                "Mean",
                _fmt_float(_mean(hota_res["HOTA"])),
                _fmt_float(_mean(hota_res["DetA"])),
                _fmt_float(_mean(hota_res["AssA"])),
                _fmt_float(_mean(hota_res["LocA"])),
            ]
        )
        lines.append(
            tabulate(
                hota_rows,
                headers=["Alpha", "HOTA", "DetA", "AssA", "LocA"],
                tablefmt="github",
                disable_numparse=True,
            )
        )

    if clear_res is not None:
        lines.extend(
            [
                "",
                "---",
                "",
                "## CLEAR Details",
                "",
            ]
        )
        clear_rows = [
            ["MOTA", _fmt_float(clear_res["MOTA"])],
            ["MOTP", _fmt_float(clear_res["MOTP"])],
            ["MT (%)", _fmt_float(100.0 * float(clear_res["MTR"]))],
            ["PT (%)", _fmt_float(100.0 * float(clear_res["PTR"]))],
            ["ML (%)", _fmt_float(100.0 * float(clear_res["MLR"]))],
            ["FP", _fmt_int(clear_res["CLR_FP"])],
            ["FN", _fmt_int(clear_res["CLR_FN"])],
            ["IDSW", _fmt_int(clear_res["IDSW"])],
        ]
        lines.append(
            tabulate(
                clear_rows,
                headers=["Metric", "Value"],
                tablefmt="github",
                disable_numparse=True,
            )
        )

    if identity_res is not None:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Identity Details",
                "",
            ]
        )
        lines.append(
            tabulate(
                [
                    [
                        _fmt_float(identity_res["IDF1"]),
                        _fmt_float(identity_res["IDP"]),
                        _fmt_float(identity_res["IDR"]),
                    ]
                ],
                headers=["IDF1", "IDP", "IDR"],
                tablefmt="github",
                disable_numparse=True,
            )
        )

    count_res = combined.get("Count")
    if count_res is not None:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Count Details",
                "",
            ]
        )
        count_rows = [
            ["GT Detections", _fmt_int(count_res["GT_Dets"])],
            ["Predicted Detections", _fmt_int(count_res["Dets"])],
            ["GT IDs", _fmt_int(count_res["GT_IDs"])],
            ["Predicted IDs", _fmt_int(count_res["IDs"])],
        ]
        lines.append(
            tabulate(
                count_rows,
                headers=["Metric", "Value"],
                tablefmt="github",
                disable_numparse=True,
            )
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Configuration",
            "",
            f"- Metrics: {', '.join(metric.upper() for metric in metrics)} (Count is added internally)",
            f"- Match threshold (CLEAR/Identity): {match_threshold}",
            "- Box format: xyxy (IoU via TrackEval x0y0x1y1)",
            "",
        ]
    )
    return "\n".join(lines)


@click.command()
@click.option(
    "--gt",
    "gt_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to GT COCO JSON.",
)
@click.option(
    "--pred",
    "pred_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to predictions COCO JSON.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for markdown report output.",
)
@click.option(
    "--tracker-name",
    type=str,
    default="tracker",
    show_default=True,
    help="Tracker display name.",
)
@click.option(
    "--metrics",
    "metrics",
    multiple=True,
    default=DEFAULT_METRICS,
    show_default=True,
    type=click.Choice(["hota", "clear", "identity"], case_sensitive=False),
    help="Metrics to evaluate (Count is always added internally by TrackEval).",
)
@click.option(
    "--match-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="IoU threshold for CLEAR and Identity matching.",
)
def main(
    gt_path: Path,
    pred_path: Path,
    output_dir: Path,
    tracker_name: str,
    metrics: tuple[str, ...],
    match_threshold: float,
) -> None:
    """Run TrackEval on cuvis-ai COCO tracking JSON files and emit markdown report."""
    if not (0.0 <= match_threshold <= 1.0):
        raise click.BadParameter("--match-threshold must be in [0.0, 1.0].")

    metrics_list = [metric.lower() for metric in metrics]
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CuvisCOCODataset(
        gt_path=gt_path,
        pred_path=pred_path,
        tracker_name=tracker_name,
        match_threshold=match_threshold,
        output_dir=output_dir,
    )

    results, messages = run_trackeval_evaluation(
        dataset,
        metric_names=metrics_list,
        match_threshold=match_threshold,
        print_results=True,
    )

    dataset_name = dataset.get_name()
    status = messages[dataset_name][tracker_name]
    if status != "Success":
        raise click.ClickException(f"TrackEval failed: {status}")

    combined = extract_combined_results(results, dataset)
    report_text = _render_markdown_report(
        tracker_name=tracker_name,
        gt_path=gt_path,
        pred_path=pred_path,
        metrics=metrics_list,
        match_threshold=match_threshold,
        combined=combined,
    )

    output_path = output_dir / f"{tracker_name}_metrics_report.md"
    output_path.write_text(report_text, encoding="utf-8")
    click.echo(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
