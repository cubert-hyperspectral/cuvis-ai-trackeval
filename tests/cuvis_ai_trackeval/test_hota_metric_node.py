from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai_trackeval.node import HOTAMetricNode


@pytest.mark.unit
def test_hota_metric_node_forward_returns_zero_tensors(synthetic_frames: list[dict]) -> None:
    node = HOTAMetricNode(iou_threshold=0.5, name="hota")

    for frame in synthetic_frames:
        out = node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
            pred_scores=frame["pred_scores"],
        )
        assert out["hota"].shape == (1,)
        assert out["deta"].shape == (1,)
        assert out["assa"].shape == (1,)
        assert out["loca"].shape == (1,)
        assert torch.equal(out["hota"], torch.zeros(1, dtype=torch.float32))


@pytest.mark.unit
def test_hota_metric_node_finalize_returns_valid_scores(synthetic_frames: list[dict]) -> None:
    node = HOTAMetricNode(iou_threshold=0.5, name="hota")
    for frame in synthetic_frames:
        node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
            pred_scores=frame["pred_scores"],
        )

    result = node.finalize()
    assert 0.0 <= result["hota"].item() <= 1.0
    assert 0.0 <= result["deta"].item() <= 1.0
    assert 0.0 <= result["assa"].item() <= 1.0
    assert 0.0 <= result["loca"].item() <= 1.0


@pytest.mark.unit
def test_hota_metric_node_finalize_perfect_case(perfect_synthetic_frames: list[dict]) -> None:
    node = HOTAMetricNode(iou_threshold=0.5, name="hota_perfect")
    for frame in perfect_synthetic_frames:
        node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
            pred_scores=frame["pred_scores"],
        )

    result = node.finalize()
    assert math.isclose(result["hota"].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(result["deta"].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(result["assa"].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(result["loca"].item(), 1.0, abs_tol=1e-6)
