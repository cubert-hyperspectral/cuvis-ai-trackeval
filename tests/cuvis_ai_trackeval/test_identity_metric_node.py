from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai_trackeval.node import IdentityMetricNode


@pytest.mark.unit
def test_identity_metric_node_forward_returns_zero_tensors(synthetic_frames: list[dict]) -> None:
    node = IdentityMetricNode(match_threshold=0.5, name="identity")

    for frame in synthetic_frames:
        out = node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
        )
        assert out["idf1"].shape == (1,)
        assert out["idp"].shape == (1,)
        assert out["idr"].shape == (1,)


@pytest.mark.unit
def test_identity_metric_node_finalize_returns_valid_scores(synthetic_frames: list[dict]) -> None:
    node = IdentityMetricNode(match_threshold=0.5, name="identity")

    for frame in synthetic_frames:
        node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
        )

    result = node.finalize()
    assert 0.0 <= result["idf1"].item() <= 1.0
    assert 0.0 <= result["idp"].item() <= 1.0
    assert 0.0 <= result["idr"].item() <= 1.0


@pytest.mark.unit
def test_identity_metric_node_finalize_perfect_case(perfect_synthetic_frames: list[dict]) -> None:
    node = IdentityMetricNode(match_threshold=0.5, name="identity_perfect")

    for frame in perfect_synthetic_frames:
        node.forward(
            frame_id=torch.tensor([frame["frame_id"]], dtype=torch.int64),
            gt_bboxes=frame["gt_bboxes"],
            gt_track_ids=frame["gt_track_ids"],
            pred_bboxes=frame["pred_bboxes"],
            pred_track_ids=frame["pred_track_ids"],
        )

    result = node.finalize()
    assert math.isclose(result["idf1"].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(result["idp"].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(result["idr"].item(), 1.0, abs_tol=1e-6)
