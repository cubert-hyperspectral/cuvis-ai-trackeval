"""HOTA metric node."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_trackeval._dataset import (
    CuvisCOCODataset,
    extract_combined_results,
    run_trackeval_evaluation,
)
from cuvis_ai_trackeval.node._tensor_utils import make_frame_record


class HOTAMetricNode(Node):
    """Accumulate per-frame tracking data and compute HOTA in finalize()."""

    _category = NodeCategory.METRIC
    _tags = frozenset({
        NodeTag.BBOX,
        NodeTag.TRACKING,
        NodeTag.EVALUATION,
        NodeTag.NUMPY,
    })

    INPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,)),
        "pred_frame_id": PortSpec(dtype=torch.int64, shape=(1,), optional=True),
        "gt_bboxes": PortSpec(dtype=torch.float32, shape=(1, -1, 4)),
        "gt_track_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "pred_bboxes": PortSpec(dtype=torch.float32, shape=(1, -1, 4)),
        "pred_track_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "pred_scores": PortSpec(dtype=torch.float32, shape=(1, -1), optional=True),
    }

    OUTPUT_SPECS = {
        "hota": PortSpec(dtype=torch.float32, shape=(1,), description="Mean HOTA"),
        "deta": PortSpec(dtype=torch.float32, shape=(1,), description="Mean DetA"),
        "assa": PortSpec(dtype=torch.float32, shape=(1,), description="Mean AssA"),
        "loca": PortSpec(dtype=torch.float32, shape=(1,), description="Mean LocA"),
    }

    def __init__(self, iou_threshold: float = 0.5, **kwargs: Any) -> None:
        self.iou_threshold = float(iou_threshold)
        self._frames: list[dict[str, Any]] = []
        self.last_results: dict[str, torch.Tensor] = {}
        super().__init__(iou_threshold=iou_threshold, **kwargs)

    def reset(self) -> None:
        self._frames.clear()
        self.last_results = {}

    def forward(
        self,
        frame_id: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_track_ids: torch.Tensor,
        pred_bboxes: torch.Tensor,
        pred_track_ids: torch.Tensor,
        pred_frame_id: torch.Tensor | None = None,
        pred_scores: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        self._frames.append(
            make_frame_record(
                frame_id=frame_id,
                pred_frame_id=pred_frame_id,
                gt_bboxes=gt_bboxes,
                gt_track_ids=gt_track_ids,
                pred_bboxes=pred_bboxes,
                pred_track_ids=pred_track_ids,
                pred_scores=pred_scores,
            )
        )
        zeros = torch.zeros(1, dtype=torch.float32)
        return {
            "hota": zeros,
            "deta": zeros,
            "assa": zeros,
            "loca": zeros,
        }

    def finalize(self) -> dict[str, torch.Tensor]:
        if not self._frames:
            raise RuntimeError("No frames accumulated. Call forward() before finalize().")

        dataset = CuvisCOCODataset.from_frames(
            self._frames,
            tracker_name=self.name,
            match_threshold=self.iou_threshold,
        )
        results, messages = run_trackeval_evaluation(
            dataset,
            metric_names=["hota"],
            match_threshold=self.iou_threshold,
            print_results=False,
        )

        status = messages[dataset.get_name()][dataset.tracker_list[0]]
        if status != "Success":
            raise RuntimeError(f"TrackEval HOTA evaluation failed: {status}")

        hota_res = extract_combined_results(results, dataset)["HOTA"]
        output = {
            "hota": torch.tensor([float(np.mean(hota_res["HOTA"]))], dtype=torch.float32),
            "deta": torch.tensor([float(np.mean(hota_res["DetA"]))], dtype=torch.float32),
            "assa": torch.tensor([float(np.mean(hota_res["AssA"]))], dtype=torch.float32),
            "loca": torch.tensor([float(np.mean(hota_res["LocA"]))], dtype=torch.float32),
        }
        self.last_results = output
        return output


__all__ = ["HOTAMetricNode"]
