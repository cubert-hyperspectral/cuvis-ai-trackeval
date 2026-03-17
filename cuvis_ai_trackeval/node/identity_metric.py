"""Identity metric node."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_trackeval._dataset import (
    CuvisCOCODataset,
    extract_combined_results,
    run_trackeval_evaluation,
)
from cuvis_ai_trackeval.node._tensor_utils import make_frame_record


class IdentityMetricNode(Node):
    """Accumulate per-frame tracking data and compute ID metrics in finalize()."""

    INPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,)),
        "gt_bboxes": PortSpec(dtype=torch.float32, shape=(1, -1, 4)),
        "gt_track_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "pred_bboxes": PortSpec(dtype=torch.float32, shape=(1, -1, 4)),
        "pred_track_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
    }

    OUTPUT_SPECS = {
        "idf1": PortSpec(dtype=torch.float32, shape=(1,), description="IDF1"),
        "idp": PortSpec(dtype=torch.float32, shape=(1,), description="IDP"),
        "idr": PortSpec(dtype=torch.float32, shape=(1,), description="IDR"),
    }

    def __init__(self, match_threshold: float = 0.5, **kwargs: Any) -> None:
        self.match_threshold = float(match_threshold)
        self._frames: list[dict[str, Any]] = []
        self.last_results: dict[str, torch.Tensor] = {}
        super().__init__(match_threshold=match_threshold, **kwargs)

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
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        self._frames.append(
            make_frame_record(
                frame_id=frame_id,
                gt_bboxes=gt_bboxes,
                gt_track_ids=gt_track_ids,
                pred_bboxes=pred_bboxes,
                pred_track_ids=pred_track_ids,
                pred_scores=None,
            )
        )
        zeros = torch.zeros(1, dtype=torch.float32)
        return {
            "idf1": zeros,
            "idp": zeros,
            "idr": zeros,
        }

    def finalize(self) -> dict[str, torch.Tensor]:
        if not self._frames:
            raise RuntimeError("No frames accumulated. Call forward() before finalize().")

        dataset = CuvisCOCODataset.from_frames(
            self._frames,
            tracker_name=self.name,
            match_threshold=self.match_threshold,
        )
        results, messages = run_trackeval_evaluation(
            dataset,
            metric_names=["identity"],
            match_threshold=self.match_threshold,
            print_results=False,
        )

        status = messages[dataset.get_name()][dataset.tracker_list[0]]
        if status != "Success":
            raise RuntimeError(f"TrackEval Identity evaluation failed: {status}")

        identity_res = extract_combined_results(results, dataset)["Identity"]
        output = {
            "idf1": torch.tensor([float(identity_res["IDF1"])], dtype=torch.float32),
            "idp": torch.tensor([float(identity_res["IDP"])], dtype=torch.float32),
            "idr": torch.tensor([float(identity_res["IDR"])], dtype=torch.float32),
        }
        self.last_results = output
        return output


__all__ = ["IdentityMetricNode"]
