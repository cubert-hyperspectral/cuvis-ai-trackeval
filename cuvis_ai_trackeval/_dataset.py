"""Shared TrackEval dataset adapter and evaluation helpers."""

from __future__ import annotations

import copy
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_numpy_aliases() -> None:
    """Patch deprecated NumPy aliases required by upstream TrackEval."""
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = np.int_  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = np.complex128  # type: ignore[attr-defined]


_ensure_numpy_aliases()

import trackeval  # noqa: E402
from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402
from trackeval.utils import TrackEvalException  # noqa: E402

DEFAULT_METRICS: tuple[str, ...] = ("hota", "clear", "identity")
_VALID_METRICS: tuple[str, ...] = ("hota", "clear", "identity")


def _as_boxes(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 4:
        raise TrackEvalException(f"Expected boxes with shape [N,4], got {arr.shape}")
    return arr.astype(float, copy=False)


def _as_ids(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    if arr.size == 0:
        return np.empty((0,), dtype=int)
    return arr.reshape(-1).astype(int, copy=False)


def _as_classes(value: Any, fallback_size: int, fallback_value: int) -> np.ndarray:
    if value is None:
        return np.full((fallback_size,), fallback_value, dtype=int)
    arr = np.asarray(value, dtype=int)
    if arr.size == 0:
        return np.empty((0,), dtype=int)
    return arr.reshape(-1).astype(int, copy=False)


def _as_scores(value: Any, fallback_size: int) -> np.ndarray:
    if value is None:
        return np.ones((fallback_size,), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    return arr.reshape(-1).astype(float, copy=False)


def _to_xyxy_from_xywh(box: Sequence[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _score_or_default(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalise_metric_names(metric_names: Iterable[str] | None) -> list[str]:
    if metric_names is None:
        return list(DEFAULT_METRICS)
    names = [str(name).lower() for name in metric_names]
    if not names:
        return list(DEFAULT_METRICS)
    if "count" in names:
        raise ValueError("'count' must not be requested explicitly; TrackEval adds it internally.")
    invalid = [name for name in names if name not in _VALID_METRICS]
    if invalid:
        raise ValueError(f"Invalid metric(s): {invalid}. Valid: {list(_VALID_METRICS)}")
    # Preserve input order but remove duplicates.
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def build_metrics_list(metric_names: Iterable[str] | None, match_threshold: float) -> list[Any]:
    names = normalise_metric_names(metric_names)
    metrics: list[Any] = []
    for name in names:
        if name == "hota":
            metrics.append(trackeval.metrics.HOTA())
        elif name == "clear":
            metrics.append(
                trackeval.metrics.CLEAR(
                    {
                        "THRESHOLD": float(match_threshold),
                        "PRINT_CONFIG": False,
                    }
                )
            )
        elif name == "identity":
            metrics.append(
                trackeval.metrics.Identity(
                    {
                        "THRESHOLD": float(match_threshold),
                        "PRINT_CONFIG": False,
                    }
                )
            )
    return metrics


def run_trackeval_evaluation(
    dataset: CuvisCOCODataset,
    metric_names: Iterable[str] | None = None,
    *,
    match_threshold: float = 0.5,
    print_results: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run TrackEval with deterministic output settings and selected metrics."""
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "PRINT_RESULTS": bool(print_results),
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )
    evaluator = trackeval.Evaluator(eval_config)
    metrics = build_metrics_list(metric_names, match_threshold)
    return evaluator.evaluate([dataset], metrics)


def extract_combined_results(
    results: dict[str, Any],
    dataset: CuvisCOCODataset,
) -> dict[str, Any]:
    dataset_name = dataset.get_name()
    tracker_name = dataset.tracker_list[0]
    class_name = dataset.class_list[0]
    return results[dataset_name][tracker_name]["COMBINED_SEQ"][class_name]


class CuvisCOCODataset(_BaseDataset):
    """TrackEval dataset adapter for cuvis-ai COCO tracking data."""

    @staticmethod
    def get_default_dataset_config() -> dict[str, Any]:
        return {
            "PRINT_CONFIG": False,
        }

    @classmethod
    def from_frames(
        cls,
        frames: list[dict[str, Any]],
        *,
        tracker_name: str = "tracker",
        match_threshold: float = 0.5,
        output_dir: str | Path | None = None,
    ) -> CuvisCOCODataset:
        return cls(
            tracker_name=tracker_name,
            match_threshold=match_threshold,
            output_dir=output_dir,
            frames=frames,
        )

    def __init__(
        self,
        gt_path: str | Path | None = None,
        pred_path: str | Path | None = None,
        tracker_name: str = "tracker",
        match_threshold: float = 0.5,
        output_dir: str | Path | None = None,
        frames: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()

        self.config = self.get_default_dataset_config()
        self.match_threshold = float(match_threshold)
        self.tracker_name = str(tracker_name)

        self.should_classes_combine = False
        self.use_super_categories = False
        self.tracker_list = [self.tracker_name]
        self.seq_list = ["sequence"]

        if frames is not None and (gt_path is not None or pred_path is not None):
            raise ValueError("Use either (gt_path + pred_path) or frames, not both.")
        if frames is None and (gt_path is None or pred_path is None):
            raise ValueError("gt_path and pred_path are required when frames are not provided.")

        if frames is not None:
            gt_raw, tracker_raw, class_id = self._build_raw_from_frames(frames)
        else:
            assert gt_path is not None
            assert pred_path is not None
            gt_raw, tracker_raw, class_id = self._build_raw_from_coco_paths(
                Path(gt_path), Path(pred_path)
            )

        self.class_name_to_class_id = {"person": int(class_id)}
        self.class_list = ["person"]

        output_root = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "cuvis_ai_trackeval"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_fol = str(output_root)
        self.output_sub_fol = ""

        self._raw_data_by_seq = {
            self.seq_list[0]: {
                "gt": gt_raw,
                "tracker": tracker_raw,
            }
        }

    def _build_raw_from_coco_paths(
        self,
        gt_path: Path,
        pred_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        pred_data = json.loads(pred_path.read_text(encoding="utf-8"))

        gt_annotations = list(gt_data.get("annotations", []))
        pred_annotations = list(pred_data.get("annotations", []))

        gt_cat_ids = sorted({int(a.get("category_id", 1)) for a in gt_annotations})
        eval_class_id = gt_cat_ids[0] if gt_cat_ids else 1

        gt_image_ids = {int(img["id"]) for img in gt_data.get("images", [])}
        pred_image_ids = {int(img["id"]) for img in pred_data.get("images", [])}
        ann_image_ids = {
            int(ann.get("image_id", -1))
            for ann in [*gt_annotations, *pred_annotations]
            if ann.get("image_id") is not None
        }
        frame_ids = sorted(gt_image_ids | pred_image_ids | ann_image_ids)
        if not frame_ids:
            raise TrackEvalException("No frames found in GT/prediction JSON files.")

        gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in gt_annotations:
            gt_by_image[int(ann["image_id"])].append(ann)
        for ann in pred_annotations:
            pred_by_image[int(ann["image_id"])].append(ann)

        gt_ids: list[np.ndarray] = []
        gt_classes: list[np.ndarray] = []
        gt_dets: list[np.ndarray] = []

        tracker_ids: list[np.ndarray] = []
        tracker_classes: list[np.ndarray] = []
        tracker_dets: list[np.ndarray] = []
        tracker_confidences: list[np.ndarray] = []

        for frame_id in frame_ids:
            gt_anns_t = gt_by_image.get(frame_id, [])
            pred_anns_t = pred_by_image.get(frame_id, [])

            gt_boxes_t: list[list[float]] = []
            gt_ids_t: list[int] = []
            gt_classes_t: list[int] = []
            for ann in gt_anns_t:
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) != 4:
                    raise TrackEvalException(f"Invalid GT bbox for frame {frame_id}: {bbox}")
                gt_boxes_t.append(_to_xyxy_from_xywh(bbox))
                gt_ids_t.append(int(ann.get("track_id", ann.get("id", 0))))
                gt_classes_t.append(int(ann.get("category_id", eval_class_id)))

            pred_boxes_t: list[list[float]] = []
            pred_ids_t: list[int] = []
            pred_classes_t: list[int] = []
            pred_scores_t: list[float] = []
            for ann in pred_anns_t:
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) != 4:
                    raise TrackEvalException(
                        f"Invalid prediction bbox for frame {frame_id}: {bbox}"
                    )
                pred_boxes_t.append(_to_xyxy_from_xywh(bbox))
                pred_ids_t.append(int(ann.get("track_id", ann.get("id", 0))))
                cat_id = int(ann.get("category_id", eval_class_id))
                if cat_id == 0:
                    cat_id = eval_class_id
                pred_classes_t.append(cat_id)
                pred_scores_t.append(_score_or_default(ann.get("score"), default=1.0))

            gt_ids.append(_as_ids(gt_ids_t))
            gt_classes.append(_as_classes(gt_classes_t, len(gt_ids_t), eval_class_id))
            gt_dets.append(_as_boxes(gt_boxes_t))

            tracker_ids.append(_as_ids(pred_ids_t))
            tracker_classes.append(_as_classes(pred_classes_t, len(pred_ids_t), eval_class_id))
            tracker_dets.append(_as_boxes(pred_boxes_t))
            tracker_confidences.append(_as_scores(pred_scores_t, len(pred_ids_t)))

        gt_raw = {
            "gt_ids": gt_ids,
            "gt_classes": gt_classes,
            "gt_dets": gt_dets,
            "num_timesteps": len(frame_ids),
            "seq": self.seq_list[0],
        }
        tracker_raw = {
            "tracker_ids": tracker_ids,
            "tracker_classes": tracker_classes,
            "tracker_dets": tracker_dets,
            "tracker_confidences": tracker_confidences,
            "num_timesteps": len(frame_ids),
            "seq": self.seq_list[0],
        }
        return gt_raw, tracker_raw, eval_class_id

    def _build_raw_from_frames(
        self,
        frames: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        if not frames:
            raise TrackEvalException("frames must contain at least one frame.")

        gt_frames: dict[int, dict[str, np.ndarray]] = {}
        pred_frames: dict[int, dict[str, np.ndarray]] = {}
        eval_class_id = 1

        for frame in frames:
            gt_frame_id = int(frame.get("gt_frame_id", frame["frame_id"]))
            pred_frame_id = int(frame.get("pred_frame_id", frame.get("frame_id", gt_frame_id)))

            gt_track_ids = _as_ids(frame.get("gt_track_ids", []))
            gt_frames[gt_frame_id] = {
                "ids": gt_track_ids,
                "classes": _as_classes(
                    frame.get("gt_category_ids"), len(gt_track_ids), eval_class_id
                ),
                "dets": _as_boxes(frame.get("gt_bboxes", [])),
            }

            pred_track_ids = _as_ids(frame.get("pred_track_ids", []))
            pred_classes = _as_classes(
                frame.get("pred_category_ids"),
                len(pred_track_ids),
                eval_class_id,
            )
            pred_classes[pred_classes == 0] = eval_class_id
            pred_frames[pred_frame_id] = {
                "ids": pred_track_ids,
                "classes": pred_classes,
                "dets": _as_boxes(frame.get("pred_bboxes", [])),
                "scores": _as_scores(frame.get("pred_scores"), len(pred_track_ids)),
            }

        frame_ids = sorted(set(gt_frames) | set(pred_frames))
        if not frame_ids:
            raise TrackEvalException("frames must contain at least one frame.")

        gt_ids: list[np.ndarray] = []
        gt_classes: list[np.ndarray] = []
        gt_dets: list[np.ndarray] = []

        tracker_ids: list[np.ndarray] = []
        tracker_classes: list[np.ndarray] = []
        tracker_dets: list[np.ndarray] = []
        tracker_confidences: list[np.ndarray] = []

        for frame_id in frame_ids:
            gt_frame = gt_frames.get(frame_id)
            if gt_frame is None:
                gt_ids.append(np.empty((0,), dtype=int))
                gt_classes.append(np.empty((0,), dtype=int))
                gt_dets.append(np.empty((0, 4), dtype=float))
            else:
                gt_ids.append(gt_frame["ids"])
                gt_classes.append(gt_frame["classes"])
                gt_dets.append(gt_frame["dets"])

            pred_frame = pred_frames.get(frame_id)
            if pred_frame is None:
                tracker_ids.append(np.empty((0,), dtype=int))
                tracker_classes.append(np.empty((0,), dtype=int))
                tracker_dets.append(np.empty((0, 4), dtype=float))
                tracker_confidences.append(np.empty((0,), dtype=float))
            else:
                tracker_ids.append(pred_frame["ids"])
                tracker_classes.append(pred_frame["classes"])
                tracker_dets.append(pred_frame["dets"])
                tracker_confidences.append(pred_frame["scores"])

        gt_raw = {
            "gt_ids": gt_ids,
            "gt_classes": gt_classes,
            "gt_dets": gt_dets,
            "num_timesteps": len(frame_ids),
            "seq": self.seq_list[0],
        }
        tracker_raw = {
            "tracker_ids": tracker_ids,
            "tracker_classes": tracker_classes,
            "tracker_dets": tracker_dets,
            "tracker_confidences": tracker_confidences,
            "num_timesteps": len(frame_ids),
            "seq": self.seq_list[0],
        }
        return gt_raw, tracker_raw, eval_class_id

    @staticmethod
    def _clone_raw(raw: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                cloned[key] = [
                    np.array(item, copy=True)
                    if isinstance(item, np.ndarray)
                    else copy.deepcopy(item)
                    for item in value
                ]
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _load_raw_file(self, tracker: str, seq: str, is_gt: bool) -> dict[str, Any]:
        if tracker != self.tracker_name:
            raise TrackEvalException(f"Unknown tracker: {tracker}")
        if seq not in self._raw_data_by_seq:
            raise TrackEvalException(f"Unknown sequence: {seq}")

        key = "gt" if is_gt else "tracker"
        return self._clone_raw(self._raw_data_by_seq[seq][key])

    def get_preprocessed_seq_data(self, raw_data: dict[str, Any], cls: str) -> dict[str, Any]:
        cls_id = self.class_name_to_class_id.get(cls)
        if cls_id is None:
            raise TrackEvalException(f"Unknown class '{cls}'")

        self._check_unique_ids(raw_data)

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}

        unique_gt_ids: list[int] = []
        unique_tracker_ids: list[int] = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id)
            tracker_class_mask = np.atleast_1d(raw_data["tracker_classes"][t] == cls_id)

            data["gt_ids"][t] = raw_data["gt_ids"][t][gt_class_mask]
            data["gt_dets"][t] = raw_data["gt_dets"][t][gt_class_mask]

            data["tracker_ids"][t] = raw_data["tracker_ids"][t][tracker_class_mask]
            data["tracker_dets"][t] = raw_data["tracker_dets"][t][tracker_class_mask]
            data["tracker_confidences"][t] = raw_data["tracker_confidences"][t][tracker_class_mask]

            similarity = raw_data["similarity_scores"][t]
            data["similarity_scores"][t] = similarity[gt_class_mask, :][:, tracker_class_mask]

            unique_gt_ids.extend([int(v) for v in np.unique(data["gt_ids"][t])])
            unique_tracker_ids.extend([int(v) for v in np.unique(data["tracker_ids"][t])])
            num_gt_dets += int(data["gt_ids"][t].shape[0])
            num_tracker_dets += int(data["tracker_ids"][t].shape[0])

        gt_id_map: dict[int, int] = {}
        tracker_id_map: dict[int, int] = {}

        if unique_gt_ids:
            for idx, track_id in enumerate(sorted(set(unique_gt_ids))):
                gt_id_map[track_id] = idx
            for t in range(raw_data["num_timesteps"]):
                if data["gt_ids"][t].size:
                    data["gt_ids"][t] = np.asarray(
                        [gt_id_map[int(track_id)] for track_id in data["gt_ids"][t]],
                        dtype=int,
                    )

        if unique_tracker_ids:
            for idx, track_id in enumerate(sorted(set(unique_tracker_ids))):
                tracker_id_map[track_id] = idx
            for t in range(raw_data["num_timesteps"]):
                if data["tracker_ids"][t].size:
                    data["tracker_ids"][t] = np.asarray(
                        [tracker_id_map[int(track_id)] for track_id in data["tracker_ids"][t]],
                        dtype=int,
                    )

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(tracker_id_map)
        data["num_gt_ids"] = len(gt_id_map)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        self._check_unique_ids(data, after_preproc=True)
        return data

    def _calculate_similarities(
        self, gt_dets_t: np.ndarray, tracker_dets_t: np.ndarray
    ) -> np.ndarray:
        return self._calculate_box_ious(
            gt_dets_t,
            tracker_dets_t,
            box_format="x0y0x1y1",
        )


__all__ = [
    "CuvisCOCODataset",
    "DEFAULT_METRICS",
    "build_metrics_list",
    "normalise_metric_names",
    "run_trackeval_evaluation",
    "extract_combined_results",
]
