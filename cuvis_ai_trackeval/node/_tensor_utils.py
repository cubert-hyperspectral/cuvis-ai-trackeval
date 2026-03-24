"""Tensor conversion helpers for metric nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy_boxes(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1, 4)


def _to_numpy_ids(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return np.empty((0,), dtype=int)
    return np.asarray(arr, dtype=int).reshape(-1)


def _to_numpy_scores(tensor: torch.Tensor | None, size: int) -> np.ndarray:
    if tensor is None:
        return np.ones((size,), dtype=float)
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)


def _to_int_scalar(tensor: torch.Tensor) -> int:
    return int(tensor.reshape(-1)[0].item())


def make_frame_record(
    *,
    frame_id: torch.Tensor,
    pred_frame_id: torch.Tensor | None,
    gt_bboxes: torch.Tensor,
    gt_track_ids: torch.Tensor,
    pred_bboxes: torch.Tensor,
    pred_track_ids: torch.Tensor,
    pred_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    pred_ids = _to_numpy_ids(pred_track_ids)
    gt_frame = _to_int_scalar(frame_id)
    pred_frame = gt_frame if pred_frame_id is None else _to_int_scalar(pred_frame_id)
    return {
        "frame_id": gt_frame,
        "gt_frame_id": gt_frame,
        "pred_frame_id": pred_frame,
        "gt_bboxes": _to_numpy_boxes(gt_bboxes),
        "gt_track_ids": _to_numpy_ids(gt_track_ids),
        "pred_bboxes": _to_numpy_boxes(pred_bboxes),
        "pred_track_ids": pred_ids,
        "pred_scores": _to_numpy_scores(pred_scores, size=int(pred_ids.shape[0])),
    }


__all__ = ["make_frame_record"]
