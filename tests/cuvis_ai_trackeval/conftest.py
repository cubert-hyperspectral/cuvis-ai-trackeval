from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _build_base_payload(category_id: int = 1) -> dict:
    images = [
        {"id": 1, "file_name": "frame_0001.png", "width": 640, "height": 480},
        {"id": 2, "file_name": "frame_0002.png", "width": 640, "height": 480},
        {"id": 3, "file_name": "frame_0003.png", "width": 640, "height": 480},
    ]
    categories = [{"id": 1, "name": "person"}]
    boxes_by_frame = {
        1: [(100.0, 120.0, 40.0, 60.0), (220.0, 180.0, 50.0, 70.0)],
        2: [(104.0, 124.0, 40.0, 60.0), (224.0, 184.0, 50.0, 70.0)],
        3: [(108.0, 128.0, 40.0, 60.0), (228.0, 188.0, 50.0, 70.0)],
    }

    annotations: list[dict] = []
    ann_id = 1
    for image_id, boxes in boxes_by_frame.items():
        for track_id, (x, y, w, h) in enumerate(boxes, start=1):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "track_id": track_id,
                    "score": None,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def gt_coco_path(tmp_path: Path) -> Path:
    payload = _build_base_payload(category_id=1)
    return _write_json(tmp_path / "gt.json", payload)


@pytest.fixture
def pred_coco_path(tmp_path: Path) -> Path:
    payload = _build_base_payload(category_id=0)
    for ann in payload["annotations"]:
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x + 2.0, y + 2.0, w, h]
        ann["score"] = 0.95
    return _write_json(tmp_path / "pred.json", payload)


@pytest.fixture
def same_coco_path(tmp_path: Path) -> Path:
    payload = _build_base_payload(category_id=1)
    for ann in payload["annotations"]:
        ann["score"] = 1.0
    return _write_json(tmp_path / "same.json", payload)


@pytest.fixture
def synthetic_frames() -> list[dict]:
    return [
        {
            "frame_id": fid,
            "gt_bboxes": torch.tensor(
                [[[10.0 + fid, 10.0 + fid, 50.0 + fid, 50.0 + fid], [60.0, 60.0, 100.0, 100.0]]],
                dtype=torch.float32,
            ),
            "gt_track_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pred_bboxes": torch.tensor(
                [[[12.0 + fid, 12.0 + fid, 52.0 + fid, 52.0 + fid], [62.0, 62.0, 102.0, 102.0]]],
                dtype=torch.float32,
            ),
            "pred_track_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pred_scores": torch.tensor([[0.9, 0.85]], dtype=torch.float32),
        }
        for fid in range(3)
    ]


@pytest.fixture
def perfect_synthetic_frames() -> list[dict]:
    return [
        {
            "frame_id": fid,
            "gt_bboxes": torch.tensor(
                [[[10.0 + fid, 10.0 + fid, 50.0 + fid, 50.0 + fid], [60.0, 60.0, 100.0, 100.0]]],
                dtype=torch.float32,
            ),
            "gt_track_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pred_bboxes": torch.tensor(
                [[[10.0 + fid, 10.0 + fid, 50.0 + fid, 50.0 + fid], [60.0, 60.0, 100.0, 100.0]]],
                dtype=torch.float32,
            ),
            "pred_track_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pred_scores": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }
        for fid in range(3)
    ]
