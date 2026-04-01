# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.0 - 2026-04-01

- Added `cuvis_ai_trackeval` plugin package with TrackEval-based metric nodes.
- Added `trackeval-cuvis` CLI for COCO tracking report generation.
- Added `CuvisCOCODataset` adapter for file-based and in-memory evaluation flows.
- Added optional `pred_frame_id` input to TrackEval metric nodes and frame-ID aware
  GT/pred timeline assembly for `CuvisCOCODataset.from_frames`.
- Modernized packaging to `pyproject.toml` and removed legacy setup files.
- Removed legacy `sys.path` insertion hacks from upstream helper scripts/tests.
