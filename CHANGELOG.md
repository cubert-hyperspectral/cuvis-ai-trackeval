# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.2 - 2026-06-10

- Require `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2` (inherits the upstream security floors transitively).
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the PyPI/TestPyPI release workflow; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests.
- Stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.1 - 2026-04-29

- Annotated `IdentityMetricNode`, `CLEARMetricNode`, and `HOTAMetricNode` with `_category = NodeCategory.METRIC` and `_tags = {BBOX, TRACKING, EVALUATION, NUMPY}` so the metric nodes surface under the correct category and filters in the cuvis-ai palette.
- Bumped minimum `cuvis-ai-schemas` from `>=0.3.0` to `>=0.4.0` (`NodeCategory` / `NodeTag` enums were added in v0.4.0).
- Stripped `hash` fields from `torch` / `torchvision` wheel entries in `uv.lock`.

## 0.1.0 - 2026-04-01

- Added `cuvis_ai_trackeval` plugin package with TrackEval-based metric nodes.
- Added `trackeval-cuvis` CLI for COCO tracking report generation.
- Added `CuvisCOCODataset` adapter for file-based and in-memory evaluation flows.
- Added optional `pred_frame_id` input to TrackEval metric nodes and frame-ID aware
  GT/pred timeline assembly for `CuvisCOCODataset.from_frames`.
- Modernized packaging to `pyproject.toml` and removed legacy setup files.
- Removed legacy `sys.path` insertion hacks from upstream helper scripts/tests.
