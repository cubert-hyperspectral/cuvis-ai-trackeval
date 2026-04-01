"""cuvis_ai_trackeval plugin package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as _np

try:
    __version__ = version("cuvis-ai-trackeval")
except PackageNotFoundError:
    __version__ = "dev"

# Compatibility aliases for upstream TrackEval code paths using deprecated NumPy symbols.
if not hasattr(_np, "float"):
    _np.float = _np.float64  # type: ignore[attr-defined]
if not hasattr(_np, "int"):
    _np.int = _np.int_  # type: ignore[attr-defined]
if not hasattr(_np, "bool"):
    _np.bool = _np.bool_  # type: ignore[attr-defined]
if not hasattr(_np, "complex"):
    _np.complex = _np.complex128  # type: ignore[attr-defined]


def register_all_nodes() -> int:
    """Register all cuvis_ai_trackeval nodes in the cuvis.ai NodeRegistry."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    return NodeRegistry().auto_register_package("cuvis_ai_trackeval.node")


__all__ = ["__version__", "register_all_nodes"]
