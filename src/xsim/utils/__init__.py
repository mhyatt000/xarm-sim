"""Cross-cutting helpers that belong to no single layer (suite stays env-only)."""

from __future__ import annotations

from xsim.utils.video import BORDER, VideoSink, tile_grid

__all__ = ["BORDER", "VideoSink", "tile_grid"]
