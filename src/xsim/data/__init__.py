"""On-disk dataset primitives shared by collection/training scripts."""

from __future__ import annotations

from xsim.data.memmap import MemmapDataset, MemmapStore, read_key

__all__ = ["MemmapDataset", "MemmapStore", "read_key"]
