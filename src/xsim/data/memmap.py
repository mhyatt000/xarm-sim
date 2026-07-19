"""Append-only flat-binary sample store + random-access readers.

One <key>.bin per key with fixed-size rows, so sample i is one memmap slice
and the OS page cache is the only caching layer. Built for image-scale
collection (DAgger, demo dumps) where the aggregate lives on disk, not RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class MemmapStore:
    """Append-only on-disk dataset: one flat <key>.bin per key + manifest.json.

    Single writer, append per recorded step, flush() publishes the manifest
    for readers.
    """

    def __init__(self, root: Path):
        import shutil

        self.root = root
        shutil.rmtree(root, ignore_errors=True)  # fresh run owns its dir
        root.mkdir(parents=True, exist_ok=True)
        self._fh: dict = {}
        self.meta: dict[str, dict] = {}

    def append(self, key: str, arr: np.ndarray) -> None:
        meta = self.meta.setdefault(
            key, {"dtype": str(arr.dtype), "shape": list(arr.shape[1:]), "n": 0})
        if key not in self._fh:
            self._fh[key] = (self.root / f"{key}.bin").open("ab")
        arr.tofile(self._fh[key])
        meta["n"] += arr.shape[0]

    def flush(self) -> None:
        for fh in self._fh.values():
            fh.flush()
        (self.root / "manifest.json").write_text(json.dumps(self.meta))

    def reader(self, key: str) -> np.memmap:
        meta = self.meta[key]
        return np.memmap(self.root / f"{key}.bin", dtype=meta["dtype"],
                         mode="r", shape=(meta["n"], *meta["shape"]))

    def __len__(self) -> int:
        return self.meta["act"]["n"] if "act" in self.meta else 0


class MemmapDataset(torch.utils.data.Dataset):
    """Random-access snapshot of a MemmapStore. Holds only the root path and
    manifest, so it pickles cheaply into spawned DataLoader workers; each
    worker opens its own memmap handles on first use."""

    def __init__(self, root: Path, keys: tuple[str, ...]):
        self.root, self.keys = root, keys
        self.meta = json.loads((root / "manifest.json").read_text())
        self.n = self.meta[keys[0]]["n"]
        self._maps: dict[str, np.memmap] | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self._maps is None:
            self._maps = {
                k: np.memmap(self.root / f"{k}.bin", dtype=m["dtype"], mode="r",
                             shape=(m["n"], *m["shape"]))
                for k in self.keys if (m := self.meta[k])
            }
        return tuple(torch.from_numpy(np.array(self._maps[k][i])) for k in self.keys)


def read_key(root: Path, key: str) -> np.memmap:
    """Reader for one key of a flushed MemmapStore dir (works across processes:
    only the manifest and bin file are touched)."""
    meta = json.loads((root / "manifest.json").read_text())[key]
    return np.memmap(root / f"{key}.bin", dtype=meta["dtype"], mode="r",
                     shape=(meta["n"], *meta["shape"]))
