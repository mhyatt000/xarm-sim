"""Crop the table volume out of the lab splat.

The real cart's dark reflective top scans as sparse see-through mush, and the scan also
bakes in the robot itself. Both live inside the table volume in world coordinates, and
both fight the sim's own mesh robot / tabletop slab when rendering. This script removes
every gaussian inside that box (using the solved splat↔world alignment from
``xsim.lift_task``) and writes the cleaned splat next to the original name in ``assets/``.

    uv run python scripts/clean_splat.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xsim.lift_task import splat_world_transform  # noqa: E402


@dataclass
class Cfg:
    src: Path = Path("/data/store/lab.ply")
    dst: Path = PROJECT_ROOT / "assets" / "lab_clean.ply"
    # world-frame crop box: the table footprint with generous margin (gaussians are
    # volumes — blobs centered outside a tight box still splat across the tabletop),
    # from just below the top over the baked robot's full reach. Also swallows the
    # rig hardware standing on the table (clamps, low-camera pole).
    box_min: tuple[float, float, float] = (-0.30, -0.55, -0.75)
    box_max: tuple[float, float, float] = (1.15, 0.75, 1.30)
    # also drop giant haze/floater gaussians: semi-transparent blobs with metre-scale
    # extents that smear across the whole scene (real surfaces use small gaussians)
    max_radius: float = 0.10
    # zero the view-dependent SH terms: the sim cameras sit inside the scene at angles
    # far outside the scan trajectory, where degree-3 SH extrapolates into white/black
    # streak garbage; DC-only colours are stable from every angle
    flatten_sh: bool = True


def main(c: Cfg) -> None:
    raw = c.src.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    n = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    props = sum(1 for ln in header.splitlines() if ln.startswith("property "))
    data = np.frombuffer(raw[end:], dtype=np.float32).reshape(n, props)

    T = splat_world_transform()
    pw = data[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3]

    lo, hi = np.asarray(c.box_min), np.asarray(c.box_max)
    inside = np.all((pw >= lo) & (pw <= hi), axis=1)
    # scale_0..2 (log) sit right after opacity, which is the property before them
    giant = np.exp(data[:, 55:58]).max(axis=1) > c.max_radius
    drop = inside | giant
    kept = data[~drop].copy()
    if c.flatten_sh:
        kept[:, 9:54] = 0.0  # f_rest_0..44
    print(f"{n} gaussians: {inside.sum()} in the table box, {giant.sum()} giant, keeping {len(kept)}")

    new_header = header.replace(f"element vertex {n}", f"element vertex {len(kept)}")
    c.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(c.dst, "wb") as f:
        f.write(new_header.encode())
        kept.astype(np.float32).tofile(f)
    print(f"wrote {c.dst} ({c.dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
