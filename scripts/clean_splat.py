"""Clean the lab splat for sim rendering.

Default mode crops the table volume out of the raw lab splat. The real cart's dark
reflective top scans as sparse see-through mush, and the scan also bakes in the robot
itself. Both live inside the table volume in world coordinates, and both fight the sim's
own mesh robot / tabletop slab when rendering.

``--keep-table`` keeps the scanned cart/table and under-table region, while removing
above-table clutter/baked robot points from the same footprint. Use that with
``--env.table-transparent`` when the desired view is the real splat table without the sim
mesh slab.

    uv run python scripts/clean_splat.py
    uv run python scripts/clean_splat.py --keep-table
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_SPLAT = PROJECT_ROOT / "assets" / "lab_clean.ply"
DEFAULT_CLEAN_W_TABLE_SPLAT = PROJECT_ROOT / "assets" / "lab_clean_w_table.ply"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# calibrated world-from-splat pose of the lab scan (was xsim.task_env, retired 2026-07)
DEFAULT_SPLAT_POS = (-0.2237, 0.7717, 0.1711)
DEFAULT_SPLAT_QUAT = (-0.501119, 0.487918, -0.50087, 0.509849)  # xyzw
DEFAULT_SPLAT_SCALE = 0.9966


def splat_world_transform(pos=DEFAULT_SPLAT_POS, quat=DEFAULT_SPLAT_QUAT, scale=DEFAULT_SPLAT_SCALE):
    """(4x4 world-from-splat transform incl. scale) for cropping/analysis tooling."""
    x, y, z, w = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = pos
    return T


@dataclass
class Cfg:
    src: Path = Path("/data/store/lab.ply")
    dst: Path = DEFAULT_CLEAN_SPLAT
    keep_table: bool = False
    # world-frame crop box: the table footprint with generous margin (gaussians are
    # volumes — blobs centered outside a tight box still splat across the tabletop),
    # from just below the top over the baked robot's full reach. Also swallows the
    # rig hardware standing on the table (clamps, low-camera pole).
    box_min: tuple[float, float, float] = (-0.30, -0.55, -0.75)
    box_max: tuple[float, float, float] = (1.15, 0.75, 1.30)
    # In keep-table mode, preserve the scanned cart/table and under-table region, but
    # remove baked robot/clamps/poles sitting above the tabletop inside the footprint.
    keep_table_remove_above_z: float = 0.04
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
    if c.keep_table:
        inside_xy = np.all((pw[:, :2] >= lo[:2]) & (pw[:, :2] <= hi[:2]), axis=1)
        inside = inside_xy & (pw[:, 2] > c.keep_table_remove_above_z) & (pw[:, 2] <= hi[2])
        inside_label = "above-table clutter"
        dst = DEFAULT_CLEAN_W_TABLE_SPLAT if c.dst == DEFAULT_CLEAN_SPLAT else c.dst
    else:
        inside = np.all((pw >= lo) & (pw <= hi), axis=1)
        inside_label = "in the table box"
        dst = c.dst

    # scale_0..2 (log) sit right after opacity, which is the property before them
    giant = np.exp(data[:, 55:58]).max(axis=1) > c.max_radius
    drop = inside | giant
    kept = data[~drop].copy()
    if c.flatten_sh:
        kept[:, 9:54] = 0.0  # f_rest_0..44
    print(f"{n} gaussians: {inside.sum()} {inside_label}, {giant.sum()} giant, keeping {len(kept)}")

    new_header = header.replace(f"element vertex {n}", f"element vertex {len(kept)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(new_header.encode())
        kept.astype(np.float32).tofile(f)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
