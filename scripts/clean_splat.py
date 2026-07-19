"""Bake the solved alignment into the lab splat and crop the table volume.

Transforms every gaussian by the arena's committed splat->world pose (so the
output PLY is world-frame: use it with an identity SplatAsset), then empties
an axis-aligned box over the table: z from the tabletop to 3 ft above, xy the
table footprint plus margin. That removes the scanned (ghost) robot, cube and
tabletop fuzz that fight the sim's own meshes, without touching the table
structure below z = -1 cm.

Gaussians whose center is outside the box but whose volume pokes in are not
dropped or shrunk uniformly (that thins the table surface into holes) — each
is replaced by the largest spheroid contained in the original gaussian that
stays outside the box: squashed only along the offending box axis, with the
center pushed away from the face when that preserves more volume. Solved per
gaussian in the whitened frame, where the optimum depends only on the
clearance/extent ratio.

    uv run python scripts/clean_splat.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from xsim.suite.models.arenas.table_arena import TableArena
from xsim.suite.models.cameras import SplatAsset
from xsim.suite.renderers.splat_bg import quat_mul_wxyz, rot_from_quat_xyzw, rots_from_quat_wxyz

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# splat->world solve for the raw lab scan: align_ransac.py seed refined by
# scripts/icp_splat.py (annealed trimmed ICP, 2026-07-17). The arena's
# DEFAULT_SPLAT is identity because this script bakes the solve into the PLY.
RAW_SCAN_SOLVE = SplatAsset(
    uri=PROJECT_ROOT / "assets" / "lab.ply",  
    pos=(-0.2711, 0.7629, 0.1824),
    quat_xyzw=(-0.542208, 0.464004, -0.464180, 0.524641),
    scale=1.0,
)

# standard 3DGS PLY layout: x y z nx ny nz f_dc(3) f_rest(45) opacity scale(3) rot(4)
_XYZ = slice(0, 3)
_F_REST = slice(9, 54)
_SCALE = slice(55, 58)
_ROT = slice(58, 62)  # wxyz


@dataclass
class Cfg:
    src: Path = Path(RAW_SCAN_SOLVE.uri)  # the full lab scan
    dst: Path = PROJECT_ROOT / "assets" / "lab_aligned.ply"
    # crop box: tabletop to 3 ft above it, xy = table footprint + margin
    z_range: tuple[float, float] = (-0.01, 0.9144)
    xy_margin: float = 0.10  # fraction of the table half-extents
    sigma: float = 2.0  # support radius (in sigmas) used for the reach test
    # giant haze/floater blobs: metre-scale semi-transparent smears
    max_radius: float = 0.10
    # DC-only colour: sim cameras sit far off the scan trajectory where
    # higher-order SH extrapolates into streak garbage (and baking the
    # alignment rotation into SH>0 isn't implemented)
    flatten_sh: bool = True


def main(c: Cfg) -> None:
    raw = c.src.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    n = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    props = sum(1 for ln in header.splitlines() if ln.startswith("property "))
    data = np.frombuffer(raw[end:], dtype=np.float32).reshape(n, props).copy()

    # bake the solved splat->world alignment into positions and rotations
    a = RAW_SCAN_SOLVE
    R = rot_from_quat_xyzw(a.quat_xyzw)
    data[:, _XYZ] = (
        a.scale * data[:, _XYZ].astype(np.float64) @ R.T + np.asarray(a.pos)
    ).astype(np.float32)
    qx, qy, qz, qw = a.quat_xyzw
    q = data[:, _ROT].astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    data[:, _ROT] = quat_mul_wxyz(np.array([qw, qx, qy, qz]), q).astype(np.float32)
    data[:, _SCALE] += np.log(a.scale)

    arena = TableArena()
    hx, hy = (1 + c.xy_margin) * np.asarray(arena.size_xy) / 2
    lo = np.array([arena.center_xy[0] - hx, arena.center_xy[1] - hy, c.z_range[0]])
    hi = np.array([arena.center_xy[0] + hx, arena.center_xy[1] + hy, c.z_range[1]])

    pw = data[:, _XYZ].astype(np.float64)
    inside = np.all((pw >= lo) & (pw <= hi), axis=1)

    # boundary gaussians: center outside, but sigma-support reaches the box.
    # world half-extent along axis i is sigma * ||B[i, :]|| for B = R_g diag(s);
    # the gaussian is separated from the box iff extent < clearance on an axis
    # where the center is out. For the rest, work in the whitened frame (old
    # ellipsoid -> unit ball, box face -> plane at distance d = clear/extent):
    # replace the ball with the max-volume spheroid inside both the ball and
    # the half-space — semi-axis a along the whitened face normal w, r perp,
    # center shifted t away from the face. a = d + t makes the face constraint
    # tight, and containment in the unit ball gives r(t) in closed form; the
    # best t is a 1-D search shared across gaussians.
    clear = np.maximum(np.maximum(lo - pw, pw - hi), 0.0)
    B = rots_from_quat_wxyz(data[:, _ROT].astype(np.float64)) * np.exp(
        data[:, _SCALE].astype(np.float64)
    )[:, None, :]
    extent = c.sigma * np.linalg.norm(B, axis=-1)
    ratio = np.where(clear > 0, clear / extent, -np.inf).max(axis=1)
    shrink = ~inside & (ratio < 1.0)

    idx = np.flatnonzero(shrink)
    k = np.argmax(np.where(clear[idx] > 0, clear[idx] / extent[idx], -np.inf), axis=1)
    d = ratio[idx]
    sign = np.where(pw[idx, k] < lo[k], 1.0, -1.0)  # toward-box direction is sign * e_k
    Btn = B[idx, k, :] * sign[:, None]  # B^T (sign e_k): rows of B
    w = Btn / np.linalg.norm(Btn, axis=-1, keepdims=True)

    grid = np.linspace(0.0, 1.0, 65)
    t = grid[None, :] * ((1.0 - d) / 2.0)[:, None]  # a + t <= 1 keeps the ball bound
    a = d[:, None] + t
    c1 = 1.0 - t**2 - a**2
    beta = (c1 + np.sqrt(np.maximum(c1**2 - 4.0 * a**2 * t**2, 0.0))) / 2.0
    r2 = a**2 + beta
    best = np.argmax(a * r2, axis=1)
    ar = np.arange(len(idx))
    t, a, r = t[ar, best], a[ar, best], np.sqrt(r2[ar, best])

    Bw = np.einsum("mij,mj->mi", B[idx], w)
    data[idx, _XYZ] -= ((c.sigma * t)[:, None] * Bw).astype(np.float32)
    # B' = B (r I + (a - r) w w^T): squash along w, full extent elsewhere
    Bp = r[:, None, None] * B[idx] + (a - r)[:, None, None] * Bw[:, :, None] * w[:, None, :]
    evals, evecs = np.linalg.eigh(Bp @ Bp.transpose(0, 2, 1))
    data[idx, _SCALE] = np.log(np.sqrt(np.maximum(evals, 1e-18))).astype(np.float32)
    evecs[:, :, 0] *= np.sign(np.linalg.det(evecs))[:, None]  # rotations, not reflections
    from scipy.spatial.transform import Rotation

    data[idx, _ROT] = Rotation.from_matrix(evecs).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    kept_vol = float(np.mean(a * r**2)) if len(idx) else 1.0

    giant = np.exp(data[:, _SCALE]).max(axis=1) > c.max_radius
    kept = data[~(inside | giant)]
    if c.flatten_sh:
        kept[:, _F_REST] = 0.0
    print(
        f"{n} gaussians: {inside.sum()} in the table box, {shrink.sum()} squashed at "
        f"the boundary (mean {100 * kept_vol:.0f}% volume kept), {giant.sum()} giant, "
        f"keeping {len(kept)}"
    )

    new_header = header.replace(f"element vertex {n}", f"element vertex {len(kept)}")
    c.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(c.dst, "wb") as f:
        f.write(new_header.encode())
        kept.astype(np.float32).tofile(f)
    print(f"wrote {c.dst} ({c.dst.stat().st_size / 1e6:.0f} MB)")
    print("output is world-frame: use SplatAsset(pos=(0,0,0), quat_xyzw=(0,0,0,1), scale=1.0)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
