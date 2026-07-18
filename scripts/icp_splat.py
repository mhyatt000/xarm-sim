"""Throwaway: refine the arena splat alignment with ICP.

Seeds from the committed RANSAC solve (table_arena.DEFAULT_SPLAT), crops the
world-frame splat centers to a band around the expected tabletop, and runs
point-to-mesh ICP against a trimesh of the table slab + PlateMount built from
the live dataclass fields. Prints the corrected SplatAsset pos/quat and writes
before/after spec-pose renders for eyeballing.

    uv run python scripts/icp_splat.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import trimesh
import tyro

from xsim.suite.models.arenas.table_arena import TableArena
from xsim.suite.models.mounts import PlateMount
from xsim.suite.renderers.splat_bg import SplatBackground, viewmats_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # crop band around the expected table/plate surfaces, sim frame
    z_band: tuple[float, float] = (-0.07, 0.05)  # only the floor cutoff is used now
    xy_pad: float = 0.1  # metres beyond the slab footprint
    trim: float = 0.06  # round-1 outlier cutoff; decays each round
    trim_decay: float = 0.75  # trim multiplier per round: anneal toward the surface
    rounds: int = 10  # trim -> fit cycles; 0.06 * 0.75^9 ~ 4.5 mm final trim
    n_icp: int = 20000  # subsampled points fed to ICP
    max_iterations: int = 60
    seed: int = 0
    res: tuple[int, int] = (320, 240)
    out: Path = PROJECT_ROOT / "outputs" / "icp_splat"


def target_mesh(arena: TableArena) -> trimesh.Trimesh:
    """Table slab + baseplate + the arm's two lowest links at their sim pose,
    from the same fields/files the sim builds from."""
    tf = trimesh.transformations.translation_matrix
    slab = trimesh.creation.box(
        extents=(*arena.size_xy, arena.slab_height),
        transform=tf((*arena.center_xy, arena.top_z - arena.slab_height / 2)),
    )
    plate = PlateMount()
    pl = trimesh.creation.box(
        extents=plate.plate_size, transform=tf((0.0, 0.0, -plate.plate_size[2] / 2))
    )
    # base at the origin; link1 at joint1's URDF origin, and the default ready
    # pose has joint1 = 0 so no rotation
    base = trimesh.load(PROJECT_ROOT / "assets" / "link_base.stl")
    link1 = trimesh.load(PROJECT_ROOT / "assets" / "link1.stl")
    link1.apply_transform(tf((0.0, 0.0, 0.267)))
    return trimesh.util.concatenate([slab, pl, base, link1])


def main(cfg: Config) -> None:
    import cv2

    arena = TableArena()
    bg = SplatBackground(arena.splat, device="cuda")  # unpruned
    pts = bg.splat["means"].cpu().numpy().astype(np.float64)
    mesh = target_mesh(arena)

    pad = np.array([cfg.xy_pad, cfg.xy_pad, cfg.xy_pad])
    bounds = mesh.bounds + np.stack([-pad, pad])
    bounds[0, 2] = cfg.z_band[0]  # keep the floor out
    sel = np.all((pts >= bounds[0]) & (pts <= bounds[1]), axis=1)
    print(f"{sel.sum():,} / {len(pts):,} splat centers in the target bbox")

    rng = np.random.default_rng(cfg.seed)
    sub = pts[sel][rng.permutation(sel.sum())[: 2 * cfg.n_icp]]
    T = np.eye(4)
    for r in range(cfg.rounds):
        trim = cfg.trim * cfg.trim_decay**r
        cur = trimesh.transform_points(sub, T)
        d = trimesh.proximity.closest_point(mesh, cur)[1]
        keep = cur[d <= trim][: cfg.n_icp]  # clutter/ghost-arm outliers out
        Tr, _, cost = trimesh.registration.icp(
            keep, mesh, max_iterations=cfg.max_iterations, reflection=False, scale=False
        )
        d1 = trimesh.proximity.closest_point(mesh, trimesh.transform_points(keep, Tr))[1]
        print(
            f"round {r + 1}: {len(keep):,} pts within {trim * 1e3:.1f} mm, "
            f"mean |dist|: {d[d <= trim].mean() * 1e3:.1f} -> {d1.mean() * 1e3:.1f} mm, "
            f"cost {cost:.6f}"
        )
        T = Tr @ T
    print("ICP correction (applied in sim frame):")
    print(f"  translation: {T[:3, 3].round(4)}")
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"  rotation: {ang:.3f} deg")

    # compose: p' = T @ (s R p + t) -> quat(T R), T@t + t_delta, same scale
    a = arena.splat
    from xsim.suite.renderers.splat_bg import rot_from_quat_xyzw

    R_new = T[:3, :3] @ rot_from_quat_xyzw(a.quat_xyzw)
    t_new = T[:3, :3] @ np.asarray(a.pos) + T[:3, 3]
    qw, qx, qy, qz = trimesh.transformations.quaternion_from_matrix(
        np.block([[R_new, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]])
    )
    new_asset = replace(a, pos=tuple(float(v) for v in t_new.round(4)), quat_xyzw=(qx, qy, qz, qw))
    print("\nSplatAsset before -> after:")
    print(f"  pos:       {tuple(a.pos)}")
    print(f"          -> {new_asset.pos}")
    print(f"  quat_xyzw: {tuple(a.quat_xyzw)}")
    print(f"          -> ({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f})")
    print(f"  scale:     {a.scale}  (unchanged)")

    cfg.out.mkdir(parents=True, exist_ok=True)
    bg_new = SplatBackground(new_asset, device="cuda")
    for spec in [s for s in arena.cameras if s.attach_link is None]:
        fy = (cfg.res[1] / 2.0) / np.tan(np.radians(spec.fov_deg) / 2.0)
        K = np.array(
            [[fy, 0.0, cfg.res[0] / 2.0], [0.0, fy, cfg.res[1] / 2.0], [0.0, 0.0, 1.0]]
        )
        vm = viewmats_cv(spec.pos, spec.lookat, spec.up)
        for tag, r in (("ransac", bg), ("icp", bg_new)):
            frame = r.render(vm, K, cfg.res)[0]
            cv2.imwrite(str(cfg.out / f"{spec.name}_{tag}.png"), frame[..., ::-1])
    print(f"wrote before/after frames -> {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
