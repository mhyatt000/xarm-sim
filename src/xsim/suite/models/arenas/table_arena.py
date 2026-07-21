"""Table workspace arena: geometry, calibrated rig cameras, splat alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np

from xsim.suite.models.arenas.arena import Arena
from xsim.suite.models.cam_space import BallLookatSampler, ShellLookatSampler
from xsim.suite.models.cameras import CameraSpec, SplatAsset, view_from_c2w_cv

_PROJECT_ROOT = Path(__file__).resolve().parents[5]

# Logitech extrinsics from /data/store/opencv_calibrated (robot-base frame),
# solved with approximate intrinsics fx=fy=515, cx=320, cy=240 -> vFOV below;
# sim cameras must render with that FOV for the extrinsics to be consistent.
LOGITECH_FOV_DEG = math.degrees(2.0 * math.atan(240.0 / 515.0))
LOW_C2W_CV = (
    (-0.6532074213027954, 0.09281682223081589, -0.7514687180519104, 1.0390468835830688),
    (0.7571737170219421, 0.07631510496139526, -0.6487403512001038, 0.48672395944595337),
    (-0.0028656369540840387, -0.9927542209625244, -0.12012804299592972, 0.23500925302505493),
    (0.0, 0.0, 0.0, 1.0),
)
SIDE_C2W_CV = (
    (-0.9901586174964905, 0.07804439961910248, -0.11616794764995575, 0.4850386679172516),
    (0.13690687716007233, 0.7123172879219055, -0.6883754134178162, 0.6458088159561157),
    (0.02902454137802124, -0.697504997253418, -0.7159919738769531, 0.9215802550315857),
    (0.0, 0.0, 0.0, 1.0),
)
_TABLE_CAMERAS = (
    view_from_c2w_cv("low", LOW_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
    view_from_c2w_cv("side", SIDE_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
)

# World-frame splat: scripts/clean_splat.py bakes the solved alignment
# (align_ransac.py seed -> scripts/icp_splat.py annealed trimmed ICP) into the
# PLY and crops the table volume, so the asset pose is identity. The pre-bake
# solve lives in clean_splat.py/git history if the raw scan needs re-cropping.
DEFAULT_SPLAT = SplatAsset(
    uri=_PROJECT_ROOT / "assets" / "lab_aligned.ply",
    pos=(0.0, 0.0, 0.0),
    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    scale=1.0,
)


@dataclass
class TableArena(Arena):
    """Flat table: infinite collision plane plus an optional visual slab."""

    # Robot base sits on a 1 cm mounting plate, so table top is 1 cm below the base origin.
    top_z: float = -0.01
    # y-centered on the robot; x placed so the table's rear edge (xmin =
    # -0.0635) is flush with the PlateMount's rear edge (5 in plate, -2.5 in)
    center_xy: tuple[float, float] = (0.3937, 0.0)
    # Real cart 3ft x 2ft top.
    size_xy: tuple[float, float] = (0.9144, 0.6096)
    color: tuple[float, float, float] = (0.13, 0.14, 0.17)
    slab: bool = True
    slab_height: float = 0.72
    # render no table at all (collision plane stays): batch compositing fills
    # the table pixels with the scanned splat table instead. Backends without
    # splat compositing (raster) show no table.
    transparent: bool = True
    cameras: tuple[CameraSpec, ...] = _TABLE_CAMERAS
    splat: SplatAsset | None = DEFAULT_SPLAT
    splat_bg: bool = True
    # attached/wrist cams drift mid-episode; static cams tolerate any cadence
    splat_resplat_every: int = 3
    # exocentric cams sampled per env/episode; False pins the calibrated rig
    # (real-robot eval parity). Sampled cams keep the rig names/FOV so obs keys
    # and downstream pipelines don't change.
    randomize_cameras: bool = True

    def __post_init__(self) -> None:
        if self.randomize_cameras:
            for spec in _TABLE_CAMERAS:
                if any(c.name == spec.name for c in self.cameras):
                    self.set_camera(self.cam_sampler(spec.name))

    def _lookat_box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(lo, hi) lookat bounds covering the cube spawn region."""
        cx = self.center_xy[0]
        return (cx - 0.15, -0.15, self.top_z + 0.01), (cx + 0.15, 0.15, self.top_z + 0.20)

    def cam_sampler(self, name: str = "rand", fov_deg: float = LOGITECH_FOV_DEG) -> ShellLookatSampler:
        """Pose sampler bounded by the calibrated rig: sphere through the
        farthest rig camera plus 10% headroom, floored at the table top, capped
        at the highest rig camera; the -1 ft x floor keeps cameras out of the
        wall behind the robot. Lookats cover the cube spawn region."""
        cam_pos = [np.asarray(c.pos) for c in _TABLE_CAMERAS]
        lookat_lo, lookat_hi = self._lookat_box()
        return ShellLookatSampler(
            name=name,
            fov_deg=fov_deg,
            radius=1.1 * max(float(np.linalg.norm(p)) for p in cam_pos),
            x_range=(-0.3048, max(float(p[0]) for p in cam_pos)),
            z_range=(self.top_z, max(float(p[2]) for p in cam_pos)),
            lookat_lo=lookat_lo,
            lookat_hi=lookat_hi,
        )

    def add_to(self, scene: gs.Scene) -> None:
        surface = gs.surfaces.Plastic(color=self.color, roughness=0.8)
        scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, self.top_z),
                visualization=not self.slab and not self.transparent,
                collision=True,
            ),
            surface=surface,
        )
        if self.slab and not self.transparent:
            # Stands in for the real cart body; occludes the under-table region.
            scene.add_entity(
                gs.morphs.Box(
                    size=(self.size_xy[0], self.size_xy[1], self.slab_height),
                    pos=(
                        self.center_xy[0],
                        self.center_xy[1],
                        self.top_z - self.slab_height / 2,
                    ),
                    fixed=True,
                    visualization=True,
                    collision=False,
                ),
                surface=surface,
            )


@dataclass
class TableEZ(TableArena):
    """TableArena with tame camera randomization: each rig camera is sampled
    in a small ball around its calibrated position instead of the workspace
    shell. Lookats are unchanged."""

    cam_radius: float = 0.10

    def cam_sampler(self, name: str = "low", fov_deg: float = LOGITECH_FOV_DEG) -> BallLookatSampler:
        centers = {c.name: tuple(c.pos) for c in _TABLE_CAMERAS}
        if name not in centers:
            raise ValueError(f"no calibrated rig camera {name!r}; have {sorted(centers)}")
        lookat_lo, lookat_hi = self._lookat_box()
        return BallLookatSampler(
            name=name,
            fov_deg=fov_deg,
            center=centers[name],
            radius=self.cam_radius,
            lookat_lo=lookat_lo,
            lookat_hi=lookat_hi,
        )
