"""Table workspace arena: geometry, calibrated rig cameras, splat alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import genesis as gs

from xsim.suite.models.arenas.arena import Arena
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

# Splat -> world alignment: align_ransac.py seed refined by scripts/icp_splat.py
# (ICP of splat centers against the table slab + PlateMount mesh); scans are
# metric so scale is pinned at 1.0.
_CLEAN_SPLAT = _PROJECT_ROOT / "assets" / "lab_clean.ply"
DEFAULT_SPLAT = SplatAsset(
    uri=_CLEAN_SPLAT if _CLEAN_SPLAT.exists() else Path("/data/store/lab.ply"),
    pos=(-0.2513, 0.767, 0.1847),
    quat_xyzw=(-0.526301, 0.471493, -0.470877, 0.528183),
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
    cameras: tuple[CameraSpec, ...] = _TABLE_CAMERAS
    splat: SplatAsset | None = DEFAULT_SPLAT

    def add_to(self, scene: gs.Scene) -> None:
        surface = gs.surfaces.Plastic(color=self.color, roughness=0.8)
        scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, self.top_z),
                visualization=not self.slab,
                collision=True,
            ),
            surface=surface,
        )
        if self.slab:
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
