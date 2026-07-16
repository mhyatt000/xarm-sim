"""Mounting fixtures: fixed rig geometry a robot bolts onto."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np
import trimesh

IN = 0.0254

_THETA = math.radians(45.0)


class Mount:
    """Fixed mounting fixture for one robot: adds its geometry to the scene and
    reports where the robot base bolts on."""

    def add_to(self, scene: gs.Scene) -> None:
        raise NotImplementedError

    def base_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Robot base (pos, wxyz quat) on this mount."""
        raise NotImplementedError


@dataclass
class VMount4040(Mount):
    """One side of the dual-arm 45 deg V rig (mirrored about y=0), built from
    4040 extrusion approximated as 2x2 in bars:

      - 2 rail halves on the floor along y; each side owns half of the shared
        2 ft rails, the halves joining flush at y=0
      - 2 sloped bars mitered 45 deg at both ends (10 in mounting face, short
        face shortened by one bar section per end), one under each plate end,
        bottom cuts resting flat on the rails, ridge cuts meeting flush at y=0
      - 5x8 in, 1 cm baseplate; the robot base sits at its top center

    Origin convention: rig xy center at the world origin with rails on the
    floor, so each mounting face is 45 deg from the floor and the two sides'
    arms 90 deg apart. The miter/ridge math is specific to the 45 deg slope.
    """

    side: int = -1  # -1: arm leaning toward -y, +1: toward +y
    bar_section: float = 2 * IN
    bar_len: float = 10 * IN  # mounting-face (longest) side of the mitered bar
    rail_len: float = 12 * IN  # this side's half of the 2 ft shared rails
    bar_spacing: float = 3 * IN  # centerline to centerline
    plate_size: tuple[float, float, float] = (5 * IN, 8 * IN, 0.01)
    color: tuple[float, float, float] = (0.62, 0.63, 0.65)

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError(f"side must be -1 or +1, got {self.side}")

    def _frame(self) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        """(face normal, plate-center point on the face, wxyz quat) of this side."""
        sin, cos = math.sin(_THETA), math.cos(_THETA)
        # Ridge height puts the bars' horizontal bottom-cut faces on the rail tops.
        ridge = np.array([0.0, 0.0, self.bar_section + self.bar_len * sin])
        n = np.array([0.0, self.side * sin, cos])
        d = np.array([0.0, self.side * cos, -sin])  # down-slope
        half = -self.side * _THETA / 2
        quat = (math.cos(half), math.sin(half), 0.0, 0.0)
        return n, ridge + d * self.bar_len / 2, quat

    def base_pose(self):
        n, mid, quat = self._frame()
        return tuple(mid + n * self.plate_size[2]), quat

    def add_to(self, scene: gs.Scene) -> None:
        surface = gs.surfaces.Plastic(color=self.color, roughness=0.6)
        n, mid, quat = self._frame()
        for x in (-self.bar_spacing / 2, self.bar_spacing / 2):
            scene.add_entity(
                gs.morphs.Box(
                    size=(self.bar_section, self.rail_len, self.bar_section),
                    pos=(x, self.side * self.rail_len / 2, self.bar_section / 2),
                    fixed=True,
                ),
                surface=surface,
            )
            scene.add_entity(
                gs.morphs.Mesh(
                    file=self._bar_mesh_file(),
                    pos=tuple(np.array([x, 0.0, 0.0]) + mid - n * self.bar_section / 2),
                    quat=quat,
                    fixed=True,
                ),
                surface=surface,
            )
        scene.add_entity(
            gs.morphs.Box(
                size=self.plate_size,
                pos=tuple(mid + n * self.plate_size[2] / 2),
                quat=quat,
                fixed=True,
            ),
            surface=surface,
        )

    def _bar_mesh_file(self) -> str:
        """Mitered-bar prism exported once per dimension set (gs.morphs.Mesh
        loads from file only): local y along the length, +z the mounting face."""
        half = self.bar_section / 2
        path = Path(tempfile.gettempdir()) / (
            f"xsim_vmount_bar_{self.bar_len:.4f}_{self.bar_section:.4f}.stl"
        )
        if not path.exists():
            profile = [
                (-self.bar_len / 2, half),
                (self.bar_len / 2, half),
                (self.bar_len / 2 - self.bar_section, -half),
                (-self.bar_len / 2 + self.bar_section, -half),
            ]
            verts = [(x, y, z) for x in (-half, half) for y, z in profile]
            trimesh.Trimesh(vertices=verts).convex_hull.export(path)
        return str(path)
