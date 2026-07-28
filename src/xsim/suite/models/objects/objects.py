"""Free-object models."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
import numpy as np
import torch
import trimesh


class GenesisObject:
    """Base free-object model: owns its entity once added to a scene."""

    name: str
    entity = None  # bound by add_to

    def add_to(self, scene: gs.Scene):
        raise NotImplementedError

    @property
    def top_offset(self) -> float:
        """Height of the object's top above its origin."""
        raise NotImplementedError

    @property
    def bottom_offset(self) -> float:
        """Height of the object's origin above its bottom (drop height that
        rests the object on a surface). Symmetric objects reuse top_offset."""
        return self.top_offset

    @property
    def xy_radius(self) -> float:
        """Circumradius of the footprint around the origin, for spacing
        objects at reset so they cannot spawn overlapping at any yaw."""
        raise NotImplementedError

    def set_pose(self, x, y, z, yaw=0.0, envs_idx=None) -> None:
        """Place the object in the selected envs (all when ``envs_idx=None``).

        ``x/y/z/yaw`` broadcast against each other; pass (K,) arrays with
        K = n_envs (or len(envs_idx)) for per-env poses.
        """
        x, y, z, yaw = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (x, y, z, yaw))
        )
        pos = np.stack([x, y, z], axis=1)
        half = yaw / 2.0
        quat = np.stack(
            [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
            axis=1,
        )
        pos_t = torch.tensor(pos, device=gs.device, dtype=gs.tc_float)
        quat_t = torch.tensor(quat, device=gs.device, dtype=gs.tc_float)
        # no skip_forward chaining: on mesh entities the skipped set_pos is
        # silently dropped by the forward pass of the following set_quat
        self.entity.set_pos(pos_t, envs_idx=envs_idx)
        self.entity.set_quat(quat_t, envs_idx=envs_idx)

    def get_pos(self) -> np.ndarray:
        """Positions (n_envs, 3)."""
        return np.asarray(self.entity.get_pos().detach().cpu())

    def get_quat(self) -> np.ndarray:
        """Orientations (n_envs, 4) as wxyz quaternions."""
        return np.asarray(self.entity.get_quat().detach().cpu())

    def get_vel(self) -> np.ndarray:
        """Linear velocities (n_envs, 3)."""
        return np.asarray(self.entity.get_vel().detach().cpu())

    def get_ang(self) -> np.ndarray:
        """Angular velocities (n_envs, 3)."""
        return np.asarray(self.entity.get_ang().detach().cpu())


@dataclass
class BoxObject(GenesisObject):
    """Rigid box object."""

    name: str
    # 1.25-inch cube.
    size: tuple[float, float, float] = (0.03175, 0.03175, 0.03175)
    color: tuple[float, float, float] = (0.48, 0.05, 0.04)
    friction: float = 2.0
    fixed: bool = False

    def add_to(self, scene: gs.Scene):
        self.entity = scene.add_entity(
            gs.morphs.Box(size=self.size, fixed=self.fixed),
            material=gs.materials.Rigid(friction=self.friction),
            surface=gs.surfaces.Plastic(color=self.color, roughness=0.6),
        )
        return self.entity

    @property
    def top_offset(self) -> float:
        return self.size[2] / 2.0

    @property
    def xy_radius(self) -> float:
        return float(np.hypot(self.size[0], self.size[1]) / 2.0)


@dataclass
class MeshObject(GenesisObject):
    """Rigid object loaded from a mesh file (STL/OBJ/GLB).

    ``max_extent`` rescales the mesh so its largest bounding-box side matches
    (on top of ``scale``); ``decompose`` runs CoACD convex decomposition so
    concave interiors (bins, cups) collide correctly — without it Genesis
    collides against the convex hull. ``color=None`` keeps the file's own
    materials/textures.
    """

    name: str
    file: str
    scale: float = 1.0
    max_extent: float | None = None
    color: tuple[float, float, float] | None = None
    friction: float = 1.0
    fixed: bool = False
    decompose: bool = False

    def __post_init__(self):
        mesh = trimesh.load(self.file, force="mesh")
        if self.max_extent is not None:
            self.scale *= self.max_extent / float(mesh.extents.max())
        self._bounds = self.scale * np.asarray(mesh.bounds)
        self._xy_radius = self.scale * float(
            np.linalg.norm(mesh.vertices[:, :2], axis=-1).max()
        )

    def add_to(self, scene: gs.Scene):
        surface = (
            gs.surfaces.Plastic(color=self.color, roughness=0.6)
            if self.color is not None
            else None
        )
        self.entity = scene.add_entity(
            gs.morphs.Mesh(
                file=self.file,
                scale=self.scale,
                fixed=self.fixed,
                # always-decompose vs plain convex hull
                decompose_object_error_threshold=0.0 if self.decompose else float("inf"),
            ),
            material=gs.materials.Rigid(friction=self.friction),
            surface=surface,
        )
        return self.entity

    @property
    def top_offset(self) -> float:
        return float(self._bounds[1, 2])

    @property
    def bottom_offset(self) -> float:
        return float(-self._bounds[0, 2])

    @property
    def xy_radius(self) -> float:
        return self._xy_radius
