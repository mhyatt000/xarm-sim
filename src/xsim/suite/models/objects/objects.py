"""Free-object models."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
import numpy as np
import torch


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
        # skip_forward chains so forward kinematics runs once for the batch
        self.entity.set_pos(pos_t, envs_idx=envs_idx, skip_forward=True)
        self.entity.set_quat(quat_t, envs_idx=envs_idx, skip_forward=False)

    def get_pos(self) -> np.ndarray:
        """Positions (n_envs, 3)."""
        return np.asarray(self.entity.get_pos().detach().cpu())

    def get_quat(self) -> np.ndarray:
        """Orientations (n_envs, 4) as wxyz quaternions."""
        return np.asarray(self.entity.get_quat().detach().cpu())


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
