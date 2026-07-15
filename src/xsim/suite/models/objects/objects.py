"""Free-object models."""

from __future__ import annotations

import math
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

    def set_pose(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        pos = torch.tensor([[x, y, z]], device=gs.device, dtype=gs.tc_float)
        quat = torch.tensor(
            [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.entity.set_pos(pos, skip_forward=True)
        self.entity.set_quat(quat, skip_forward=False)

    def get_pos(self) -> np.ndarray:
        return np.asarray(self.entity.get_pos().detach().cpu()).reshape(-1)

    def get_quat(self) -> np.ndarray:
        """Orientation as wxyz quaternion."""
        return np.asarray(self.entity.get_quat().detach().cpu()).reshape(-1)


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
