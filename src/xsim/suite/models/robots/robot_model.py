"""Base robot description and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import genesis as gs

from xsim.suite.models.cameras import CameraSpec

ROBOT_MODEL_REGISTRY: dict[str, type[RobotModel]] = {}


@dataclass
class RobotModel:
    """Morph source + dof layout + gains for one robot. Pure description:
    the entity is bound by Task.add_to and consumed by the runtime Robot."""

    name: str
    morph_kind: Literal["urdf", "mjcf"]
    morph_file: str
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    fixed: bool = True
    merge_fixed_links: bool = False
    arm_dofs: int = 7
    default_arm_qpos: tuple[float, ...] = ()
    ee_link_name: str = ""
    arm_kp: tuple[float, ...] = ()
    arm_kv: tuple[float, ...] = ()
    arm_force_limit: float = 50.0
    gripper_name: str | None = None
    ik_init_at_home: bool = True
    ik_max_samples: int = 50
    ik_max_solver_iters: int = 40
    ik_damping: float = 0.01
    # robot-mounted cameras (robosuite keeps eye-in-hand cams in the robot XML)
    cameras: tuple[CameraSpec, ...] = ()
    entity: object = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ROBOT_MODEL_REGISTRY[cls.__name__] = cls

    def make_morph(self):
        """Genesis loads the robot description directly — one morph, no XML merge."""
        if self.morph_kind == "urdf":
            return gs.morphs.URDF(
                file=self.morph_file,
                pos=self.base_pos,
                quat=self.base_quat,
                fixed=self.fixed,
                merge_fixed_links=self.merge_fixed_links,
            )
        return gs.morphs.MJCF(file=self.morph_file, pos=self.base_pos, quat=self.base_quat)
