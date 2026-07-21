"""Base robot description and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import genesis as gs

from xsim.suite.models.cameras import CameraSpec
from xsim.suite.models.mounts import Mount

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
    # IK backend selector. "genesis" -> the built-in RigidEntity.inverse_kinematics
    # (sample+DLS; multi-branch, can jump for redundant 7-DOF arms). "softcost" ->
    # Robot.ik_softcost, a batched weighted-soft-cost Gauss-Newton/LM solve that
    # folds a HOME rest-pose preference (and optional manipulability) into the same
    # least-squares problem as the pose task, so EE-pose -> joint-target is a
    # near-single-valued continuous map (kills IK-branch label multimodality).
    ik_backend: Literal["genesis", "softcost"] = "genesis"
    # softcost weights (see Robot.ik_softcost). Defaults: pose tracking dominates,
    # home is a gentle regularizer (arm still reaches the table), a firm joint-limit
    # barrier, manipulability off.
    ik_w_pos: float = 4.0        # position residual weight  [m]
    ik_w_rot: float = 2.0        # orientation residual weight [rad, angle-axis]
    ik_w_home: float = 0.01      # rest-pose (q - q_home) regularizer weight
    ik_w_limit: float = 1.0      # soft joint-limit barrier weight
    ik_w_manip: float = 0.0      # manipulability ascent weight (0 = off; approximate)
    ik_iters: int = 25           # Gauss-Newton/LM iterations
    ik_sc_damping: float = 0.01  # LM damping lambda added to the normal-eqn diagonal
    # robot-mounted cameras (robosuite keeps eye-in-hand cams in the robot XML)
    cameras: tuple[CameraSpec, ...] = ()
    # fixed rig the robot bolts onto; its geometry is added alongside the robot
    mount: Mount | None = None
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
