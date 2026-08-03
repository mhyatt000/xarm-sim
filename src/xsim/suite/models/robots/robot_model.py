"""Base robot description and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import genesis as gs
import numpy as np

from xsim.suite.models.cameras import CameraSpec, rots_from_quat_wxyz
from xsim.suite.models.grippers import GripperModel, gripper_factory
from xsim.suite.models.mounts import Mount

ROBOT_MODEL_REGISTRY: dict[str, type[RobotModel]] = {}


def compose_pose(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Rigid-transform composition T1 ∘ T2 on (pos, wxyz quat) tuples."""
    R1 = rots_from_quat_wxyz(np.asarray(q1, dtype=np.float64)[None])[0]
    pos = np.asarray(p1, dtype=np.float64) + R1 @ np.asarray(p2, dtype=np.float64)
    a, b = np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64)
    quat = np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )
    return tuple(pos), tuple(quat)


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
    # scene-level self-collision opt-out: dexterous hands with hundreds of
    # intra-hand pairs jam at curled postures (teleop drops all 231 RUKA pairs)
    self_collision: bool = True
    arm_dofs: int = 7
    default_arm_qpos: tuple[float, ...] = ()
    ee_link_name: str = ""
    arm_kp: tuple[float, ...] = ()
    arm_kv: tuple[float, ...] = ()
    arm_force_limit: float = 50.0
    gripper_name: str | None = None
    # attached-entity grippers: the flange link the gripper entity mounts on,
    # plus the robot-side flange->mount transform (robosuite mounts everything
    # at link7 ∘ ((0,0,-0.027), 180° about z)). Empty link name = the gripper
    # is baked into the robot morph (native xArm).
    gripper_mount_link: str = ""
    gripper_mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
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
    ik_backend: Literal["genesis", "softcost"] = "softcost"
    # softcost weights (see Robot.ik_softcost). Defaults: pose tracking dominates,
    # home is a gentle regularizer (arm still reaches the table), a firm joint-limit
    # barrier, manipulability off.
    ik_w_pos: float = 4.0        # position residual weight  [m]
    ik_w_rot: float = 2.0        # orientation residual weight [rad, angle-axis]
    ik_w_home: float = 0.01      # rest-pose (q - q_home) regularizer weight
    # Optional per-joint scaling of the home block, applied as
    # ``ik_w_home * ik_home_hierarchy``. Parking the big proximal joints harder
    # than the wrist makes tracking recruit the wrist first, so the arm keeps a
    # natural elbow instead of the nearest-branch contortion a uniform weight
    # allows (xclients/pyroki uses (4,4,2,4,2,1,1) on the xArm7). None = uniform.
    ik_home_hierarchy: tuple[float, ...] | None = None
    # Rest pose the home block pulls toward; None = the model's home qpos.
    ik_home_qpos: tuple[float, ...] | None = None
    ik_w_limit: float = 1.0      # soft joint-limit barrier weight
    ik_w_manip: float = 0.0      # manipulability ascent weight (0 = off; approximate)
    ik_iters: int = 25           # Gauss-Newton/LM iterations
    ik_sc_damping: float = 0.01  # LM damping lambda added to the normal-eqn diagonal
    # robot-mounted cameras (robosuite keeps eye-in-hand cams in the robot XML)
    cameras: tuple[CameraSpec, ...] = ()
    # fixed rig the robot bolts onto; its geometry is added alongside the robot
    mount: Mount | None = None
    entity: object = field(default=None, repr=False, compare=False)
    gripper_entity: object = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ROBOT_MODEL_REGISTRY[cls.__name__] = cls

    @cached_property
    def gripper_model(self) -> GripperModel | None:
        """The one GripperModel instance shared by build_entity and the runtime
        Robot (per-robot-model, so multi-robot rigs don't share state)."""
        return gripper_factory(self.gripper_name) if self.gripper_name else None

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

    def build_entity(self, scene):
        """Add this robot to the scene and return its entity. Default is a plain
        rigid entity; robots with soft parts (e.g. an MPM-skinned hand) override this
        to build a hybrid entity. Called by Task.add_to.

        A gripper with its own morph is added as a second entity attached to
        ``gripper_mount_link`` (pre-build; the mount transform rides on the
        child morph pos/quat — the child XML's base body must be identity)."""
        entity = scene.add_entity(material=gs.materials.Rigid(), morph=self.make_morph())
        g = self.gripper_model
        if g is not None and g.morph_file is not None:
            pos, quat = compose_pose(
                self.gripper_mount_pos, self.gripper_mount_quat,
                g.mount_pos, g.mount_quat,
            )
            gripper = scene.add_entity(
                material=gs.materials.Rigid(),
                morph=gs.morphs.MJCF(
                    file=str(g.morph_file), pos=pos, quat=quat,
                    batch_fixed_verts=True,
                ),
            )
            gripper.attach(entity, self.gripper_mount_link)
            self.gripper_entity = gripper
        return entity

    def mpm_options(self):
        """kwargs for gs.options.MPMOptions if this robot has an MPM soft part, else
        None. The env reads this before Scene construction to enable/size the MPM
        solver (world-fixed, set at build time). None -> no MPM solver."""
        return None
