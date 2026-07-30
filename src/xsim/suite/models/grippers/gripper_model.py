"""Base gripper description and registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GRIPPER_REGISTRY: dict[str, type[GripperModel]] = {}


@dataclass
class GripperModel:
    """Dof layout + setpoints for an end-effector. The fingers may be baked
    into the robot's URDF; Genesis then exposes them as trailing dofs of the
    same entity, which is where the runtime GripperController points."""

    name: str
    n_dofs: int
    open_dof: float  # dof value with fingers fully open
    close_dof: float  # hard mechanical stop
    grasp_dof: float  # task grasp target (less than full closure)
    finger_link_names: tuple[str, str]
    kp: float
    kv: float
    force_limit: float
    # grasp geometry and timing consumed by the scripted experts
    grasp_dz: float  # TCP height above the held object's center at grasp
    max_open_width: float  # jaw opening at open_dof
    held_radius: float  # object-center-to-TCP distance that still counts as held
    close_min_s: float  # finger travel time before a grasp can register
    close_timeout_s: float  # abort a close attempt that never seats
    open_s: float  # finger opening dwell
    # gripper_norm interval meaning "fingers seated on an object" (below:
    # closed on air; above: still open)
    hold_norm_lo: float
    hold_norm_hi: float
    # contact friction applied to the gripper's entity at setup (None = keep
    # the morph/Genesis default). URDF hand ports carry no friction spec and
    # the default is too slippery to carry a cube through a transport.
    finger_friction: float | None = None
    # gripper action commanded during the stack expert's OPEN dwell and the
    # RETREAT that follows. 1.0 (full open) is right for parallel jaws, whose
    # fingers splay laterally; a curling hand's fingers EXTEND several cm
    # through the just-placed cube, so hands set a partial uncurl (~0.8) —
    # enough clearance to shed the cube without the extension sweep. The next
    # APPROACH restores full open at transport height.
    release_a: float = 1.0
    # stack-place geometry consumed by the stack expert (per-gripper: a hand's
    # curled digits need different hover/shed margins than a jaw's pads)
    place_clearance: float = 0.002  # held-cube bottom above the seat at release
    release_rise: float = 0.0  # OPEN-dwell vertical creep per tick, m (0 = hold)
    # attached-entity grippers (robosuite MJCF ports): a separate Genesis
    # entity attached to the robot's flange. None = fingers baked into the
    # robot's own URDF (the native xArm case).
    morph_file: Path | None = None
    # the gripper MJCF's original base-body pose, zeroed out of the patched XML
    # (a non-identity base breaks RigidEntity.attach); composed with the
    # robot's flange->mount constant into the child morph pos/quat
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # gripper base -> TCP (robosuite's eef body). None = the robot's own
    # ee link already is the TCP
    tcp_pos: tuple[float, float, float] | None = None
    tcp_quat: tuple[float, float, float, float] | None = None
    # per-dof setpoints for mirrored fingers / linkage ratios / hand postures;
    # () broadcasts the open_dof/grasp_dof scalars to all dofs
    open_dofs: tuple[float, ...] = ()
    grasp_dofs: tuple[float, ...] = ()
    drive_dof: int = 0  # dof the open/close/grasp_dof scalars describe
    # per-dof gains: linkage grippers servo their driven knuckles stiffly and
    # their spring-coupled followers softly (the follower target then acts as
    # the spring's neutral pose, so contact can back-drive the pad flat onto
    # the object, like the real tendon spring). () broadcasts kp/kv.
    kp_dofs: tuple[float, ...] = ()
    kv_dofs: tuple[float, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GRIPPER_REGISTRY[cls.__name__] = cls

    @property
    def kp_vec(self) -> tuple[float, ...]:
        return self.kp_dofs or (self.kp,) * self.n_dofs

    @property
    def kv_vec(self) -> tuple[float, ...]:
        return self.kv_dofs or (self.kv,) * self.n_dofs

    @property
    def open_vec(self) -> tuple[float, ...]:
        return self.open_dofs or (self.open_dof,) * self.n_dofs

    @property
    def grasp_vec(self) -> tuple[float, ...]:
        return self.grasp_dofs or (self.grasp_dof,) * self.n_dofs

    @property
    def default_dofs(self) -> tuple[float, ...]:
        return self.open_vec

    def holding_band(self, obj_width: float) -> tuple[float, float]:
        """``gripper_norm`` interval for fingers seated on an ``obj_width``-wide
        object. Parallel-jaw grippers use the class band; hands override."""
        return self.hold_norm_lo, self.hold_norm_hi
