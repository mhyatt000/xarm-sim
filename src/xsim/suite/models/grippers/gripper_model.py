"""Base gripper description and registry."""

from __future__ import annotations

from dataclasses import dataclass

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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GRIPPER_REGISTRY[cls.__name__] = cls

    @property
    def default_dofs(self) -> tuple[float, ...]:
        return (self.open_dof,) * self.n_dofs

    def holding_band(self, obj_width: float) -> tuple[float, float]:
        """``gripper_norm`` interval for fingers seated on an ``obj_width``-wide
        object. Parallel-jaw grippers use the class band; hands override."""
        return self.hold_norm_lo, self.hold_norm_hi
