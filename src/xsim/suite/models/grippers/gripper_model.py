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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GRIPPER_REGISTRY[cls.__name__] = cls

    @property
    def default_dofs(self) -> tuple[float, ...]:
        return (self.open_dof,) * self.n_dofs
