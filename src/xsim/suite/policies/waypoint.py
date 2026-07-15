"""Waypoint-following scripted policy over the suite's public robot surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import genesis as gs
import numpy as np
import torch

from xsim.suite.robots import Robot

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0


@dataclass
class Waypoint:
    pose: torch.Tensor  # [1,7] world EE pose [x,y,z, qw,qx,qy,qz]
    gripper: float  # GRIPPER_OPEN / GRIPPER_CLOSED
    steps: int  # control ticks to hold this target


class WaypointPolicy:
    """Gym-style scripted policy: reset() builds a waypoint plan via the
    waypoints() hook; act() emits [j0..j6, g] actions (np.float32, shape (8,)),
    solving IK once per waypoint at the step it becomes active and holding each
    target for its ``steps`` ticks. When the plan is exhausted it keeps
    emitting the final hold action."""

    def __init__(self, robot: Robot, steps_per_segment: int = 20):
        self.robot = robot
        self.steps_per_segment = steps_per_segment
        self._commands = None

    def waypoints(self) -> list[Waypoint]:
        """Return the episode's plan, built from the sim's current state."""
        raise NotImplementedError

    def reset(self, obs: Any = None) -> None:
        self._commands = self._generator(self.waypoints())

    def act(self, obs: Any = None) -> np.ndarray:
        return next(self._commands)

    def _generator(self, wps: list[Waypoint]):
        action = None
        for wp in wps:
            # IK is solved when the waypoint becomes active, not upfront:
            # later segments depend on physics only through the plan built at reset.
            action = self._action(wp)
            for _ in range(max(1, wp.steps)):
                yield action
        while True:
            yield action

    def _action(self, wp: Waypoint) -> np.ndarray:
        joints = self.robot.ik(wp.pose.to(device=gs.device, dtype=gs.tc_float))
        return np.append(joints, wp.gripper).astype(np.float32)
