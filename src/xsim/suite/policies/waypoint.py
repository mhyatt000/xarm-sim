"""Waypoint-following scripted policy over the suite's public robot surface."""

from __future__ import annotations

import math
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
    gripper: float  # GRIPPER_OPEN / GRIPPER_CLOSED, commanded while travelling here
    steps: int  # control ticks to travel here from the previous waypoint


def _slerp(qa: torch.Tensor, qb: torch.Tensor, t: float) -> torch.Tensor:
    """Shortest-arc quaternion slerp; falls back to nlerp when nearly parallel."""
    qa = qa / qa.norm()
    qb = qb / qb.norm()
    dot = float((qa * qb).sum())
    if dot < 0.0:  # antipode: q and -q are the same rotation, take the short way
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + t * (qb - qa)
        return q / q.norm()
    theta = math.acos(min(dot, 1.0))
    s = math.sin(theta)
    return (math.sin((1.0 - t) * theta) / s) * qa + (math.sin(t * theta) / s) * qb


class WaypointPolicy:
    """Gym-style scripted policy: reset() builds a waypoint plan via the
    waypoints() hook; act() emits [j0..j6, g] actions (np.float32, shape (8,)).

    The commanded EE pose interpolates smoothly (position lerp, quaternion
    slerp) from the previous waypoint to the active one over its ``steps``
    ticks — arriving exactly on the waypoint at the last tick and immediately
    departing into the next segment. IK is solved per tick on the interpolated
    pose. A segment whose pose equals the previous one degenerates to a dwell
    (how grasp-close holds are expressed). When the plan is exhausted the final
    action repeats forever.
    """

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
        prev = wps[0].pose  # plans start at the current EE pose
        for wp in wps:
            steps = max(1, wp.steps)
            for k in range(1, steps + 1):
                t = k / steps
                pos = prev[0, :3] + t * (wp.pose[0, :3] - prev[0, :3])
                quat = _slerp(prev[0, 3:7], wp.pose[0, 3:7], t)
                action = self._action(torch.cat([pos, quat]).reshape(1, 7), wp.gripper)
                yield action
            prev = wp.pose
        while True:
            yield action

    def _action(self, pose: torch.Tensor, gripper: float) -> np.ndarray:
        joints = self.robot.ik(pose.to(device=gs.device, dtype=gs.tc_float))
        return np.append(joints, gripper).astype(np.float32)
