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
    pose: torch.Tensor  # [n_envs,7] world EE poses [x,y,z, qw,qx,qy,qz]
    gripper: float  # GRIPPER_OPEN / GRIPPER_CLOSED, commanded while travelling here
    steps: int  # control ticks to travel here from the previous waypoint


def _slerp(qa: torch.Tensor, qb: torch.Tensor, t: float) -> torch.Tensor:
    """Batched shortest-arc quaternion slerp over [n_envs, 4]; falls back to
    nlerp per row when nearly parallel."""
    qa = qa / qa.norm(dim=-1, keepdim=True)
    qb = qb / qb.norm(dim=-1, keepdim=True)
    dot = (qa * qb).sum(dim=-1, keepdim=True)
    # antipode: q and -q are the same rotation, take the short way
    qb = torch.where(dot < 0.0, -qb, qb)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    parallel = dot > 0.9995
    # nan from sin/s where s ~ 0 lands only in rows where() discards
    wa = torch.where(parallel, 1.0 - t, torch.sin((1.0 - t) * theta) / s)
    wb = torch.where(parallel, torch.full_like(dot, t), torch.sin(t * theta) / s)
    q = wa * qa + wb * qb
    return q / q.norm(dim=-1, keepdim=True)


class WaypointPolicy:
    """Gym-style scripted policy: reset() builds a per-env waypoint plan via the
    waypoints() hook; act() emits batched [j0..j6, g] actions (np.float32,
    shape (n_envs, 8)). All envs share the segment schedule; poses differ.

    The commanded EE pose interpolates smoothly (position lerp, quaternion
    slerp) from the previous waypoint to the active one over its ``steps``
    ticks — arriving exactly on the waypoint at the last tick and immediately
    departing into the next segment. IK is solved per tick on the interpolated
    pose. A segment whose pose equals the previous one degenerates to a dwell
    (how grasp-close holds are expressed). When the plan is exhausted the final
    action repeats forever.
    """

    def __init__(self, robot: Robot, steps_per_segment: int = 20,
                 cartesian: bool = False):
        self.robot = robot
        self.steps_per_segment = steps_per_segment
        # cartesian: emit [x,y,z, qw..qz, g] pose actions (CartesianActionWrapper's
        # space) instead of solving IK here — pose labels are branch-unambiguous
        self.cartesian = cartesian
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
        prev = wps[0].pose  # plans start at the current EE poses
        for wp in wps:
            steps = max(1, wp.steps)
            for k in range(1, steps + 1):
                t = k / steps
                pos = prev[:, :3] + t * (wp.pose[:, :3] - prev[:, :3])
                quat = _slerp(prev[:, 3:7], wp.pose[:, 3:7], t)
                action = self._action(torch.cat([pos, quat], dim=-1), wp.gripper)
                yield action
            prev = wp.pose
        while True:
            yield action

    def _action(self, pose: torch.Tensor, gripper: float) -> np.ndarray:
        if self.cartesian:
            p = np.asarray(pose.detach().cpu(), dtype=np.float64)
            g = np.full((p.shape[0], 1), gripper)
            return np.concatenate([p, g], axis=-1).astype(np.float32)
        joints = self.robot.ik(pose.to(device=gs.device, dtype=gs.tc_float))
        g = np.full((joints.shape[0], 1), gripper)
        return np.concatenate([joints, g], axis=-1).astype(np.float32)
