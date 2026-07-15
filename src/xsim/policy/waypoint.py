"""One waypoint-following policy; scripted tasks differ only in their waypoint list.

Subclasses override :meth:`WaypointPolicy.waypoints` to return the trajectory as
data. This base owns everything else: per-segment straight-line lerp, quaternion
renormalization, segment timing from per-waypoint weights, and the hold-forever
open tail after the last waypoint. It is privileged — it keeps the env handle so
``waypoints()`` can read sim state (cube pose, drop target, ...) at reset time —
but ``act`` itself never touches the env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from xsim.policy.base import GRIPPER_CLOSED, GRIPPER_OPEN, Action


@dataclass
class Waypoint:
    name: str
    pose: torch.Tensor  # [1,7] = [x,y,z, qw,qx,qy,qz]
    open_gripper: bool
    # duration of the segment arriving at this waypoint, x steps_per_segment
    # (ignored on the first waypoint, which is the initial hold)
    weight: float = 1.0


class WaypointPolicy:
    def __init__(self, env, steps_per_segment: int = 40):
        self.env = env
        self.steps_per_segment = steps_per_segment
        self._commands = None
        # Trajectory metadata for runners (previews, weld timing, recording length);
        # not part of the Policy protocol.
        self.waypoint_names: list[str] = []
        self.grasp_lock_step = 0  # step after the first closed waypoint is reached
        self.release_step = 0  # first tail step (the gripper opens here)
        self.n_steps = 0

    def waypoints(self) -> list[Waypoint]:
        """Return the episode's trajectory, built from ``self.env``'s current state."""
        raise NotImplementedError

    def reset(self, obs: Any = None) -> None:
        wps = self.waypoints()
        self._segment_steps = [
            max(2, round(w.weight * self.steps_per_segment)) for w in wps[1:]
        ]
        self._commands = self._generator(wps)

        self.waypoint_names = [w.name for w in wps]
        closed = [i for i, w in enumerate(wps) if not w.open_gripper]
        # step 0 is the initial hold; waypoint i is reached at 1 + sum of the
        # segment steps before it
        self.grasp_lock_step = 1 + sum(self._segment_steps[: closed[0]]) if closed else 0
        self.release_step = 1 + sum(self._segment_steps)
        self.n_steps = self.release_step

    def act(self, obs: Any = None) -> Action:
        return next(self._commands)

    def _generator(self, wps: list[Waypoint]):
        last = wps[0]
        yield self._action(last.pose, last.open_gripper)
        for target, seg_steps in zip(wps[1:], self._segment_steps, strict=True):
            for i in range(1, seg_steps + 1):
                pose = last.pose.lerp(target.pose, i / seg_steps).clone()
                pose[:, 3:7] = pose[:, 3:7] / torch.linalg.norm(pose[:, 3:7])
                yield self._action(pose, target.open_gripper)
            last = target
        # trajectory over: hold the final pose with the gripper open, forever
        while True:
            yield self._action(wps[-1].pose, True)

    @staticmethod
    def _action(pose: torch.Tensor, open_gripper: bool) -> Action:
        g = GRIPPER_OPEN if open_gripper else GRIPPER_CLOSED
        return torch.cat([pose, pose.new_full((pose.shape[0], 1), g)], dim=-1)
