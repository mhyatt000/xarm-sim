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

    Closed-loop (default): each act() re-anchors to the robot's *actual* EE pose
    and commands one step of the remaining fraction toward the active waypoint —
    ``ee + (wp - ee) / remaining`` (slerp for the quat) — landing exactly on the
    waypoint at the segment's last tick. Because the anchor is the measured pose,
    the command is a valid correction from wherever the arm actually is, so the
    policy stays meaningful when it is queried for DAgger labels (or takes over
    mid-episode) while another policy has been driving off-schedule.

    Open-loop (``closed_loop=False``, the previous behavior) interpolates from
    the *nominal* previous waypoint on a fixed clock, ignoring the sim entirely
    after reset.

    IK is solved per tick on the commanded pose. A segment whose pose equals the
    previous one degenerates to a dwell (how grasp-close holds are expressed).
    When the plan is exhausted the final waypoint is commanded forever.
    """

    def __init__(self, robot: Robot, steps_per_segment: int = 20,
                 cartesian: bool = False, closed_loop: bool = True,
                 lookahead: float = 2.0):
        self.robot = robot
        self.steps_per_segment = steps_per_segment
        # cartesian: emit [x,y,z, qw..qz, g] pose actions (CartesianActionWrapper's
        # space) instead of solving IK here — pose labels are branch-unambiguous
        self.cartesian = cartesian
        self.closed_loop = closed_loop
        # closed-loop only: command lookahead/remaining of the gap instead of
        # 1/remaining. The PD arm realizes only ~55% of a commanded step per tick,
        # so the exact-fraction command lands ~steps^-0.55 short of the waypoint;
        # a 2x lead keeps commands bounded while compensating tracking lag.
        self.lookahead = lookahead
        self._plan: list[Waypoint] | None = None

    def waypoints(self) -> list[Waypoint]:
        """Return the episode's plan, built from the sim's current state."""
        raise NotImplementedError

    def reset(self, obs: Any = None) -> None:
        self._plan = self.waypoints()
        self._seg = 0
        self._remaining = max(1, self._plan[0].steps)
        self._prev = self._plan[0].pose  # open-loop anchor; plans start at the current EE poses
        self._hold_action: np.ndarray | None = None

    def _ee_pose(self) -> torch.Tensor:
        r = self.robot
        return torch.cat(
            [torch.as_tensor(r.ee_pos), torch.as_tensor(r.ee_quat)], dim=-1
        ).to(device=gs.device, dtype=torch.float32)

    def act(self, obs: Any = None) -> np.ndarray:
        if self._hold_action is not None:  # plan exhausted: terminal hold
            return self._hold_action
        wp = self._plan[self._seg]
        steps = max(1, wp.steps)
        if self.closed_loop:
            anchor = self._ee_pose()
            t = min(1.0, self.lookahead / self._remaining)
        else:
            anchor = self._prev
            t = (steps - self._remaining + 1) / steps
        pos = anchor[:, :3] + t * (wp.pose[:, :3] - anchor[:, :3])
        quat = _slerp(anchor[:, 3:7], wp.pose[:, 3:7], t)
        action = self._action(torch.cat([pos, quat], dim=-1), wp.gripper)
        self._remaining -= 1
        if self._remaining == 0:
            self._prev = wp.pose
            self._seg += 1
            if self._seg < len(self._plan):
                self._remaining = max(1, self._plan[self._seg].steps)
            else:
                # the last tick commanded wp.pose exactly (t == 1); repeat it forever
                self._hold_action = action
        return action

    def _action(self, pose: torch.Tensor, gripper: float) -> np.ndarray:
        if self.cartesian:
            p = np.asarray(pose.detach().cpu(), dtype=np.float64).copy()
            # canonicalize the double cover: q and -q are one rotation, but MSE
            # labels split across hemispheres regress toward the zero quat. All
            # task grasps are qx-dominant (top-down family), so pin qx >= 0.
            flip = p[:, 4] < 0.0
            p[flip, 3:7] *= -1.0
            g = np.full((p.shape[0], 1), gripper)
            return np.concatenate([p, g], axis=-1).astype(np.float32)
        joints = self.robot.ik(pose.to(device=gs.device, dtype=gs.tc_float))
        g = np.full((joints.shape[0], 1), gripper)
        return np.concatenate([joints, g], axis=-1).astype(np.float32)
