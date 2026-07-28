"""Multi-arm composition for the single-arm FSM experts.

One task FSM drives the whole rig: per env, exactly one arm is active and the
FSM sees that arm through ``ActiveArmView`` (a per-env gather across
``env.robots``), so the swept single-arm cores run unmodified. Every tick the
nearest arm to the core's current grasp target is chosen, but a switch only
applies on envs whose FSM sits in APPROACH — mid-grasp state never migrates
between arms. Initial assignment, StackRGY per-move handoffs (retreat ends in
APPROACH for the next move), and re-grasp reassignment after a dropped cube
all fall out of that one rule.

Inactive arms park: a rate-limited joint-space glide to the model's home qpos
with the gripper open. Gliding home (rather than holding pose) matters at a
stack handoff — the outgoing arm retreats hovering right above the tower,
where the incoming arm must place the next cube.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from xsim.suite.policies.waypoint import GRIPPER_OPEN

if TYPE_CHECKING:
    from xsim.suite.environments.robot_env import RobotEnv
    from xsim.suite.robots.robot import Robot


class ActiveArmView:
    """The FSM-facing surface of ``Robot``, gathered per env across a robot
    list by the active-arm index. ``ik`` solves every robot full-batch —
    inactive rows target that robot's own live EE pose (a seeded no-op) — and
    gathers the active rows."""

    def __init__(self, robots: list[Robot], n_envs: int):
        assert len({type(r.gripper) for r in robots}) == 1, (
            "mixed grippers across arms are unsupported"
        )
        assert len({r.action_dim for r in robots}) == 1, (
            "arms must share one action layout"
        )
        self.robots = robots
        self.gripper = robots[0].gripper
        self.active = np.zeros(n_envs, dtype=np.int64)
        self._rows = np.arange(n_envs)

    def _gather(self, per_robot: list[np.ndarray]) -> np.ndarray:
        return np.stack(per_robot)[self.active, self._rows]

    @property
    def ee_pos(self) -> np.ndarray:
        return self._gather([r.ee_pos for r in self.robots])

    @property
    def ee_quat(self) -> np.ndarray:
        return self._gather([r.ee_quat for r in self.robots])

    @property
    def ee_vel(self) -> np.ndarray:
        return self._gather([r.ee_vel for r in self.robots])

    @property
    def gripper_norm(self) -> np.ndarray:
        return self._gather([r.gripper_norm for r in self.robots])

    @property
    def joint_positions(self) -> np.ndarray:
        return self._gather([r.joint_positions for r in self.robots])

    def ik(self, pose: torch.Tensor, from_current: bool = False) -> np.ndarray:
        out = None
        for i, r in enumerate(self.robots):
            own = torch.as_tensor(
                np.concatenate([r.ee_pos, r.ee_quat], axis=-1),
                device=pose.device, dtype=pose.dtype,
            )
            mine = torch.as_tensor(self.active == i, device=pose.device)
            q = r.ik(torch.where(mine[:, None], pose, own), from_current=from_current)
            out = q if out is None else np.where((self.active == i)[:, None], q, out)
        return out


class DualArmPolicy:
    """Nearest-arm dispatch of a single-arm FSM expert over ``env.robots``.

    ``make_core`` builds the task FSM bound to the ``ActiveArmView``; the core
    must expose ``grasp_target_pos()`` and ``reassignable()`` (joint-action
    mode only). act() -> (n_envs, env.action_dim): the core's block scattered
    into the active robot's slice, glide-home park elsewhere.
    """

    def __init__(
        self,
        env: RobotEnv,
        make_core: Callable[[RobotEnv, ActiveArmView], object],
        park_step_rad: float = 0.04,
    ):
        self.env = env
        self.park_step_rad = park_step_rad
        self.view = ActiveArmView(env.robots, env.n_envs)
        self._home = [
            np.asarray(r.model.default_arm_qpos, dtype=np.float64)
            for r in env.robots
        ]
        self.core = make_core(env, self.view)
        self.reset()

    def reset(self, obs=None) -> None:
        self.core.reset(obs)
        # static per-arm assignment reference: the reset (home) TCP — measured
        # once so nearest-arm never chatters as the active arm moves
        self._ref = np.stack([r.ee_pos for r in self.env.robots])  # (R, n, 3)
        self.view.active = self._nearest()

    def _nearest(self) -> np.ndarray:
        target = self.core.grasp_target_pos()
        d = np.linalg.norm(self._ref[:, :, :2] - target[None, :, :2], axis=-1)
        return d.argmin(axis=0)

    def act(self, obs=None) -> np.ndarray:
        cand = self._nearest()
        switch = self.core.reassignable() & (cand != self.view.active)
        self.view.active[switch] = cand[switch]
        block = np.asarray(self.core.act(obs), dtype=np.float32)
        n = self.env.n_envs
        out = np.empty((n, self.env.action_dim), dtype=np.float32)
        off = 0
        for i, r in enumerate(self.env.robots):
            q = np.asarray(r.joint_positions, dtype=np.float64)
            step = np.clip(
                self._home[i][None] - q, -self.park_step_rad, self.park_step_rad
            )
            park = np.concatenate([q + step, np.full((n, 1), GRIPPER_OPEN)], axis=1)
            mine = (self.view.active == i)[:, None]
            out[:, off : off + r.action_dim] = np.where(mine, block, park)
            off += r.action_dim
        return out
