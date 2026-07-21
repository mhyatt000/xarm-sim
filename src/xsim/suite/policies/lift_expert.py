"""Reactive scripted lift expert: phase from live state, never from a clock.

``LiftPolicy`` (waypoint) plans at reset and paces segments on a shared tick
countdown, so its label at a given instant depends on hidden schedule state —
unfittable for a student that only sees the instantaneous observation (img-v8:
round-0 BC loss 1000x the mlp-teacher baseline). This expert keeps the same
grasp geometry (above -> at -> close -> lift -> release, face-aligned yaw) but
derives each env's phase every tick from the measured world state, tracking
the *live* cube pose. Labels are a function of state by construction, the plan
waits for the arm under beta-mixing, and a fumbled cube demotes the phase, so
the aggregate collects recovery corrections instead of schedule garbage. After
a completed lift the expert releases at height (env success fires mid-LIFT at
5cm, so eval is unaffected); the drop demotes the env back to APPROACH, so
fixed-horizon rollouts cycle grasp -> lift -> drop -> re-grasp instead of
logging redundant hover states.

Residual hidden state is only a per-env close-attempt counter (abort a grasp
that isn't seating) — tiny and strongly state-correlated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch

from xsim.suite.policies.lift import APPROACH_HEIGHT, GRASP_TCP_OFFSET, LIFT_HEIGHT
from xsim.suite.policies.waypoint import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    _slerp,
    format_action,
)

if TYPE_CHECKING:
    from xsim.suite.environments.manipulation.lift import Lift

APPROACH, DESCEND, CLOSE, LIFT, RELEASE = range(5)


def side_grasp_quats(cube_yaw: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    """Batched face-aligned top-down grasp quats (n, 4) wxyz: of the four
    equivalent side grasps of a square block, the one nearest ``ref_quat``."""
    k = np.arange(-4, 5)
    h = (cube_yaw[:, None] + k[None, :] * (np.pi / 2.0)) / 2.0  # (n, 9)
    cand = np.zeros((cube_yaw.shape[0], k.shape[0], 4))
    cand[..., 1] = np.cos(h)
    cand[..., 2] = np.sin(h)
    score = np.abs(np.einsum("nkq,nq->nk", cand, ref_quat))
    return cand[np.arange(cube_yaw.shape[0]), score.argmax(axis=1)]


class LiftExpertPolicy:
    """Batched reactive lift expert over the suite's public surface.

    act() -> (n_envs, 8) float32 [j0..j6, g] (or pose actions when
    ``cartesian``): a ``max_step``-capped step from the measured EE toward the
    active phase's target, orientation slerped ``rot_frac`` toward the
    face-aligned grasp quat. Phase transitions read only measured state; the
    only memory is the phase itself plus a close-attempt tick counter.
    """

    def __init__(
        self,
        env: Lift,
        cartesian: bool = False,
        # only meaningful when cartesian: "fixed" pins the grasp to a top-down
        # quat and emits 4-dim [x,y,z, g] labels (matching CartesianActionWrapper
        # orientation="fixed"); "yaw" keeps top-down but adds a free yaw DOF,
        # emitting 6-dim [x,y,z, c,s, g] where (c,s) is the (qw,qy) pair of the
        # face-aligned grasp quat; "free" keeps the 8-dim pose labels.
        cartesian_orientation: str = "free",
        # delta-joint labels matching DeltaActionWrapper: emit
        # [clip((q_target - qpos)/max_delta_rad, -1, 1), 2*g - 1] instead of the
        # absolute [j0..j6, g]. Mutually exclusive with cartesian; max_delta_rad
        # MUST equal the wrapper's (both are fed cfg.delta_max_rad by the caller).
        delta: bool = False,
        max_delta_rad: float = 0.10,
        # 0.025/0.15 swept on the v8 spawn/init distribution: 98.4-99.8% at 512
        # envs; larger steps destabilize per-tick IK near the base column
        max_step: float = 0.025,   # commanded EE translation per tick, m
        rot_frac: float = 0.15,    # slerp fraction toward the grasp quat per tick
        tol_xy: float = 0.02,      # xy alignment gate, m
        tol_z: float = 0.015,      # z arrival gate, m
        grasp_r: float = 0.035,    # cube-to-TCP distance that counts as held, m
        # gripping band: closed-on-air drives gripper_norm toward 0, open is 1,
        # seated on the 31.75mm cube it plateaus in between
        grip_lo: float = 0.20,
        grip_hi: float = 0.85,
        close_ticks_min: int = 12, # finger travel time before a grasp can count
        close_ticks_max: int = 30, # abort a close attempt that never seats
    ):
        assert cartesian_orientation in ("free", "fixed", "yaw")
        assert not (cartesian and delta), "cartesian and delta are mutually exclusive"
        self.env = env
        self.robot = env.robots[0]
        self.cartesian = cartesian
        self.cartesian_orientation = cartesian_orientation
        self.delta = delta
        self.max_delta_rad = float(max_delta_rad)
        self.max_step = max_step
        self.rot_frac = rot_frac
        self.tol_xy = tol_xy
        self.tol_z = tol_z
        self.grasp_r = grasp_r
        self.grip_lo = grip_lo
        self.grip_hi = grip_hi
        self.close_ticks_min = close_ticks_min
        self.close_ticks_max = close_ticks_max
        # xy clamp for chase targets: stay over the table even if the cube
        # leaves it (never labels a dive off the edge)
        cx, cy = env.arena.center_xy
        sx, sy = env.arena.size_xy
        m = 0.03
        self._xy_lo = np.array([cx - sx / 2 + m, cy - sy / 2 + m])
        self._xy_hi = np.array([cx + sx / 2 - m, cy + sy / 2 - m])
        self.reset()

    def reset(self, obs=None) -> None:
        n = self.env.n_envs
        self.phase = np.full(n, APPROACH, dtype=np.int64)
        self._close_ticks = np.zeros(n, dtype=np.int64)

    def act(self, obs=None) -> np.ndarray:
        r = self.robot
        ee = np.asarray(r.ee_pos, dtype=np.float64)
        ee_quat = np.asarray(r.ee_quat, dtype=np.float64)
        gnorm = np.asarray(r.gripper_norm, dtype=np.float64)
        cube = np.asarray(self.env.cube.get_pos(), dtype=np.float64)
        q = np.asarray(self.env.cube.get_quat(), dtype=np.float64)
        top_z = self.env.arena.top_z
        grasp_z = top_z + GRASP_TCP_OFFSET
        lift_z = grasp_z + LIFT_HEIGHT

        cube_xy = np.clip(cube[:, :2], self._xy_lo, self._xy_hi)
        xy_err = np.linalg.norm(ee[:, :2] - cube_xy, axis=1)
        # gripping = cube at the TCP with the fingers seated on it (norm in the
        # cube-width band; below it they closed on air, above they're still open)
        held = ((np.linalg.norm(cube - ee, axis=1) < self.grasp_r)
                & (gnorm > self.grip_lo) & (gnorm < self.grip_hi))
        near_at = (xy_err < self.tol_xy) & (np.abs(ee[:, 2] - grasp_z) < self.tol_z)
        near_above = (xy_err < self.tol_xy) & (
            np.abs(ee[:, 2] - (grasp_z + APPROACH_HEIGHT)) < 2 * self.tol_z
        )

        p = self.phase
        # demotions first: lost the cube, or drifted off it while descending
        p[(p >= LIFT) & ~held] = APPROACH
        p[(p == DESCEND) & (xy_err > 2 * self.tol_xy)] = APPROACH
        abort = (p == CLOSE) & ~held & (self._close_ticks >= self.close_ticks_max)
        p[abort] = APPROACH
        # promotions
        p[(p == APPROACH) & near_above] = DESCEND
        starting_close = (p == DESCEND) & near_at
        p[starting_close] = CLOSE
        # fingers need travel time: a grasp only counts once the dwell has run
        # and the norm sits in the gripping band (not still sweeping through it)
        p[(p == CLOSE) & held & (self._close_ticks >= self.close_ticks_min)] = LIFT
        p[(p == LIFT) & held & (ee[:, 2] > lift_z - self.tol_z)] = RELEASE
        self._close_ticks[starting_close | abort] = 0
        self._close_ticks[p == CLOSE] += 1

        z = np.choose(p, [grasp_z + APPROACH_HEIGHT, grasp_z, grasp_z, lift_z, lift_z])
        target = np.concatenate([cube_xy, z[:, None]], axis=1)
        # RELEASE opens at height: the dropped cube's bounce diversifies re-grasp
        # poses, and the existing ~held demotion recycles the env to APPROACH
        grip = np.where((p >= CLOSE) & (p != RELEASE), GRIPPER_CLOSED, GRIPPER_OPEN)

        delta = target - ee
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        pos_cmd = ee + delta * np.minimum(1.0, self.max_step / np.maximum(dist, 1e-9))
        if self.cartesian and self.cartesian_orientation == "fixed":
            # orientation is pinned by the wrapper; emit only [x,y,z, g]
            g = np.broadcast_to(np.asarray(grip, dtype=np.float64).reshape(-1, 1),
                                (pos_cmd.shape[0], 1))
            return np.concatenate([pos_cmd, g], axis=-1).astype(np.float32)
        grasp_quat = side_grasp_quats(2.0 * np.arctan2(q[:, 3], q[:, 0]), ee_quat)
        if self.cartesian and self.cartesian_orientation == "yaw":
            # top-down grasp, free yaw: emit [x,y,z, c,s, g] where (c,s) is the
            # (qw,qy) pair of the face-aligned grasp quat wxyz=[0,cos(h),sin(h),0]
            # (side_grasp_quats fills cand[...,1]=cos(h), cand[...,2]=sin(h)).
            c = grasp_quat[:, 1:2]
            s = grasp_quat[:, 2:3]
            g = np.asarray(grip, dtype=np.float64).reshape(-1, 1)
            return np.concatenate([pos_cmd, c, s, g], axis=-1).astype(np.float32)
        quat_cmd = _slerp(
            torch.as_tensor(ee_quat, device=gs.device, dtype=torch.float32),
            torch.as_tensor(grasp_quat, device=gs.device, dtype=torch.float32),
            self.rot_frac,
        )
        pose = torch.cat(
            [torch.as_tensor(pos_cmd, device=gs.device, dtype=torch.float32), quat_cmd],
            dim=-1,
        )
        # IK seeded at the live qpos: home seeding returns far-branch joint
        # targets from randomized starts (identity-gap 0.45 rad^2, privileged
        # probe floor 0.23 on img-v8b) — labels must be continuous in state
        action = format_action(self.robot, pose, grip, self.cartesian, ik_from_current=True)
        if self.delta:  # invert DeltaActionWrapper: absolute [j..,g] -> [-1,1]^(n+1)
            return self._to_delta(action)
        return action

    def _to_delta(self, action: np.ndarray) -> np.ndarray:
        """Map the expert's absolute joint action ``[j0..jn, g in [0,1]]`` into
        DeltaActionWrapper's ``[-1, 1]^(arm+1)`` space (its exact inverse):
        arm -> clip((q_target - qpos)/max_delta_rad, -1, 1), gripper -> 2g - 1."""
        qpos = np.asarray(self.robot.joint_positions, dtype=np.float64)
        n = qpos.shape[1]
        arm = np.clip((action[:, :n] - qpos) / self.max_delta_rad, -1.0, 1.0)
        gripper = 2.0 * np.clip(action[:, n:], 0.0, 1.0) - 1.0
        return np.concatenate([arm, gripper], axis=-1).astype(np.float32)
