"""Reactive scripted stacking expert: the lift-expert FSM, per cube.

Same design contract as ``LiftExpertPolicy``: every phase is derived from the
measured world state each tick (never a clock), targets track live object
poses, and a fumbled cube demotes the phase, so the policy recovers instead of
executing a stale schedule. Residual hidden state is per-env: which move of
the stacking sequence is active, the phase, and small dwell counters (close /
open finger-travel time).

The stacking sequence comes from the env's first allowed order
(``stack_orders[0]``): the base cube stays put; each subsequent cube is
grasped, lifted to a transport height that clears the tower, carried over the
cube below it, lowered until the held cube hovers ~2 mm above the target's
top face, released, and the hand retreats straight up before the next move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch

from xsim.suite.policies.lift import APPROACH_HEIGHT
from xsim.suite.policies.lift_expert import side_grasp_quats
from xsim.suite.policies.waypoint import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    _slerp,
    format_action,
)

if TYPE_CHECKING:
    from xsim.suite.environments.manipulation.stack import Stack
    from xsim.suite.robots.robot import Robot

APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, PLACE, OPEN, RETREAT = range(8)


class StackExpertPolicy:
    """Batched reactive stacking expert over the suite's public surface.

    act() -> (n_envs, 8) float32 [j0..j6, g]: a ``max_step``-capped step from
    the measured EE toward the active phase's target, orientation slerped
    toward the face-aligned grasp quat of the active cube (target cube once
    carrying).
    """

    def __init__(
        self,
        env: Stack,
        robot: Robot | None = None,
        max_step: float = 0.025,
        rot_frac: float = 0.15,
        tol_xy: float = 0.02,
        tol_z: float = 0.015,
        # tighter gates for the drop: centering within stack_xy_tol with margin
        place_tol_xy: float = 0.008,
        place_tol_z: float = 0.008,
        # TCP height above the table while carrying; None = the gripper
        # model's transport_height if it declares one, else 0.15
        transport_height: float | None = None,
        # release geometry; None = the gripper model's values (jaw defaults:
        # 2 mm hover, no creep). release_rise is the vertical creep per tick
        # during the OPEN dwell — hands shed the cube gently along the rise
        # instead of dwelling in contact and then jerking away at retreat
        # speed (the drag that knocks the tower).
        place_clearance: float | None = None,
        release_rise: float | None = None,
    ):
        self.env = env
        self.robot = env.robots[0] if robot is None else robot
        self.max_step = max_step
        self.rot_frac = rot_frac
        self.tol_xy = tol_xy
        self.tol_z = tol_z
        self.place_tol_xy = place_tol_xy
        self.place_tol_z = place_tol_z
        # grasp geometry and timing from the gripper (see LiftExpertPolicy)
        g = self.robot.gripper
        if transport_height is None:
            transport_height = 0.15 if g.transport_height is None else g.transport_height
        self.transport_height = transport_height
        if g.rot_frac is not None:
            self.rot_frac = g.rot_frac
        self.carry_step_scale = 0.8 if g.carry_step_scale is None else g.carry_step_scale
        self.place_clearance = g.place_clearance if place_clearance is None else place_clearance
        self.release_rise = g.release_rise if release_rise is None else release_rise
        self.grasp_r = g.held_radius
        self.grip_lo, self.grip_hi = g.holding_band(env.cube_size)
        self.close_ticks_min = round(g.close_min_s * env.control_freq)
        self.close_ticks_max = round(g.close_timeout_s * env.control_freq)
        self.open_ticks_min = round(g.open_s * env.control_freq)
        base, mid, top = env.stack_orders[0]
        self.moves = np.array([(mid, base), (top, mid)], dtype=np.int64)
        cx, cy = env.arena.center_xy
        sx, sy = env.arena.size_xy
        m = 0.03
        self._xy_lo = np.array([cx - sx / 2 + m, cy - sy / 2 + m])
        self._xy_hi = np.array([cx + sx / 2 - m, cy + sy / 2 - m])
        self.reset()

    def reset(self, obs=None) -> None:
        n = self.env.n_envs
        self.stage = np.zeros(n, dtype=np.int64)
        self.phase = np.full(n, APPROACH, dtype=np.int64)
        self._close_ticks = np.zeros(n, dtype=np.int64)
        self._open_ticks = np.zeros(n, dtype=np.int64)
        self._open_target = np.zeros((n, 3))

    def grasp_target_pos(self) -> np.ndarray:
        """(n, 3) position of each env's current move's cube (arm assignment)."""
        rows = np.arange(self.env.n_envs)
        mv = self.moves[np.minimum(self.stage, len(self.moves) - 1)]
        pos_all = np.stack([c.get_pos() for c in self.env.cubes])
        return np.asarray(pos_all[mv[:, 0], rows], dtype=np.float64)

    def reassignable(self) -> np.ndarray:
        """(n,) envs between grasps, where switching the active arm is safe."""
        return self.phase == APPROACH

    def act(self, obs=None) -> np.ndarray:
        env, r = self.env, self.robot
        n = env.n_envs
        rows = np.arange(n)
        ee = np.asarray(r.ee_pos, dtype=np.float64)
        ee_quat = np.asarray(r.ee_quat, dtype=np.float64)
        gnorm = np.asarray(r.gripper_norm, dtype=np.float64)
        pos_all = np.stack([c.get_pos() for c in env.cubes])  # (3, n, 3)
        quat_all = np.stack([c.get_quat() for c in env.cubes])  # (3, n, 4)
        size = env.cube_size
        tcp_rel = r.gripper.grasp_dz  # TCP height above the held cube's center
        top_z = env.arena.top_z

        mv = self.moves[np.minimum(self.stage, len(self.moves) - 1)]  # (n, 2)
        cube = pos_all[mv[:, 0], rows]
        targ = pos_all[mv[:, 1], rows]
        cube_q = quat_all[mv[:, 0], rows]
        targ_q = quat_all[mv[:, 1], rows]
        done = self.stage >= len(self.moves)

        cube_xy = np.clip(cube[:, :2], self._xy_lo, self._xy_hi)
        targ_xy = np.clip(targ[:, :2], self._xy_lo, self._xy_hi)
        grasp_z = cube[:, 2] + tcp_rel
        approach_z = grasp_z + APPROACH_HEIGHT
        transport_z = np.full(n, top_z + self.transport_height)
        place_z = targ[:, 2] + size + tcp_rel + self.place_clearance

        xy_err_cube = np.linalg.norm(ee[:, :2] - cube_xy, axis=1)
        # placement is servoed on the HELD CUBE, not the TCP: the cube seats
        # off-center in the fingers by up to ~1.5 cm, which is enough to drop
        # it off the tower edge if the TCP is what gets centered
        xy_err_place = np.linalg.norm(cube[:, :2] - targ_xy, axis=1)
        grip_band = (gnorm > self.grip_lo) & (gnorm < self.grip_hi)
        cube_dist = np.linalg.norm(cube - ee, axis=1)
        held = (cube_dist < self.grasp_r) & grip_band
        # demotion uses a looser radius: a one-tick distance blip during a fast
        # carry must not open the gripper (that IS what drops the cube)
        held_carry = (cube_dist < 1.6 * self.grasp_r) & grip_band
        held_off = np.clip(ee - cube, -self.grasp_r, self.grasp_r)  # TCP minus cube center
        placed = (
            np.linalg.norm(cube[:, :2] - targ[:, :2], axis=1) < env.stack_xy_tol
        ) & (np.abs(cube[:, 2] - targ[:, 2] - size) < env.stack_z_tol)

        p = self.phase
        # demotions first: lost the cube mid-carry, drifted off it descending,
        # or a close attempt that never seats
        p[~done & (p >= LIFT) & (p <= PLACE) & ~held_carry & ~placed] = APPROACH
        p[~done & (p == DESCEND) & (xy_err_cube > 2 * self.tol_xy)] = APPROACH
        abort = ~done & (p == CLOSE) & ~held & (self._close_ticks >= self.close_ticks_max)
        p[abort] = APPROACH
        # a cube that reads placed while the hand thinks it lost the grip (a
        # lucky drop, or a grip-pressure blip as the cube seats on the tower):
        # release in place — never re-approach or yank away from the tower
        slipped_in_place = ~done & (p <= PLACE) & placed & ~held
        p[slipped_in_place] = OPEN
        # promotions
        near_above = (xy_err_cube < self.tol_xy) & (
            np.abs(ee[:, 2] - approach_z) < 2 * self.tol_z
        )
        p[~done & (p == APPROACH) & near_above] = DESCEND
        near_at = (xy_err_cube < self.tol_xy) & (np.abs(ee[:, 2] - grasp_z) < self.tol_z)
        starting_close = ~done & (p == DESCEND) & near_at
        p[starting_close] = CLOSE
        p[~done & (p == CLOSE) & held & (self._close_ticks >= self.close_ticks_min)] = LIFT
        p[~done & (p == LIFT) & held & (ee[:, 2] > transport_z - self.tol_z)] = TRANSPORT
        p[~done & (p == TRANSPORT) & held & (xy_err_place < self.place_tol_xy)] = PLACE
        starting_open = (
            ~done
            & (p == PLACE)
            & held
            & (xy_err_place < 1.5 * self.place_tol_xy)
            & (cube[:, 2] - targ[:, 2] - size < self.place_tol_z + self.place_clearance)
        )
        p[starting_open] = OPEN
        starting_open |= slipped_in_place
        # zero the dwell BEFORE the finished gate: a stale count from the
        # previous move's open would otherwise skip this move's dwell entirely,
        # retreating with the fingers still physically closed (dragging the
        # just-placed cube through the tower)
        self._open_ticks[starting_open] = 0
        finished_open = ~done & (p == OPEN) & (self._open_ticks >= self.open_ticks_min)
        p[finished_open] = RETREAT
        self._close_ticks[starting_close | abort] = 0
        self._close_ticks[p == CLOSE] += 1
        self._open_ticks[p == OPEN] += 1
        # retreat complete: next move (or hover done above the tower)
        retreat_done = ~done & (p == RETREAT) & (ee[:, 2] > transport_z - 2 * self.tol_z)
        self.stage[retreat_done] += 1
        p[retreat_done & (self.stage < len(self.moves))] = APPROACH

        z = np.choose(
            p,
            [approach_z, grasp_z, grasp_z, transport_z, transport_z, place_z,
             place_z, transport_z],
        )
        xy = np.where((p >= TRANSPORT)[:, None], targ_xy, cube_xy)
        target = np.concatenate([xy, z[:, None]], axis=1)
        # carrying: compensate the in-hand offset so the CUBE lands on target
        carrying = (p == TRANSPORT) | (p == PLACE)
        target[:, :2] = np.where(
            carrying[:, None], target[:, :2] + held_off[:, :2], target[:, :2]
        )
        target[p == PLACE, 2] = (
            targ[:, 2] + size + self.place_clearance + held_off[:, 2]
        )[p == PLACE]
        # hold still while the fingers open — at the pose LATCHED when OPEN
        # began, not the live ee: retargeting the measured ee each tick lets
        # the finger-opening reaction walk the arm (a hand's uncurl shifted
        # the TCP 3.5 cm sideways chasing its own drift, carrying the wedged
        # cube off the tower before it dropped)
        self._open_target[starting_open] = ee[starting_open]
        target[p == OPEN] = self._open_target[p == OPEN]
        target[p == OPEN, 2] += np.minimum(
            self._open_ticks[p == OPEN] * self.release_rise, 0.06
        )
        # RETREAT rises VERTICALLY from the release pose: its choose-row xy is
        # the tower center, but the TCP released laterally offset from it
        # (held-cube compensation), so chasing center-xy while rising sweeps
        # the fingers diagonally through the tower
        target[p == RETREAT, :2] = self._open_target[p == RETREAT, :2]
        grip = np.where((p >= CLOSE) & (p <= PLACE), GRIPPER_CLOSED, GRIPPER_OPEN)
        # OPEN and RETREAT command the gripper's release action: full open for
        # jaws, a partial uncurl for hands (a full open EXTENDS curled fingers
        # through the just-placed cube — during the retreat rise too, where
        # the extra uncurl drags the seated cube off the tower). The next
        # APPROACH restores full open at transport height.
        grip = np.where(p >= OPEN, self.robot.gripper.release_a, grip)

        delta = target - ee
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        # gentler while carrying: full-speed transports shake the cube loose
        step = np.where(((p >= LIFT) & (p <= PLACE))[:, None],
                        self.carry_step_scale * self.max_step, self.max_step)
        pos_cmd = ee + delta * np.minimum(1.0, step / np.maximum(dist, 1e-9))
        # align to the active cube's faces while grasping, to the target cube's
        # once carrying (so the placed faces meet flush)
        yaw_q = np.where((p >= TRANSPORT)[:, None], targ_q, cube_q)
        yaw = 2.0 * np.arctan2(yaw_q[:, 3], yaw_q[:, 0])
        grasp_quat = side_grasp_quats(yaw, ee_quat)
        quat_cmd = _slerp(
            torch.as_tensor(ee_quat, device=gs.device, dtype=torch.float32),
            torch.as_tensor(grasp_quat, device=gs.device, dtype=torch.float32),
            self.rot_frac,
        )
        pose = torch.cat(
            [torch.as_tensor(pos_cmd, device=gs.device, dtype=torch.float32), quat_cmd],
            dim=-1,
        )
        return format_action(self.robot, pose, grip, cartesian=False, ik_from_current=True)
