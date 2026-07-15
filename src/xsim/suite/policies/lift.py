"""Scripted lift plan built from the Lift env's public surface.

home -> above the cube (high, yaw-aligned to the cube faces, gripper straight
down) -> vertical plunge -> close -> lift quickly -> transport to the drop
target -> release -> retreat. The square block has 90-degree rotational
symmetry, so side-grasp yaw selection picks the equivalent face-aligned yaw
closest to the current wrist orientation instead of forcing a needless
quarter-turn.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import genesis as gs
import torch

from xsim.suite.policies.waypoint import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    Waypoint,
    WaypointPolicy,
)

if TYPE_CHECKING:
    from xsim.suite.environments.manipulation.lift import Lift

# The cube is 31.75 mm tall. Keep the TCP near the upper half of the cube;
# driving it near the table plane makes the gripper visibly clip through the block.
GRASP_TCP_OFFSET = 0.018
APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.09
# Per-waypoint hold durations (x steps_per_segment): approach, plunge, close,
# lift, transport+settle, release, retreat. Approach keeps the real demos' slow
# start; lift is deliberately fast.
SEGMENT_WEIGHTS = (2.6, 0.8, 0.8, 0.4, 1.6, 0.6, 1.0)


def yawed_top_down_quat(yaw: float) -> tuple[float, float, float, float]:
    """Top-down grasp quat (gripper straight down) twisted about world z by ``yaw`` (w,x,y,z)."""
    # q = qz(yaw) * q_topdown, with q_topdown = (0,1,0,0) -> (0, cos(yaw/2), sin(yaw/2), 0)
    h = yaw / 2.0
    return (0.0, math.cos(h), math.sin(h), 0.0)


def nearest_side_grasp_quat(cube_yaw: float, reference_quat) -> tuple[float, float, float, float]:
    """Face-aligned top-down grasp requiring minimal rotation from ``reference_quat``.

    ``cube_yaw + k*pi/2`` are equivalent side grasps of a square block; choose
    the one closest to the current wrist orientation.
    """
    if hasattr(reference_quat, "detach"):
        reference_quat = reference_quat.detach().cpu().tolist()
    ref = tuple(float(v) for v in reference_quat)
    best_q = yawed_top_down_quat(cube_yaw)
    best_score = -1.0
    for k in range(-4, 5):
        q = yawed_top_down_quat(cube_yaw + k * math.pi / 2.0)
        score = abs(sum(a * b for a, b in zip(q, ref, strict=True)))
        if score > best_score:
            best_score = score
            best_q = q
    return best_q


class LiftPolicy(WaypointPolicy):
    """Scripted lift: hover over the cube aligned to its yaw, descend, close,
    lift, transport to drop_xy, release, retreat. Reads only the public suite
    surface (env.cube, env.arena, env.robots[0])."""

    def __init__(
        self,
        env: Lift,
        steps_per_segment: int = 20,
        drop_xy: tuple[float, float] = (0.35, 0.0),
    ):
        super().__init__(env.robots[0], steps_per_segment)
        self.env = env
        self.drop_xy = drop_xy

    def waypoints(self) -> list[Waypoint]:
        r = self.robot
        ee = torch.cat(
            [torch.as_tensor(r.ee_pos), torch.as_tensor(r.ee_quat)]
        ).to(device=gs.device, dtype=torch.float32)
        home_quat = ee[3:7].clone()
        cube = torch.as_tensor(
            self.env.cube.get_pos(), device=gs.device, dtype=torch.float32
        )
        q = self.env.cube.get_quat()  # wxyz; cube spawn rotation is pure-z
        cube_yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        drop_x, drop_y = self.drop_xy
        top_z = self.env.arena.top_z

        grasp_quat = torch.as_tensor(
            nearest_side_grasp_quat(cube_yaw, home_quat),
            device=gs.device,
            dtype=torch.float32,
        )

        def pose(xyz, quat):
            return torch.cat(
                [torch.as_tensor(xyz, device=gs.device, dtype=torch.float32), quat]
            ).reshape(1, 7)

        grasp_z = top_z + GRASP_TCP_OFFSET
        lift_z = grasp_z + LIFT_HEIGHT
        above = pose([cube[0], cube[1], grasp_z + APPROACH_HEIGHT], grasp_quat)
        at = pose([cube[0], cube[1], grasp_z], grasp_quat)
        lift = pose([cube[0], cube[1], lift_z], grasp_quat)
        # transport level with the lift; release happens over the drop (no lowering)
        over_drop = pose([drop_x, drop_y, lift_z], grasp_quat)
        retreat = pose([drop_x, drop_y, grasp_z + APPROACH_HEIGHT], grasp_quat)

        def hold(weight: float) -> int:
            return max(2, round(weight * self.steps_per_segment))

        w = SEGMENT_WEIGHTS
        return [
            Waypoint(ee.reshape(1, 7), GRIPPER_OPEN, 1),
            Waypoint(above, GRIPPER_OPEN, hold(w[0])),
            Waypoint(at, GRIPPER_OPEN, hold(w[1])),
            Waypoint(at, GRIPPER_CLOSED, hold(w[2])),
            Waypoint(lift, GRIPPER_CLOSED, hold(w[3])),
            Waypoint(over_drop, GRIPPER_CLOSED, hold(w[4])),
            Waypoint(over_drop, GRIPPER_OPEN, hold(w[5])),
            Waypoint(retreat, GRIPPER_OPEN, hold(w[6])),
        ]
