"""Scripted waypoint policy for the block lift task.

Sequence (grifflee's 2026-07-02 protocol): home → above the cube (high, yaw-aligned to
the cube faces, gripper straight down) → vertical plunge → close → lift quickly →
transport to the drop target on the table centerline → open (cube drops). The episode
ends at the release; there is no lowering-to-place and no retreat — the robot must never
be recorded moving away from a cube that is on the table.

Drives the env's ``Manipulator`` via ``go_to_goal`` with straight-line lerp per segment.
Waypoint quats differ (yaw alignment happens over the approach segment), so the lerped
quat is renormalized each step (nlerp; twist is ≤45° so this is fine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class LiftCommand:
    pose: torch.Tensor      # [1,7] = [x,y,z, qw,qx,qy,qz]
    open_gripper: bool


# Per-segment duration weights (× steps_per_segment): approach, plunge, close, lift,
# transport. Approach keeps the real demos' slow start; lift is deliberately fast
# ("lift quickly up"); the plunge is short so it reads as a vertical drop.
SEGMENT_WEIGHTS = (2.6, 0.8, 0.8, 0.4, 1.0)

# Gripper pointing straight down (180° about world x from identity), so the approach and
# grasp are exactly vertical instead of inheriting the ready pose's slight tilt.
TOP_DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def _yawed_top_down_quat(yaw: float) -> tuple[float, float, float, float]:
    """Top-down grasp quat twisted about world z by ``yaw`` (w,x,y,z)."""
    # q = qz(yaw) ⊗ q_topdown, with q_topdown = (0,1,0,0):
    # (cos·0 - 0, cos·1 + sin·0, ... ) → (0, cos(yaw/2), sin(yaw/2), 0)
    h = yaw / 2.0
    return (0.0, math.cos(h), math.sin(h), 0.0)


class ScriptedLiftPolicy:
    def __init__(self, env, steps_per_segment: int = 40, grasp_tcp_offset: float = 0.018,
                 approach_height: float = 0.12, lift_height: float = 0.09,
                 segment_weights: tuple[float, ...] = SEGMENT_WEIGHTS):
        self.env = env
        self.steps_per_segment = steps_per_segment
        self.grasp_tcp_offset = grasp_tcp_offset
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.segment_weights = segment_weights
        self._commands = None
        self.waypoint_names = []
        self.n_steps = 0
        self.release_step = 0   # global step index at which the open command begins
        self.grasp_lock_step = 0  # global step index at which the close segment completes

    def reset(self) -> None:
        env = self.env
        device = env.device
        ee = env.robot.ee_pose.clone().reshape(-1)          # [7]
        home_quat = ee[3:7].clone()
        cube = torch.as_tensor(env.cube_pos(), device=device, dtype=ee.dtype)  # [3]
        drop_x, drop_y = env.current_drop_xy
        top_z = env.cfg.table.top_z

        # align the fingers with the cube faces: twist the top-down grasp about z by the
        # cube yaw folded into (-45°, 45°] so the wrist never turns more than a quarter face
        yaw = float(env.cube_yaw())
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        grasp_quat = torch.as_tensor(_yawed_top_down_quat(yaw), device=device, dtype=ee.dtype)
        if torch.dot(home_quat, grasp_quat) < 0:  # avoid lerping across the antipode
            grasp_quat = -grasp_quat

        def pose(xyz, quat):
            return torch.cat([torch.as_tensor(xyz, device=device, dtype=ee.dtype), quat]).reshape(1, 7)

        # The cube is 31.75 mm tall. Keep the TCP near the upper half of the cube;
        # driving it near the table plane makes the gripper visibly clip through the block.
        grasp_z = top_z + self.grasp_tcp_offset
        lift_z = grasp_z + self.lift_height
        above = pose([cube[0], cube[1], grasp_z + self.approach_height], grasp_quat)
        at = pose([cube[0], cube[1], grasp_z], grasp_quat)
        lift = pose([cube[0], cube[1], lift_z], grasp_quat)
        # transport level with the lift; the release happens here (no lowering, cube drops)
        over_drop = pose([drop_x, drop_y, lift_z], grasp_quat)

        self.waypoint_names = [
            "home",
            "above_block",
            "at_block_open",
            "at_block_closed",
            "lift",
            "over_drop",
        ]

        # (target_pose, open_gripper) waypoints
        self._waypoints = [
            (ee.reshape(1, 7), True),   # home
            (above, True),              # high above the cube, yaw-aligned, straight down
            (at, True),                 # vertical plunge
            (at, False),                # close (grasp)
            (lift, False),              # lift quickly
            (over_drop, False),         # transport to the drop target
        ]
        n_seg = len(self._waypoints) - 1
        weights = self.segment_weights if len(self.segment_weights) == n_seg else (1.0,) * n_seg
        self._segment_steps = [max(2, round(w * self.steps_per_segment)) for w in weights]
        self._commands = self._make_generator()
        # step 0 is the initial hold; segments follow in order (close is segment 3)
        self.grasp_lock_step = 1 + sum(self._segment_steps[:3])
        self.release_step = 1 + sum(self._segment_steps)
        self.n_steps = self.release_step

    def _make_generator(self):
        last_pose, last_open = self._waypoints[0]
        yield LiftCommand(last_pose, last_open)
        for (target_pose, target_open), seg_steps in zip(self._waypoints[1:], self._segment_steps):
            for i in range(1, seg_steps + 1):
                alpha = i / seg_steps
                p = last_pose.lerp(target_pose, alpha).clone()
                p[:, 3:7] = p[:, 3:7] / torch.linalg.norm(p[:, 3:7])
                yield LiftCommand(p, target_open)
            last_pose = target_pose
        # release: hold the drop pose and open; the runner records the fingers opening
        # for a short tail and then ends the episode
        while True:
            yield LiftCommand(self._waypoints[-1][0], True)

    def step(self) -> LiftCommand:
        return next(self._commands)
