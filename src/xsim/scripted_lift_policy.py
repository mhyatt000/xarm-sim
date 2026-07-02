"""Scripted waypoint policy for the block lift task.

Sequence: home → above block → descend → close → lift → transport to the central drop
zone → open (drop) → retreat. Drives the env's ``Manipulator`` via ``go_to_goal``, using
a fixed top-down (gripper pointing down) orientation captured at reset, interpolating
between waypoints with a straight-line lerp per segment (like ``GraspPolicy``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LiftCommand:
    pose: torch.Tensor      # [1,7] = [x,y,z, qw,qx,qy,qz]
    open_gripper: bool


class ScriptedLiftPolicy:
    def __init__(self, env, steps_per_segment: int = 40, grasp_tcp_offset: float = 0.016,
                 approach_height: float = 0.14, lift_height: float = 0.22):
        self.env = env
        self.steps_per_segment = steps_per_segment
        self.grasp_tcp_offset = grasp_tcp_offset
        self.approach_height = approach_height
        self.lift_height = lift_height
        self._commands = None
        self.waypoint_names = []
        self.n_steps = 0

    def reset(self) -> None:
        env = self.env
        device = env.device
        ee = env.robot.ee_pose.clone().reshape(-1)          # [7]
        quat = ee[3:7].clone()                               # keep top-down orientation
        cube = torch.as_tensor(env.cube_pos(), device=device, dtype=ee.dtype)  # [3]
        drop = torch.as_tensor(env.cfg.drop_zone, device=device, dtype=ee.dtype)
        top_z = env.cfg.table.top_z

        def pose(xyz):
            return torch.cat([torch.as_tensor(xyz, device=device, dtype=ee.dtype), quat]).reshape(1, 7)

        # The cube is 31.75 mm tall. Keep the TCP near the upper half of the cube;
        # driving it near the table plane makes the gripper visibly clip through the block.
        grasp_z = top_z + self.grasp_tcp_offset
        above = pose([cube[0], cube[1], grasp_z + self.approach_height])
        at = pose([cube[0], cube[1], grasp_z])
        lift = pose([cube[0], cube[1], grasp_z + self.lift_height])
        over_drop = pose([drop[0], drop[1], drop[2] + self.lift_height])
        at_drop = pose([drop[0], drop[1], drop[2]])
        retreat = pose([drop[0], drop[1], drop[2] + self.lift_height])

        self.waypoint_names = [
            "home",
            "above_block",
            "at_block_open",
            "at_block_closed",
            "lift",
            "over_drop",
            "at_drop_closed",
            "release",
            "retreat",
        ]

        # (target_pose, open_gripper) waypoints
        self._waypoints = [
            (ee.reshape(1, 7), True),   # home
            (above, True),              # above block
            (at, True),                 # descend
            (at, False),                # close (grasp)
            (lift, False),              # lift
            (over_drop, False),         # transport above drop zone
            (at_drop, False),           # lower into drop zone
            (at_drop, True),            # release
            (retreat, True),            # retreat
        ]
        self._commands = self._make_generator()
        # total steps = 1 (initial) + (n_wp-1)*steps_per_segment, then hold
        self.n_steps = 1 + (len(self._waypoints) - 1) * self.steps_per_segment

    def _make_generator(self):
        last_pose, last_open = self._waypoints[0]
        yield LiftCommand(last_pose, last_open)
        for target_pose, target_open in self._waypoints[1:]:
            for i in range(1, self.steps_per_segment + 1):
                alpha = i / self.steps_per_segment
                p = last_pose.lerp(target_pose, alpha)
                yield LiftCommand(p, target_open)
            last_pose = target_pose
        while True:
            yield LiftCommand(self._waypoints[-1][0], self._waypoints[-1][1])

    def step(self) -> LiftCommand:
        return next(self._commands)
