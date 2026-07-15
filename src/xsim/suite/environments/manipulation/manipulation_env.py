"""Shared base for tabletop manipulation environments."""

from __future__ import annotations

import numpy as np

from xsim.suite.environments.robot_env import RobotEnv


class ManipulationEnv(RobotEnv):
    """Shared helpers for tabletop manipulation tasks."""

    def _gripper_to_target_dist(self, target_pos, robot_idx: int = 0) -> float:
        return float(
            np.linalg.norm(
                self.robots[robot_idx].ee_pos - np.asarray(target_pos, dtype=np.float64)
            )
        )
