"""Delta-joint action interface over an absolute joint-position env."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DeltaActionWrapper(gym.Wrapper):
    """Exposes ``[-1, 1]^(arm+1)`` relative actions over a single-robot suite env
    whose canonical action space is absolute joint targets plus a [0, 1] gripper.

    - arm: ``target = clip(qpos + a[:n] * max_delta_rad, joint_limits)``, so |a|=1
      moves each joint target by at most ``max_delta_rad`` per control tick;
    - gripper: affine ``[-1, 1] -> [0, 1]`` (continuous, +1 = open).
    """

    def __init__(self, env: gym.Env, max_delta_rad: float = 0.10):
        super().__init__(env)
        robots = env.unwrapped.robots
        assert len(robots) == 1 and robots[0].gripper_controller is not None, (
            "DeltaActionWrapper assumes one robot with an arm + gripper action layout"
        )
        self.robot = robots[0]
        self.max_delta_rad = float(max_delta_rad)
        self._q_lo, self._q_hi = self.robot.arm_controller.joint_limits
        self._n_arm = self.robot.arm_controller.action_dim
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self._n_arm + 1,), dtype=np.float32
        )

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        qpos = self.robot.joint_positions
        target = np.clip(
            qpos + a[: self._n_arm] * self.max_delta_rad, self._q_lo, self._q_hi
        )
        gripper = (a[self._n_arm] + 1.0) / 2.0
        return self.env.step(np.append(target, gripper))

    def absolute_to_delta(self, action: np.ndarray) -> np.ndarray:
        """Inverse map: absolute ``[j0..jn, g in [0,1]]`` -> the wrapper's delta
        action, saturating joint moves the arm cannot make in one tick. The one
        shared path for converting scripted-policy outputs into agent actions."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        qpos = self.robot.joint_positions
        delta = np.clip((a[: self._n_arm] - qpos) / self.max_delta_rad, -1.0, 1.0)
        gripper = 2.0 * np.clip(a[self._n_arm], 0.0, 1.0) - 1.0
        return np.append(delta, gripper).astype(np.float32)
