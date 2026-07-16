"""Cartesian EE-pose action interface over an absolute joint-position env."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch


class CartesianActionWrapper(gym.Wrapper):
    """Exposes absolute EE-pose actions ``[x,y,z, qw,qx,qy,qz, g]`` over a
    single-robot suite env whose canonical action space is absolute joint
    targets plus a [0, 1] gripper.

    The wrapper owns the IK. By default it solves with the inner env's robot
    (Genesis batched IK, seeded per the robot model's ``ik_*`` options); pass
    ``ik`` to substitute another solver (e.g. pyroki) with the same contract:
    poses (n_envs, 7) ``[x,y,z, qw,qx,qy,qz]`` -> joint targets
    (n_envs, n_arm) ndarray. Action quaternions are normalized before solving.
    """

    def __init__(self, env: gym.Env, ik: Callable[[torch.Tensor], np.ndarray] | None = None):
        super().__init__(env)
        robots = env.unwrapped.robots
        assert len(robots) == 1 and robots[0].gripper_controller is not None, (
            "CartesianActionWrapper assumes one robot with an arm + gripper action layout"
        )
        self.robot = robots[0]
        self.ik = ik if ik is not None else self.robot.ik
        low = np.array([-1.0, -1.0, -0.2, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, env.unwrapped.n_envs
        )

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1, 8).copy()
        norm = np.linalg.norm(a[:, 3:7], axis=-1, keepdims=True).clip(1e-8)
        a[:, 3:7] /= norm
        joints = self.ik(torch.as_tensor(a[:, :7], dtype=torch.float32))
        return self.env.step(np.concatenate([joints, a[:, 7:8]], axis=-1))
