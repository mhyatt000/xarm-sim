"""Cartesian EE-pose action interface over an absolute joint-position env."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch


# top-down, zero-yaw grasp quat (wxyz), gripper pointing straight down. Matches
# side_grasp_quats(cube_yaw=0) at k=0: [0, cos0, sin0, 0] = [0, 1, 0, 0].
FIXED_GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


class CartesianActionWrapper(gym.Wrapper):
    """Exposes absolute EE-pose actions ``[x,y,z, qw,qx,qy,qz, g]`` over a
    single-robot suite env whose canonical action space is absolute joint
    targets plus a [0, 1] gripper.

    The wrapper owns the IK. By default it solves with the inner env's robot
    (Genesis batched IK, seeded per the robot model's ``ik_*`` options); pass
    ``ik`` to substitute another solver (e.g. pyroki) with the same contract:
    poses (n_envs, 7) ``[x,y,z, qw,qx,qy,qz]`` -> joint targets
    (n_envs, n_arm) ndarray. Action quaternions are normalized before solving.

    ``orientation="fixed"`` pins the EE quat to a canonical top-down grasp
    (``FIXED_GRASP_QUAT``) and shrinks the action to 4-dim ``[x,y,z, g]``: the
    policy only commands position + gripper, removing the quaternion double-cover
    multimodality. ``step`` reinserts the fixed quat before solving IK, so the
    inner env is unchanged. ``orientation="free"`` is the default 8-dim behavior.

    ``orientation="yaw"`` keeps the gripper top-down but adds a single free yaw
    DOF, so the EE can align to a rotated cube (which "fixed" cannot). The action
    is 6-dim ``[x,y,z, c,s, g]``: ``(c, s)`` is a 2D unit-vector encoding of the
    half-yaw, i.e. the (qw, qy) pair of the top-down grasp quat family
    ``wxyz = [0, cos(theta/2), sin(theta/2), 0]``. Carrying yaw as this smooth
    2D vector avoids the +-pi wraparound cliff a raw scalar theta would have
    under MSE/flow regression. ``step`` normalizes ``(c, s)``, rebuilds
    ``wxyz = [0, c, s, 0]``, then solves IK exactly as the other branches.
    """

    def __init__(self, env: gym.Env, ik: Callable[[torch.Tensor], np.ndarray] | None = None,
                 orientation: str = "free"):
        super().__init__(env)
        assert orientation in ("free", "fixed", "yaw"), f"unknown orientation {orientation!r}"
        robots = env.unwrapped.robots
        assert len(robots) == 1 and robots[0].gripper_controller is not None, (
            "CartesianActionWrapper assumes one robot with an arm + gripper action layout"
        )
        self.robot = robots[0]
        # seed IK at the live qpos (near-branch) — home seeding jumps IK
        # branches on ~3% of poses (teacher ceiling 0.97, unstable), matching the
        # joint expert's ik_from_current=True path (0.99).
        self.ik = ik if ik is not None else (
            lambda p: self.robot.ik(p, from_current=True)
        )
        self.orientation = orientation
        if orientation == "fixed":
            low = np.array([-1.0, -1.0, -0.2, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.2, 1.0], dtype=np.float32)
        elif orientation == "yaw":
            low = np.array([-1.0, -1.0, -0.2, -1.0, -1.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.2, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([-1.0, -1.0, -0.2, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, env.unwrapped.n_envs
        )

    def step(self, action):
        if self.orientation == "fixed":
            a = np.asarray(action, dtype=np.float64).reshape(-1, 4)
            quat = np.broadcast_to(FIXED_GRASP_QUAT, (a.shape[0], 4))
            a = np.concatenate([a[:, :3], quat, a[:, 3:4]], axis=-1)
        elif self.orientation == "yaw":
            # [x,y,z, c,s, g]: (c,s) is the (qw,qy) pair of a top-down grasp quat
            # wxyz = [0, c, s, 0]. Normalize (c,s) so the half-yaw is a unit 2D
            # vector, then let the shared quat-normalize below handle the full 4D.
            a = np.asarray(action, dtype=np.float64).reshape(-1, 6)
            cs = a[:, 3:5]
            cs = cs / np.linalg.norm(cs, axis=-1, keepdims=True).clip(1e-8)
            zeros = np.zeros((a.shape[0], 1))
            quat = np.concatenate([zeros, cs[:, 0:1], cs[:, 1:2], zeros], axis=-1)
            a = np.concatenate([a[:, :3], quat, a[:, 5:6]], axis=-1)
        else:
            a = np.asarray(action, dtype=np.float64).reshape(-1, 8).copy()
        norm = np.linalg.norm(a[:, 3:7], axis=-1, keepdims=True).clip(1e-8)
        a[:, 3:7] = a[:, 3:7] / norm
        joints = self.ik(torch.as_tensor(a[:, :7], dtype=torch.float32))
        return self.env.step(np.concatenate([joints, a[:, 7:8]], axis=-1))
