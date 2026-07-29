"""Shared base for tabletop manipulation environments."""

from __future__ import annotations

import numpy as np

from xsim.suite.environments.robot_env import RobotEnv
from xsim.suite.models.cameras import rots_from_quat_wxyz


def pose_mats(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(n, 3) positions + (n, 4) wxyz quats -> (n, 4, 4) homogeneous poses."""
    pos = np.asarray(pos, dtype=np.float64)
    T = np.tile(np.eye(4), (pos.shape[0], 1, 1))
    T[:, :3, :3] = rots_from_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float64))
    T[:, :3, 3] = pos
    return T


class ManipulationEnv(RobotEnv):
    """Shared helpers for tabletop manipulation tasks."""

    def _gripper_to_target_dist(self, target_pos, robot_idx: int = 0) -> np.ndarray:
        """Per-env EE-to-target distances, shape (n_envs,)."""
        return np.linalg.norm(
            self.robots[robot_idx].ee_pos - np.asarray(target_pos, dtype=np.float64),
            axis=-1,
        )

    def _contact_mask(self, obj, other_entity) -> np.ndarray:
        """Per-env: is ``obj`` (a GenesisObject) in contact with ``other_entity``?"""
        contacts = obj.entity.get_contacts(with_entity=other_entity)
        mask = contacts.get("valid_mask") if isinstance(contacts, dict) else None
        if mask is None:
            return np.zeros(self.n_envs, dtype=bool)
        mask = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask)
        if mask.ndim == 1:  # non-parallelized scene: (n_contacts,)
            return np.full(self.n_envs, bool(mask.any()))
        return mask.any(axis=-1)

    def _robot_contact(self, obj, robot_idx: int | None = None) -> np.ndarray:
        """Per-env: is ``obj`` in contact with any body of the ``robot_idx``-th
        robot (or of ANY robot, the default)? A robot may span several entities
        (arm + attached gripper)."""
        robots = self.robots if robot_idx is None else [self.robots[robot_idx]]
        return np.any(
            [self._contact_mask(obj, e) for r in robots for e in r.entities], axis=0
        )

    def _grasped(self, obj, clearance: float = 0.005) -> np.ndarray:
        """Per-env: is ``obj`` held by any robot? Contact paired with THAT
        robot's fingers not fully open (gripper_norm < 0.9; seated on an object
        it plateaus well below — a parked open hand brushing the object doesn't
        count), and the object's bottom clear of the table top by ``clearance``."""
        off_surface = (
            obj.get_pos()[:, 2] > self.arena.top_z + obj.bottom_offset + clearance
        )
        holding = np.any(
            [
                np.any([self._contact_mask(obj, e) for e in r.entities], axis=0)
                & (r.gripper_norm < 0.9)
                for r in self.robots
            ],
            axis=0,
        )
        return holding & off_surface

    # -- mimicgen-style generation state ------------------------------------------
    def datagen_info(self) -> dict:
        """Per-step generation state, all batched over envs:

        - ``eef_pose``: (n_envs, 4, 4) robot-0 EEF poses
        - ``object_poses``: {name: (n_envs, 4, 4)} via :meth:`_datagen_object_poses`
        - ``gripper_action``: (n_envs, 1) last commanded gripper channel
          (GRIPPER_OPEN before any action)
        - ``subtask_term_signals``: {name: (n_envs,) bool} via
          :meth:`_datagen_term_signals`
        """
        robot = self.robots[0]
        return {
            "eef_pose": pose_mats(robot.ee_pos, robot.ee_quat),
            "object_poses": self._datagen_object_poses(),
            "gripper_action": self._last_gripper_action(),
            "subtask_term_signals": self._datagen_term_signals(),
        }

    def _last_gripper_action(self) -> np.ndarray:
        if self._last_action is None:
            return np.ones((self.n_envs, 1))
        g = self.robots[0].action_dim - 1  # gripper is robot 0's last channel
        return self._last_action[:, g : g + 1].copy()

    def _datagen_object_poses(self) -> dict[str, np.ndarray]:
        return {}

    def _datagen_term_signals(self) -> dict[str, np.ndarray]:
        return {}
