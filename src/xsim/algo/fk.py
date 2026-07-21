"""Differentiable forward kinematics for the FK-space auxiliary BC loss.

Absolute-joint BC minimizes joint MSE, but joint error != TCP/task error: the
redundant 7-DOF arm maps a fixed joint error to a pose error that varies wildly
with configuration, so the joint metric is misaligned with the task. This wraps
``pytorch_kinematics`` into a batched, autograd-friendly FK from arm joint angles
to the EE-link pose, so training can add a task-space term on FK(q_pred) vs
FK(q_target) while keeping the joint-space output.

The serial chain is built from the same URDF Genesis loads and terminated at the
robot's own EE link (``link_tcp``); every joint on that path is one of the 7 arm
revolutes (the gripper/tcp joints are fixed), so FK is a pure function of the
(B, 7) arm targets. Chains are cached per (urdf, link, device, dtype).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = _PROJECT_ROOT / "xarm7_standalone.urdf"
DEFAULT_EE_LINK = "link_tcp"


@lru_cache(maxsize=None)
def _build_chain(urdf: str, ee_link: str, device: str, dtype: torch.dtype):
    import pytorch_kinematics as pk

    chain = pk.build_serial_chain_from_urdf(Path(urdf).read_bytes(), ee_link)
    return chain.to(dtype=dtype, device=torch.device(device))


class FKChain:
    """Batched differentiable FK to the xArm7 EE link.

    Call with arm joint targets ``q`` of shape (B, 7) (radians) and get the EE
    pose as a (B, 4, 4) homogeneous transform. ``position`` / ``rotation`` slice
    out the translation and 3x3 rotation for the pose loss. The underlying
    ``pytorch_kinematics`` chain is cached, so repeated construction is free.
    """

    def __init__(self, urdf: str | Path = DEFAULT_URDF,
                 ee_link: str = DEFAULT_EE_LINK,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.chain = _build_chain(str(urdf), ee_link, str(torch.device(device)), dtype)
        self.n_joints = len(self.chain.get_joint_parameter_names())

    def matrix(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, n_joints) arm joint angles -> (B, 4, 4) EE transform."""
        return self.chain.forward_kinematics(q).get_matrix()

    def position(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, n_joints) -> (B, 3) EE position in the base frame."""
        return self.matrix(q)[:, :3, 3]

    def rotation(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, n_joints) -> (B, 3, 3) EE rotation in the base frame."""
        return self.matrix(q)[:, :3, :3]

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return self.position(q)
