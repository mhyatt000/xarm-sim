"""Proportional open/grasp gripper control."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from xsim.suite.controllers.controller import Controller
from xsim.suite.models.grippers import GripperModel


class GripperController(Controller):
    """Proportional open/grasp: one channel in [0, 1] interpolating the fingers
    between the model's grasp (0) and open (1) setpoints."""

    def __init__(
        self,
        entity,
        dofs_idx: torch.Tensor,
        gripper: GripperModel,
    ):
        super().__init__(entity, dofs_idx)
        self.gripper = gripper

    @property
    def action_dim(self) -> int:
        return 1

    def setup(self) -> None:
        n = int(self.dofs_idx.shape[0])
        kp = torch.full((n,), self.gripper.kp, device=gs.device, dtype=gs.tc_float)
        kv = torch.full((n,), self.gripper.kv, device=gs.device, dtype=gs.tc_float)
        self.entity.set_dofs_kp(kp, dofs_idx_local=self.dofs_idx)
        self.entity.set_dofs_kv(kv, dofs_idx_local=self.dofs_idx)
        f = torch.full((n,), self.gripper.force_limit, device=gs.device, dtype=gs.tc_float)
        self.entity.set_dofs_force_range(-f, f, dofs_idx_local=self.dofs_idx)

    def run(self, action: np.ndarray) -> None:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), 0.0, 1.0)
        dof = self.gripper.grasp_dof + a * (self.gripper.open_dof - self.gripper.grasp_dof)
        n = int(self.dofs_idx.shape[0])
        t = torch.tensor(
            np.repeat(dof[:, None], n, axis=1), device=gs.device, dtype=gs.tc_float
        )
        self.entity.control_dofs_position(position=t, dofs_idx_local=self.dofs_idx)
