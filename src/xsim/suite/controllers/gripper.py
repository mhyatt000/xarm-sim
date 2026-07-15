"""Binary open/grasp gripper control."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from xsim.suite.controllers.controller import Controller
from xsim.suite.models.grippers import GripperModel


class GripperController(Controller):
    """Binary open/grasp: one channel, value > threshold means open; fingers
    snap to the model's open/grasp setpoints (no proportional finger control)."""

    def __init__(
        self,
        entity,
        dofs_idx: torch.Tensor,
        gripper: GripperModel,
        open_threshold: float = 0.5,
    ):
        super().__init__(entity, dofs_idx)
        self.gripper = gripper
        self.open_threshold = open_threshold

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
        is_open = (
            float(np.asarray(action, dtype=np.float64).reshape(-1)[0]) > self.open_threshold
        )
        dof = self.gripper.open_dof if is_open else self.gripper.grasp_dof
        n = int(self.dofs_idx.shape[0])
        t = torch.full((1, n), float(dof), device=gs.device, dtype=gs.tc_float)
        self.entity.control_dofs_position(position=t, dofs_idx_local=self.dofs_idx)
