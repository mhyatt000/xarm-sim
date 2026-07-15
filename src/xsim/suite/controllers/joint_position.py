"""Absolute joint-position control clipped to joint limits."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from xsim.suite.controllers.controller import Controller


class JointPositionController(Controller):
    """Absolute joint-position targets clipped to joint limits."""

    def __init__(
        self,
        entity,
        dofs_idx: torch.Tensor,
        kp: tuple[float, ...],
        kv: tuple[float, ...],
        force_limit: float,
    ):
        super().__init__(entity, dofs_idx)
        self.kp = kp
        self.kv = kv
        self.force_limit = force_limit
        self._q_lo: np.ndarray | None = None
        self._q_hi: np.ndarray | None = None

    @property
    def action_dim(self) -> int:
        return int(self.dofs_idx.shape[0])

    def setup(self) -> None:
        self.entity.set_dofs_kp(
            torch.tensor(self.kp, device=gs.device, dtype=gs.tc_float),
            dofs_idx_local=self.dofs_idx,
        )
        self.entity.set_dofs_kv(
            torch.tensor(self.kv, device=gs.device, dtype=gs.tc_float),
            dofs_idx_local=self.dofs_idx,
        )
        f = torch.full(
            (self.action_dim,), float(self.force_limit), device=gs.device, dtype=gs.tc_float
        )
        self.entity.set_dofs_force_range(-f, f, dofs_idx_local=self.dofs_idx)
        lower, upper = self.entity.get_dofs_limit(self.dofs_idx)
        self._q_lo = np.asarray(lower.detach().cpu(), dtype=np.float64).reshape(-1)
        self._q_hi = np.asarray(upper.detach().cpu(), dtype=np.float64).reshape(-1)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_lo, self._q_hi

    def run(self, action: np.ndarray) -> None:
        target = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), self._q_lo, self._q_hi
        )
        t = torch.tensor(target[None], device=gs.device, dtype=gs.tc_float)
        self.entity.control_dofs_position(position=t, dofs_idx_local=self.dofs_idx)
