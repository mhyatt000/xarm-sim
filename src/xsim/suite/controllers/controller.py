"""Base controller for one controllable part of a robot entity."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Controller(ABC):
    """One controllable part (arm, gripper) of a robot entity. Each controller
    commands only its own dofs via dofs_idx_local; envs never touch actuators."""

    def __init__(self, entity, dofs_idx: torch.Tensor):
        self.entity = entity
        self.dofs_idx = dofs_idx

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    def setup(self) -> None:
        """Post-build hook: set gains/limits once the solver exists."""

    def reset(self) -> None:
        """Per-episode hook."""

    @abstractmethod
    def run(self, action: np.ndarray) -> None:
        """Apply one control tick's batched (n_envs, action_dim) action to this
        part's dofs."""
