"""Torch port of ogbench's goal-conditioned behavioral cloning (ogbench/agents/gcbc.py)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xsim.suite.algo.gcdataset import GCDatasetConfig
from xsim.suite.algo.nets import GCActor


@dataclass
class GCBCConfig(GCDatasetConfig):
    """Configuration for the GCBC agent."""

    lr: float = 3e-4
    batch_size: int = 1024
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    const_std: bool = True  # constant (unit) std for the actor


class GCBCAgent(nn.Module):
    """Goal-conditioned behavioral cloning (GCBC) agent.

    Batches are dicts of tensors with keys ``observations``, ``actions``, and
    ``actor_goals``. ``ob_dim`` is the dimension after ``encoder`` (if any).
    """

    def __init__(
        self,
        cfg: GCBCConfig,
        ob_dim: int,
        action_dim: int,
        goal_dim: int,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.actor = GCActor(
            ob_dim=ob_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            state_dependent_std=False,
            const_std=cfg.const_std,
            encoder=encoder,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def actor_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the BC actor loss."""
        dist = self.actor(batch["observations"], batch["actor_goals"])
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -log_prob.mean()

        info = {
            "actor_loss": actor_loss.item(),
            "bc_log_prob": log_prob.mean().item(),
            "mse": ((dist.mode - batch["actions"]) ** 2).mean().item(),
            "std": dist.base_dist.scale.mean().item(),
        }
        return actor_loss, info

    def total_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        actor_loss, actor_info = self.actor_loss(batch)
        info = {f"actor/{k}": v for k, v in actor_info.items()}
        return actor_loss, info

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Take one gradient step and return the info dictionary."""
        loss, info = self.total_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: torch.Tensor,
        goals: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample actions from the actor."""
        dist = self.actor(observations, goals, temperature=temperature)
        return dist.sample().clamp(-1, 1)
