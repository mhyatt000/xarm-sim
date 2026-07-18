"""Torch port of ogbench's goal-conditioned flow BC (ogbench/agents/gcfbc.py)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xsim.suite.algo.gcdataset import GCDatasetConfig
from xsim.suite.algo.nets import GCActorVectorField


@dataclass
class GCFBCConfig(GCDatasetConfig):
    """Configuration for the GCFBC agent."""

    lr: float = 3e-4
    batch_size: int = 1024
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    flow_steps: int = 16  # Euler integration steps at sampling time


class GCFBCAgent(nn.Module):
    """Goal-conditioned flow behavioral cloning (GCFBC) agent.

    Behavioral flow matching: the actor is a vector field v(s, x_t, t, g)
    trained on linear interpolants between noise and dataset actions, and
    actions are sampled by Euler integration from N(0, I).

    Batches are dicts of tensors with keys ``observations``, ``actions``, and
    ``actor_goals``. ``ob_dim`` is the dimension after ``encoder`` (if any).
    """

    def __init__(
        self,
        cfg: GCFBCConfig,
        ob_dim: int,
        action_dim: int,
        goal_dim: int,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.actor_flow = GCActorVectorField(
            ob_dim=ob_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            goal_dim=goal_dim,
            layer_norm=cfg.actor_layer_norm,
            encoder=encoder,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def actor_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the flow BC loss."""
        x_1 = batch["actions"]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, device=x_1.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.actor_flow(batch["observations"], x_t, t, batch["actor_goals"])
        actor_loss = ((pred - vel) ** 2).mean()

        return actor_loss, {"actor_loss": actor_loss.item()}

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
    ) -> torch.Tensor:
        """Sample actions by Euler-integrating the learned vector field."""
        if self.actor_flow.encoder is not None:
            observations = self.actor_flow.encoder(observations)
        actions = torch.randn(*observations.shape[:-1], self.action_dim, device=observations.device)
        for i in range(self.cfg.flow_steps):
            t = torch.full((*observations.shape[:-1], 1), i / self.cfg.flow_steps, device=observations.device)
            vels = self.actor_flow(observations, actions, t, goals, is_encoded=True)
            actions = actions + vels / self.cfg.flow_steps
        return actions.clamp(-1, 1)
