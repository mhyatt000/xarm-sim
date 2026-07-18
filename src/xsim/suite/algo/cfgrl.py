"""Torch port of ogbench's classifier-free guidance RL (ogbench/agents/cfgrl.py)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xsim.suite.algo.gcdataset import GCDatasetConfig
from xsim.suite.algo.nets import GCActorVectorField, UnconditionalEmbedding


@dataclass
class CFGRLConfig(GCDatasetConfig):
    """Configuration for the CFGRL agent."""

    lr: float = 3e-4
    batch_size: int = 1024
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    flow_steps: int = 16  # Euler integration steps at sampling time
    cfg: float = 3.0  # CFG coefficient
    uncond_prob: float = 0.1  # P(goal dropped for the unconditional embedding)


class CFGRLAgent(nn.Module):
    """Classifier-free guidance RL (CFGRL) agent.

    GCFBC with classifier-free guidance: during training the goal is replaced
    by a learned unconditional embedding with probability ``uncond_prob``; at
    sampling time the velocity is ``unc + cfg * (cond - unc)``.

    Batches are dicts of tensors with keys ``observations``, ``actions``, and
    ``actor_goals``. ``ob_dim`` is the dimension after ``encoder`` (if any).
    """

    def __init__(
        self,
        cfg: CFGRLConfig,
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
        self.unc_embed = UnconditionalEmbedding(goal_dim=goal_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def actor_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the behavioral flow-matching actor loss."""
        x_1 = batch["actions"]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, device=x_1.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        do_cfg = torch.rand(x_1.shape[0], 1, device=x_1.device) < self.cfg.uncond_prob
        goals = torch.where(do_cfg, self.unc_embed(), batch["actor_goals"])

        pred = self.actor_flow(batch["observations"], x_t, t, goals)
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
        """Sample actions by Euler-integrating the CFG-combined vector field."""
        if self.actor_flow.encoder is not None:
            observations = self.actor_flow.encoder(observations)
        actions = torch.randn(*observations.shape[:-1], self.action_dim, device=observations.device)
        unc_goals = self.unc_embed()[0].expand_as(goals)
        for i in range(self.cfg.flow_steps):
            t = torch.full((*observations.shape[:-1], 1), i / self.cfg.flow_steps, device=observations.device)
            unc_vels = self.actor_flow(observations, actions, t, unc_goals, is_encoded=True)
            cond_vels = self.actor_flow(observations, actions, t, goals, is_encoded=True)
            vels = unc_vels + self.cfg.cfg * (cond_vels - unc_vels)
            actions = actions + vels / self.cfg.flow_steps
        return actions.clamp(-1, 1)
