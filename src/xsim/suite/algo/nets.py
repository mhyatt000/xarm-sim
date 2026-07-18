"""Torch ports of ogbench's goal-conditioned networks (ogbench/utils/networks.py).

All modules take explicit input dims (torch has no shape inference); when an
encoder is used, ``ob_dim`` is the dimension *after* encoding.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import distributions as D
from torch import nn


def variance_scaling_(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Flax ``default_init``: variance scaling, fan_avg, uniform."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    limit = math.sqrt(3.0 * scale / ((fan_in + fan_out) / 2.0))
    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def dense(in_dim: int, out_dim: int, init_scale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    variance_scaling_(layer.weight, init_scale)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """GELU MLP with optional post-activation layer norm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.activate_final = activate_final
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim, *hidden_dims]
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(dense(d_in, d_out))
            is_activated = i + 2 < len(dims) or activate_final
            self.norms.append(nn.LayerNorm(d_out) if layer_norm and is_activated else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = norm(self.act(x))
        return x


class TransformedWithMode(D.TransformedDistribution):
    """Transformed distribution whose mode maps the base mode through the bijector."""

    @property
    def mode(self) -> torch.Tensor:
        x = self.base_dist.mode
        for transform in self.transforms:
            x = transform(x)
        return x


class GCActor(nn.Module):
    """Goal-conditioned Gaussian actor; returns a torch distribution over actions."""

    def __init__(
        self,
        ob_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        tanh_squash: bool = False,
        state_dependent_std: bool = False,
        const_std: bool = True,
        final_fc_init_scale: float = 1e-2,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.encoder = encoder

        self.actor_net = MLP(ob_dim + goal_dim, hidden_dims, activate_final=True)
        self.mean_net = dense(hidden_dims[-1], action_dim, final_fc_init_scale)
        if state_dependent_std:
            self.log_std_net = dense(hidden_dims[-1], action_dim, final_fc_init_scale)
        elif not const_std:
            self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        observations: torch.Tensor,
        goals: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> D.Distribution:
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = observations if goals is None else torch.cat([observations, goals], dim=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = torch.zeros_like(means)
        else:
            log_stds = self.log_stds.expand_as(means)
        log_stds = log_stds.clamp(self.log_std_min, self.log_std_max)

        dist = D.Independent(D.Normal(means, log_stds.exp() * temperature), 1)
        if self.tanh_squash:
            dist = TransformedWithMode(dist, D.TanhTransform(cache_size=1))
        return dist


class GCActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Input is the concat of [observations, actions, times, goals?, goal_steps?];
    output is the velocity in action space.
    """

    def __init__(
        self,
        ob_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        goal_dim: int = 0,
        step_dim: int = 0,
        layer_norm: bool = False,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = ob_dim + action_dim + 1 + goal_dim + step_dim
        self.mlp = MLP(in_dim, (*hidden_dims, action_dim), activate_final=False, layer_norm=layer_norm)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        times: torch.Tensor,
        goals: torch.Tensor | None = None,
        goal_steps: torch.Tensor | None = None,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        if self.encoder is not None and not is_encoded:
            observations = self.encoder(observations)
        inputs = [x for x in (observations, actions, times, goals, goal_steps) if x is not None]
        return self.mlp(torch.cat(inputs, dim=-1))


class UnconditionalEmbedding(nn.Module):
    """Learned embedding table; row 0 is the unconditional (null) token."""

    def __init__(self, goal_dim: int, num_embeddings: int = 1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, goal_dim)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(goal_dim))

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            x = torch.zeros(1, dtype=torch.long, device=self.embed.weight.device)
        return self.embed(x)
