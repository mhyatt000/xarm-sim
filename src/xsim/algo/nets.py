"""BC/DAgger student networks.

Checkpoints are plain state_dicts: obs/action normalization stats and action
limits live in buffers so a saved policy is self-contained and loadable by
eval, play, and real-robot inference without the training script.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Student(nn.Module):
    """MLP over the GymWrapper's flat obs -> absolute [j0..j6, g] action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 act_low: np.ndarray, act_high: np.ndarray):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.obs_mean = nn.Buffer(torch.zeros(obs_dim))
        self.obs_std = nn.Buffer(torch.ones(obs_dim))
        self.act_low = nn.Buffer(torch.as_tensor(act_low, dtype=torch.float32))
        self.act_high = nn.Buffer(torch.as_tensor(act_high, dtype=torch.float32))

    def set_obs_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std.clamp_min(1e-6))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        device = self.obs_mean.device
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        a = self(x).clamp(self.act_low, self.act_high)
        return a.cpu().numpy()


def rand_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style random shift: replicate-pad then translate up to +-pad px,
    one offset per sample. x: (N, C, H, W) float."""
    n, _, h, w = x.shape
    shift = torch.randint(-pad, pad + 1, (n, 2), device=x.device, dtype=torch.float32)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift[:, 0] / w
    theta[:, 1, 2] = 2.0 * shift[:, 1] / h
    xp = F.pad(x, (pad,) * 4, mode="replicate")
    grid = F.affine_grid(theta, (n, x.shape[1], h + 2 * pad, w + 2 * pad),
                         align_corners=False)
    return F.grid_sample(xp, grid, align_corners=False)[:, :, pad:-pad, pad:-pad]


class ImageStudent(nn.Module):
    """CNN over (V, 3, H, W) rgb + MLP over proprio -> [j0..j6, gripper logit],
    plus an auxiliary cube pos+yaw head off the fused trunk.

    ``shared`` encoder folds V into the batch dim and adds a learned per-view
    embedding (new views = new embedding rows, encoder untouched); ``separate``
    trains V independent CNNs. The gripper is a logit trained with BCE (labels
    are 0/1) and snapped to the extremes at act() — an MSE-hedged half-open
    command is the one action error this task cannot absorb.
    """

    def __init__(self, proprio_dim: int, act_dim: int, n_views: int, hw: int,
                 act_low: np.ndarray, act_high: np.ndarray,
                 encoder: str = "shared", hidden: int = 256, feat_dim: int = 64):
        super().__init__()
        self.n_views = n_views
        self.shared = encoder == "shared"
        c = hw // 16  # four stride-2 convs

        def make_enc() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * c * c, feat_dim), nn.LayerNorm(feat_dim), nn.Tanh(),
            )

        if self.shared:
            self.encoder = make_enc()
            self.view_emb = nn.Parameter(torch.zeros(n_views, feat_dim))
        else:
            self.encoders = nn.ModuleList([make_enc() for _ in range(n_views)])
        self.prop_net = nn.Sequential(nn.Linear(proprio_dim, 128), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(n_views * feat_dim + 128, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, act_dim)
        self.aux_head = nn.Linear(hidden, 5)  # cube xyz + sin/cos of 4*yaw
        self.prop_mean = nn.Buffer(torch.zeros(proprio_dim))
        self.prop_std = nn.Buffer(torch.ones(proprio_dim))
        self.act_low = nn.Buffer(torch.as_tensor(act_low, dtype=torch.float32))
        self.act_high = nn.Buffer(torch.as_tensor(act_high, dtype=torch.float32))

    def set_obs_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.prop_mean.copy_(mean)
        self.prop_std.copy_(std.clamp_min(1e-6))

    def features(self, rgb: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """rgb: (N, V, 3, H, W) float in [-0.5, 0.5]; prop: (N, P) raw."""
        n = rgb.shape[0]
        if self.shared:
            f = self.encoder(rgb.reshape(n * self.n_views, *rgb.shape[2:]))
            f = f.reshape(n, self.n_views, -1) + self.view_emb
        else:
            f = torch.stack(
                [enc(rgb[:, i]) for i, enc in enumerate(self.encoders)], dim=1)
        p = self.prop_net((prop - self.prop_mean) / self.prop_std)
        return self.trunk(torch.cat([f.reshape(n, -1), p], dim=-1))

    def forward(self, rgb: torch.Tensor, prop: torch.Tensor):
        h = self.features(rgb, prop)
        return self.head(h), self.aux_head(h)

    @torch.no_grad()
    def act(self, obs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        prop, rgb = obs
        device = self.prop_mean.device
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float() / 255.0 - 0.5
        p = torch.from_numpy(np.asarray(prop, dtype=np.float32)).to(device)
        a, _ = self(x, p)
        joints = a[:, :-1].clamp(self.act_low[:-1], self.act_high[:-1])
        grip = (torch.sigmoid(a[:, -1:]) > 0.5).float()
        return torch.cat([joints, grip], dim=-1).cpu().numpy()


def time_features(t: torch.Tensor) -> torch.Tensor:
    """Fourier features of flow time t in [0, 1]. t: (N, 1) -> (N, 16)."""
    ang = t * (math.pi * 2.0 ** torch.arange(8, device=t.device))
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FlowImageStudent(ImageStudent):
    """ImageStudent trunk + a rectified-flow head over a ``chunk``-step plan.

    ``vel_net`` predicts the denoising velocity field v(x_t, t | h) in
    normalized action-chunk space — the straight noise->data direction of
    rectified flow, not joint-space velocity. act() Euler-integrates it from
    N(0, I) and returns a (B, chunk, act_dim) plan of absolute joint targets;
    the gripper stays a smooth [0, 1] value (clamped, never thresholded).
    Action normalization stats live in buffers so a checkpoint stays
    self-contained.
    """

    def __init__(self, proprio_dim: int, act_dim: int, n_views: int, hw: int,
                 act_low: np.ndarray, act_high: np.ndarray,
                 encoder: str = "shared", hidden: int = 256, feat_dim: int = 64,
                 chunk: int = 50, flow_steps: int = 10):
        super().__init__(proprio_dim, act_dim, n_views, hw, act_low, act_high,
                         encoder, hidden, feat_dim)
        del self.head  # the flow head replaces the direct regression head
        self.chunk, self.flow_steps, self.act_dim = chunk, flow_steps, act_dim
        d = chunk * act_dim
        self.vel_net = nn.Sequential(
            nn.Linear(hidden + d + 16, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d),
        )
        self.act_mean = nn.Buffer(torch.zeros(act_dim))
        self.act_std = nn.Buffer(torch.ones(act_dim))

    def set_act_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.act_mean.copy_(mean)
        self.act_std.copy_(std.clamp_min(1e-6))

    def velocity(self, h: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """h: (N, hidden); x: (N, chunk*act_dim) normalized; t: (N, 1)."""
        return self.vel_net(torch.cat([h, x, time_features(t)], dim=-1))

    def forward(self, rgb: torch.Tensor, prop: torch.Tensor,
                x_t: torch.Tensor, t: torch.Tensor):
        """Training forward: one call touches every parameter, so the DDP wrap
        syncs gradients (calling features/velocity on the raw module would
        bypass it). Returns (v_pred, aux_pred, h)."""
        h = self.features(rgb, prop)
        return self.velocity(h, x_t, t), self.aux_head(h), h

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """Integrate noise -> plan; returns (N, chunk, act_dim) in action units."""
        n = h.shape[0]
        x = torch.randn(n, self.chunk * self.act_dim, device=h.device)
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full((n, 1), k * dt, device=h.device)
            x = x + dt * self.velocity(h, x, t)
        a = x.reshape(n, self.chunk, self.act_dim) * self.act_std + self.act_mean
        return a.clamp(self.act_low, self.act_high)

    @torch.no_grad()
    def act(self, obs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        prop, rgb = obs
        device = self.prop_mean.device
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float() / 255.0 - 0.5
        p = torch.from_numpy(np.asarray(prop, dtype=np.float32)).to(device)
        return self.sample(self.features(x, p)).cpu().numpy()
