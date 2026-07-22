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


class _FlowHead:
    """Rectified-flow chunk head shared by the image and state flow students.

    ``flow_net`` predicts the clean action chunk x1 (the flow-path endpoint) in
    normalized action-chunk space, conditioned on the noised chunk x_t, the flow
    time t, and the trunk hidden ``h`` — endpoint (x1) prediction, not velocity
    prediction. The two parametrize the same noise->data probability-flow ODE,
    but the velocity target (a - eps) carries irreducible eps-noise that is
    stiff to fit near t=1 and collapses training to the chunk mean; the endpoint
    target is the clean chunk, so it fits like a plain regression. sample()
    Euler-integrates the induced velocity (x1 - x)/(1 - t) from N(0, I) and
    returns a (B, chunk, act_dim) plan of absolute joint targets; the gripper
    stays a smooth [0, 1] value (clamped, never thresholded). Action
    normalization stats live in buffers so a checkpoint stays self-contained.
    Mixed into a student that already owns ``act_low`` / ``act_high`` buffers
    and supplies the trunk hidden vector ``h``.
    """

    def _build_flow_head(self, hidden: int, act_dim: int,
                         chunk: int, flow_steps: int) -> None:
        self.chunk, self.flow_steps, self.act_dim = chunk, flow_steps, act_dim
        d = chunk * act_dim
        self.flow_net = nn.Sequential(
            nn.Linear(hidden + d + 16, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d),
        )
        self.act_mean = nn.Buffer(torch.zeros(act_dim))
        self.act_std = nn.Buffer(torch.ones(act_dim))

    def set_act_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.act_mean.copy_(mean)
        self.act_std.copy_(std.clamp_min(1e-6))

    def predict_x1(self, h: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicted clean chunk x1 given the noised chunk x_t and time t.
        h: (N, hidden); x: (N, chunk*act_dim) normalized; t: (N, 1)."""
        return self.flow_net(torch.cat([h, x, time_features(t)], dim=-1))

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """Integrate noise -> plan; returns (N, chunk, act_dim) in action units."""
        n = h.shape[0]
        x = torch.randn(n, self.chunk * self.act_dim, device=h.device)
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full((n, 1), k * dt, device=h.device)
            # velocity of the linear path from the endpoint estimate; the last
            # step (t = 1 - dt) lands exactly on x1, so 1 - t stays >= dt > 0
            x = x + dt * (self.predict_x1(h, x, t) - x) / (1.0 - k * dt)
        a = x.reshape(n, self.chunk, self.act_dim) * self.act_std + self.act_mean
        return a.clamp(self.act_low, self.act_high)


class FlowImageStudent(_FlowHead, ImageStudent):
    """ImageStudent trunk + the rectified-flow chunk head (see ``_FlowHead``).

    act() returns a (B, chunk, act_dim) plan the collector executes
    receding-horizon.
    """

    def __init__(self, proprio_dim: int, act_dim: int, n_views: int, hw: int,
                 act_low: np.ndarray, act_high: np.ndarray,
                 encoder: str = "shared", hidden: int = 256, feat_dim: int = 64,
                 chunk: int = 50, flow_steps: int = 10):
        super().__init__(proprio_dim, act_dim, n_views, hw, act_low, act_high,
                         encoder, hidden, feat_dim)
        del self.head  # the flow head replaces the direct regression head
        self._build_flow_head(hidden, act_dim, chunk, flow_steps)

    def forward(self, rgb: torch.Tensor, prop: torch.Tensor,
                x_t: torch.Tensor, t: torch.Tensor):
        """Training forward: one call touches every parameter, so the DDP wrap
        syncs gradients (calling features/predict_x1 on the raw module would
        bypass it). Returns (x1_pred, aux_pred, h)."""
        h = self.features(rgb, prop)
        return self.predict_x1(h, x_t, t), self.aux_head(h), h

    @torch.no_grad()
    def act(self, obs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        prop, rgb = obs
        device = self.prop_mean.device
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float() / 255.0 - 0.5
        p = torch.from_numpy(np.asarray(prop, dtype=np.float32)).to(device)
        return self.sample(self.features(x, p)).cpu().numpy()


class _Block(nn.Module):
    """Pre-norm transformer block; with ``cond_dim`` it becomes a DiT-style
    AdaLN-zero block (shift/scale/gate from the conditioning vector, modulation
    zero-initialized so conditioning phases in from identity)."""

    def __init__(self, dim: int, heads: int, cond_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=cond_dim == 0)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=cond_dim == 0)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.ada = None
        if cond_dim:
            self.ada = nn.Linear(cond_dim, 6 * dim)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if self.ada is None:
            h = self.norm1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))
        s1, g1, a1, s2, g2, a2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1) + s1
        x = x + a1 * self.attn(h, h, h, need_weights=False)[0]
        return x + a2 * self.mlp(self.norm2(x) * (1 + g2) + s2)


class ViTEncoder(nn.Module):
    """ViT-Tiny per the LeWM recipe: 12 layers, 3 heads, dim 192, [CLS] token
    embedding -> 1-layer MLP projection with BatchNorm. Patch size adapts to
    the frame (their 14 doesn't tile 64px; 8 gives an 8x8 grid)."""

    def __init__(self, hw: int, dim: int = 192, depth: int = 12, heads: int = 3,
                 patch: int = 8, proj_dim: int = 192):
        super().__init__()
        assert hw % patch == 0, f"patch {patch} must tile {hw}px frames"
        n_tok = (hw // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        # BN'd projection off the [CLS] token (the paper's anti-collapse
        # plumbing; kept for checkpoint compatibility with LeWM encoders)
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.BatchNorm1d(proj_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) float -> (N, proj_dim)."""
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.proj(self.norm(t[:, 0]))


class ViTFlowImageStudent(_FlowHead, nn.Module):
    """ViT-T encoder per view + a DiT-style flow transformer over the action
    chunk (LeWM-predictor-sized: 6 layers, 16 heads, dim 320, dropout 0.1,
    AdaLN-zero conditioning on trunk features + flow time). Same training and
    act() surface as FlowImageStudent: forward(rgb, prop, x_t, t) ->
    (x1_pred, aux, h); act() returns a (B, chunk, act_dim) plan."""

    def __init__(self, proprio_dim: int, act_dim: int, n_views: int, hw: int,
                 act_low: np.ndarray, act_high: np.ndarray,
                 chunk: int = 50, flow_steps: int = 10,
                 flow_dim: int = 320, flow_depth: int = 6, flow_heads: int = 16,
                 dropout: float = 0.1):
        nn.Module.__init__(self)
        self.n_views = n_views
        self.encoder = ViTEncoder(hw)
        proj = self.encoder.proj[0].out_features
        self.view_emb = nn.Parameter(torch.zeros(n_views, proj))
        self.prop_net = nn.Sequential(nn.Linear(proprio_dim, 128), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(n_views * proj + 128, 2 * flow_dim), nn.ReLU(),
            nn.Linear(2 * flow_dim, flow_dim),
        )
        self.aux_head = nn.Linear(flow_dim, 5)
        self.prop_mean = nn.Buffer(torch.zeros(proprio_dim))
        self.prop_std = nn.Buffer(torch.ones(proprio_dim))
        self.act_low = nn.Buffer(torch.as_tensor(act_low, dtype=torch.float32))
        self.act_high = nn.Buffer(torch.as_tensor(act_high, dtype=torch.float32))
        # flow transformer replaces _FlowHead's MLP: chunk steps are tokens
        self.chunk, self.flow_steps, self.act_dim = chunk, flow_steps, act_dim
        self.tok_in = nn.Linear(act_dim, flow_dim)
        self.tok_pos = nn.Parameter(torch.zeros(1, chunk, flow_dim))
        nn.init.trunc_normal_(self.tok_pos, std=0.02)
        self.t_embed = nn.Sequential(nn.Linear(16, flow_dim), nn.SiLU(),
                                     nn.Linear(flow_dim, flow_dim))
        self.flow_blocks = nn.ModuleList(
            _Block(flow_dim, flow_heads, cond_dim=flow_dim, dropout=dropout)
            for _ in range(flow_depth))
        self.flow_norm = nn.LayerNorm(flow_dim, elementwise_affine=False)
        self.flow_ada = nn.Linear(flow_dim, 2 * flow_dim)
        self.flow_out = nn.Linear(flow_dim, act_dim)
        nn.init.zeros_(self.flow_ada.weight)
        nn.init.zeros_(self.flow_ada.bias)
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)
        self.act_mean = nn.Buffer(torch.zeros(act_dim))
        self.act_std = nn.Buffer(torch.ones(act_dim))

    def set_obs_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.prop_mean.copy_(mean)
        self.prop_std.copy_(std.clamp_min(1e-6))

    def features(self, rgb: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """rgb: (N, V, 3, H, W) float in [-0.5, 0.5]; prop: (N, P) raw."""
        n = rgb.shape[0]
        f = self.encoder(rgb.reshape(n * self.n_views, *rgb.shape[2:]))
        f = f.reshape(n, self.n_views, -1) + self.view_emb
        p = self.prop_net((prop - self.prop_mean) / self.prop_std)
        return self.trunk(torch.cat([f.reshape(n, -1), p], dim=-1))

    def predict_x1(self, h: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = h + self.t_embed(time_features(t))
        tok = self.tok_in(x.reshape(-1, self.chunk, self.act_dim)) + self.tok_pos
        for blk in self.flow_blocks:
            tok = blk(tok, c)
        s, g = self.flow_ada(c).unsqueeze(1).chunk(2, dim=-1)
        tok = self.flow_out(self.flow_norm(tok) * (1 + g) + s)
        return tok.reshape(-1, self.chunk * self.act_dim)

    def forward(self, rgb: torch.Tensor, prop: torch.Tensor,
                x_t: torch.Tensor, t: torch.Tensor):
        """One call touches every parameter (DDP grad sync)."""
        h = self.features(rgb, prop)
        return self.predict_x1(h, x_t, t), self.aux_head(h), h

    @torch.no_grad()
    def act(self, obs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # BN and dropout live in this net (unlike the CNN student): inference
        # must run in eval mode or plans get batch-stat drift + dropout noise
        was_training = self.training
        self.eval()
        try:
            prop, rgb = obs
            device = self.prop_mean.device
            x = torch.from_numpy(
                np.ascontiguousarray(rgb)).to(device).float() / 255.0 - 0.5
            p = torch.from_numpy(np.asarray(prop, dtype=np.float32)).to(device)
            return self.sample(self.features(x, p)).cpu().numpy()
        finally:
            self.train(was_training)


class FlowStateStudent(_FlowHead, Student):
    """State-obs MLP trunk + the rectified-flow chunk head (see ``_FlowHead``).

    The same flow head as ``FlowImageStudent``, but fed by an MLP over the flat
    privileged state instead of the CNN+proprio trunk. act() returns a
    (B, chunk, act_dim) plan of absolute joint targets the collector executes
    receding-horizon — identical surface to the image flow student minus rgb.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 act_low: np.ndarray, act_high: np.ndarray,
                 chunk: int = 50, flow_steps: int = 10):
        super().__init__(obs_dim, act_dim, hidden, act_low, act_high)
        del self.net  # the flow trunk + head replace the direct regression net
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self._build_flow_head(hidden, act_dim, chunk, flow_steps)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (N, obs_dim) raw -> (N, hidden) trunk features."""
        return self.trunk((obs - self.obs_mean) / self.obs_std)

    def forward(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Training forward (one call touches every parameter for DDP).
        Returns (x1_pred, h)."""
        h = self.features(obs)
        return self.predict_x1(h, x_t, t), h

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        device = self.obs_mean.device
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        return self.sample(self.features(x)).cpu().numpy()
