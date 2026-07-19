"""SimpleDAgger on the xsim.suite Lift task: distill a teacher into a student.

Port of imitation's SimpleDAggerTrainer loop to the suite's batched Genesis envs
(imitation itself pins gymnasium 0.29 and can't share an env with Genesis). Each
round rolls out a per-env beta-mixture of teacher and student, labels every
visited state with the teacher's action, aggregates the dataset, and continues
BC training. Round 0 is pure teacher (plain BC warm-up); beta ramps linearly to
``beta_floor`` over ``beta_rampdown_rounds``. The waypoint teacher is
closed-loop: act() re-anchors to the measured EE pose, so its labels are valid
corrections from wherever the student wandered.

Two students (``--policy``):
- ``state``: MLP on the GymWrapper's flat privileged obs -> absolute [j0..j6, g]
  (b4096-v7: 93% on the 15cm box).
- ``image``: CNN on ``rgb`` (B, V, 3, H, W) from all cameras + proprio MLP
  (no cube state) -> joints + gripper logit, with an auxiliary cube pos+yaw
  head. Rides the madrona batch renderer (img-v2 trained at B=256); frames
  come composited over live splat backgrounds with per-reset camera
  randomization (arena defaults).

Three teachers (``--teacher``):
- ``waypoint``: the scripted LiftPolicy (closed-loop anchor, but segment pacing
  rides a shared clock — labels are NOT a function of state; unfittable from
  randomized starts without vel, see img-v8).
- ``expert``: LiftExpertPolicy — reactive FSM, phase from live state, tracks a
  fumbled cube and recovers. The label is a function of observable state.
- ``mlp:<ckpt.pt>``: a frozen state-mode Student checkpoint (labels any state,
  no clock; capped at that policy's own success rate).

Two loss schemes (``--loss``), orthogonal to the obs mode:
- ``mse``: per-step regression (the students above, unchanged).
- ``flow`` (image student, v9): rectified-flow matching over a ``--chunk``-step
  plan of absolute joint actions, executed receding-horizon (``--replan``).
  Chunk labels are the per-step teacher labels stitched along the visited
  trajectory (an approximation under beta-mixing); the gripper stays a smooth
  [0, 1] value.

    uv run python scripts/simpledagger.py --n-envs 4096 --batch-size 4096
    uv run python scripts/simpledagger.py --policy image --teacher mlp:outputs/dagger/b4096-v7/best.pt
    uv run python scripts/simpledagger.py --policy image --loss flow --teacher mlp:outputs/dagger/b4096-v7/best.pt
    # multi-GPU: one rank per GPU, each owning n_envs envs AND a DDP replica
    # (Genesis binds one GPU per process, so ranks collect and train symmetrically)
    uv run torchrun --standalone --nproc-per-node=4 scripts/simpledagger.py --policy image --n-envs 2048
    # multi-host: torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=host0:29500 ...
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tyro
from rich import print

from xsim.suite.algo.distributed import Distributed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # run
    rounds: int = 10
    seed: int = 0
    exp_name: str = "default"
    out: Path = PROJECT_ROOT / "outputs" / "dagger"
    # dagger
    episodes_per_round: int = 1       # env-batch rollouts per round (n_envs episodes each)
    beta_rampdown_rounds: int = 5     # beta = max(floor, 1 - round/rampdown)
    # never collect with the pure student: beta=0 collection poisoned the aggregate
    # (b4096: 81% at beta=0.2, then 0% after beta=0)
    beta_floor: float = 0.2
    steps_per_segment: int = 20       # waypoint teacher pacing
    # teacher: "waypoint" (scripted LiftPolicy) or "mlp:<checkpoint.pt>" (frozen
    # state-mode Student; labels any state, no clock)
    teacher: str = "waypoint"
    # bc
    epochs_per_round: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    # per-round cosine decay lr -> lr_final: late rounds (largest dataset, best
    # policy) take refining steps instead of reshaping ones
    lr_final: float = 1e-4
    hidden_dim: int = 256
    # eval/checkpoint an EMA of the student: the live net swings round-to-round
    # (b4096-v3 eval oscillated 2%-82% at stable BC loss)
    ema_decay: float = 0.999
    # student
    policy: Literal["state", "image"] = "state"
    image_hw: int = 64                # square rgb obs size (64 trains fast)
    encoder: Literal["shared", "separate"] = "shared"  # one CNN + per-view embedding, or V CNNs
    feat_dim: int = 64                # per-view feature size
    aux_pose_coef: float = 0.1        # aux cube pos+yaw loss weight (0 = off; sim-only supervision)
    grip_coef: float = 0.1            # BCE weight on the gripper logit (mse loss only; flow keeps grip continuous)
    aug_pad: int = 4                  # DrQ random-shift padding, px (0 = off)
    # loss scheme (orthogonal to --policy; flow currently rides the image student)
    loss: Literal["mse", "flow"] = "mse"
    chunk: int = 50                   # flow: action-chunk length (stitched teacher labels)
    replan: int = 10                  # flow: student actions executed per inference (<= chunk)
    flow_steps: int = 10              # flow: Euler steps integrating the denoising velocity at act()
    # record every kth visited state in image mode (labels stay per-step for
    # control; 1 = keep every frame — the aggregate lives on disk, not RAM)
    frame_stride: int = 1
    # image-mode dataset store: flat-binary memmap per key under
    # <data_dir>/<exp_name> (~36KB/sample rgb at 3x64x64 -> ~5.5GB/round at
    # B=1024 stride 1; point at the big NVMe, not /home)
    data_dir: Path = Path("/data/fast/xarm-dagger")
    num_workers: int = 8              # BC DataLoader workers (0 = main process)
    # legacy PNG background plates (scripts/make_plates.py); None = the arena's
    # own live splat compositing in render_views, which also covers the wrist cam
    plates_dir: Path | None = None
    # madrona rasterizer instead of the raytracer; the raytracer is the suite
    # default and the domain the realigned splat/cameras target (rasterizer =
    # img-v2's light-calibrated legacy domain)
    batch_rasterizer: bool = False
    # eval
    eval_batches: int = 1             # student-only eval rollouts per round (n_envs each)
    eval_seed: int = 51_000
    eval_video: bool = True           # tile the first eval rollout (image mode) -> eval_rNN.mp4
    # wandb (rank 0 only under torchrun): every log() line streams as
    # <kind>/<key>, eval videos attach per round; --wandb-project None = off
    wandb_project: str | None = "xarm-sim"
    # env. n_envs is PER RANK: under torchrun each rank owns n_envs envs and a
    # DDP replica, so a rollout visits world_size * n_envs envs (and batch_size
    # is likewise per rank). Launch via torchrun for multi-GPU; no flag needed.
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    # EE-pose actions via CartesianActionWrapper (wrapper owns IK). Joint space is
    # the performance default: v7 (joints) 93% vs v6b (poses, quat-canonical) 73%
    # at equal budget.
    cartesian: bool = False
    # cube spawn: x 200-400mm, y +-1ft minus the cube half-extent (table is
    # exactly 2ft wide, y +-0.3048; +-0.288 keeps the 1.25in cube fully on it)
    cube_x_range: tuple[float, float] = (0.20, 0.40)
    cube_y_range: tuple[float, float] = (-0.288, 0.288)
    # per-reset arm start: TCP uniform in this box (x 100-400mm, y +-1ft,
    # z table-top..300mm), home orientation, IK-seated; None = home pose only
    init_tcp_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = (
        (0.10, 0.40), (-0.3048, 0.3048), (-0.01, 0.30)
    )
    # per-reset exo-cam + wrist-mount pose resampling (arena default; pin for
    # fixed-rig eval/play)
    randomize_cameras: bool = True
    horizon: int = 200
    control_freq: float = 30.0
    noslip_iterations: int = 10
    # play mode: load a checkpoint, roll the student out once (seeded, nyx-rendered),
    # write an all-envs grid video (green border = success tick, red = timeout) and
    # a per-env spawn/outcome table; no training
    play: Path | None = None
    play_video: Path | None = None    # default: <checkpoint dir>/rollout.mp4
    cameras: tuple[str, ...] = ("low", "side", "wrist")
    spp: int = 8
    video_max_width: int = 1280       # per-camera grid width cap, px


def image_proprio_keys(keys) -> list[str]:
    """Image-student proprio: non-privileged keys, minus joint velocities.
    Sim velocity profiles (Genesis PD at 30 Hz) don't transfer to the real
    arm — the img-v5 IRL sensitivity probe showed vel as the strongest input —
    so the image student never sees them. State-mode students/teachers keep
    the full flat obs including vel."""
    return [k for k in keys if "cube" not in k and "vel" not in k]


def flat(obs: dict, keys: list[str]) -> np.ndarray:
    """Concatenate dict obs values for ``keys`` into (n_envs, D) float32."""
    b = np.asarray(obs[keys[0]]).shape[0]
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).reshape(b, -1) for k in keys], axis=-1
    )


def aux_targets(obs: dict) -> np.ndarray:
    """Privileged aux regression targets: cube pos + (sin, cos) of 4*yaw
    (the cube has 90-degree rotational symmetry)."""
    q = np.asarray(obs["cube_quat"], dtype=np.float32)
    yaw4 = 4.0 * 2.0 * np.arctan2(q[:, 3], q[:, 0])
    return np.concatenate(
        [np.asarray(obs["cube_pos"], dtype=np.float32),
         np.sin(yaw4)[:, None], np.cos(yaw4)[:, None]], axis=-1
    )


class MemmapStore:
    """Append-only on-disk dataset: one flat <key>.bin per key + manifest.json.

    Rows are fixed-size, so sample i of a key is one memmap slice; the OS page
    cache is the only caching layer. Single writer, append per recorded step,
    flush() publishes the manifest for readers.
    """

    def __init__(self, root: Path):
        import shutil

        self.root = root
        shutil.rmtree(root, ignore_errors=True)  # fresh run owns its dir
        root.mkdir(parents=True, exist_ok=True)
        self._fh: dict = {}
        self.meta: dict[str, dict] = {}

    def append(self, key: str, arr: np.ndarray) -> None:
        meta = self.meta.setdefault(
            key, {"dtype": str(arr.dtype), "shape": list(arr.shape[1:]), "n": 0})
        if key not in self._fh:
            self._fh[key] = (self.root / f"{key}.bin").open("ab")
        arr.tofile(self._fh[key])
        meta["n"] += arr.shape[0]

    def flush(self) -> None:
        for fh in self._fh.values():
            fh.flush()
        (self.root / "manifest.json").write_text(json.dumps(self.meta))

    def reader(self, key: str) -> np.memmap:
        meta = self.meta[key]
        return np.memmap(self.root / f"{key}.bin", dtype=meta["dtype"],
                         mode="r", shape=(meta["n"], *meta["shape"]))

    def __len__(self) -> int:
        return self.meta["act"]["n"] if "act" in self.meta else 0


class MemmapDataset(torch.utils.data.Dataset):
    """Random-access snapshot of a MemmapStore. Holds only the root path and
    manifest, so it pickles cheaply into spawned DataLoader workers; each
    worker opens its own memmap handles on first use."""

    def __init__(self, root: Path, keys: tuple[str, ...]):
        self.root, self.keys = root, keys
        self.meta = json.loads((root / "manifest.json").read_text())
        self.n = self.meta[keys[0]]["n"]
        self._maps: dict[str, np.memmap] | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self._maps is None:
            self._maps = {
                k: np.memmap(self.root / f"{k}.bin", dtype=m["dtype"], mode="r",
                             shape=(m["n"], *m["shape"]))
                for k in self.keys if (m := self.meta[k])
            }
        return tuple(torch.from_numpy(np.array(self._maps[k][i])) for k in self.keys)


class Student(nn.Module):
    """MLP over the GymWrapper's flat obs -> absolute [j0..j6, g] action.

    Obs normalization stats and action limits live in buffers so a checkpoint
    is self-contained.
    """

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
        """Fused trunk feature. rgb: (N, V, 3, H, W) float in [-0.5, 0.5];
        prop: (N, P) raw."""
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


class MLPTeacher:
    """Frozen state-mode Student checkpoint as a DAgger teacher. Structure is
    inferred from the checkpoint; input is the sorted-key flat state vector
    (GymWrapper's layout)."""

    def __init__(self, ckpt: Path, device: torch.device):
        sd = torch.load(ckpt, map_location="cpu")
        obs_dim, hidden = sd["net.0.weight"].shape[1], sd["net.0.weight"].shape[0]
        act_dim = sd["net.4.weight"].shape[0]
        self.net = Student(obs_dim, act_dim, hidden,
                           sd["act_low"].numpy(), sd["act_high"].numpy())
        self.net.load_state_dict(sd)
        self.net.eval().to(device)

    def reset(self, obs=None) -> None:
        pass

    def act(self, state_flat: np.ndarray) -> np.ndarray:
        return self.net.act(state_flat)


def build_env(cfg: Config, render: bool = False):
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.renderers import BatchConfig, NyxConfig
    from xsim.suite.wrappers import CartesianActionWrapper, GymWrapper, ImageObsWrapper

    image = cfg.policy == "image"
    with_cams = render or image
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7",
        camera_names=list(cfg.cameras) if with_cams else [],
        camera_res=(cfg.image_hw, cfg.image_hw) if image else (640, 480),
        # image obs ride the madrona batch renderer (~100k env-cam fps at 64px);
        # nyx stays the photo-real path for state-mode play videos
        render_backend="batch" if image else ("nyx" if with_cams else "raster"),
        renderer_config=(BatchConfig(use_rasterizer=cfg.batch_rasterizer) if image
                         else NyxConfig(spp=cfg.spp) if with_cams else None),
        x_range=cfg.cube_x_range, y_range=cfg.cube_y_range,
        init_tcp_box=cfg.init_tcp_box,
        randomize_cameras=cfg.randomize_cameras,
        horizon=cfg.horizon, n_envs=cfg.n_envs,
        control_freq=cfg.control_freq,
        noslip_iterations=cfg.noslip_iterations,
    )
    if cfg.cartesian:
        env = CartesianActionWrapper(env)
    if image:
        plates = None
        if cfg.plates_dir is not None:
            plates = {p.stem: p for p in sorted(cfg.plates_dir.glob("*.png"))}
        return ImageObsWrapper(env, plates=plates)  # dict obs; trainer flattens per key set
    return GymWrapper(env)


def env_spec(cfg: Config, env) -> dict:
    """The picklable space facts build_student needs, so collector shards can
    ship them to a trainer process that never builds an env."""
    act_space = env.get_wrapper_attr("single_action_space")
    spec = {"act_dim": act_space.shape[0],
            "act_low": act_space.low, "act_high": act_space.high}
    if cfg.policy == "state":
        spec["obs_dim"] = env.single_observation_space.shape[0]
    else:
        base_spaces = env.unwrapped.single_observation_space.spaces
        proprio_keys = sorted(k for k in base_spaces if "cube" not in k)
        spec["proprio_dim"] = sum(
            int(np.prod(base_spaces[k].shape)) for k in proprio_keys)
        spec["n_views"] = len(env.views)
    return spec


def build_student(cfg: Config, spec_or_env, device: torch.device):
    spec = spec_or_env if isinstance(spec_or_env, dict) else env_spec(cfg, spec_or_env)
    if cfg.policy == "state":
        return Student(
            obs_dim=spec["obs_dim"], act_dim=spec["act_dim"], hidden=cfg.hidden_dim,
            act_low=spec["act_low"], act_high=spec["act_high"],
        ).to(device)
    kwargs = dict(
        proprio_dim=spec["proprio_dim"], act_dim=spec["act_dim"],
        n_views=spec["n_views"], hw=cfg.image_hw,
        act_low=spec["act_low"], act_high=spec["act_high"],
        encoder=cfg.encoder, hidden=cfg.hidden_dim, feat_dim=cfg.feat_dim,
    )
    if cfg.loss == "flow":
        return FlowImageStudent(**kwargs, chunk=cfg.chunk,
                                flow_steps=cfg.flow_steps).to(device)
    return ImageStudent(**kwargs).to(device)


# ---------------------------------------------------------------------------------------
# video tiling (eval rollouts + play mode)
# ---------------------------------------------------------------------------------------

BORDER = {0: (150, 150, 150), 1: (0, 200, 0), 2: (220, 30, 30)}  # live/success/fail


def _grid(frames: np.ndarray, status: np.ndarray, max_width: int) -> np.ndarray:
    """(B,H,W,3) -> near-square grid, tiles bordered by per-env status."""
    import cv2

    b, h, w, _ = frames.shape
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)
    tw = max(2, max_width // cols) // 2 * 2
    th = max(2, round(h * tw / w)) // 2 * 2
    interp = cv2.INTER_AREA if tw <= w else cv2.INTER_NEAREST
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    t = max(2, tw // 64)
    for i in range(b):
        r, c = divmod(i, cols)
        tile = cv2.resize(frames[i], (tw, th), interpolation=interp)
        tile[:t], tile[-t:], tile[:, :t], tile[:, -t:] = (BORDER[int(status[i])],) * 4
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
    return canvas


class VideoSink:
    """Streamed h264 mp4 writer: RGB uint8 frames piped to an ffmpeg
    subprocess, opened lazily on the first frame so callers never hold a full
    rollout of grids in RAM. h264 (not cv2's mp4v) so the wandb web UI can
    play the video inline."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps = path, fps
        self._p = None

    def add(self, frame: np.ndarray) -> None:
        if self._p is None:
            import shutil
            import subprocess

            exe = shutil.which("ffmpeg")
            if exe is None:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._p = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.fps}",
                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 str(self.path)],
                stdin=subprocess.PIPE)
        self._p.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._p is not None:
            self._p.stdin.close()
            self._p.wait()


def read_key(root: Path, key: str) -> np.memmap:
    """Reader for one key of a flushed MemmapStore dir (works across processes:
    only the manifest and bin file are touched)."""
    meta = json.loads((root / "manifest.json").read_text())[key]
    return np.memmap(root / f"{key}.bin", dtype=meta["dtype"], mode="r",
                     shape=(meta["n"], *meta["shape"]))


class Collector:
    """One process's collection stack: env + teacher + recording.

    The single-process trainer owns one directly; with --collect-gpus each
    spawned _collect_worker owns one pinned to its GPU.
    """

    def __init__(self, cfg: Config, store_root: Path | None, seed_offset: int = 0):
        from xsim.suite.policies import LiftPolicy

        self.cfg = cfg
        self.image = cfg.policy == "image"
        self.flow = cfg.loss == "flow"
        if self.flow:
            if not self.image:
                raise ValueError("loss='flow' currently rides the image student")
            if cfg.frame_stride != 1:
                raise ValueError("flow chunk labels need every step: frame_stride=1")
            if not 1 <= cfg.replan <= cfg.chunk:
                raise ValueError("need 1 <= replan <= chunk")
        self.env = build_env(cfg)
        base = self.env.unwrapped
        self.state_keys = sorted(base.single_observation_space.spaces)
        self.proprio_keys = image_proprio_keys(self.state_keys)
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
        self.spec = env_spec(cfg, self.env)
        if cfg.teacher == "waypoint":
            self.teacher = LiftPolicy(
                base, steps_per_segment=cfg.steps_per_segment, cartesian=cfg.cartesian)
        elif cfg.teacher == "expert":
            from xsim.suite.policies import LiftExpertPolicy

            self.teacher = LiftExpertPolicy(base, cartesian=cfg.cartesian)
        elif cfg.teacher.startswith("mlp:"):
            self.teacher = MLPTeacher(Path(cfg.teacher[4:]), self.device)
        else:
            raise ValueError(f"unknown teacher {cfg.teacher!r}")
        self.rng = np.random.default_rng(cfg.seed + seed_offset)
        self.store = MemmapStore(store_root) if (self.image and store_root) else None
        self._chunks: dict[str, list[np.ndarray]] = {}

    # -- obs adapters --------------------------------------------------------------
    def _teacher_obs(self, obs):
        """Sorted-key flat state (the waypoint teacher ignores it)."""
        return flat(obs, self.state_keys) if self.image else obs

    def _student_obs(self, obs):
        if self.image:
            return flat(obs, self.proprio_keys), obs["rgb"]
        return obs

    def _record(self, obs, teacher_a: np.ndarray, live: np.ndarray) -> None:
        if self.image:
            prop, rgb = self._student_obs(obs)
            self.store.append("act", teacher_a[live].astype(np.float32))
            self.store.append("prop", np.ascontiguousarray(prop[live]))
            self.store.append("rgb", np.ascontiguousarray(rgb[live]))
            self.store.append("aux", np.ascontiguousarray(aux_targets(obs)[live]))
        else:
            self._chunks.setdefault("act", []).append(
                teacher_a[live].astype(np.float32, copy=True))
            self._chunks.setdefault("obs", []).append(
                obs[live].astype(np.float32, copy=True))

    def pop_chunks(self) -> dict[str, np.ndarray] | None:
        """Drain state-mode recordings (the shard->trainer pipe payload)."""
        if not self._chunks:
            return None
        out = {k: np.concatenate(v) for k, v in self._chunks.items()}
        self._chunks.clear()
        return out

    def _append_chunk_labels(self, acts: np.ndarray, lives: np.ndarray) -> None:
        """Stitched chunk labels: for each recorded row (env e live at tick t),
        the next ``chunk`` per-step teacher labels along the visited trajectory,
        the final pre-death label repeated past episode end (hold pose). Rows
        append in _record's tick-major live-masked order, so chunk.bin stays
        row-aligned with rgb.bin. acts: (T, B, A); lives: (T, B)."""
        last = lives.sum(axis=0) - 1  # (B,) each env's final live tick
        ar = np.arange(self.cfg.chunk)
        for t in range(acts.shape[0]):
            envs = np.flatnonzero(lives[t])
            idx = np.minimum(t + ar[None, :], last[envs, None])  # (n_live, chunk)
            self.store.append("chunk", acts[idx, envs[:, None]])

    def rollout(self, beta: float, record: bool, student: nn.Module,
                seed: int | None = None, video_path: Path | None = None) -> dict:
        """One synchronous env-batch episode; per-env Bernoulli(beta) picks the
        teacher's action over the student's — per step, or per replan window in
        flow mode, where the student emits a chunk executed receding-horizon.
        Every visited state is labeled with the teacher action when ``record``
        is on. ``video_path`` (image mode) streams the policy's own frames as a
        play-style bordered grid mp4."""
        env, B, cfg = self.env, self.cfg.n_envs, self.cfg
        obs, _ = env.reset(seed=seed)
        self.teacher.reset()
        live = np.ones(B, dtype=bool)
        success = np.zeros(B, dtype=bool)
        ep_len = np.zeros(B, dtype=np.int64)
        tick = 0
        plan = use_teacher = None
        acts_hist: list[np.ndarray] = []  # flow: per-tick teacher labels for stitching
        live_hist: list[np.ndarray] = []
        sink = status = None
        if video_path is not None:
            sink = VideoSink(video_path, 1.0 / env.unwrapped.control_dt)
            status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail

        def snap(o) -> None:
            rgb = o["rgb"].transpose(0, 1, 3, 4, 2)  # (B, V, H, W, 3)
            sink.add(np.concatenate(
                [_grid(rgb[:, i], status, cfg.video_max_width)
                 for i in range(rgb.shape[1])], axis=1))

        if sink is not None:
            snap(obs)
        while live.any():
            teacher_a = self.teacher.act(self._teacher_obs(obs))
            if beta >= 1.0:
                action = teacher_a
            else:
                if self.flow:
                    if tick % cfg.replan == 0:
                        plan = student.act(self._student_obs(obs))  # (B, chunk, A)
                        use_teacher = self.rng.random(B) < beta
                    student_a = plan[:, tick % cfg.replan]
                else:
                    student_a = student.act(self._student_obs(obs))
                    use_teacher = self.rng.random(B) < beta
                action = np.where(use_teacher[:, None], teacher_a, student_a)
            if record and (not self.image or tick % cfg.frame_stride == 0):
                self._record(obs, teacher_a, live)
                if self.flow:
                    acts_hist.append(teacher_a.astype(np.float32, copy=True))
                    live_hist.append(live.copy())
            tick += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            if sink is not None:
                status[live & done & info["success"]] = 1
                status[live & done & ~info["success"]] = 2
            success |= live & info["success"]
            ep_len += live
            live &= ~done
            if sink is not None:
                snap(obs)
        if sink is not None:
            sink.close()
        if record and self.flow:
            self._append_chunk_labels(np.stack(acts_hist), np.stack(live_hist))
        return {"success": float(success.mean()), "ep_len": float(ep_len.mean())}


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.image = cfg.policy == "image"
        self.flow = cfg.loss == "flow"
        self.work_dir = cfg.out / cfg.exp_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.work_dir / "metrics.jsonl"

        self.dist = Distributed()  # pins this rank's GPU; must precede gs.init
        self.wandb = None
        if cfg.wandb_project and self.dist.main:
            import wandb

            self.wandb = wandb.init(
                project=cfg.wandb_project, name=cfg.exp_name,
                config=json.loads(json.dumps(asdict(cfg), default=str)))
        store_base = cfg.data_dir / cfg.exp_name
        self.store_root = (store_base / f"shard_{self.dist.rank}"
                           if self.dist.enabled else store_base)
        # identical init weights on every rank (DDP requirement), then
        # per-rank env reset spacing and beta/shuffle streams
        torch.manual_seed(cfg.seed)
        self.collector = Collector(
            cfg, store_root=self.store_root if self.image else None,
            seed_offset=100003 * self.dist.rank)
        self.device = self.collector.device
        self.student = build_student(cfg, self.collector.spec, self.device)
        self.net = self.dist.wrap(self.student)  # BC fwd/bwd path
        self.ema = copy.deepcopy(self.student)
        self.optim = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        torch.manual_seed(cfg.seed + 7919 * self.dist.rank)  # loader/aug streams
        self.rng = np.random.default_rng(cfg.seed + 7919 * self.dist.rank)
        # aggregated DAgger dataset. image mode appends every recorded step to
        # this rank's on-disk MemmapStore; state mode stays in RAM (tiny rows).
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._stats_set = False
        self._act_stats_set = False
        self.best_success = -1.0
        self._start = time.time()

    def log(self, kind: str, d: dict) -> None:
        if not self.dist.main:
            return
        d = {"kind": kind, "elapsed_s": round(time.time() - self._start, 1), **d}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(d) + "\n")
        if self.wandb is not None:
            self.wandb.log({f"{kind}/{k}": v for k, v in d.items() if k != "kind"})
        pretty = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in d.items() if k != "kind")
        color = {"collect": "white", "bc": "cyan", "eval": "green"}.get(kind, "white")
        print(f"[{color}]\\[{kind}][/] {pretty}")

    # -- collection --------------------------------------------------------------
    def _rollouts(self, beta: float, record: bool, student: nn.Module,
                  seed: int | None = None, video_path: Path | None = None) -> dict:
        """One env-batch episode on this rank, stats averaged across ranks.
        Rank reset seeds are spaced so no two ranks replay the same spawns."""
        stats = self.collector.rollout(
            beta=beta, record=record, student=student,
            seed=None if seed is None else seed + 100003 * self.dist.rank,
            video_path=video_path)
        if record and not self.image:
            payload = self.collector.pop_chunks()
            if payload:
                for k, arr in payload.items():
                    self._chunks.setdefault(k, []).append(arr)
        return {"success": self.dist.mean(stats["success"]),
                "ep_len": self.dist.mean(stats["ep_len"])}

    # -- bc ----------------------------------------------------------------------
    def _freeze_stats(self, x: torch.Tensor) -> None:
        if not self._stats_set:
            self.student.set_obs_stats(*self.dist.obs_stats(x))
            self._stats_set = True

    def train_bc(self) -> dict:
        if not self.image:
            return self._train_bc_state()
        return self._train_bc_flow() if self.flow else self._train_bc_image()

    def _train_bc_state(self) -> dict:
        cfg = self.cfg
        X = torch.from_numpy(np.concatenate(self._chunks["obs"])).to(self.device)
        Y = torch.from_numpy(np.concatenate(self._chunks["act"])).to(self.device)
        self._freeze_stats(X)
        n = X.shape[0]
        steps = self.dist.min_int(math.ceil(n / cfg.batch_size))
        last_loss = 0.0
        for _ in range(cfg.epochs_per_round):
            perm = torch.randperm(n, device=self.device)
            losses = []
            for s in range(steps):
                idx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size]
                loss = F.mse_loss(self.net(X[idx]), Y[idx])
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                self._ema_update()
                losses.append(loss.item())
            last_loss = self.dist.mean(float(np.mean(losses)))
        samples = int(self.dist.mean(float(n)) * self.dist.world)
        return {"samples": samples, "bc_loss": last_loss}

    def _train_bc_image(self) -> dict:
        cfg = self.cfg
        self.collector.store.flush()
        ds = MemmapDataset(self.store_root, ("rgb", "prop", "act", "aux"))
        if not self._stats_set:
            self._freeze_stats(torch.from_numpy(
                np.asarray(read_key(self.store_root, "prop"))).to(self.device))
        g = torch.Generator().manual_seed(int(self.rng.integers(2**31)))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, generator=g,
            num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
            persistent_workers=cfg.num_workers > 0,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        )
        steps = self.dist.min_int(len(loader))
        stats = {"bc_loss": 0.0, "grip_bce": 0.0, "aux_mse": 0.0}
        for _ in range(cfg.epochs_per_round):
            sums = {k: 0.0 for k in stats}
            batches = 0
            # explicit cpu: gs.init makes cuda the torch default device, but the
            # loader's sampler/collate must build cpu tensors
            with torch.device("cpu"):
                for rgb, prop, y, aux_y in loader:
                    if batches >= steps:
                        break
                    x = rgb.to(self.device, non_blocking=True).float() / 255.0 - 0.5
                    prop = prop.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    aux_y = aux_y.to(self.device, non_blocking=True)
                    if cfg.aug_pad > 0:
                        nv = x.shape[0] * x.shape[1]
                        x = rand_shift(x.reshape(nv, *x.shape[2:]), cfg.aug_pad).reshape(x.shape)
                    pred, aux = self.net(x, prop)
                    joints = F.mse_loss(pred[:, :-1], y[:, :-1])
                    grip = F.binary_cross_entropy_with_logits(pred[:, -1], y[:, -1])
                    aux_l = F.mse_loss(aux, aux_y)
                    loss = joints + cfg.grip_coef * grip + cfg.aux_pose_coef * aux_l
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)
                    self._ema_update()
                    sums["bc_loss"] += joints.item()
                    sums["grip_bce"] += grip.item()
                    sums["aux_mse"] += aux_l.item()
                    batches += 1
            stats = {k: self.dist.mean(v / max(batches, 1)) for k, v in sums.items()}
        del loader  # release persistent workers before the next collect
        samples = int(self.dist.mean(float(len(ds))) * self.dist.world)
        return {"samples": samples, **stats}

    def _train_bc_flow(self) -> dict:
        cfg = self.cfg
        self.collector.store.flush()
        ds = MemmapDataset(self.store_root, ("rgb", "prop", "chunk", "aux"))
        if not self._stats_set:
            self._freeze_stats(torch.from_numpy(
                np.asarray(read_key(self.store_root, "prop"))).to(self.device))
        if not self._act_stats_set:
            a = torch.from_numpy(
                np.asarray(read_key(self.store_root, "act"))).to(self.device)
            self.student.set_act_stats(*self.dist.obs_stats(a))
            self._act_stats_set = True
        g = torch.Generator().manual_seed(int(self.rng.integers(2**31)))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, generator=g,
            num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
            persistent_workers=cfg.num_workers > 0,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        )
        steps = self.dist.min_int(len(loader))
        stats = {"flow_loss": 0.0, "a0_mse": 0.0, "aux_mse": 0.0}
        for _ in range(cfg.epochs_per_round):
            sums = {k: 0.0 for k in stats}
            batches = 0
            # explicit cpu: gs.init makes cuda the torch default device, but the
            # loader's sampler/collate must build cpu tensors
            with torch.device("cpu"):
                for rgb, prop, y, aux_y in loader:
                    if batches >= steps:
                        break
                    x = rgb.to(self.device, non_blocking=True).float() / 255.0 - 0.5
                    prop = prop.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    aux_y = aux_y.to(self.device, non_blocking=True)
                    if cfg.aug_pad > 0:
                        nv = x.shape[0] * x.shape[1]
                        x = rand_shift(x.reshape(nv, *x.shape[2:]), cfg.aug_pad).reshape(x.shape)
                    n = y.shape[0]
                    a = ((y - self.student.act_mean) / self.student.act_std).reshape(n, -1)
                    eps = torch.randn_like(a)
                    t = torch.rand(n, 1, device=self.device)
                    x_t = (1.0 - t) * eps + t * a
                    v_pred, aux, h = self.net(x, prop, x_t, t)
                    fm = F.mse_loss(v_pred, a - eps)
                    aux_l = F.mse_loss(aux, aux_y)
                    loss = fm + cfg.aux_pose_coef * aux_l
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)
                    self._ema_update()
                    with torch.no_grad():  # diagnostic comparable to mse bc_loss
                        a0 = self.student.sample(h.detach())[:, 0]
                        sums["a0_mse"] += F.mse_loss(a0, y[:, 0]).item()
                    sums["flow_loss"] += fm.item()
                    sums["aux_mse"] += aux_l.item()
                    batches += 1
            stats = {k: self.dist.mean(v / max(batches, 1)) for k, v in sums.items()}
        del loader  # release persistent workers before the next collect
        samples = int(self.dist.mean(float(len(ds))) * self.dist.world)
        return {"samples": samples, **stats}

    @torch.no_grad()
    def _ema_update(self) -> None:
        d = self.cfg.ema_decay
        for pe, p in zip(self.ema.parameters(), self.student.parameters()):
            pe.lerp_(p, 1.0 - d)
        for be, b in zip(self.ema.buffers(), self.student.buffers()):
            be.copy_(b)

    # -- loop --------------------------------------------------------------------
    def evaluate(self, rnd: int) -> dict:
        video = (self.work_dir / f"eval_r{rnd:02d}.mp4"
                 if self.image and self.cfg.eval_video and self.dist.main else None)
        stats = [self._rollouts(beta=0.0, record=False, student=self.ema,
                                seed=self.cfg.eval_seed + b,
                                video_path=video if b == 0 else None)
                 for b in range(self.cfg.eval_batches)]
        if video is not None and self.wandb is not None:
            import wandb

            self.wandb.log({"eval/video": wandb.Video(str(video), format="mp4")})
        return {
            "eval_success": float(np.mean([s["success"] for s in stats])),
            "eval_len": float(np.mean([s["ep_len"] for s in stats])),
        }

    def train(self) -> None:
        cfg = self.cfg
        for rnd in range(cfg.rounds):
            beta = max(cfg.beta_floor, 1.0 - rnd / cfg.beta_rampdown_rounds)
            frac = rnd / max(1, cfg.rounds - 1)
            lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1.0 + math.cos(math.pi * frac))
            for g in self.optim.param_groups:
                g["lr"] = lr
            for _ in range(cfg.episodes_per_round):
                stats = self._rollouts(beta=beta, record=True, student=self.student,
                                       seed=cfg.seed + rnd if rnd == 0 else None)
                self.log("collect", {"round": rnd, "beta": beta, **stats})
            self.log("bc", {"round": rnd, "lr": lr, **self.train_bc()})
            eval_metrics = self.evaluate(rnd)
            self.log("eval", {"round": rnd, **eval_metrics})
            if self.dist.main:
                torch.save(self.ema.state_dict(), self.work_dir / "latest.pt")
                # per-round snapshot so any training fraction can be replayed
                torch.save(self.ema.state_dict(),
                           self.work_dir / f"round_{rnd + 1:02d}.pt")
                if eval_metrics["eval_success"] >= self.best_success:
                    self.best_success = eval_metrics["eval_success"]
                    torch.save(self.ema.state_dict(), self.work_dir / "best.pt")
        self.dist.close()
        if self.wandb is not None:
            self.wandb.finish()
        if self.dist.main:
            print(f"\\[done] best eval success: {self.best_success:.0%}")


# ---------------------------------------------------------------------------------------
# play mode
# ---------------------------------------------------------------------------------------


def play(cfg: Config) -> None:
    env = build_env(cfg, render=True)
    B = cfg.n_envs
    image = cfg.policy == "image"
    device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
    student = build_student(cfg, env, device)
    student.load_state_dict(torch.load(cfg.play, map_location=device))
    student.eval()
    state_keys = sorted(env.unwrapped.single_observation_space.spaces)
    proprio_keys = image_proprio_keys(state_keys)

    obs, _ = env.reset(seed=cfg.eval_seed)
    spawn = np.asarray(env.unwrapped.cube.get_pos(), dtype=np.float64).copy()
    q = np.asarray(env.unwrapped.cube.get_quat(), dtype=np.float64)
    spawn_yaw = 2.0 * np.arctan2(q[:, 3], q[:, 0])
    status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail
    ep_len = np.zeros(B, dtype=np.int64)
    live = np.ones(B, dtype=bool)
    out = cfg.play_video or cfg.play.parent / "rollout.mp4"
    sink = VideoSink(out, 1.0 / env.unwrapped.control_dt)

    def snap(obs) -> None:
        if image:  # reuse the policy's own frames (image_hw px, upscaled)
            rgb = obs["rgb"].transpose(0, 1, 3, 4, 2)  # (B, V, H, W, 3)
            views = [rgb[:, i] for i in range(rgb.shape[1])]
        else:
            d = env.unwrapped.render_views(all_envs=True)
            views = [d[k] for k in sorted(d)]
        sink.add(np.concatenate(
            [_grid(v, status, cfg.video_max_width) for v in views], axis=1))

    snap(obs)
    flow = cfg.loss == "flow"
    tick, plan = 0, None
    while live.any():
        so = (flat(obs, proprio_keys), obs["rgb"]) if image else obs
        if flow:
            if tick % cfg.replan == 0:
                plan = student.act(so)
            a = plan[:, tick % cfg.replan]
        else:
            a = student.act(so)
        tick += 1
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated | truncated
        status[live & done & info["success"]] = 1
        status[live & done & ~info["success"]] = 2
        ep_len += live
        live &= ~done
        snap(obs)

    sink.close()

    print(f"\ncheckpoint: {cfg.play}   success {int((status == 1).sum())}/{B}")
    print(f"{'env':>3} {'outcome':>8} {'len':>4} {'cube_x':>7} {'cube_y':>7} {'yaw_deg':>8}")
    for i in range(B):
        print(f"{i:>3} {'success' if status[i] == 1 else 'FAIL':>8} {ep_len[i]:>4} "
              f"{spawn[i, 0]:>7.3f} {spawn[i, 1]:>7.3f} {np.degrees(spawn_yaw[i]):>8.1f}")
    print(f"video -> {out}")


def main(cfg: Config) -> None:
    if cfg.play is not None:
        play(cfg)
        return
    if int(os.environ.get("RANK", 0)) == 0:
        (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
        with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
            json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
