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
  head. Needs a batched render backend; nyx works for smoke tests only
  (~1 step/s at B=32) — training-scale collection waits on the madrona batch
  renderer (gs-madrona wheel is CUDA 12.4; Blackwell needs a cu128 build).

Two teachers (``--teacher``):
- ``waypoint``: the scripted LiftPolicy (closed-loop, replans at reset).
- ``mlp:<ckpt.pt>``: a frozen state-mode Student checkpoint (labels any state,
  no clock; capped at that policy's own success rate).

    uv run python scripts/simpledagger.py --n-envs 4096 --batch-size 4096
    uv run python scripts/simpledagger.py --policy image --teacher mlp:outputs/dagger/b4096-v7/best.pt
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tyro
from rich import print

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
    grip_coef: float = 0.1            # BCE weight on the gripper logit (labels are 0/1)
    aug_pad: int = 4                  # DrQ random-shift padding, px (0 = off)
    # record every kth visited state in image mode (labels stay per-step for
    # control; adjacent 30Hz frames are near-duplicates and RAM is the binding
    # constraint at B >= 256: ~36KB/sample at 3x64x64)
    frame_stride: int = 1
    # splat background plates composited behind static cameras wherever madrona's
    # segmentation says sky (scripts/make_plates.py); None = raw batch frames
    plates_dir: Path | None = None
    # madrona rasterizer for image obs (the light calibration was matched under
    # it); False = raytracer (2.7x faster, real shadows, different visual domain)
    batch_rasterizer: bool = True
    # eval
    eval_batches: int = 1             # student-only eval rollouts per round (n_envs each)
    eval_seed: int = 51_000
    # env
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    # EE-pose actions via CartesianActionWrapper (wrapper owns IK). Joint space is
    # the performance default: v7 (joints) 93% vs v6b (poses, quat-canonical) 73%
    # at equal budget.
    cartesian: bool = False
    # 15cm x 15cm spawn box around the home TCP (x=0.34, y=0)
    cube_x_range: tuple[float, float] = (0.275, 0.425)
    cube_y_range: tuple[float, float] = (-0.075, 0.075)
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

    def forward(self, rgb: torch.Tensor, prop: torch.Tensor):
        """rgb: (N, V, 3, H, W) float in [-0.5, 0.5]; prop: (N, P) raw."""
        n = rgb.shape[0]
        if self.shared:
            f = self.encoder(rgb.reshape(n * self.n_views, *rgb.shape[2:]))
            f = f.reshape(n, self.n_views, -1) + self.view_emb
        else:
            f = torch.stack(
                [enc(rgb[:, i]) for i, enc in enumerate(self.encoders)], dim=1)
        p = self.prop_net((prop - self.prop_mean) / self.prop_std)
        h = self.trunk(torch.cat([f.reshape(n, -1), p], dim=-1))
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


def build_student(cfg: Config, env, device: torch.device):
    act_space = env.get_wrapper_attr("single_action_space")
    if cfg.policy == "state":
        return Student(
            obs_dim=env.single_observation_space.shape[0],
            act_dim=act_space.shape[0], hidden=cfg.hidden_dim,
            act_low=act_space.low, act_high=act_space.high,
        ).to(device)
    base_spaces = env.unwrapped.single_observation_space.spaces
    proprio_keys = sorted(k for k in base_spaces if "cube" not in k)
    proprio_dim = sum(int(np.prod(base_spaces[k].shape)) for k in proprio_keys)
    return ImageStudent(
        proprio_dim=proprio_dim, act_dim=act_space.shape[0],
        n_views=len(env.views), hw=cfg.image_hw,
        act_low=act_space.low, act_high=act_space.high,
        encoder=cfg.encoder, hidden=cfg.hidden_dim, feat_dim=cfg.feat_dim,
    ).to(device)


class Trainer:
    def __init__(self, cfg: Config):
        from xsim.suite.policies import LiftPolicy

        self.cfg = cfg
        self.image = cfg.policy == "image"
        self.work_dir = cfg.out / cfg.exp_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.work_dir / "metrics.jsonl"

        self.env = build_env(cfg)
        base = self.env.unwrapped
        self.state_keys = sorted(base.single_observation_space.spaces)
        self.proprio_keys = [k for k in self.state_keys if "cube" not in k]
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
        if cfg.teacher == "waypoint":
            self.teacher = LiftPolicy(
                base, steps_per_segment=cfg.steps_per_segment, cartesian=cfg.cartesian)
        elif cfg.teacher.startswith("mlp:"):
            self.teacher = MLPTeacher(Path(cfg.teacher[4:]), self.device)
        else:
            raise ValueError(f"unknown teacher {cfg.teacher!r}")
        self.student = build_student(cfg, self.env, self.device)
        self.ema = copy.deepcopy(self.student)
        self.optim = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        # aggregated DAgger dataset, one chunk appended per recorded step.
        # image mode keeps rgb as uint8 on cpu (36KB/step at 3x64x64); at
        # madrona scale (B >= 256) this wants a disk-backed ring instead.
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._stats_set = False
        self.best_success = -1.0
        self._start = time.time()

    def log(self, kind: str, d: dict) -> None:
        d = {"kind": kind, "elapsed_s": round(time.time() - self._start, 1), **d}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(d) + "\n")
        pretty = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in d.items() if k != "kind")
        color = {"collect": "white", "bc": "cyan", "eval": "green"}.get(kind, "white")
        print(f"[{color}]\\[{kind}][/] {pretty}")

    # -- obs adapters --------------------------------------------------------------
    def _teacher_obs(self, obs):
        """Sorted-key flat state (the waypoint teacher ignores it)."""
        return flat(obs, self.state_keys) if self.image else obs

    def _student_obs(self, obs):
        if self.image:
            return flat(obs, self.proprio_keys), obs["rgb"]
        return obs

    def _append(self, key: str, arr: np.ndarray) -> None:
        self._chunks.setdefault(key, []).append(arr)

    def _record(self, obs, teacher_a: np.ndarray, live: np.ndarray) -> None:
        self._append("act", teacher_a[live].astype(np.float32, copy=True))
        if self.image:
            prop, rgb = self._student_obs(obs)
            self._append("prop", prop[live].copy())
            self._append("rgb", rgb[live].copy())
            self._append("aux", aux_targets(obs)[live].copy())
        else:
            self._append("obs", obs[live].astype(np.float32, copy=True))

    # -- collection --------------------------------------------------------------
    def rollout(self, beta: float, record: bool, seed: int | None = None,
                student: nn.Module | None = None) -> dict:
        """One synchronous env-batch episode; per-env, per-step Bernoulli(beta)
        picks the teacher's action over the student's. Every visited state is
        labeled with the teacher action when ``record`` is on."""
        env, B = self.env, self.cfg.n_envs
        student = student or self.student
        obs, _ = env.reset(seed=seed)
        self.teacher.reset()
        live = np.ones(B, dtype=bool)
        success = np.zeros(B, dtype=bool)
        ep_len = np.zeros(B, dtype=np.int64)
        tick = 0
        while live.any():
            teacher_a = self.teacher.act(self._teacher_obs(obs))
            if beta >= 1.0:
                action = teacher_a
            else:
                student_a = student.act(self._student_obs(obs))
                use_teacher = self.rng.random(B) < beta
                action = np.where(use_teacher[:, None], teacher_a, student_a)
            if record and (not self.image or tick % self.cfg.frame_stride == 0):
                self._record(obs, teacher_a, live)
            tick += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            success |= live & info["success"]
            ep_len += live
            live &= ~done
        return {"success": float(success.mean()), "ep_len": float(ep_len.mean())}

    # -- bc ----------------------------------------------------------------------
    def _freeze_stats(self, x: torch.Tensor) -> None:
        if not self._stats_set:
            self.student.set_obs_stats(x.mean(dim=0), x.std(dim=0))
            self._stats_set = True

    def train_bc(self) -> dict:
        return self._train_bc_image() if self.image else self._train_bc_state()

    def _train_bc_state(self) -> dict:
        cfg = self.cfg
        X = torch.from_numpy(np.concatenate(self._chunks["obs"])).to(self.device)
        Y = torch.from_numpy(np.concatenate(self._chunks["act"])).to(self.device)
        self._freeze_stats(X)
        n = X.shape[0]
        last_loss = 0.0
        for _ in range(cfg.epochs_per_round):
            perm = torch.randperm(n, device=self.device)
            losses = []
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                loss = F.mse_loss(self.student(X[idx]), Y[idx])
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                self._ema_update()
                losses.append(loss.item())
            last_loss = float(np.mean(losses))
        return {"samples": n, "bc_loss": last_loss}

    def _train_bc_image(self) -> dict:
        cfg = self.cfg
        rgb = torch.from_numpy(np.concatenate(self._chunks["rgb"]))          # cpu uint8
        prop = torch.from_numpy(np.concatenate(self._chunks["prop"])).to(self.device)
        Y = torch.from_numpy(np.concatenate(self._chunks["act"])).to(self.device)
        aux_y = torch.from_numpy(np.concatenate(self._chunks["aux"])).to(self.device)
        self._freeze_stats(prop)
        n = rgb.shape[0]
        stats = {"bc_loss": 0.0, "grip_bce": 0.0, "aux_mse": 0.0}
        for _ in range(cfg.epochs_per_round):
            # explicit cpu: gs.init makes cuda the default device, but rgb is cpu
            perm = torch.randperm(n, device="cpu")
            sums = {k: 0.0 for k in stats}
            batches = 0
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                x = rgb[idx].to(self.device, non_blocking=True).float() / 255.0 - 0.5
                if cfg.aug_pad > 0:
                    nv = x.shape[0] * x.shape[1]
                    x = rand_shift(x.reshape(nv, *x.shape[2:]), cfg.aug_pad).reshape(x.shape)
                pred, aux = self.student(x, prop[idx.to(self.device)])
                y = Y[idx.to(self.device)]
                joints = F.mse_loss(pred[:, :-1], y[:, :-1])
                grip = F.binary_cross_entropy_with_logits(pred[:, -1], y[:, -1])
                aux_l = F.mse_loss(aux, aux_y[idx.to(self.device)])
                loss = joints + cfg.grip_coef * grip + cfg.aux_pose_coef * aux_l
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                self._ema_update()
                sums["bc_loss"] += joints.item()
                sums["grip_bce"] += grip.item()
                sums["aux_mse"] += aux_l.item()
                batches += 1
            stats = {k: v / max(batches, 1) for k, v in sums.items()}
        return {"samples": n, **stats}

    @torch.no_grad()
    def _ema_update(self) -> None:
        d = self.cfg.ema_decay
        for pe, p in zip(self.ema.parameters(), self.student.parameters()):
            pe.lerp_(p, 1.0 - d)
        for be, b in zip(self.ema.buffers(), self.student.buffers()):
            be.copy_(b)

    # -- loop --------------------------------------------------------------------
    def evaluate(self) -> dict:
        stats = [self.rollout(beta=0.0, record=False, seed=self.cfg.eval_seed + b,
                              student=self.ema)
                 for b in range(self.cfg.eval_batches)]
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
                stats = self.rollout(beta=beta, record=True,
                                     seed=cfg.seed + rnd if rnd == 0 else None)
                self.log("collect", {"round": rnd, "beta": beta, **stats})
            self.log("bc", {"round": rnd, "lr": lr, **self.train_bc()})
            eval_metrics = self.evaluate()
            self.log("eval", {"round": rnd, **eval_metrics})
            torch.save(self.ema.state_dict(), self.work_dir / "latest.pt")
            if eval_metrics["eval_success"] >= self.best_success:
                self.best_success = eval_metrics["eval_success"]
                torch.save(self.ema.state_dict(), self.work_dir / "best.pt")
        print(f"\\[done] best eval success: {self.best_success:.0%}")


# ---------------------------------------------------------------------------------------
# play mode
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


def play(cfg: Config) -> None:
    import cv2

    env = build_env(cfg, render=True)
    B = cfg.n_envs
    image = cfg.policy == "image"
    device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
    student = build_student(cfg, env, device)
    student.load_state_dict(torch.load(cfg.play, map_location=device))
    student.eval()
    state_keys = sorted(env.unwrapped.single_observation_space.spaces)
    proprio_keys = [k for k in state_keys if "cube" not in k]

    obs, _ = env.reset(seed=cfg.eval_seed)
    spawn = np.asarray(env.unwrapped.cube.get_pos(), dtype=np.float64).copy()
    q = np.asarray(env.unwrapped.cube.get_quat(), dtype=np.float64)
    spawn_yaw = 2.0 * np.arctan2(q[:, 3], q[:, 0])
    status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail
    ep_len = np.zeros(B, dtype=np.int64)
    live = np.ones(B, dtype=bool)
    frames = []

    def snap(obs) -> None:
        if image:  # reuse the policy's own frames (image_hw px, upscaled)
            rgb = obs["rgb"].transpose(0, 1, 3, 4, 2)  # (B, V, H, W, 3)
            views = [rgb[:, i] for i in range(rgb.shape[1])]
        else:
            d = env.unwrapped.render_views(all_envs=True)
            views = [d[k] for k in sorted(d)]
        frames.append(np.concatenate(
            [_grid(v, status, cfg.video_max_width) for v in views], axis=1))

    snap(obs)
    while live.any():
        a = student.act((flat(obs, proprio_keys), obs["rgb"]) if image else obs)
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated | truncated
        status[live & done & info["success"]] = 1
        status[live & done & ~info["success"]] = 2
        ep_len += live
        live &= ~done
        snap(obs)

    out = cfg.play_video or cfg.play.parent / "rollout.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / env.unwrapped.control_dt, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        writer.write(f[:, :, ::-1])
    writer.release()

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
    torch.manual_seed(cfg.seed)
    (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
        json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
