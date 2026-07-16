"""SimpleDAgger on the xsim.suite Lift task: distill the scripted waypoint expert.

Port of imitation's SimpleDAggerTrainer loop to the suite's batched Genesis envs
(imitation itself pins gymnasium 0.29 and can't share an env with Genesis). Each
round rolls out a per-env beta-mixture of the LiftPolicy expert and the MLP
student, labels every visited state with the expert's action, aggregates the
dataset, and continues BC training (MSE on the canonical absolute [j0..j6, g]
action). Round 0 is pure expert, so the first round is plain BC warm-up; beta
ramps linearly to 0 over ``beta_rampdown_rounds``.

The waypoint expert is open-loop in time (it replans from sim state at reset and
plays the schedule forward), so its labels on student-visited states are the
scripted trajectory's action at that tick — time-indexed references, not
closed-loop corrections. All envs reset together, keeping every env on the same
tick of its own plan.

    uv run python scripts/simpledagger.py                      # 16 envs, 10 rounds
    uv run python scripts/simpledagger.py --n-envs 4096        # scale collection
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
    # never collect with the pure student: the expert labels by tick index, so once
    # the student's timing drifts off the script the labels contradict the obs and
    # the aggregate poisons itself (b4096: 81% at beta=0.2, then 0% after beta=0)
    beta_floor: float = 0.2
    steps_per_segment: int = 20       # expert waypoint pacing
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
    # eval
    eval_batches: int = 1             # student-only eval rollouts per round (n_envs each)
    eval_seed: int = 51_000
    # env
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    # EE-pose actions [x,y,z, qw..qz, g] via CartesianActionWrapper: labels are
    # poses (branch-unambiguous) and the wrapper owns the IK; False = raw joints
    cartesian: bool = True
    # 5cm x 5cm spawn box centered on (x=300mm, y=0)
    cube_x_range: tuple[float, float] = (0.275, 0.325)
    cube_y_range: tuple[float, float] = (-0.025, 0.025)
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


def build_env(cfg: Config, render: bool = False):
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.renderers import NyxConfig
    from xsim.suite.wrappers import CartesianActionWrapper, GymWrapper

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7",
        camera_names=list(cfg.cameras) if render else [],
        render_backend="nyx" if render else "raster",
        renderer_config=NyxConfig(spp=cfg.spp) if render else None,
        x_range=cfg.cube_x_range, y_range=cfg.cube_y_range,
        horizon=cfg.horizon, n_envs=cfg.n_envs,
        control_freq=cfg.control_freq,
        noslip_iterations=cfg.noslip_iterations,
    )
    if cfg.cartesian:
        env = CartesianActionWrapper(env)
    return GymWrapper(env)


class Trainer:
    def __init__(self, cfg: Config):
        from xsim.suite.policies import LiftPolicy

        self.cfg = cfg
        self.work_dir = cfg.out / cfg.exp_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.work_dir / "metrics.jsonl"

        self.env = build_env(cfg)
        self.expert = LiftPolicy(
            self.env.unwrapped, steps_per_segment=cfg.steps_per_segment,
            cartesian=cfg.cartesian)
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
        # single_action_space lives on the innermost action wrapper (Cartesian or raw)
        act_space = self.env.get_wrapper_attr("single_action_space")
        self.student = Student(
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=act_space.shape[0],
            hidden=cfg.hidden_dim,
            act_low=act_space.low,
            act_high=act_space.high,
        ).to(self.device)
        self.ema = copy.deepcopy(self.student)
        self.optim = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        # aggregated DAgger dataset: per-rollout (obs, expert_action) chunks
        self._obs_chunks: list[np.ndarray] = []
        self._act_chunks: list[np.ndarray] = []
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

    # -- collection --------------------------------------------------------------
    def rollout(self, beta: float, record: bool, seed: int | None = None,
                student: Student | None = None) -> dict:
        """One synchronous env-batch episode; per-env, per-step Bernoulli(beta)
        picks the expert's action over the student's. Every visited state is
        labeled with the expert action when ``record`` is on."""
        env, B = self.env, self.cfg.n_envs
        student = student or self.student
        obs, _ = env.reset(seed=seed)
        self.expert.reset()
        live = np.ones(B, dtype=bool)
        success = np.zeros(B, dtype=bool)
        ep_len = np.zeros(B, dtype=np.int64)
        while live.any():
            expert_a = self.expert.act()  # advances the plan one tick for all envs
            if beta >= 1.0:
                action = expert_a
            else:
                student_a = student.act(obs)
                use_expert = self.rng.random(B) < beta
                action = np.where(use_expert[:, None], expert_a, student_a)
            if record and live.any():
                self._obs_chunks.append(obs[live].astype(np.float32, copy=True))
                self._act_chunks.append(expert_a[live].astype(np.float32, copy=True))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            success |= live & info["success"]
            ep_len += live
            live &= ~done
        return {"success": float(success.mean()), "ep_len": float(ep_len.mean())}

    # -- bc ----------------------------------------------------------------------
    def train_bc(self) -> dict:
        cfg = self.cfg
        X = torch.from_numpy(np.concatenate(self._obs_chunks)).to(self.device)
        Y = torch.from_numpy(np.concatenate(self._act_chunks)).to(self.device)
        # freeze normalization after the first (pure-expert) fit: refitting on later
        # rounds shifts the input space under the trained net
        if not self._stats_set:
            self.student.set_obs_stats(X.mean(dim=0), X.std(dim=0))
            self._stats_set = True
        n = X.shape[0]
        last_loss = 0.0
        for _ in range(cfg.epochs_per_round):
            perm = torch.randperm(n, device=self.device)
            losses = []
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                loss = nn.functional.mse_loss(self.student(X[idx]), Y[idx])
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                self._ema_update()
                losses.append(loss.item())
            last_loss = float(np.mean(losses))
        return {"samples": n, "bc_loss": last_loss}

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
    import math

    b, h, w, _ = frames.shape
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)
    tw = max(2, min(w, max_width // cols)) // 2 * 2
    th = max(2, round(h * tw / w)) // 2 * 2
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    t = max(2, tw // 64)
    for i in range(b):
        r, c = divmod(i, cols)
        tile = cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA)
        tile[:t], tile[-t:], tile[:, :t], tile[:, -t:] = (BORDER[int(status[i])],) * 4
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
    return canvas


def play(cfg: Config) -> None:
    import cv2

    env = build_env(cfg, render=True)
    B = cfg.n_envs
    act_space = env.get_wrapper_attr("single_action_space")
    student = Student(
        obs_dim=env.single_observation_space.shape[0],
        act_dim=act_space.shape[0],
        hidden=cfg.hidden_dim,
        act_low=act_space.low,
        act_high=act_space.high,
    )
    student.load_state_dict(torch.load(cfg.play, map_location="cpu"))
    student.eval()

    obs, _ = env.reset(seed=cfg.eval_seed)
    spawn = np.asarray(env.unwrapped.cube.get_pos(), dtype=np.float64).copy()
    q = np.asarray(env.unwrapped.cube.get_quat(), dtype=np.float64)
    spawn_yaw = 2.0 * np.arctan2(q[:, 3], q[:, 0])
    status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail
    ep_len = np.zeros(B, dtype=np.int64)
    live = np.ones(B, dtype=bool)
    frames = []

    def snap() -> None:
        views = env.unwrapped.render_views(all_envs=True)
        frames.append(np.concatenate(
            [_grid(views[k], status, cfg.video_max_width) for k in sorted(views)], axis=1))

    snap()
    while live.any():
        obs, reward, terminated, truncated, info = env.step(student.act(obs))
        done = terminated | truncated
        status[live & done & info["success"]] = 1
        status[live & done & ~info["success"]] = 2
        ep_len += live
        live &= ~done
        snap()

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
