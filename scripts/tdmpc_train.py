"""TD-MPC2 online training on the xsim.suite Lift task (sparse reward).

Drives the vendored ``xsim.suite.algo.tdmpc2`` agent on the suite stack without
hydra/wandb:

- env: :class:`xsim.suite.Lift` (dict state obs, absolute joint actions) under the
  suite's ``DeltaActionWrapper`` (``[-1,1]^8`` relative joint targets + continuous
  gripper) and ``GymWrapper`` (flat obs vector), plus a script-local ``TensorShim``
  giving tdmpc2 its old-gym 4-tuple tensor contract and lift guard rails (max_rise
  tracking, fell/flew early termination). Success = cube 5 cm above the table.
- agent/buffer: stock ``TDMPC2`` + torchrl ``Buffer``, driven by a flat config
  dataclass built here (replacing ``common.parser.parse_cfg``); model_size 5 defaults.
- bootstrapping: the replay buffer is pre-seeded with scripted-policy demos collected
  by ``scripts/tdmpc_demos.py`` (same action space, same dynamics), the agent is
  pretrained on them, and (optionally) a fresh scripted episode is injected every
  ``demo_every``-th training episode so sparse successes keep flowing while the agent
  is weak.
- loop: per-step interleaved act/step/update like ``OnlineTrainer``, with periodic
  seeded eval, best/latest checkpoints, eval videos (low cam), and jsonl metrics.

    uv run python scripts/tdmpc_demos.py --episodes 100          # once
    uv run python scripts/tdmpc_train.py --steps 200000
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import dataclasses
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import torch
import tyro
from rich import print

import xsim.suite.algo  # noqa: F401  # puts the vendored tdmpc2 package on sys.path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@dataclass
class Config:
    # run
    steps: int = 200_000              # online env steps (demo-injected steps count too)
    seed: int = 1
    exp_name: str = "default"
    out: Path = PROJECT_ROOT / "outputs" / "tdmpc"
    resume: Path | None = None        # checkpoint to load into the agent before training
    # bootstrap
    demos: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "demos.pt"
    pretrain_updates: int = 10_000
    seed_steps: int = 500             # random-action env steps before the agent acts
    demo_every: int = 10              # inject a scripted episode every Nth episode (0 = off)
    steps_per_segment: int = 20       # scripted policy pacing for injected demos
    # training
    utd: float = 1.0                  # gradient updates per env step
    batch_size: int = 256
    horizon: int = 3                  # tdmpc2 planning horizon
    max_std: float = 2.0              # planner sampling std; lower = smoother exploration
    discount: float = 0.99
    buffer_size: int = 500_000
    model_size: int = 5
    mpc: bool = True
    compile: bool = True                    # 46ms -> 3.7ms per update on a 5090; coexists with Genesis
    # eval
    eval_freq: int = 5_000
    eval_episodes: int = 10
    eval_seed: int = 51_000
    save_video: bool = True
    # env
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0        # v1 control regime (suite default is 30 Hz)
    max_steps: int = 150              # episode truncation, in control steps
    max_delta_rad: float = 0.10       # joint-target delta per control tick at |a| = 1
    noslip_iterations: int = 10       # weld-free grasping needs the noslip solver


# ---------------------------------------------------------------------------------------
# tdmpc2 config: flat dataclass with every field the vendored modules read
# (grep 'cfg\.' over common/ + tdmpc2.py), replacing hydra + common.parser.parse_cfg.
# ---------------------------------------------------------------------------------------

TDMPC2_DEFAULTS: dict[str, Any] = dict(
    obs="state", episodic=True,
    # optimization
    reward_coef=0.1, value_coef=0.1, termination_coef=1.0, consistency_coef=20.0,
    rho=0.5, lr=3e-4, enc_lr_scale=0.3, grad_clip_norm=20.0, tau=0.01,
    # planning
    iterations=6, num_samples=512, num_elites=64, num_pi_trajs=24,
    min_std=0.05, max_std=2.0, temperature=0.5,
    # actor / critic
    log_std_min=-10.0, log_std_max=2.0, entropy_coef=1e-4,
    num_bins=101, vmin=-10, vmax=+10,
    # architecture (model_size 5 values are applied below)
    num_enc_layers=2, enc_dim=256, num_channels=32, mlp_dim=512, latent_dim=512,
    task_dim=0, num_q=5, dropout=0.01, simnorm_dim=8,
    # single-task bookkeeping
    multitask=False, episode_lengths=[], obs_shapes={}, action_dims=[],
)


def make_tdmpc_cfg(cfg: Config, obs_dim: int, action_dim: int, work_dir: Path):
    from common import MODEL_SIZE

    d = dict(TDMPC2_DEFAULTS)
    d.update(MODEL_SIZE[cfg.model_size])
    d.update(
        task="xarm-lift", tasks=["xarm-lift"], task_title="Xarm Lift",
        # Buffer capacity = min(buffer_size, steps); pad steps so preloaded demo
        # transitions never wrap the ring storage (wrapped episodes -> garbage slices).
        steps=cfg.steps + 25_000, batch_size=cfg.batch_size, horizon=cfg.horizon,
        max_std=cfg.max_std,
        buffer_size=cfg.buffer_size, mpc=cfg.mpc, compile=cfg.compile, seed=cfg.seed,
        # pin the discount: sparse terminal reward ~130 steps out needs gamma ~ 0.99,
        # not the episode-length heuristic's 0.967
        discount_denom=5, discount_min=cfg.discount, discount_max=cfg.discount,
        obs_shape={"state": (obs_dim,)}, action_dim=action_dim,
        episode_length=cfg.max_steps,
        seed_steps=cfg.seed_steps,
        bin_size=(d["vmax"] - d["vmin"]) / (d["num_bins"] - 1),
        work_dir=str(work_dir), model_size=cfg.model_size,
    )
    fields = [(k, Any, dataclasses.field(default_factory=lambda v=v: v)) for k, v in d.items()]
    dc = dataclasses.make_dataclass("TDMPC2Config", fields)
    dc.get = lambda self, k, default=None: getattr(self, k, default)
    return dc()


def make_agent(tdcfg):
    from tdmpc2 import TDMPC2

    return TDMPC2(tdcfg)


# ---------------------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------------------


class TensorShim:
    """tdmpc2's env contract over the wrapped suite env: tensor obs/actions, old-gym
    4-tuple step with ``info['terminated']``, ``rand_act`` — plus lift guard rails
    (max_rise tracking, fell/flew early termination) kept out of the suite env."""

    def __init__(self, env, delta_wrapper):
        self.env = env
        self.delta_wrapper = delta_wrapper
        self.unwrapped = env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._cube = self.unwrapped.cube
        self._top_z = self.unwrapped.arena.top_z
        self._start_z = 0.0
        self._max_rise = 0.0

    def rand_act(self) -> torch.Tensor:
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def reset(self, seed: int | None = None) -> torch.Tensor:
        obs, _ = self.env.reset(seed=seed)
        self._start_z = float(self._cube.get_pos()[2])
        self._max_rise = 0.0
        return torch.from_numpy(obs)

    def step(self, action) -> tuple[torch.Tensor, float, bool, dict]:
        a = action.detach().cpu().numpy() if torch.is_tensor(action) else action
        obs, reward, terminated, truncated, info = self.env.step(a)
        cube = np.asarray(self._cube.get_pos(), dtype=np.float64).reshape(-1)
        self._max_rise = max(self._max_rise, float(cube[2]) - self._start_z)
        fell = float(cube[2]) < self._top_z - 0.05
        flew = math.hypot(float(cube[0]), float(cube[1])) > 0.9
        terminated = bool(terminated or fell or flew)
        info = dict(info, terminated=terminated, fell=fell, flew=flew,
                    max_rise=self._max_rise)
        return torch.from_numpy(obs), float(reward), terminated or truncated, info

    def absolute_to_delta(self, action: np.ndarray) -> np.ndarray:
        return self.delta_wrapper.absolute_to_delta(action)

    def render_views(self):
        return self.unwrapped.render_views()


def build_env(cfg: Config) -> TensorShim:
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.wrappers import DeltaActionWrapper, GymWrapper

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7", camera_names=["low"], render_backend="raster",
        physics_dt=1.0 / cfg.sim_hz, control_freq=cfg.control_freq,
        horizon=cfg.max_steps, noslip_iterations=cfg.noslip_iterations,
    )
    delta = DeltaActionWrapper(env, cfg.max_delta_rad)
    return TensorShim(GymWrapper(delta), delta)


def to_td(obs, action=None, reward=None, terminated=None, action_dim: int = 8):
    """One buffer frame in tdmpc2's episode layout (cf. OnlineTrainer.to_td)."""
    from tensordict import TensorDict

    # explicit cpu everywhere: gs.init sets torch's default device to cuda, so bare
    # torch.full/torch.tensor would otherwise mix devices with act()'s cpu tensors
    return TensorDict(
        obs=obs.unsqueeze(0).cpu(),
        action=torch.full((1, action_dim), float("nan"), device="cpu") if action is None
        else action.unsqueeze(0).cpu(),
        reward=torch.tensor([float("nan") if reward is None else float(reward)], device="cpu"),
        terminated=torch.tensor(
            [float("nan") if terminated is None else float(terminated)], device="cpu"),
        batch_size=(1,),
    )


# ---------------------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.work_dir = cfg.out / cfg.exp_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "videos").mkdir(exist_ok=True)
        self.metrics_path = self.work_dir / "metrics.jsonl"

        self.env = build_env(cfg)
        tdcfg = make_tdmpc_cfg(
            cfg, self.env.observation_space.shape[0], self.env.action_space.shape[0],
            self.work_dir)

        from common.buffer import Buffer

        self.agent = make_agent(tdcfg)
        if cfg.resume is not None:
            self.agent.load(str(cfg.resume))
            print(f"[resume] loaded {cfg.resume}")
        self.buffer = Buffer(tdcfg)
        self.tdcfg = tdcfg
        self.step = 0
        self.episode = 0
        self.best_success = -1.0
        self._start = time.time()
        self._update_debt = 0.0
        self._scripted = None  # lazily built demo-injection policy

    # -- logging -------------------------------------------------------------------
    def log(self, kind: str, d: dict) -> None:
        d = {"kind": kind, "step": self.step, "episode": self.episode,
             "elapsed_s": round(time.time() - self._start, 1), **d}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(d) + "\n")
        pretty = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in d.items() if k != "kind")
        color = {"train": "white", "eval": "green", "pretrain": "cyan"}.get(kind, "white")
        print(f"[{color}]\\[{kind}][/] {pretty}")

    @staticmethod
    def _scalars(metrics) -> dict:
        out = {}
        for k, v in dict(metrics).items():
            v = v.item() if hasattr(v, "item") else v
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    # -- demo handling --------------------------------------------------------------
    def preload_demos(self) -> None:
        blob = torch.load(self.cfg.demos, weights_only=False)
        n = 0
        for ep in blob["episodes"]:
            if len(ep) >= self.cfg.horizon + 1:
                self.buffer.add(ep)
                n += 1
        print(f"[demos] preloaded {n} episodes "
              f"({sum(len(e) - 1 for e in blob['episodes'])} transitions) from {self.cfg.demos}")

    def scripted_episode(self) -> tuple[list, dict]:
        """Roll one scripted-policy episode (through the full wrapper stack) for
        online injection."""
        from xsim.suite.policies import LiftPolicy

        if self._scripted is None:
            self._scripted = LiftPolicy(
                self.env.unwrapped, steps_per_segment=self.cfg.steps_per_segment)
        obs = self.env.reset()
        self._scripted.reset()
        tds = [to_td(obs)]
        done, info = False, {}
        while not done:
            a = torch.from_numpy(self.env.absolute_to_delta(self._scripted.act()))
            obs, reward, done, info = self.env.step(a)
            tds.append(to_td(obs, a, reward, info["terminated"]))
        return tds, info

    # -- phases ----------------------------------------------------------------------
    def pretrain(self) -> None:
        if self.cfg.pretrain_updates <= 0:
            return
        print(f"[pretrain] {self.cfg.pretrain_updates} updates on demo buffer...")
        t0 = time.time()
        for i in range(self.cfg.pretrain_updates):
            metrics = self.agent.update(self.buffer)
            if (i + 1) % 1000 == 0 or i == 0:
                self.log("pretrain", {"update": i + 1, **self._scalars(metrics),
                                      "ups": (i + 1) / (time.time() - t0)})

    def evaluate(self) -> dict:
        successes, rewards, lengths, rises = [], [], [], []
        frames = []
        for i in range(self.cfg.eval_episodes):
            record = self.cfg.save_video and i == 0
            obs = self.env.reset(seed=self.cfg.eval_seed + i)
            done, t, ep_reward, info = False, 0, 0.0, {}
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += float(reward)
                t += 1
                if record:
                    frames.append(self.env.render_views()["low"])
            successes.append(float(info["success"]))
            rewards.append(ep_reward)
            lengths.append(t)
            rises.append(float(info.get("max_rise", 0.0)))
        if frames:
            self._save_video(frames)
        return dict(
            eval_success=float(np.mean(successes)), eval_reward=float(np.mean(rewards)),
            eval_length=float(np.mean(lengths)), eval_max_rise=float(np.mean(rises)),
        )

    def _save_video(self, frames: list) -> None:
        try:
            import imageio.v3 as iio

            out = self.work_dir / "videos" / f"step_{self.step:07d}.mp4"
            iio.imwrite(out, np.stack(frames), fps=round(self.cfg.control_freq))
        except Exception as e:  # video is best-effort; never kill training over it
            print(f"[video] failed: {e}")

    def maybe_update(self, n_env_steps: int = 1) -> dict:
        self._update_debt += self.cfg.utd * n_env_steps
        metrics = {}
        while self._update_debt >= 1.0:
            torch.compiler.cudagraph_mark_step_begin()
            metrics = self.agent.update(self.buffer)
            self._update_debt -= 1.0
        return metrics

    def train(self) -> None:
        cfg = self.cfg
        self.preload_demos()
        self.pretrain()

        next_eval = 0
        train_metrics: dict = {}
        while self.step < cfg.steps:
            if self.step >= next_eval:
                eval_metrics = self.evaluate()
                self.log("eval", eval_metrics)
                self.agent.save(str(self.work_dir / "latest.pt"))
                if eval_metrics["eval_success"] >= self.best_success:
                    self.best_success = eval_metrics["eval_success"]
                    self.agent.save(str(self.work_dir / "best.pt"))
                next_eval += cfg.eval_freq

            inject = cfg.demo_every > 0 and self.episode > 0 and self.episode % cfg.demo_every == 0
            if inject:
                tds, info = self.scripted_episode()
                for _ in range(len(tds) - 1):
                    self.step += 1
                    train_metrics.update(self._scalars(self.maybe_update()))
            else:
                obs = self.env.reset()
                tds = [to_td(obs)]
                done, info = False, {}
                while not done:
                    if self.step < cfg.seed_steps:
                        action = self.env.rand_act()
                    else:
                        torch.compiler.cudagraph_mark_step_begin()
                        action = self.agent.act(obs, t0=len(tds) == 1)
                    obs, reward, done, info = self.env.step(action)
                    tds.append(to_td(obs, action, reward, info["terminated"]))
                    self.step += 1
                    train_metrics.update(self._scalars(self.maybe_update()))

            if len(tds) >= cfg.horizon + 1:
                self.buffer.add(torch.cat(tds))
            self.episode += 1
            ep_reward = float(np.nansum([td["reward"].item() for td in tds[1:]]))
            self.log("train", {
                "ep_reward": ep_reward, "ep_len": len(tds) - 1,
                "success": float(info.get("success", False)), "scripted": int(inject),
                "max_rise": float(info.get("max_rise", 0.0)),
                "sps": self.step / max(time.time() - self._start, 1e-9),
                **train_metrics,
            })

        eval_metrics = self.evaluate()
        self.log("eval", eval_metrics)
        self.agent.save(str(self.work_dir / "latest.pt"))
        if eval_metrics["eval_success"] >= self.best_success:
            self.best_success = eval_metrics["eval_success"]
            self.agent.save(str(self.work_dir / "best.pt"))
        print(f"[done] best eval success: {self.best_success:.0%}")


def main(cfg: Config) -> None:
    assert torch.cuda.is_available() or cfg.backend == "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
        json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
