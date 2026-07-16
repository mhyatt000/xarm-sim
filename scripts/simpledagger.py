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

from dataclasses import asdict, dataclass
import json
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
    beta_rampdown_rounds: int = 5     # beta = max(0, 1 - round/rampdown); 0 -> pure BC
    steps_per_segment: int = 20       # expert waypoint pacing
    # bc
    epochs_per_round: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256
    # eval
    eval_batches: int = 1             # student-only eval rollouts per round (n_envs each)
    eval_seed: int = 51_000
    # env
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    horizon: int = 200
    control_freq: float = 30.0
    noslip_iterations: int = 10


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


def build_env(cfg: Config):
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.wrappers import GymWrapper

    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7", camera_names=[],
        horizon=cfg.horizon, n_envs=cfg.n_envs,
        control_freq=cfg.control_freq,
        noslip_iterations=cfg.noslip_iterations,
    )
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
            self.env.unwrapped, steps_per_segment=cfg.steps_per_segment)
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
        self.student = Student(
            obs_dim=self.env.single_observation_space.shape[0],
            act_dim=self.env.unwrapped.single_action_space.shape[0],
            hidden=cfg.hidden_dim,
            act_low=self.env.unwrapped.single_action_space.low,
            act_high=self.env.unwrapped.single_action_space.high,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        # aggregated DAgger dataset: per-rollout (obs, expert_action) chunks
        self._obs_chunks: list[np.ndarray] = []
        self._act_chunks: list[np.ndarray] = []
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
    def rollout(self, beta: float, record: bool, seed: int | None = None) -> dict:
        """One synchronous env-batch episode; per-env, per-step Bernoulli(beta)
        picks the expert's action over the student's. Every visited state is
        labeled with the expert action when ``record`` is on."""
        env, B = self.env, self.cfg.n_envs
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
                student_a = self.student.act(obs)
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
        self.student.set_obs_stats(X.mean(dim=0), X.std(dim=0))
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
                losses.append(loss.item())
            last_loss = float(np.mean(losses))
        return {"samples": n, "bc_loss": last_loss}

    # -- loop --------------------------------------------------------------------
    def evaluate(self) -> dict:
        stats = [self.rollout(beta=0.0, record=False, seed=self.cfg.eval_seed + b)
                 for b in range(self.cfg.eval_batches)]
        return {
            "eval_success": float(np.mean([s["success"] for s in stats])),
            "eval_len": float(np.mean([s["ep_len"] for s in stats])),
        }

    def train(self) -> None:
        cfg = self.cfg
        for rnd in range(cfg.rounds):
            beta = max(0.0, 1.0 - rnd / cfg.beta_rampdown_rounds)
            for _ in range(cfg.episodes_per_round):
                stats = self.rollout(beta=beta, record=True,
                                     seed=cfg.seed + rnd if rnd == 0 else None)
                self.log("collect", {"round": rnd, "beta": beta, **stats})
            self.log("bc", {"round": rnd, **self.train_bc()})
            eval_metrics = self.evaluate()
            self.log("eval", {"round": rnd, **eval_metrics})
            torch.save(self.student.state_dict(), self.work_dir / "latest.pt")
            if eval_metrics["eval_success"] >= self.best_success:
                self.best_success = eval_metrics["eval_success"]
                torch.save(self.student.state_dict(), self.work_dir / "best.pt")
        print(f"\\[done] best eval success: {self.best_success:.0%}")


def main(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
        json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
