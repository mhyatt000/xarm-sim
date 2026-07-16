"""TD-MPC2 online training on the xsim.suite Lift task (sparse reward, batched envs).

Drives the vendored ``xsim.suite.algo.tdmpc2`` agent on the suite stack without
hydra/wandb:

- env: :class:`xsim.suite.Lift` at ``n_envs`` parallel Genesis envs (dict state obs,
  absolute joint actions) under the suite's ``DeltaActionWrapper`` (``[-1,1]^8``
  relative joint targets + continuous gripper) and ``GymWrapper`` (flat obs vectors),
  plus a script-local ``TensorShim`` giving tdmpc2 tensors, the old-gym 4-tuple
  contract, per-env lift guard rails (max_rise, fell/flew), and optional autoreset
  (done envs restart in place via the suite's partial reset).
- agent: ``BatchedTDMPC2`` — stock TD-MPC2 with the MPPI planner re-derived over a
  leading env dim (per-env elites/warm-starts, one compiled CUDA graph for all envs);
  the vendored code is not modified.
- bootstrapping: the replay buffer is pre-seeded with scripted-policy demos collected
  by ``scripts/tdmpc_demos.py``, the agent is pretrained on them, and
  ``demo_rounds_per_eval`` scripted rounds are injected after each eval while
  collection is already paused.
- loop: continuous async collection (episodes close per env and the env restarts
  immediately); before each eval the loop drains in-flight episodes instead of
  discarding them. ``step`` counts recorded transitions; ``utd`` gradient updates run
  per recorded transition, as in the single-env version.

    uv run python scripts/tdmpc_demos.py --episodes 100          # once
    uv run python scripts/tdmpc_train.py --steps 200000 --n-envs 16
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import dataclasses
import json
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
    steps: int = 200_000              # recorded env transitions (scripted rounds count too)
    seed: int = 1
    exp_name: str = "default"
    out: Path = PROJECT_ROOT / "outputs" / "tdmpc"
    resume: Path | None = None        # checkpoint to load into the agent before training
    # bootstrap
    demos: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "demos.pt"
    pretrain_updates: int = 10_000
    seed_steps: int = 500             # random-action transitions before the agent acts
    demo_rounds_per_eval: int = 1     # scripted sync rounds (n_envs episodes each) at injection time
    demo_every_evals: int = 2         # inject after every Nth eval (v5's every-eval gave a 41%-scripted buffer)
    steps_per_segment: int = 20       # scripted policy pacing for injected demos
    # training
    utd: float = 1.0                  # gradient updates per recorded transition
    batch_size: int = 256
    horizon: int = 3                  # tdmpc2 planning horizon
    max_std: float = 2.0              # planner sampling std; lower = smoother exploration
    discount: float = 0.99
    # TD value-target bootstrap length: 1 = stock 1-step TD; n>1 sums n discounted
    # rewards before bootstrapping, so sparse terminal reward jumps n transitions of
    # value credit per update instead of one (ignition fix for the lift reward)
    nstep: int = 25
    buffer_size: int = 500_000
    model_size: int = 5
    mpc: bool = True
    compile: bool = True                    # 46ms -> 3.7ms per update on a 5090; coexists with Genesis
    # eval
    eval_freq: int = 5_000
    eval_episodes: int = 16
    eval_seed: int = 51_000
    save_video: bool = True
    # env
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0        # v1 control regime (suite default is 30 Hz)
    max_steps: int = 150              # episode truncation, in control steps
    max_delta_rad: float = 0.10       # joint-target delta per control tick at |a| = 1
    # snap the wrapper's gripper channel to open/grasp extremes: v1's exploration
    # geometry (any a[7]<0 = full grasp). v4/v5 showed the continuous channel delays
    # sparse-reward grasp discovery ~4x (v5 ignition 167k vs v1 45k) — sustained
    # near-(-1) closure is too rare under planner noise. Env stays continuous.
    binary_gripper: bool = True
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
        # Buffer capacity = min(buffer_size, steps); pad steps so preloaded demos and
        # the drain overshoot never wrap the ring storage (wrapped episodes -> garbage).
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


def make_agent(tdcfg, n_envs: int):
    """Stock TD-MPC2 with the MPPI planner re-derived over a leading env dim.

    Same math as the vendored ``_plan``/``_estimate_value`` (single-task branch), with
    the env batch folded into the sample dim and elites/scores/warm-start means kept
    per env. One compiled CUDA graph serves all envs, so act stays O(1) in n_envs.
    There is no per-env t0 flag: ``reset_envs`` zeroes the env's warm-start rows, and
    shifting zeros is zeros — identical to the stock t0 branch.
    """
    from common import math as tdmath
    from tdmpc2 import TDMPC2
    from tensordict import TensorDict
    import torch.nn.functional as F

    class BatchedTDMPC2(TDMPC2):
        def __init__(self, cfg, n_envs: int):
            super().__init__(cfg)
            self.n_envs = n_envs
            self._prev_means = torch.nn.Buffer(
                torch.zeros(n_envs, cfg.horizon, cfg.action_dim, device=self.device))

        def reset_envs(self, envs_idx) -> None:
            self._prev_means[torch.as_tensor(envs_idx, device=self.device)] = 0.0

        @torch.no_grad()
        def act(self, obs, t0=False, eval_mode=False, task=None):
            """Batched act: obs (B, obs_dim) -> actions (B, action_dim) on cpu."""
            obs = obs.to(self.device, non_blocking=True)
            if self.cfg.mpc:
                return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
            z = self.model.encode(obs, task)
            action, info = self.model.pi(z, task)
            if eval_mode:
                action = info["mean"]
            return action.cpu()

        @torch.no_grad()
        def _estimate_value(self, z, actions, task):
            # stock body, minus multitask, with a rollout-count-agnostic termination shape
            G, discount = 0, 1
            termination = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=z.device)
            for t in range(self.cfg.horizon):
                reward = tdmath.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
                z = self.model.next(z, actions[t], task)
                G = G + discount * (1 - termination) * reward
                discount = discount * self.discount
                if self.cfg.episodic:
                    termination = torch.clip(
                        termination + (self.model.termination(z, task) > 0.5).float(), max=1.0)
            action, _ = self.model.pi(z, task)
            return G + discount * (1 - termination) * self.model.Q(z, action, task, return_type="avg")

        @torch.no_grad()
        def _plan(self, obs, t0=False, eval_mode=False, task=None):
            cfg = self.cfg
            B, H, A = obs.shape[0], cfg.horizon, cfg.action_dim
            S, P, E = cfg.num_samples, cfg.num_pi_trajs, cfg.num_elites
            z = self.model.encode(obs, task)  # (B, L)
            if P > 0:
                pi_actions = torch.empty(H, B, P, A, device=self.device)
                _z = z.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
                for t in range(H - 1):
                    a, _ = self.model.pi(_z, task)
                    pi_actions[t] = a.view(B, P, A)
                    _z = self.model.next(_z, a, task)
                a, _ = self.model.pi(_z, task)
                pi_actions[-1] = a.view(B, P, A)

            zs = z.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
            mean = torch.zeros(H, B, A, device=self.device)
            mean[:-1] = self._prev_means.permute(1, 0, 2)[1:]
            std = torch.full((H, B, A), cfg.max_std, dtype=torch.float, device=self.device)
            actions = torch.empty(H, B, S, A, device=self.device)
            if P > 0:
                actions[:, :, :P] = pi_actions

            for _ in range(cfg.iterations):
                r = torch.randn(H, B, S - P, A, device=std.device)
                actions[:, :, P:] = (mean.unsqueeze(2) + std.unsqueeze(2) * r).clamp(-1, 1)
                value = self._estimate_value(zs, actions.view(H, B * S, A), task).nan_to_num(0)
                value = value.view(B, S)
                elite_idxs = torch.topk(value, E, dim=1).indices              # (B, E)
                elite_value = torch.gather(value, 1, elite_idxs)              # (B, E)
                gather_idx = elite_idxs.reshape(1, B, E, 1).expand(H, B, E, A)
                elite_actions = torch.gather(actions, 2, gather_idx)          # (H, B, E, A)

                max_value = elite_value.max(dim=1, keepdim=True).values
                score = torch.exp(cfg.temperature * (elite_value - max_value))
                score = score / score.sum(dim=1, keepdim=True)                # (B, E)
                w = score.reshape(1, B, E, 1)
                mean = (w * elite_actions).sum(dim=2)                         # (H, B, A)
                std = ((w * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2)).sqrt()
                std = std.clamp(cfg.min_std, cfg.max_std)

            rand_idx = tdmath.gumbel_softmax_sample(score, dim=1)             # (B,)
            sel = rand_idx.reshape(1, B, 1, 1).expand(H, B, 1, A)
            a = torch.gather(elite_actions, 2, sel).squeeze(2)[0]             # (B, A)
            if not eval_mode:
                a = a + std[0] * torch.randn(B, A, device=std.device)
            self._prev_means.copy_(mean.permute(1, 0, 2))
            return a.clamp(-1, 1)

        def update(self, buffer):
            """Stock update, unpacking NStepBuffer's extra n-step fields."""
            obs, action, reward, terminated, nstep, task = buffer.sample()
            kwargs = {}
            if task is not None:
                kwargs["task"] = task
            torch.compiler.cudagraph_mark_step_begin()
            return self._update(obs, action, reward, terminated, *nstep, **kwargs)

        def _update(self, obs, action, reward, terminated,
                    nstep_reward, nstep_obs, nstep_terminated, nstep_discount, task=None):
            # vendored _update (single-task branch), with one change: the value target
            # bootstraps n steps ahead (augment_nstep fields) instead of one.
            with torch.no_grad():
                next_z = self.model.encode(obs[1:], task)  # consistency still needs 1-step latents
                boot_z = self.model.encode(nstep_obs, task)
                a, _ = self.model.pi(boot_z, task)
                td_targets = nstep_reward + nstep_discount * (1 - nstep_terminated) * \
                    self.model.Q(boot_z, a, task, return_type="min", target=True)

            # Prepare for update
            self.model.train()

            # Latent rollout
            zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim,
                             device=self.device)
            z = self.model.encode(obs[0], task)
            zs[0] = z
            consistency_loss = 0
            for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
                z = self.model.next(z, _action, task)
                consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
                zs[t + 1] = z

            # Predictions
            _zs = zs[:-1]
            qs = self.model.Q(_zs, action, task, return_type="all")
            reward_preds = self.model.reward(_zs, action, task)
            if self.cfg.episodic:
                termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

            # Compute losses
            reward_loss, value_loss = 0, 0
            for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
                    zip(reward_preds.unbind(0), reward.unbind(0),
                        td_targets.unbind(0), qs.unbind(1))):
                reward_loss = reward_loss + \
                    tdmath.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
                for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                    value_loss = value_loss + \
                        tdmath.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() \
                        * self.cfg.rho**t

            consistency_loss = consistency_loss / self.cfg.horizon
            reward_loss = reward_loss / self.cfg.horizon
            if self.cfg.episodic:
                termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
            else:
                termination_loss = 0.
            value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
            total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef * reward_loss +
                self.cfg.termination_coef * termination_loss +
                self.cfg.value_coef * value_loss
            )

            # Update model
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            # Update policy
            pi_info = self.update_pi(zs.detach(), task)

            # Update target Q-functions
            self.model.soft_update_target_Q()

            # Return training statistics
            self.model.eval()
            info = TensorDict({
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "termination_loss": termination_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            })
            if self.cfg.episodic:
                info.update(tdmath.termination_statistics(
                    torch.sigmoid(termination_pred[-1]), terminated[-1]))
            info.update(pi_info)
            return info.detach().mean()

    return BatchedTDMPC2(tdcfg, n_envs)


def augment_nstep(td, n: int, gamma: float):
    """Add n-step bootstrap fields to a whole-episode TensorDict (T+1 frames).

    For frame i in 1..T, with e = min(i+n-1, T) (clamped to the episode end):

        nstep_reward[i]     = sum_{j=i..e} gamma^(j-i) * reward[j]
        nstep_obs[i]        = obs[e]            (bootstrap observation)
        nstep_terminated[i] = terminated[e]
        nstep_discount[i]   = gamma^(e-i+1)     (bootstrap discount)

    Frame 0 gets zero placeholders (its reward/terminated are NaN and it is never a
    target; the fields only need to exist for stacking). n=1 reduces exactly to the
    stock 1-step quantities: nstep_reward=reward, nstep_obs=obs[i], nstep_discount=gamma.
    """
    T = td.shape[0] - 1
    reward = td["reward"].to("cpu", torch.float32)
    terminated = td["terminated"].to("cpu", torch.float32)
    obs = td["obs"].cpu()
    nstep_reward = torch.zeros(T + 1, dtype=torch.float32, device="cpu")
    nstep_obs = torch.zeros_like(obs)
    nstep_terminated = torch.zeros(T + 1, dtype=torch.float32, device="cpu")
    nstep_discount = torch.zeros(T + 1, dtype=torch.float32, device="cpu")
    for i in range(1, T + 1):
        e = min(i + n - 1, T)
        k = torch.arange(e - i + 1, dtype=torch.float32, device="cpu")  # gs.init defaults cuda
        nstep_reward[i] = (gamma**k * reward[i : e + 1]).sum()
        nstep_obs[i] = obs[e]
        nstep_terminated[i] = terminated[e]
        nstep_discount[i] = gamma ** (e - i + 1)
    td["nstep_reward"] = nstep_reward
    td["nstep_obs"] = nstep_obs
    td["nstep_terminated"] = nstep_terminated
    td["nstep_discount"] = nstep_discount
    return td


def make_buffer(tdcfg):
    """Stock tdmpc2 Buffer, with sampled batches also carrying the n-step fields."""
    from common.buffer import Buffer

    class NStepBuffer(Buffer):
        def _prepare_batch(self, td):
            # stock _prepare_batch, extended: also select/slice the four nstep fields
            # (episodes are augmented via augment_nstep before entering the buffer)
            td = td.select(
                "obs", "action", "reward", "terminated", "task",
                "nstep_reward", "nstep_obs", "nstep_terminated", "nstep_discount",
                strict=False,
            ).to(self._device, non_blocking=True)
            obs = td.get("obs").contiguous()
            action = td.get("action")[1:].contiguous()
            reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
            terminated = td.get("terminated")[1:].unsqueeze(-1).contiguous()
            nstep = (
                td.get("nstep_reward")[1:].unsqueeze(-1).contiguous(),     # (H, B, 1)
                td.get("nstep_obs")[1:].contiguous(),                      # (H, B, obs)
                td.get("nstep_terminated")[1:].unsqueeze(-1).contiguous(), # (H, B, 1)
                td.get("nstep_discount")[1:].unsqueeze(-1).contiguous(),   # (H, B, 1)
            )
            task = td.get("task", None)
            if task is not None:
                task = task[0].contiguous()
            return obs, action, reward, terminated, nstep, task

    return NStepBuffer(tdcfg)


# ---------------------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------------------


class TensorShim:
    """tdmpc2's env contract over the wrapped suite env, batched: tensor obs/actions,
    old-gym 4-tuple step with per-env ``info`` arrays, ``rand_act`` — plus lift guard
    rails (max_rise tracking, fell/flew early termination) and optional autoreset.

    With ``autoreset`` on, done envs restart in place via the suite's partial reset:
    the returned obs rows for those envs are the NEW episodes' first observations,
    the finished episodes' terminal rows ride in ``info["final_obs"]`` (indexed by
    ``info["reset_envs"]``).
    """

    def __init__(self, env, delta_wrapper):
        self.env = env
        self.delta_wrapper = delta_wrapper
        self.unwrapped = env.unwrapped
        self.n_envs = self.unwrapped.n_envs
        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = delta_wrapper.single_action_space.shape[0]
        self.autoreset = True
        self._cube = self.unwrapped.cube
        self._top_z = self.unwrapped.arena.top_z
        self._start_z = np.zeros(self.n_envs)
        self._max_rise = np.zeros(self.n_envs)

    def rand_act(self) -> torch.Tensor:
        return torch.rand(self.n_envs, self.action_dim, device="cpu") * 2.0 - 1.0

    def _cube_z(self) -> np.ndarray:
        return np.asarray(self._cube.get_pos(), dtype=np.float64)[:, 2]

    def reset(self, seed: int | None = None) -> torch.Tensor:
        obs, _ = self.env.reset(seed=seed)
        self._start_z[:] = self._cube_z()
        self._max_rise[:] = 0.0
        return torch.from_numpy(obs)

    def step(self, action) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict]:
        a = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        obs, reward, terminated, truncated, info = self.env.step(a)
        cube = np.asarray(self._cube.get_pos(), dtype=np.float64)
        self._max_rise = np.maximum(self._max_rise, cube[:, 2] - self._start_z)
        fell = cube[:, 2] < self._top_z - 0.05
        flew = np.hypot(cube[:, 0], cube[:, 1]) > 0.9
        terminated = np.asarray(terminated, dtype=bool) | fell | flew
        done = terminated | np.asarray(truncated, dtype=bool)
        info = dict(info, terminated=terminated, fell=fell, flew=flew,
                    max_rise=self._max_rise.copy())
        obs_t = torch.from_numpy(obs)

        if self.autoreset and done.any():
            idx = np.flatnonzero(done)
            info["reset_envs"] = idx
            info["final_obs"] = obs_t[idx].clone()
            new_obs, _ = self.env.reset(options={"envs_idx": idx})
            obs_t[idx] = torch.from_numpy(new_obs[idx])
            self._start_z[idx] = self._cube_z()[idx]
            self._max_rise[idx] = 0.0
        return obs_t, np.asarray(reward, dtype=np.float64), done, info

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
        horizon=cfg.max_steps, n_envs=cfg.n_envs,
        noslip_iterations=cfg.noslip_iterations,
    )
    delta = DeltaActionWrapper(env, cfg.max_delta_rad, binary_gripper=cfg.binary_gripper)
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
        tdcfg = make_tdmpc_cfg(cfg, self.env.obs_dim, self.env.action_dim, self.work_dir)

        self.agent = make_agent(tdcfg, cfg.n_envs)
        if cfg.resume is not None:
            self.agent.load(str(cfg.resume))
            print(f"[resume] loaded {cfg.resume}")
        self.buffer = make_buffer(tdcfg)
        self.tdcfg = tdcfg
        self.step = 0
        self.episode = 0
        self.best_success = -1.0
        self._start = time.time()
        self._update_debt = 0.0
        self._scripted = None       # lazily built demo-injection policy
        self._window: list[dict] = []  # per-episode stats awaiting an aggregate log row
        self._train_metrics: dict = {}

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

    def _add_episode(self, td) -> None:
        """Single buffer entry point: every episode gets its n-step fields here."""
        self.buffer.add(augment_nstep(td, self.cfg.nstep, self.cfg.discount))

    def _close_episode(self, tds: list, stats: dict) -> None:
        """Buffer + account one finished episode; aggregate-log every n_envs closes."""
        if len(tds) >= self.cfg.horizon + 1:
            self._add_episode(torch.cat(tds))
        self.episode += 1
        stats["ep_reward"] = float(np.nansum([td["reward"].item() for td in tds[1:]]))
        stats["ep_len"] = len(tds) - 1
        self._window.append(stats)
        if len(self._window) >= self.cfg.n_envs:
            w = self._window
            self.log("train", {
                "episodes": len(w),
                "ep_reward": float(np.mean([s["ep_reward"] for s in w])),
                "ep_len": float(np.mean([s["ep_len"] for s in w])),
                "success": float(np.mean([s["success"] for s in w])),
                "max_rise": float(np.mean([s["max_rise"] for s in w])),
                "scripted": float(np.mean([s["scripted"] for s in w])),
                "tps": self.step / max(time.time() - self._start, 1e-9),
                **self._train_metrics,
            })
            self._window = []

    # -- demo handling --------------------------------------------------------------
    def preload_demos(self) -> None:
        blob = torch.load(self.cfg.demos, weights_only=False)
        n = 0
        for ep in blob["episodes"]:
            if len(ep) >= self.cfg.horizon + 1:
                self._add_episode(ep)
                n += 1
        print(f"[demos] preloaded {n} episodes "
              f"({sum(len(e) - 1 for e in blob['episodes'])} transitions) from {self.cfg.demos}")

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

    def maybe_update(self, n_transitions: int) -> None:
        self._update_debt += self.cfg.utd * n_transitions
        while self._update_debt >= 1.0:
            torch.compiler.cudagraph_mark_step_begin()
            self._train_metrics = self._scalars(self.agent.update(self.buffer))
            self._update_debt -= 1.0

    def _actions(self, obs: torch.Tensor) -> torch.Tensor:
        if self.step < self.cfg.seed_steps:
            return self.env.rand_act()
        torch.compiler.cudagraph_mark_step_begin()
        return self.agent.act(obs)

    def collect_until(self, until: int) -> None:
        """Async collection (autoreset) until `step` reaches `until`, then drain."""
        if self.step >= until:
            return
        B = self.cfg.n_envs
        env = self.env
        env.autoreset = True
        obs = env.reset()
        self.agent.reset_envs(np.arange(B))
        tds = [[to_td(obs[i])] for i in range(B)]

        while self.step < until:
            actions = self._actions(obs)
            obs, reward, done, info = env.step(actions)
            reset_idx = np.asarray(info.get("reset_envs", []), dtype=int)
            pos = {int(e): k for k, e in enumerate(reset_idx)}
            for i in range(B):
                row = info["final_obs"][pos[i]] if i in pos else obs[i]
                tds[i].append(to_td(row, actions[i], reward[i], info["terminated"][i]))
            self.step += B
            self.maybe_update(B)
            for i in reset_idx:
                self._close_episode(tds[i], dict(
                    success=float(info["success"][i]), max_rise=float(info["max_rise"][i]),
                    scripted=0.0))
                tds[i] = [to_td(obs[i])]  # obs row i is already the new episode's first obs
            if len(reset_idx):
                self.agent.reset_envs(reset_idx)

        # drain: let in-flight episodes finish (recorded) instead of discarding them
        env.autoreset = False
        live = np.ones(B, dtype=bool)
        while live.any():
            actions = self._actions(obs)
            obs, reward, done, info = env.step(actions)
            for i in np.flatnonzero(live):
                tds[i].append(to_td(obs[i], actions[i], reward[i], info["terminated"][i]))
            self.step += int(live.sum())
            self.maybe_update(int(live.sum()))
            for i in np.flatnonzero(live & done):
                self._close_episode(tds[i], dict(
                    success=float(info["success"][i]), max_rise=float(info["max_rise"][i]),
                    scripted=0.0))
            live &= ~done

    def scripted_rounds(self, n: int) -> None:
        """Sync scripted rounds (B demo episodes each), run while collection is paused."""
        from xsim.suite.policies import LiftPolicy

        if n <= 0:
            return
        B = self.cfg.n_envs
        env = self.env
        env.autoreset = False
        if self._scripted is None:
            self._scripted = LiftPolicy(
                env.unwrapped, steps_per_segment=self.cfg.steps_per_segment)
        for _ in range(n):
            obs = env.reset()
            self._scripted.reset()
            tds = [[to_td(obs[i])] for i in range(B)]
            live = np.ones(B, dtype=bool)
            while live.any():
                actions = torch.from_numpy(env.absolute_to_delta(self._scripted.act()))
                obs, reward, done, info = env.step(actions)
                for i in np.flatnonzero(live):
                    tds[i].append(to_td(obs[i], actions[i], reward[i], info["terminated"][i]))
                self.step += int(live.sum())
                self.maybe_update(int(live.sum()))
                for i in np.flatnonzero(live & done):
                    self._close_episode(tds[i], dict(
                        success=float(info["success"][i]), max_rise=float(info["max_rise"][i]),
                        scripted=1.0))
                live &= ~done

    def evaluate(self) -> dict:
        B = self.cfg.n_envs
        env = self.env
        env.autoreset = False
        successes, rewards, lengths, rises = [], [], [], []
        frames = []
        rounds = max(1, -(-self.cfg.eval_episodes // B))
        for rd in range(rounds):
            record = self.cfg.save_video and rd == 0
            obs = env.reset(seed=self.cfg.eval_seed + rd)
            self.agent.reset_envs(np.arange(B))
            live = np.ones(B, dtype=bool)
            ep_reward = np.zeros(B)
            ep_len = np.zeros(B, dtype=int)
            while live.any():
                torch.compiler.cudagraph_mark_step_begin()
                actions = self.agent.act(obs, eval_mode=True)
                obs, reward, done, info = env.step(actions)
                ep_reward[live] += reward[live]
                ep_len[live] += 1
                if record and live[0]:
                    frames.append(env.render_views()["low"])
                for i in np.flatnonzero(live & done):
                    successes.append(float(info["success"][i]))
                    rewards.append(float(ep_reward[i]))
                    lengths.append(int(ep_len[i]))
                    rises.append(float(info["max_rise"][i]))
                live &= ~done
        if frames:
            self._save_video(frames)
        return dict(
            eval_success=float(np.mean(successes)), eval_reward=float(np.mean(rewards)),
            eval_length=float(np.mean(lengths)), eval_max_rise=float(np.mean(rises)),
            eval_episodes=len(successes),
        )

    def _save_video(self, frames: list) -> None:
        try:
            import imageio.v3 as iio

            out = self.work_dir / "videos" / f"step_{self.step:07d}.mp4"
            iio.imwrite(out, np.stack(frames), fps=round(self.cfg.control_freq))
        except Exception as e:  # video is best-effort; never kill training over it
            print(f"[video] failed: {e}")

    def _eval_and_save(self) -> None:
        eval_metrics = self.evaluate()
        self.log("eval", eval_metrics)
        self.agent.save(str(self.work_dir / "latest.pt"))
        if eval_metrics["eval_success"] >= self.best_success:
            self.best_success = eval_metrics["eval_success"]
            self.agent.save(str(self.work_dir / "best.pt"))

    def train(self) -> None:
        cfg = self.cfg
        self.preload_demos()
        self.pretrain()

        next_eval = 0
        n_evals = 0
        while self.step < cfg.steps:
            if self.step >= next_eval:
                self._eval_and_save()
                if n_evals % cfg.demo_every_evals == 0:
                    self.scripted_rounds(cfg.demo_rounds_per_eval)
                n_evals += 1
                next_eval += cfg.eval_freq
            self.collect_until(min(next_eval, cfg.steps))

        self._eval_and_save()
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
