"""Plain behavior cloning on a suite task: collect once with an expert, train once.

The DAgger degenerate case (beta=1, single round) as its own script: roll the
teacher for ``data_episodes`` episodes, label every visited state with the
teacher's action, then train a student on the frozen dataset for
``train_steps`` optimizer steps, evaluating the EMA student on the live envs
every ``eval_every`` steps. No aggregation, no beta mixture — for the full
DAgger loop or multi-GPU, use scripts/simpledagger.py.

Students and teachers are simpledagger's (``--policy state|image``,
``--teacher expert|waypoint|mlp:<ckpt.pt>``); the defaults here are the vision
student taught by the reactive LiftExpertPolicy.

    uv run python scripts/bc.py --task LiftEZ --exp-name ez-v1
    uv run python scripts/bc.py --task Lift --policy state --n-envs 4096
"""

from __future__ import annotations

import contextlib
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

from xsim.algo import (
    Collector, FlowImageStudent, ImageStudent, MLPTeacher,
    Student, image_proprio_keys, rand_shift,
)
from xsim.data import AugmentedDataset, MemmapDataset, read_key, sim2real_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # run
    seed: int = 0
    exp_name: str = "default"
    out: Path = PROJECT_ROOT / "outputs" / "bc"
    # env: any registered suite env name (Lift, LiftEZ, ...)
    task: str = "Lift"
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    cartesian: bool = False
    horizon: int = 200
    control_freq: float = 30.0
    noslip_iterations: int = 10
    randomize_cameras: bool = True
    cameras: tuple[str, ...] = ("low", "side", "wrist")
    batch_rasterizer: bool = False    # madrona rasterizer instead of the raytracer
    # cube spawn / arm start overrides; None = the task's own defaults (LiftEZ
    # narrows y to +-3 in and always starts at HOME regardless of init_tcp_box)
    cube_x_range: tuple[float, float] | None = None
    cube_y_range: tuple[float, float] | None = None
    init_tcp_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    # collection: total episodes, rounded up to full env batches (n_envs each)
    data_episodes: int = 512
    # "expert" (reactive FSM), "waypoint" (scripted clock-paced LiftPolicy), or
    # "mlp:<ckpt.pt>" (frozen state-mode Student)
    teacher: str = "expert"
    # IK backend for teacher-label generation (and CartesianActionWrapper when
    # --cartesian; they share robot.ik): "softcost" = batched weighted
    # soft-cost (GN/LM) solver (default), "genesis" = built-in sample+DLS
    ik_backend: Literal["genesis", "softcost"] = "softcost"
    # softcost cost-block weights (only used when ik_backend="softcost")
    ik_w_pos: float = 4.0
    ik_w_rot: float = 2.0
    ik_w_home: float = 0.01
    ik_w_limit: float = 1.0
    ik_w_manip: float = 0.0
    ik_iters: int = 25
    ik_damping: float = 0.01          # softcost LM lambda (normal-eqn diagonal)
    steps_per_segment: int = 20       # waypoint teacher pacing
    frame_stride: int = 1             # record every kth visited state (image mode)
    # image-mode dataset store: flat-binary memmap per key under
    # <data_dir>/<exp_name>; point at the big NVMe, not /home
    data_dir: Path = Path("/data/fast/xarm-bc")
    # train on the existing <data_dir>/<exp_name> store, skipping collection
    # (image mode only — creating the writer would wipe the store)
    reuse_data: bool = False
    # store dir name under data_dir; None = exp_name. Lets a new experiment
    # (fresh outputs/wandb name) train on another run's collected store
    store_name: str | None = None
    # collect + flush the store, then exit; train in a fresh process with
    # --reuse-data. Starting loader workers in a process fattened by a full
    # collection (genesis + madrona + the whole store mapped) gets silently
    # killed on multi-M-sample runs regardless of mp_context; two-phase runs
    # never hit it (liftez-bc-v2/flow-v1, Jul 2026)
    collect_only: bool = False
    # student
    policy: Literal["state", "image"] = "image"
    # mse: per-step regression. flow (image only): rectified-flow matching over
    # a ``chunk``-step plan of absolute joint actions, executed receding-horizon
    # (``replan``) — chunk labels are stitched at collection, so flow data must
    # be collected with --loss flow (an mse store has no "chunk" key)
    loss: Literal["mse", "flow"] = "mse"
    chunk: int = 50                   # flow: action-chunk length
    replan: int = 10                  # flow: student actions executed per inference (<= chunk)
    flow_steps: int = 10              # flow: Euler steps integrating the velocity at act()
    hidden_dim: int = 256
    image_hw: int = 64                # square rgb obs size
    encoder: Literal["shared", "separate"] = "shared"
    feat_dim: int = 64                # per-view feature size
    # train
    train_steps: int = 50_000
    batch_size: int = 256
    # gradient accumulation: one optimizer step per ``accum`` micro-batches
    # (effective batch = batch_size * accum). Large LOADER batches trigger the
    # loader-startup kills (pinned-memory spike) — accumulate instead
    accum: int = 1
    lr: float = 1e-3
    lr_final: float = 1e-4            # cosine decay lr -> lr_final over train_steps
    aux_pose_coef: float = 0.1        # aux cube pos+yaw loss weight (0 = off)
    grip_coef: float = 0.1            # BCE weight on the gripper logit
    aug_pad: int = 4                  # DrQ random-shift padding, px (0 = off)
    # sim2real photometric albumentations (xsim.data.augs) in the loader
    # workers: exposure/WB drift, sensor noise, blur, JPEG, occlusion cutouts
    augs: bool = False
    aug_strength: float = 1.0         # scales the albumentations ranges/probs
    ema_decay: float = 0.999          # eval/checkpoint the EMA student
    # warm-start: load this checkpoint into the student (and EMA) before
    # training — finetune/continue a compatible prior run
    init_from: Path | None = None
    save_every: int = 1000            # save EMA -> latest.pt every n steps (0 = eval-time only)
    num_workers: int = 8              # image-mode DataLoader workers (0 = main process)
    # fork workers read the memmap without re-importing this script; spawn
    # workers (re-import + cv2/torch init in a CUDA-parent) got the process
    # silently SIGKILLed on the 2.4M-sample store (liftez-bc-v2, Jul 2026)
    mp_context: Literal["fork", "spawn"] = "fork"
    # eval: EMA-student rollouts on the live n_envs batch during training
    eval_every: int = 10_000          # optimizer steps between evals (0 = final only)
    eval_batches: int = 1             # env-batch rollouts per eval (n_envs each)
    eval_seed: int = 51_000
    eval_video: bool = True           # tile the first eval rollout (image mode) -> eval_sNNNNNN.mp4
    video_envs: int = 16              # tile only the first k envs into the grid
    video_max_width: int = 1280       # per-camera grid width cap, px
    # logging
    wandb_project: str | None = "xarm-sim"
    log_every: int = 1000             # stream bc losses to wandb every n steps (0 = off)
    print_every: int = 100            # print windowed losses + rates to stdout every n steps (0 = off)


def build_env(cfg: Config):
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.renderers import BatchConfig
    from xsim.suite.wrappers import CartesianActionWrapper, GymWrapper, ImageObsWrapper

    image = cfg.policy == "image"
    gs.init(backend=gs.gpu if cfg.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    kwargs = {}
    if cfg.cube_x_range is not None:
        kwargs["x_range"] = cfg.cube_x_range
    if cfg.cube_y_range is not None:
        kwargs["y_range"] = cfg.cube_y_range
    env = make(
        cfg.task, robots="XArm7",
        camera_names=list(cfg.cameras) if image else [],
        camera_res=(cfg.image_hw, cfg.image_hw) if image else (640, 480),
        render_backend="batch" if image else "raster",
        renderer_config=BatchConfig(use_rasterizer=cfg.batch_rasterizer) if image else None,
        init_tcp_box=cfg.init_tcp_box,
        randomize_cameras=cfg.randomize_cameras,
        horizon=cfg.horizon, n_envs=cfg.n_envs,
        control_freq=cfg.control_freq,
        noslip_iterations=cfg.noslip_iterations,
        **kwargs,
    )
    rm = env.robots[0].model
    rm.ik_backend = cfg.ik_backend
    rm.ik_w_pos, rm.ik_w_rot = cfg.ik_w_pos, cfg.ik_w_rot
    rm.ik_w_home, rm.ik_w_limit, rm.ik_w_manip = cfg.ik_w_home, cfg.ik_w_limit, cfg.ik_w_manip
    rm.ik_iters, rm.ik_sc_damping = cfg.ik_iters, cfg.ik_damping
    if cfg.cartesian:
        env = CartesianActionWrapper(env)
    return ImageObsWrapper(env) if image else GymWrapper(env)


def make_teacher(cfg: Config, env, device: torch.device):
    from xsim.suite.policies import LiftExpertPolicy, LiftPolicy

    base = env.unwrapped
    if cfg.teacher == "expert":
        return LiftExpertPolicy(base, cartesian=cfg.cartesian)
    if cfg.teacher == "waypoint":
        return LiftPolicy(base, steps_per_segment=cfg.steps_per_segment,
                          cartesian=cfg.cartesian)
    if cfg.teacher.startswith("mlp:"):
        return MLPTeacher(Path(cfg.teacher[4:]), device)
    raise ValueError(f"unknown teacher {cfg.teacher!r}")


def build_student(cfg: Config, env, device: torch.device) -> nn.Module:
    act_space = env.get_wrapper_attr("single_action_space")
    if cfg.policy == "state":
        return Student(
            obs_dim=env.single_observation_space.shape[0],
            act_dim=act_space.shape[0], hidden=cfg.hidden_dim,
            act_low=act_space.low, act_high=act_space.high,
        ).to(device)
    base_spaces = env.unwrapped.single_observation_space.spaces
    proprio_keys = image_proprio_keys(sorted(base_spaces))
    kwargs = dict(
        proprio_dim=sum(int(np.prod(base_spaces[k].shape)) for k in proprio_keys),
        act_dim=act_space.shape[0], n_views=len(env.views), hw=cfg.image_hw,
        act_low=act_space.low, act_high=act_space.high,
        encoder=cfg.encoder, hidden=cfg.hidden_dim, feat_dim=cfg.feat_dim,
    )
    if cfg.loss == "flow":
        return FlowImageStudent(**kwargs, chunk=cfg.chunk,
                                flow_steps=cfg.flow_steps).to(device)
    return ImageStudent(**kwargs).to(device)


class Timer:
    """Wall-clock windows for the train loop: total time plus named segments
    (e.g. loader wait) accumulated since the last ``window()`` call."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._seg: dict[str, float] = {}

    @contextlib.contextmanager
    def track(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self._seg[name] = self._seg.get(name, 0.0) + time.perf_counter() - t

    def window(self) -> dict[str, float]:
        """{"wall": s, <segment>: s} since the last call, then reset."""
        now = time.perf_counter()
        out = {"wall": now - self._t0, **self._seg}
        self._t0, self._seg = now, {}
        return out


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.image = cfg.policy == "image"
        self.flow = cfg.loss == "flow"
        self.work_dir = cfg.out / cfg.exp_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.work_dir / "metrics.jsonl"
        self.wandb = None
        if cfg.wandb_project:
            import wandb

            self.wandb = wandb.init(
                project=cfg.wandb_project, name=cfg.exp_name,
                config=json.loads(json.dumps(asdict(cfg), default=str)))
            self.wandb.define_metric("bc/step")
            self.wandb.define_metric("bc/*", step_metric="bc/step")
        self.store_root = cfg.data_dir / (cfg.store_name or cfg.exp_name)
        torch.manual_seed(cfg.seed)
        env = build_env(cfg)
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
        if cfg.reuse_data and not self.image:
            raise ValueError("--reuse-data only applies to the image-mode memmap store")
        write = self.image and not cfg.reuse_data
        self.collector = Collector(cfg, env, make_teacher(cfg, env, self.device),
                                   store_root=self.store_root if write else None)
        self.student = build_student(cfg, env, self.device)
        if cfg.init_from is not None:
            self.student.load_state_dict(torch.load(cfg.init_from, map_location=self.device))
        self.ema = copy.deepcopy(self.student)
        self.optim = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self._chunks: dict[str, np.ndarray] = {}  # state-mode dataset (RAM)
        self.best_success = -1.0
        self.bc_step = 0
        self._win: dict[str, float] = {}
        self._win_n = 0
        self._pwin: dict[str, float] = {}
        self._pwin_n = 0
        self.timer = Timer()
        self._start = time.time()

    def log(self, kind: str, d: dict) -> None:
        d = {"kind": kind, "elapsed_s": round(time.time() - self._start, 1), **d}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(d) + "\n")
        if self.wandb is not None:
            self.wandb.log({f"{kind}/{k}": v for k, v in d.items() if k != "kind"})
        pretty = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in d.items() if k != "kind")
        color = {"collect": "white", "bc": "cyan", "eval": "green"}.get(kind, "white")
        print(f"[{color}]\\[{kind}][/] {pretty}")

    def _step_metrics(self, vals: dict[str, float]) -> None:
        """Windowed per-step loss streams: stdout every cfg.print_every
        optimizer steps (with wall-clock rates from the Timer), wandb every
        cfg.log_every."""
        self.bc_step += 1
        cfg = self.cfg
        if cfg.print_every > 0:
            for k, v in vals.items():
                self._pwin[k] = self._pwin.get(k, 0.0) + v
            self._pwin_n += 1
            if self.bc_step % cfg.print_every == 0:
                t = self.timer.window()
                steps_s = self._pwin_n / t["wall"]
                eta_h = (cfg.train_steps - self.bc_step) / steps_s / 3600
                losses = " ".join(f"{k}={v / self._pwin_n:.5g}" for k, v in self._pwin.items())
                extra = "".join(f" {k}%={100 * s / t['wall']:.0f}"
                                for k, s in t.items() if k != "wall")
                print(f"[dim]\\[bc][/] step={self.bc_step}/{cfg.train_steps} {losses} "
                      f"steps/s={steps_s:.2f} sps={steps_s * cfg.batch_size * cfg.accum:.0f}"
                      f"{extra} eta={eta_h:.1f}h")
                self._pwin, self._pwin_n = {}, 0
        if self.wandb is None or cfg.log_every <= 0:
            return
        for k, v in vals.items():
            self._win[k] = self._win.get(k, 0.0) + v
        self._win_n += 1
        if self.bc_step % cfg.log_every == 0:
            d = {f"bc/{k}": v / self._win_n for k, v in self._win.items()}
            self.wandb.log({**d, "bc/step": self.bc_step})
            self._win, self._win_n = {}, 0

    # -- collection --------------------------------------------------------------
    def collect(self) -> None:
        cfg = self.cfg
        batches = math.ceil(cfg.data_episodes / cfg.n_envs)
        chunks: dict[str, list[np.ndarray]] = {}
        for b in range(batches):
            stats = self.collector.rollout(beta=1.0, record=True, student=None,
                                           seed=cfg.seed + 1000 * b)
            if not self.image:
                for k, arr in (self.collector.pop_chunks() or {}).items():
                    chunks.setdefault(k, []).append(arr)
            self.log("collect", {"batch": b, "episodes": (b + 1) * cfg.n_envs, **stats})
        if self.image:
            self.collector.store.flush()
        else:
            self._chunks = {k: np.concatenate(v) for k, v in chunks.items()}

    # -- bc ----------------------------------------------------------------------
    def _obs_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=0)
        return mean, (x * x).mean(dim=0).sub(mean * mean).clamp_min(0.0).sqrt()

    def _setup_state(self):
        self._X = torch.from_numpy(self._chunks["obs"]).to(self.device)
        self._Y = torch.from_numpy(self._chunks["act"]).to(self.device)
        self.student.set_obs_stats(*self._obs_stats(self._X))
        self._perm = torch.randperm(self._X.shape[0], device=self.device)
        self._pos = 0
        return self._X.shape[0]

    def _step_state(self) -> dict[str, float]:
        cfg = self.cfg
        if self._pos + cfg.batch_size > self._X.shape[0]:
            self._perm, self._pos = torch.randperm(self._X.shape[0], device=self.device), 0
        idx = self._perm[self._pos : self._pos + cfg.batch_size]
        self._pos += cfg.batch_size
        loss = F.mse_loss(self.student(self._X[idx]), self._Y[idx])
        (loss / cfg.accum).backward()
        return {"bc_loss": loss.item()}

    def _setup_image(self):
        cfg = self.cfg
        label_key = "chunk" if self.flow else "act"
        ds = MemmapDataset(self.store_root, ("rgb", "prop", label_key, "aux"))
        if cfg.augs:
            ds = AugmentedDataset(ds, sim2real_transform(cfg.aug_strength))
        self.student.set_obs_stats(*self._obs_stats(torch.from_numpy(
            np.asarray(read_key(self.store_root, "prop"))).to(self.device)))
        if self.flow:  # flow matches in normalized action space
            self.student.set_act_stats(*self._obs_stats(torch.from_numpy(
                np.asarray(read_key(self.store_root, "act"))).to(self.device)))
        g = torch.Generator().manual_seed(int(self.rng.integers(2**31)))
        self._loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, generator=g,
            num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
            persistent_workers=cfg.num_workers > 0,
            multiprocessing_context=cfg.mp_context if cfg.num_workers > 0 else None,
        )
        with torch.device("cpu"):  # iterator seeding builds cpu tensors
            self._it = iter(self._loader)
        return len(ds)

    def _next_batch(self):
        # explicit cpu: gs.init makes cuda the torch default device, but the
        # loader's sampler/collate must build cpu tensors
        with self.timer.track("data"), torch.device("cpu"):
            try:
                return next(self._it)
            except StopIteration:  # cycle the dataset until the budget is spent
                self._it = iter(self._loader)
                return next(self._it)

    def _step_image(self) -> dict[str, float]:
        cfg = self.cfg
        rgb, prop, y, aux_y = self._next_batch()
        x = rgb.to(self.device, non_blocking=True).float() / 255.0 - 0.5
        prop = prop.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        aux_y = aux_y.to(self.device, non_blocking=True)
        if cfg.aug_pad > 0:
            nv = x.shape[0] * x.shape[1]
            x = rand_shift(x.reshape(nv, *x.shape[2:]), cfg.aug_pad).reshape(x.shape)
        pred, aux = self.student(x, prop)
        joints = F.mse_loss(pred[:, :-1], y[:, :-1])
        grip = F.binary_cross_entropy_with_logits(pred[:, -1], y[:, -1])
        aux_l = F.mse_loss(aux, aux_y)
        ((joints + cfg.grip_coef * grip + cfg.aux_pose_coef * aux_l) / cfg.accum).backward()
        return {"bc_loss": joints.item(), "grip_bce": grip.item(), "aux_mse": aux_l.item()}

    def _step_flow(self) -> dict[str, float]:
        cfg = self.cfg
        rgb, prop, y, aux_y = self._next_batch()
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
        x1_pred, aux, h = self.student(x, prop, x_t, t)
        fm = F.mse_loss(x1_pred, a)  # endpoint (x1) prediction, not velocity
        aux_l = F.mse_loss(aux, aux_y)
        ((fm + cfg.aux_pose_coef * aux_l) / cfg.accum).backward()
        with torch.no_grad():  # diagnostic comparable to mse bc_loss
            a0 = self.student.sample(h.detach())[:, 0]
            a0_mse = F.mse_loss(a0, y[:, 0]).item()
        return {"flow_loss": fm.item(), "a0_mse": a0_mse, "aux_mse": aux_l.item()}

    @torch.no_grad()
    def _ema_update(self) -> None:
        d = self.cfg.ema_decay
        for pe, p in zip(self.ema.parameters(), self.student.parameters()):
            pe.lerp_(p, 1.0 - d)
        for be, b in zip(self.ema.buffers(), self.student.buffers()):
            be.copy_(b)

    # -- eval --------------------------------------------------------------------
    def evaluate(self, step: int) -> None:
        cfg = self.cfg
        video = (self.work_dir / f"eval_s{step:06d}.mp4"
                 if self.image and cfg.eval_video else None)
        stats = [self.collector.rollout(beta=0.0, record=False, student=self.ema,
                                        seed=cfg.eval_seed + b,
                                        video_path=video if b == 0 else None)
                 for b in range(cfg.eval_batches)]
        if video is not None and self.wandb is not None:
            import wandb

            self.wandb.log({"eval/video": wandb.Video(str(video), format="mp4")})
        success = float(np.mean([s["success"] for s in stats]))
        self.log("eval", {"step": step, "eval_success": success,
                          "eval_len": float(np.mean([s["ep_len"] for s in stats]))})
        torch.save(self.ema.state_dict(), self.work_dir / "latest.pt")
        if success >= self.best_success:
            self.best_success = success
            torch.save(self.ema.state_dict(), self.work_dir / "best.pt")

    # -- loop --------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        if not cfg.reuse_data:
            self.collect()
            if cfg.collect_only:
                if self.wandb is not None:
                    self.wandb.finish()
                print("\\[done] collect-only: store flushed")
                return
        samples = self._setup_image() if self.image else self._setup_state()
        self.log("bc", {"samples": samples})
        step_fn = (self._step_flow if self.flow else self._step_image) if self.image else self._step_state
        sums: dict[str, float] = {}
        for step in range(1, cfg.train_steps + 1):
            frac = (step - 1) / max(1, cfg.train_steps - 1)
            lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1.0 + math.cos(math.pi * frac))
            for g in self.optim.param_groups:
                g["lr"] = lr
            vals: dict[str, float] = {}
            for _ in range(cfg.accum):
                for k, v in step_fn().items():
                    vals[k] = vals.get(k, 0.0) + v / cfg.accum
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self._ema_update()
            self._step_metrics(vals)
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            if cfg.save_every > 0 and step % cfg.save_every == 0:
                torch.save(self.ema.state_dict(), self.work_dir / "latest.pt")
            if cfg.eval_every > 0 and step % cfg.eval_every == 0:
                self.evaluate(step)
        self.log("bc", {"step": cfg.train_steps, "lr": lr,
                        **{k: v / cfg.train_steps for k, v in sums.items()}})
        if cfg.eval_every <= 0 or cfg.train_steps % cfg.eval_every != 0:
            self.evaluate(cfg.train_steps)
        if self.wandb is not None:
            self.wandb.finish()
        print(f"\\[done] best eval success: {self.best_success:.0%}")


def main(cfg: Config) -> None:
    (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
        json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
