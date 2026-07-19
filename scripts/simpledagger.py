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

from xsim.algo import (
    Collector, Distributed, FlowImageStudent, ImageStudent, MLPTeacher,
    Student, image_proprio_keys, rand_shift,
)
from xsim.data import MemmapDataset, read_key

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
    # bc: a fixed optimizer-step budget per round, not epochs — the aggregate
    # grows every round, so epochs would mean linearly growing round cost; a
    # step budget keeps rounds constant-cost and is what lr/log cadence key off.
    # Each rank runs exactly this many steps (cycling its shard), so DDP ranks
    # stay in lockstep regardless of shard-size imbalance.
    steps_per_round: int = 50_000
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
    # tile only the first k envs into rollout videos (the rollout itself keeps
    # n_envs; a 16-tile grid stays readable in wandb, 2048 does not)
    video_envs: int = 16
    # wandb (rank 0 only under torchrun): every log() line streams as
    # <kind>/<key>, eval videos attach per round; --wandb-project None = off
    wandb_project: str | None = "xarm-sim"
    # stream bc losses to wandb every n optimizer steps (windowed mean over the
    # last n, x-axis bc/step; rank 0's local values). 0 = per-round summaries only
    log_every: int = 1000
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
    # play mode (scripts/play.py; shares this Config): load a checkpoint, roll
    # the student out once (seeded), write an all-envs grid video (green border
    # = success tick, red = timeout) and a per-env spawn/outcome table
    play: Path | None = None
    play_video: Path | None = None    # default: <checkpoint dir>/rollout.mp4
    cameras: tuple[str, ...] = ("low", "side", "wrist")
    spp: int = 8
    video_max_width: int = 1280       # per-camera grid width cap, px


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
        proprio_keys = image_proprio_keys(sorted(base_spaces))
        spec["proprio_dim"] = sum(
            int(np.prod(base_spaces[k].shape)) for k in proprio_keys)
        spec["n_views"] = len(env.views)
    return spec


def make_teacher(cfg: Config, env, device: torch.device):
    """Teacher wiring stays in the script: scripted teachers come from the
    suite (env-side), the mlp teacher from a checkpoint."""
    from xsim.suite.policies import LiftExpertPolicy, LiftPolicy

    base = env.unwrapped
    if cfg.teacher == "waypoint":
        return LiftPolicy(base, steps_per_segment=cfg.steps_per_segment,
                          cartesian=cfg.cartesian)
    if cfg.teacher == "expert":
        return LiftExpertPolicy(base, cartesian=cfg.cartesian)
    if cfg.teacher.startswith("mlp:"):
        return MLPTeacher(Path(cfg.teacher[4:]), device)
    raise ValueError(f"unknown teacher {cfg.teacher!r}")


def make_collector(cfg: Config, store_root: Path | None,
                   seed_offset: int = 0) -> Collector:
    env = build_env(cfg)
    device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
    return Collector(cfg, env, make_teacher(cfg, env, device),
                     store_root=store_root, seed_offset=seed_offset)


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
            # step-stream bc metrics get their own x-axis so they don't fight
            # the per-round log() calls over wandb's global step counter
            self.wandb.define_metric("bc/step")
            self.wandb.define_metric("bc/*", step_metric="bc/step")
        store_base = cfg.data_dir / cfg.exp_name
        self.store_root = (store_base / f"shard_{self.dist.rank}"
                           if self.dist.enabled else store_base)
        # identical init weights on every rank (DDP requirement), then
        # per-rank env reset spacing and beta/shuffle streams
        torch.manual_seed(cfg.seed)
        self.collector = make_collector(
            cfg, store_root=self.store_root if self.image else None,
            seed_offset=100003 * self.dist.rank)
        self.device = self.collector.device
        self.student = build_student(cfg, self.collector.env, self.device)
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
        self.bc_step = 0  # optimizer steps across rounds (bc/step wandb axis)
        self._win: dict[str, float] = {}
        self._win_n = 0
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

    def _step_metrics(self, vals: dict[str, float]) -> None:
        """Per-optimizer-step loss stream: windowed mean to wandb every
        cfg.log_every steps. Rank 0's local values — no cross-rank sync in the
        hot loop; the per-round log() summary stays globally reduced."""
        self.bc_step += 1
        if self.wandb is None or self.cfg.log_every <= 0:
            return
        for k, v in vals.items():
            self._win[k] = self._win.get(k, 0.0) + v
        self._win_n += 1
        if self.bc_step % self.cfg.log_every == 0:
            d = {f"bc/{k}": v / self._win_n for k, v in self._win.items()}
            self.wandb.log({**d, "bc/step": self.bc_step})
            self._win, self._win_n = {}, 0

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
        perm, pos = torch.randperm(n, device=self.device), 0
        losses = []
        for _ in range(cfg.steps_per_round):
            if pos + cfg.batch_size > n:  # reshuffle when the shard is exhausted
                perm, pos = torch.randperm(n, device=self.device), 0
            idx = perm[pos : pos + cfg.batch_size]
            pos += cfg.batch_size
            loss = F.mse_loss(self.net(X[idx]), Y[idx])
            loss.backward()
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self._ema_update()
            lv = loss.item()
            self._step_metrics({"bc_loss": lv})
            losses.append(lv)
        samples = int(self.dist.mean(float(n)) * self.dist.world)
        return {"samples": samples,
                "bc_loss": self.dist.mean(float(np.mean(losses)))}

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
        sums = {"bc_loss": 0.0, "grip_bce": 0.0, "aux_mse": 0.0}
        # explicit cpu: gs.init makes cuda the torch default device, but the
        # loader's iterator seeding and sampler/collate must build cpu tensors
        with torch.device("cpu"):
            it = iter(loader)
            for _ in range(cfg.steps_per_round):
                try:
                    rgb, prop, y, aux_y = next(it)
                except StopIteration:  # cycle the shard until the budget is spent
                    it = iter(loader)
                    rgb, prop, y, aux_y = next(it)
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
                vals = {"bc_loss": joints.item(), "grip_bce": grip.item(),
                        "aux_mse": aux_l.item()}
                self._step_metrics(vals)
                for k, v in vals.items():
                    sums[k] += v
        del it, loader  # release persistent workers before the next collect
        stats = {k: self.dist.mean(v / cfg.steps_per_round) for k, v in sums.items()}
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
        sums = {"flow_loss": 0.0, "a0_mse": 0.0, "aux_mse": 0.0}
        # explicit cpu: gs.init makes cuda the torch default device, but the
        # loader's iterator seeding and sampler/collate must build cpu tensors
        with torch.device("cpu"):
            it = iter(loader)
            for _ in range(cfg.steps_per_round):
                try:
                    rgb, prop, y, aux_y = next(it)
                except StopIteration:  # cycle the shard until the budget is spent
                    it = iter(loader)
                    rgb, prop, y, aux_y = next(it)
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
                    a0_mse = F.mse_loss(a0, y[:, 0]).item()
                vals = {"flow_loss": fm.item(), "a0_mse": a0_mse,
                        "aux_mse": aux_l.item()}
                self._step_metrics(vals)
                for k, v in vals.items():
                    sums[k] += v
        del it, loader  # release persistent workers before the next collect
        stats = {k: self.dist.mean(v / cfg.steps_per_round) for k, v in sums.items()}
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


def main(cfg: Config) -> None:
    if cfg.play is not None:
        raise SystemExit("play mode moved: uv run python scripts/play.py --play <ckpt.pt> ...")
    if int(os.environ.get("RANK", 0)) == 0:
        (cfg.out / cfg.exp_name).mkdir(parents=True, exist_ok=True)
        with (cfg.out / cfg.exp_name / "config.json").open("w") as f:
            json.dump(json.loads(json.dumps(asdict(cfg), default=str)), f, indent=2)
    Trainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(Config))
