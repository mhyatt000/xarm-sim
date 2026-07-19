"""DAgger collection layer: obs adapters, teachers-as-labelers, and the
per-process Collector that rolls a beta-mixture of teacher and student.

Env and teacher are injected — this module never builds environments or
imports xsim.suite (the suite stays env-only). The script owning the run
wires them up and keeps the training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from xsim.algo.nets import Student
from xsim.data import MemmapStore
from xsim.utils.timer import Timer
from xsim.utils.video import VideoSink, tile_grid


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


class Collector:
    """One process's collection stack: env + teacher + recording.

    ``cfg`` is duck-typed — any object with the attrs read here (policy, loss,
    chunk, replan, frame_stride, n_envs, backend, seed, video_max_width). The
    single-process trainer owns one directly; under torchrun each rank owns
    one pinned to its GPU.
    """

    def __init__(self, cfg, env, teacher, store_root: Path | None,
                 seed_offset: int = 0, n_envs: int | None = None):
        self.cfg = cfg
        self.n_envs = n_envs if n_envs is not None else cfg.n_envs
        self.image = cfg.policy == "image"
        self.flow = cfg.loss == "flow"
        if self.flow:
            if self.image and cfg.frame_stride != 1:
                raise ValueError("flow chunk labels need every step: frame_stride=1")
            if not 1 <= cfg.replan <= cfg.chunk:
                raise ValueError("need 1 <= replan <= chunk")
        self.env = env
        self.teacher = teacher
        base = env.unwrapped
        self.state_keys = sorted(base.single_observation_space.spaces)
        self.proprio_keys = image_proprio_keys(self.state_keys)
        self.device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
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
        append in _record's tick-major live-masked order, so chunk labels stay
        row-aligned with the obs (image: chunk.bin with rgb.bin on disk; state:
        the RAM ``chunk`` list with ``obs``). acts: (T, B, A); lives: (T, B)."""
        last = lives.sum(axis=0) - 1  # (B,) each env's final live tick
        ar = np.arange(self.cfg.chunk)
        for t in range(acts.shape[0]):
            envs = np.flatnonzero(lives[t])
            idx = np.minimum(t + ar[None, :], last[envs, None])  # (n_live, chunk)
            labels = acts[idx, envs[:, None]]  # (n_live, chunk, A)
            if self.store is not None:
                self.store.append("chunk", labels)
            else:
                self._chunks.setdefault("chunk", []).append(
                    labels.astype(np.float32, copy=True))

    def rollout(self, beta: float, record: bool, student: nn.Module,
                seed: int | None = None, video_path: Path | None = None,
                on_metrics=None) -> dict:
        """One synchronous env-batch episode; per-env Bernoulli(beta) picks the
        teacher's action over the student's — per step, or per replan window in
        flow mode, where the student emits a chunk executed receding-horizon.
        Every visited state is labeled with the teacher action when ``record``
        is on. ``video_path`` (image mode) streams the policy's own frames as a
        play-style bordered grid mp4. ``on_metrics`` receives windowed t/*
        means + live-env count every cfg.tick_log_every ticks (live streaming;
        the returned stats keep the full-rollout means)."""
        env, B, cfg = self.env, self.n_envs, self.cfg
        timer = Timer()
        with timer("reset"):
            obs, _ = env.reset(seed=seed)
        self.teacher.reset()
        # eval rollouts (pure student, nothing recorded) never consume the
        # teacher's action — skip the expert entirely
        need_teacher = record or beta > 0.0
        live = np.ones(B, dtype=bool)
        success = np.zeros(B, dtype=bool)
        ep_len = np.zeros(B, dtype=np.int64)
        tick = 0
        plan = use_teacher = None
        teacher_a = None
        stream_k = getattr(cfg, "tick_log_every", 0) or 0
        prev_times: dict[str, float] = {}
        prev_counts: dict[str, int] = {}
        acts_hist: list[np.ndarray] = []  # flow: per-tick teacher labels for stitching
        live_hist: list[np.ndarray] = []
        sink = status = None
        if video_path is not None:
            sink = VideoSink(video_path, 1.0 / env.unwrapped.control_dt)
            status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail
        # video shows only the first k envs; the rollout itself keeps all B
        k = min(getattr(cfg, "video_envs", B) or B, B)

        def snap(o) -> None:
            rgb = o["rgb"][:k].transpose(0, 1, 3, 4, 2)  # (k, V, H, W, 3)
            sink.add(np.concatenate(
                [tile_grid(rgb[:, i], cfg.video_max_width, status[:k], upscale=True)
                 for i in range(rgb.shape[1])], axis=1))

        if sink is not None:
            snap(obs)
        while live.any():
            if need_teacher:
                with timer("teacher"):
                    teacher_a = self.teacher.act(self._teacher_obs(obs))
            if beta >= 1.0:
                action = teacher_a
            else:
                with timer("student"):
                    if self.flow:
                        if tick % cfg.replan == 0:
                            plan = student.act(self._student_obs(obs))  # (B, chunk, A)
                            use_teacher = self.rng.random(B) < beta
                        student_a = plan[:, tick % cfg.replan]
                    else:
                        student_a = student.act(self._student_obs(obs))
                        use_teacher = self.rng.random(B) < beta
                action = (np.where(use_teacher[:, None], teacher_a, student_a)
                          if need_teacher else student_a)
            if record and (not self.image or tick % cfg.frame_stride == 0):
                with timer("record"):
                    self._record(obs, teacher_a, live)
                if self.flow:
                    acts_hist.append(teacher_a.astype(np.float32, copy=True))
                    live_hist.append(live.copy())
            tick += 1
            with timer("env_step"):
                obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            if sink is not None:
                status[live & done & info["success"]] = 1
                status[live & done & ~info["success"]] = 2
            success |= live & info["success"]
            ep_len += live
            live &= ~done
            if sink is not None:
                with timer("video"):
                    snap(obs)
            if on_metrics is not None and stream_k and tick % stream_k == 0:
                # windowed means since the last emission, without resetting the
                # rollout-total accumulators behind the returned stats
                win = {f"t/{k}": round(
                    (timer.times[k] - prev_times.get(k, 0.0))
                    / max(1, timer.counts[k] - prev_counts.get(k, 0)), 4)
                    for k in timer.counts if k != "reset"}
                prev_times, prev_counts = dict(timer.times), dict(timer.counts)
                on_metrics({**win, "live": int(live.sum())})
        if sink is not None:
            sink.close()
        if record and self.flow:
            self._append_chunk_labels(np.stack(acts_hist), np.stack(live_hist))
        # per-tick phase averages (reset is per rollout); "t/" keys ride the
        # collect/eval log lines
        timing = {f"t/{k}": round(v, 4)
                  for k, v in timer.get_average_times().items()}
        return {"success": float(success.mean()), "ep_len": float(ep_len.mean()),
                **timing}
