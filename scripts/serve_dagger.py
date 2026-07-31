"""Serve a DAgger ImageStudent checkpoint over webpolicy for real-robot rollout.

    uv run python scripts/serve_dagger.py --ckpt outputs/dagger/img-v2/best.pt

Client contract (msgpack dict per step; single env, no batch dim):
  obs["low"], obs["side"], obs["wrist"]: (H, W, 3) uint8 RGB — any resolution,
      resized here to the checkpoint's training size (64x64 for img-v2)
  obs["robot0_joint_pos"]:      (7,) arm joint angles, rad
  obs["robot0_joint_vel"]:      (7,) arm joint velocities, rad/s — only for
      legacy 22-dim ckpts (img-v5 and earlier); v8+ no-vel ckpts ignore it
      and the client may omit it (the served key set is in the metadata)
  obs["robot0_eef_pos"]:        (3,) TCP position in the robot base frame, m
  obs["robot0_eef_quat"]:       (4,) TCP orientation, (w, x, y, z)
  obs["robot0_gripper_norm"]:   (1,) gripper opening in [0, 1], 1 = fully open

Returns {"action": (8,) float32} — absolute joint targets j0..j6 (rad, clamped
to the limits baked into the checkpoint) + gripper command in {0, 1} with
1 = open, 0 = close. The policy was trained at control_freq = 30 Hz.
"""

from __future__ import annotations
import time 

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Literal

import cv2
import numpy as np
import torch
import tyro
from rich import print
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simpledagger import FlowImageStudent, ImageStudent, ViTFlowImageStudent  # noqa: E402

from xsim.algo import RTCGuidance  # noqa: E402

VIEWS = ("low", "side", "wrist")  # sorted camera names = training view order
PROPRIO_KEYS = (  # sorted non-cube obs keys, the ImageStudent proprio layout
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_norm",
    "robot0_joint_pos",
    "robot0_joint_vel",
)
# checkpoint proprio_dim -> key set: v8+ image students drop joint_vel (15 dims)
PROPRIO_BY_DIM = {
    22: PROPRIO_KEYS,
    15: tuple(k for k in PROPRIO_KEYS if "vel" not in k),
}


def load_student(ckpt: Path, device: torch.device,
                 flow_steps: int = 10) -> tuple[torch.nn.Module, int]:
    """Rebuild a student from a checkpoint alone (shapes carry the config):
    ``encoder.patch_embed`` keys mark the ViT flow student, ``flow_net`` keys
    the CNN flow student, else the plain CNN mse student."""
    sd = torch.load(ckpt, map_location="cpu")
    act_dim = sd["act_low"].shape[0]
    common = dict(
        proprio_dim=sd["prop_mean"].shape[0],
        n_views=sd["view_emb"].shape[0],
        act_low=sd["act_low"].numpy(),
        act_high=sd["act_high"].numpy(),
    )
    if "encoder.patch_embed.weight" in sd:  # ViT-T + DiT-style flow transformer
        patch = sd["encoder.patch_embed.weight"].shape[-1]
        hw = patch * int(math.isqrt(sd["encoder.pos"].shape[1] - 1))
        net = ViTFlowImageStudent(**common, act_dim=act_dim, hw=hw,
                                  chunk=sd["tok_pos"].shape[1],
                                  flow_steps=flow_steps)
    else:
        hw = 16 * int(math.isqrt(sd["encoder.9.weight"].shape[1] // 32))
        cnn = dict(**common, hw=hw, hidden=sd["trunk.0.weight"].shape[0],
                   feat_dim=sd["view_emb"].shape[1])
        if "flow_net.0.weight" in sd:
            chunk = sd["flow_net.4.weight"].shape[0] // act_dim
            net = FlowImageStudent(**cnn, act_dim=act_dim, chunk=chunk,
                                   flow_steps=flow_steps)
        else:
            net = ImageStudent(**cnn, act_dim=sd["head.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval().to(device)
    return net, hw


class DaggerPolicy(BasePolicy):
    def __init__(self, ckpt: Path, device: torch.device,
                 resize: Literal["squash", "ccrop"] = "squash",
                 flow_steps: int = 10, ema: float = 0.5,
                 mode: Literal["rhc", "rtc", "replan"] = "rhc", execute: int = 10,
                 rtc_delay: int = 2, rtc_beta: float = 10.0,
                 rtc_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp"):
        self.net, self.hw = load_student(ckpt, device, flow_steps=flow_steps)
        self.flow = isinstance(self.net, (FlowImageStudent, ViTFlowImageStudent))
        if mode != "rhc" and not self.flow:
            raise SystemExit(f"--mode {mode} needs a flow checkpoint (chunked plans)")
        self.mode = mode
        self.resize = resize
        self.ema = ema
        # rhc flow: blended (chunk, 8) plan whose row 0 is the current step;
        # each step() shifts it by one and EMA-folds it into the fresh plan, so
        # a prediction made k replans ago carries weight ema * (1 - ema)^k.
        # rtc: the active chunk, executed from cursor self._i.
        self._plan: np.ndarray | None = None
        # rtc: one background worker infers the next chunk (guided toward the
        # executing plan) while step() keeps popping from the current one
        self.execute, self.rtc_beta, self.rtc_schedule = execute, rtc_beta, rtc_schedule
        self._delay0 = max(1, rtc_delay)
        self._pool = ThreadPoolExecutor(max_workers=1) if mode == "rtc" else None
        self._future: Future | None = None
        self._i = self._launch_i = 0  # plan cursor / cursor when inference launched
        self._t = self._launch_t = 0  # steps served / steps when inference launched
        self._delays: deque[int] = deque(maxlen=10)  # measured delays, in steps
        self.prop_keys = PROPRIO_BY_DIM[self.net.prop_mean.shape[0]]
        # remaining rows of the active plan at the last step(): row 0 is the
        # action just returned, row -1 the chunk's final action — read by the
        # viser wrapper, which otherwise only sees the single returned row
        self.horizon: np.ndarray | None = None

    def reset(self, payload: dict | None = None) -> None:
        if self._future is not None:  # drain the in-flight inference, discard it
            self._future.result()
            self._future = None
        self._plan = None
        self.horizon = None
        self._i = self._launch_i = 0
        self._t = self._launch_t = 0

    def _to_hw(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (h, w) == (self.hw, self.hw):
            return img
        if self.resize == "ccrop" and h != w:
            # crop to square before scaling: uniform scale, and 640x480 D435
            # frames land at ~42.6 deg hFOV = the sim camera's, vs ~54.7 raw
            s = min(h, w)
            y, x = (h - s) // 2, (w - s) // 2
            img = img[y : y + s, x : x + s]
        return cv2.resize(img, (self.hw, self.hw), interpolation=cv2.INTER_AREA)

    def _preprocess(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        prop = np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.prop_keys]
        )[None]
        frames = [self._to_hw(np.asarray(obs[name], dtype=np.uint8)) for name in VIEWS]
        rgb = np.stack(frames)[None].transpose(0, 1, 4, 2, 3)  # (1, V, 3, H, W)
        return prop, rgb

    def step(self, obs: dict) -> dict:
        if self.mode == "rtc":
            return {"action": self._step_rtc(obs)}
        if self.mode == "replan":
            return {"action": self._step_replan(obs)}
        action = self.net.act(self._preprocess(obs))[0]
        if self.flow:  # receding horizon: replan every step, execute row 0 of
            # the fresh (chunk, 8) plan EMA-blended with last step's plan
            plan = np.asarray(action, dtype=np.float32)
            if self._plan is not None and len(self._plan) > 1:
                prev = self._plan[1:]  # rows covering this step onward
                plan[: len(prev)] *= self.ema
                plan[: len(prev)] += (1.0 - self.ema) * prev
            self._plan = plan
            self.horizon = plan
            action = plan[0]
        else:
            self.horizon = np.asarray(action, dtype=np.float32)[None]
        return {"action": action}

    def _step_replan(self, obs: dict) -> np.ndarray:
        """Open-loop chunking (the rtc ablation): infer synchronously, execute
        the plan's first `execute` rows blind, then infer again."""
        # time.sleep(0.1)
        if self._plan is None or self._i >= min(self.execute, len(self._plan)):
            self._plan = self.net.act(self._preprocess(obs))[0]
            self._i = 0
        action = self._plan[self._i]
        self.horizon = self._plan[self._i:]
        self._i += 1
        return action

    def _d_est(self) -> int:
        """Frozen-prefix width d: the max of recent measured delays (the
        paper's rule) floored at rtc_delay, bounded by the execution horizon
        (d <= s). Raising rtc_delay above the true delay over-freezes: the
        extra rows are weight-1 copies of the committed plan that DO execute,
        smoothing the boundary at the cost of reacting that many steps later."""
        d = max(self._delays) if self._delays else 0
        return int(np.clip(max(d, self._delay0), 1, self.execute))

    def _swap_in(self, plan: np.ndarray) -> None:
        d = min(self._i - self._launch_i, self.net.chunk - 1)
        self._delays.append(max(1, d))
        # rows [0, d) were frozen to the old plan and have already executed
        self._plan, self._i, self._future = plan, d, None

    def _step_rtc(self, obs: dict) -> np.ndarray:
        """Real-time chunking (Black et al. 2025): pop one action per call from
        the active chunk and launch a background inference every `execute`
        steps; its sampler inpaints the new chunk onto the plan tail that keeps
        executing meanwhile, so the inference latency is hidden from the
        client and chunk boundaries stay consistent."""
        chunk = self.net.chunk
        if self._plan is None:  # first chunk: plain flow sample, no guidance
            self._plan = self.net.act(self._preprocess(obs))[0]
            self._i, self._launch_t = 0, self._t
        if self._future is not None and (
                self._future.done() or self._i >= len(self._plan)):
            self._swap_in(self._future.result()[0])  # blocks only when run dry
        if self._i >= len(self._plan):  # dry with nothing in flight: hard replan
            self._plan = self.net.act(self._preprocess(obs))[0]
            self._i, self._launch_t = 0, self._t
        if self._future is None and self._t - self._launch_t >= self.execute:
            prev = self._plan[self._i:]  # row 0 = the action popped below
            if len(prev) < chunk:  # right-pad; rows past `horizon` weigh zero
                prev = np.concatenate(
                    [prev, np.repeat(prev[-1:], chunk - len(prev), axis=0)])
            rtc = RTCGuidance(prev=prev[None], delay=self._d_est(),
                              horizon=max(0, chunk - self.execute),
                              schedule=self.rtc_schedule, beta=self.rtc_beta)
            self._launch_i, self._launch_t = self._i, self._t
            self._future = self._pool.submit(
                self.net.act, self._preprocess(obs), rtc)
        action = self._plan[self._i]
        self.horizon = self._plan[self._i:]
        self._i += 1
        self._t += 1
        return action


@dataclass
class Config:
    ckpt: Path = Path("outputs/dagger/img-v2/best.pt")
    host: str = "0.0.0.0"
    port: int = 8000
    viser: bool = True                # live viser viewer of each step's inputs/outputs
    viser_port: int = 8080
    urdf: Path = Path(__file__).resolve().parents[1] / "xarm7_standalone.urdf"
    # squash: plain resize to hw x hw (training-era behavior, distorts 4:3 frames)
    # ccrop: center-crop square first (uniform scale, matches the sim camera FOV)
    resize: Literal["squash", "ccrop"] = "squash"
    # flow checkpoints: Euler steps integrating the sampler at act()
    flow_steps: int = 10
    # rhc:    replan every step, execute row 0 of the EMA-blended plan (fully
    #         closed-loop; pays one inference per control step)
    # rtc:    real-time chunking (Black et al. 2025, arXiv:2506.07339) — infer
    #         the next chunk in a background thread while the current one
    #         executes, soft-inpainting its overlap onto the committed plan;
    #         one inference per `execute` steps, latency hidden from the client
    # replan: the rtc ablation — synchronous inference, execute the first
    #         `execute` rows open-loop, infer again (blocks for one inference
    #         at every chunk boundary, no cross-chunk consistency)
    mode: Literal["rhc", "rtc", "replan"] = "rhc"
    # rhc: weight of the fresh plan when EMA-blending with the previous step's
    # plan (temporal ensemble across replans); 1.0 = no smoothing
    ema: float = 0.5
    # rtc/replan: execution horizon s — rows each chunk executes before the
    # next takes over; rtc also cuts the guidance soft mask at chunk - execute
    execute: int = 10
    # rtc: floor on the frozen-prefix width d (true delay is measured and takes
    # over when larger); raise above the real delay to over-freeze — extra rows
    # execute as exact copies of the committed plan, smoothing swap boundaries
    rtc_delay: int = 2
    rtc_beta: float = 10.0   # rtc: max guidance weight clip
    rtc_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp"


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner = DaggerPolicy(cfg.ckpt, device, resize=cfg.resize,
                         flow_steps=cfg.flow_steps, ema=cfg.ema, mode=cfg.mode,
                         execute=cfg.execute, rtc_delay=cfg.rtc_delay,
                         rtc_beta=cfg.rtc_beta, rtc_schedule=cfg.rtc_schedule)
    policy: BasePolicy = inner
    if cfg.viser:
        from xsim.run.viewer import PolicyViewer, ViserWrappedPolicy

        viewer = PolicyViewer(host=cfg.host, port=cfg.viser_port, urdf_path=cfg.urdf)
        policy = ViserWrappedPolicy(inner, viewer, views=VIEWS)
    print(f"serving {cfg.ckpt} on {device} (image {inner.hw}px, views {VIEWS})")
    Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata={
            "ckpt": str(cfg.ckpt),
            "views": list(VIEWS),
            "image_hw": inner.hw,
            "proprio_keys": list(inner.prop_keys),
            "action": "abs joints j0..j6 (rad) + gripper, 1=open"
                      + ("" if not inner.flow else
                         " (receding horizon: replan every step, EMA-blend "
                         "with the previous plan, execute row 0)"
                         if cfg.mode == "rhc" else
                         " (real-time chunking: background inference every "
                         "`execute` steps, guided onto the executing plan)"
                         if cfg.mode == "rtc" else
                         " (open-loop: execute `execute` rows per chunk, "
                         "blocking replan between chunks)"),
            "mode": cfg.mode if inner.flow else "rhc",
            "chunk": inner.net.chunk if inner.flow else 1,
            "ema": cfg.ema if inner.flow else 1.0,
            "execute": cfg.execute if cfg.mode in ("rtc", "replan") else 1,
            "control_freq": 30.0,
        },
    ).serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
