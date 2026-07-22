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
from simpledagger import FlowImageStudent, ImageStudent  # noqa: E402

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
                 flow_steps: int = 10) -> tuple[ImageStudent, int]:
    """Rebuild an Image/FlowImage student from a checkpoint alone (shapes carry
    the config; ``flow_net`` keys mark the rectified-flow chunk head)."""
    sd = torch.load(ckpt, map_location="cpu")
    n_views, feat_dim = sd["view_emb"].shape
    hw = 16 * int(math.isqrt(sd["encoder.9.weight"].shape[1] // 32))
    kwargs = dict(
        proprio_dim=sd["prop_mean"].shape[0],
        n_views=n_views,
        hw=hw,
        act_low=sd["act_low"].numpy(),
        act_high=sd["act_high"].numpy(),
        hidden=sd["trunk.0.weight"].shape[0],
        feat_dim=feat_dim,
    )
    if "flow_net.0.weight" in sd:
        act_dim = sd["act_low"].shape[0]
        chunk = sd["flow_net.4.weight"].shape[0] // act_dim
        net = FlowImageStudent(**kwargs, act_dim=act_dim, chunk=chunk,
                               flow_steps=flow_steps)
    else:
        net = ImageStudent(**kwargs, act_dim=sd["head.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval().to(device)
    return net, hw


class DaggerPolicy(BasePolicy):
    def __init__(self, ckpt: Path, device: torch.device,
                 resize: Literal["squash", "ccrop"] = "squash",
                 flow_steps: int = 10, replan: int = 10):
        self.net, self.hw = load_student(ckpt, device, flow_steps=flow_steps)
        self.flow = isinstance(self.net, FlowImageStudent)
        self.resize = resize
        self.replan = replan
        # flow: receding-horizon queue — the first `replan` rows of the latest
        # plan; one row pops per step() call, empty queue = run inference again.
        # The client keeps the plain one-(8,)-action-per-step contract.
        self._queue: list[np.ndarray] = []
        self.prop_keys = PROPRIO_BY_DIM[self.net.prop_mean.shape[0]]

    def reset(self, payload: dict | None = None) -> None:
        self._queue = []

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

    def step(self, obs: dict) -> dict:
        if self.flow and self._queue:
            action = self._queue.pop(0)
            print(action)
            return {"action": action}
        prop = np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.prop_keys]
        )[None]
        frames = [self._to_hw(np.asarray(obs[name], dtype=np.uint8)) for name in VIEWS]
        rgb = np.stack(frames)[None].transpose(0, 1, 4, 2, 3)  # (1, V, 3, H, W)
        action = self.net.act((prop, rgb))[0]
        if self.flow:  # (chunk, 8) plan -> queue the first `replan` rows
            self._queue = list(action[: self.replan])
            action = self._queue.pop(0)
        print(action)
        return {"action": action}


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
    # flow checkpoints: Euler steps integrating the sampler at act(); the
    # recommended client execution horizon before re-querying is `replan`
    flow_steps: int = 10
    replan: int = 10


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner = DaggerPolicy(cfg.ckpt, device, resize=cfg.resize,
                         flow_steps=cfg.flow_steps, replan=cfg.replan)
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
                      + (" (server-side receding horizon: inference every "
                         "`replan` calls)" if inner.flow else ""),
            "chunk": inner.net.chunk if inner.flow else 1,
            "replan": cfg.replan if inner.flow else 1,
            "control_freq": 30.0,
        },
    ).serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
