"""Verify the lab-splat ↔ robot-base alignment against the calibration captures.

Renders the Nyx-composited scene (splat room + mesh robot) through the two calibrated
Logitech cameras with the robot posed at a `cap.npz` calibration joint config, and writes
real | sim | blend panels. Re-run this whenever the room is rescanned or the cameras are
recalibrated; tweak --yaw-deg / --base-* until the room lines up, then bake the values
into `xsim.lift_task.SPLAT_YAW_DEG` / `SPLAT_ROBOT_BASE`.

    uv run python scripts/verify_splat_alignment.py --tag check
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.lift_task import (  # noqa: E402
    DEFAULT_CAMERAS,
    SPLAT_ROBOT_BASE,
    SPLAT_YAW_DEG,
    CameraView,
    LiftBlockEnv,
    LiftEnvCfg,
    splat_world_transform,
)

CAP_NPZ = Path("/data/store/opencv_calibrated/cap.npz")
SERIAL_BY_CAM = {"low": "1e9c6aae", "side": "ad3f052e"}


@dataclass
class Cfg:
    yaw_deg: float = SPLAT_YAW_DEG
    base: tuple[float, float, float] = SPLAT_ROBOT_BASE  # robot base in splat coords
    qi: int = 0                                          # cap.npz joint config index
    tag: str = "align"
    out_dir: Path = PROJECT_ROOT / "outputs" / "splat_align"
    nyx_spp: int = 8


def main(c: Cfg) -> None:
    pos, quat = splat_world_transform(c.yaw_deg, c.base)
    print(f"splat_pos={tuple(round(v, 4) for v in pos)} splat_quat(xyzw)={tuple(round(v, 5) for v in quat)}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    cams = list(DEFAULT_CAMERAS) + [
        # oblique overview: the mesh robot at the origin must sit on the splat's baked robot
        CameraView("oblique", pos=(1.6, 1.1, 1.3), lookat=(0.2, 0.0, 0.0), fov_deg=75.0),
    ]
    env = LiftBlockEnv(
        LiftEnvCfg(render_backend="nyx", splat_pos=pos, splat_quat=quat, nyx_spp=c.nyx_spp),
        cameras=cams,
    )

    d = np.load(CAP_NPZ, allow_pickle=True)
    qpos = torch.zeros(13, dtype=torch.float32, device=env.device)
    qpos[:7] = torch.tensor(d["q"][c.qi, :7], dtype=torch.float32)
    ent = env.robot._robot_entity
    ent.set_qpos(qpos, zero_velocity=True, skip_forward=False)
    ent.control_dofs_position(qpos.reshape(1, -1))  # hold against gravity
    env.step()  # scene.step refreshes the renderer's visual state

    c.out_dir.mkdir(parents=True, exist_ok=True)
    imgs = env.render()
    cv2.imwrite(str(c.out_dir / f"{c.tag}_oblique.png"), cv2.cvtColor(imgs["oblique"], cv2.COLOR_RGB2BGR))
    for cam, serial in SERIAL_BY_CAM.items():
        real = d[f"img__camera_logitech_{serial}"][c.qi]
        blend = cv2.addWeighted(real, 0.5, imgs[cam], 0.5, 0)
        panel = np.concatenate([real, imgs[cam], blend], axis=1)
        cv2.imwrite(str(c.out_dir / f"{c.tag}_{cam}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"wrote {c.tag}_{{low,side,oblique}}.png -> {c.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
