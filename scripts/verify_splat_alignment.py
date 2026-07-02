"""Verify the lab-splat ↔ robot-base alignment against the calibration captures.

Renders the Nyx-composited scene (splat room + mesh robot) through the two calibrated
Logitech cameras with the robot posed at a `cap.npz` calibration joint config, and writes
real | sim | blend panels. Re-run this whenever the room is rescanned or the cameras are
recalibrated; re-solve with scripts/align_ransac.py and bake the result into
`xsim.lift_task.DEFAULT_SPLAT_POS/QUAT/SCALE` (overridable here via --pos/--quat/--scale).

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
    DEFAULT_SPLAT_POS,
    DEFAULT_SPLAT_QUAT,
    DEFAULT_SPLAT_SCALE,
    CameraView,
    LiftBlockEnv,
    LiftEnvCfg,
)

CAP_NPZ = Path("/data/store/opencv_calibrated/cap.npz")
SERIAL_BY_CAM = {"low": "1e9c6aae", "side": "ad3f052e"}


@dataclass
class Cfg:
    pos: tuple[float, float, float] = DEFAULT_SPLAT_POS
    quat: tuple[float, float, float, float] = DEFAULT_SPLAT_QUAT  # xyzw
    scale: float = DEFAULT_SPLAT_SCALE
    splat_uri: Path | None = None
    qi: int = 0                                          # cap.npz joint config index
    tag: str = "align"
    out_dir: Path = PROJECT_ROOT / "outputs" / "splat_align"
    nyx_spp: int = 8


def main(c: Cfg) -> None:
    pos, quat = c.pos, c.quat
    print(f"splat_pos={tuple(round(v, 4) for v in pos)} splat_quat(xyzw)={tuple(round(v, 5) for v in quat)} scale={c.scale}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    cams = list(DEFAULT_CAMERAS) + [
        # oblique overview: the mesh robot at the origin must sit on the splat's baked robot
        CameraView("oblique", pos=(1.6, 1.1, 1.3), lookat=(0.2, 0.0, 0.0), fov_deg=75.0),
    ]
    env_cfg = LiftEnvCfg(render_backend="nyx", splat_pos=pos, splat_quat=quat, splat_scale=c.scale, nyx_spp=c.nyx_spp)
    if c.splat_uri is not None:
        env_cfg.splat_uri = c.splat_uri
    env = LiftBlockEnv(env_cfg, cameras=cams)

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
