"""Generate synthetic block-lift episodes as Foxglove MCAP.

Drives ``LiftBlockEnv`` with ``ScriptedLiftPolicy``, records at a fixed rate (decimated
from the physics step), and writes one ``<episode>.mcap`` per rollout via
``EpisodeMcapWriter`` — matching the reference `_base.mcap` schema plus camera
intrinsics/extrinsics and joint states. Grasp success is computed per episode; by default
only successful episodes are kept.

    uv run python scripts/generate_lift_dataset.py --n-episodes 3 --backend gpu
    uv run python scripts/generate_lift_dataset.py --mode preview --backend cpu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Literal

import cv2
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from xsim.lift_task import LiftBlockEnv, LiftEnvCfg  # noqa: E402
from xsim.mcap_writer import CameraSpec, EpisodeMcapWriter  # noqa: E402
from xsim.scripted_lift_policy import ScriptedLiftPolicy  # noqa: E402


@dataclass
class Config:
    mode: Literal["generate", "preview"] = "generate"
    out_dir: Path = PROJECT_ROOT / "outputs" / "sim_mcap" / "lift"
    preview_dir: Path = PROJECT_ROOT / "outputs" / "sim_preview" / "lift_env"
    n_episodes: int = 1
    backend: Literal["gpu", "cpu"] = "gpu"
    seed: int = 0
    steps_per_segment: int = 108    # 8 segments at 120 Hz → ~7.5 s episodes (like real)
    hold_steps: int = 48            # extra settle steps after the sequence
    lift_threshold: float = 0.05    # min cube rise (m) for a successful grasp
    deliver_radius: float = 0.12    # max xy dist (m) from drop zone at episode end
    save_failures: bool = False
    env: LiftEnvCfg = field(default_factory=LiftEnvCfg)


def _spec_dict(env: LiftBlockEnv) -> dict[str, CameraSpec]:
    specs = {}
    for name, (w, h, fx, fy, cx, cy) in env.camera_specs().items():
        specs[name] = CameraSpec(name=name, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return specs


PREVIEW_CAMERA_ORDER = ("low", "side", "wrist", "over")


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write {path}")


def _preview_camera_names(images: dict[str, np.ndarray]) -> list[str]:
    ordered = [name for name in PREVIEW_CAMERA_ORDER if name in images]
    return ordered + [name for name in images if name not in ordered]


def save_preview_frame(env: LiftBlockEnv, preview_dir: Path, step_idx: int, label: str) -> None:
    images = env.render()
    safe = _safe_label(label)
    names = _preview_camera_names(images)
    for name in names:
        _save_rgb_png(preview_dir / f"{step_idx:04d}_{safe}_{name}.png", images[name])

    annotated = []
    for name in names:
        frame = images[name].copy()
        cv2.putText(
            frame,
            f"{step_idx:04d} {label} {name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated.append(frame)
    _save_rgb_png(preview_dir / f"{step_idx:04d}_{safe}_contact.png", np.concatenate(annotated, axis=1))


def run_preview(env: LiftBlockEnv, cfg: Config) -> None:
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=cfg.seed)
    policy = ScriptedLiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
    policy.reset()

    save_preview_frame(env, cfg.preview_dir, 0, "reset")

    waypoint_names = policy.waypoint_names or ["home"]
    milestone_steps = {0: waypoint_names[0]}
    for waypoint_idx, name in enumerate(waypoint_names[1:], start=1):
        milestone_steps[waypoint_idx * cfg.steps_per_segment] = name

    total = policy.n_steps + cfg.hold_steps
    with torch.no_grad():
        for step_idx in range(total):
            cmd = policy.step()
            env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
            env.step()
            if step_idx in milestone_steps:
                save_preview_frame(env, cfg.preview_dir, step_idx, milestone_steps[step_idx])

    save_preview_frame(env, cfg.preview_dir, total - 1, "settled")
    print(f"wrote preview frames to {cfg.preview_dir}")


def run_episode(env: LiftBlockEnv, cfg: Config, episode_idx: int, path: Path) -> dict:
    env.reset(seed=cfg.seed + episode_idx)
    policy = ScriptedLiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
    policy.reset()

    cube_start = env.cube_pos().copy()
    max_rise = 0.0
    record_dt_ns = int(round(env.record_dt * 1e9))
    base_ns = 1_000_000_000
    rec = 0

    specs = _spec_dict(env)
    total = policy.n_steps + cfg.hold_steps
    with EpisodeMcapWriter(path, specs) as writer:
        with torch.no_grad():
            for i in range(total):
                cmd = policy.step()
                env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
                env.step()
                cube = env.cube_pos()
                max_rise = max(max_rise, float(cube[2] - cube_start[2]))
                if i % env.cfg.record_every == 0:
                    imgs = env.render()
                    pos, vel, eff, ee = env.proprio()
                    extr = {name: env.extrinsic_base_cam(name) for name in specs}
                    writer.log_step(base_ns + rec * record_dt_ns, imgs, pos, vel, eff, extr,
                                    ee_pose=ee, gripper_norm=env.gripper_norm())
                    rec += 1

    cube_end = env.cube_pos()
    drop = np.asarray(env.cfg.drop_zone)
    deliver_dist = float(np.linalg.norm(cube_end[:2] - drop[:2]))
    lifted = max_rise >= cfg.lift_threshold
    delivered = deliver_dist <= cfg.deliver_radius
    return {
        "episode": episode_idx, "frames": rec, "max_rise": max_rise,
        "deliver_dist": deliver_dist, "lifted": lifted, "delivered": delivered,
        "success": lifted and delivered,
    }


def main(cfg: Config) -> None:
    backend = gs.gpu if cfg.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    env = LiftBlockEnv(cfg.env)
    if cfg.mode == "preview":
        run_preview(env, cfg)
        return

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    n_success = 0
    for ep in range(cfg.n_episodes):
        path = cfg.out_dir / f"episode_{ep:06d}.mcap"
        stats = run_episode(env, cfg, ep, path)
        keep = stats["success"] or cfg.save_failures
        if not keep:
            path.unlink(missing_ok=True)
        n_success += int(stats["success"])
        flag = "OK " if stats["success"] else ("kept" if keep else "drop")
        print(f"[{flag}] ep{ep}: frames={stats['frames']} rise={stats['max_rise']:.3f} "
              f"deliver={stats['deliver_dist']:.3f} lifted={stats['lifted']} delivered={stats['delivered']}")

    print(f"\nsuccess rate: {n_success}/{cfg.n_episodes}  -> {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
