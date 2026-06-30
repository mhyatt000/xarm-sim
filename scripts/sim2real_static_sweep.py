import csv
from dataclasses import dataclass
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypeVar

import av
import numpy as np
import torch
import tyro
from PIL import Image

import genesis as gs
import scripted_grasp_policy_room as room


T = TypeVar("T")


@dataclass
class Config:
    """Configuration for the sim-to-real static sweep."""

    output_dir: Path = Path("outputs/sim2real_static_sweep")  # Directory for rendered cases and the CSV summary.
    reference: Path = Path("left.png")  # Optional reference image for image statistics.
    backend: Literal["cpu", "gpu"] = "gpu"  # Genesis backend to initialize.
    envmap_multipliers: str = "0.8,1.2,1.6"  # Comma-separated environment map multipliers.
    envmap_rotations: str = "0,30,60"  # Comma-separated environment map rotations in degrees.
    camera_fovs: str = str(room.ZED_LEFT_FHD_VERTICAL_FOV_DEG)  # Comma-separated camera FOV values.
    floor_roughnesses: str = "0.25,0.45"  # Comma-separated floor roughness values.
    floor_metallics: str = "0.0,0.3"  # Comma-separated floor metallic values.
    floor_color: str = "#6f767b"  # Floor color as a hex RGB string.
    wall_color: str = "#aaabab"  # Wall color as a hex RGB string.
    steps: int = 3  # Simulation steps to render for each static case.
    single_case: bool = False  # Render one case instead of launching the full sweep.
    case_name: str | None = None  # Single-case output stem.
    envmap_multiplier: float | None = None  # Single-case environment map multiplier.
    envmap_rotation: float | None = None  # Single-case environment map rotation in degrees.
    camera_fov: float | None = None  # Single-case camera FOV.
    floor_roughness: float | None = None  # Single-case floor roughness.
    floor_metallic: float | None = None  # Single-case floor metallic value.


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def require_single_case_value(value: T | None, name: str) -> T:
    if value is None:
        raise SystemExit(f"--{name.replace('_', '-')} is required with --single-case.")
    return value


def render_case(
    output_dir: Path,
    case_name: str,
    envmap_multiplier: float,
    envmap_rotation_deg: float,
    camera_fov: float,
    floor_roughness: float,
    floor_metallic: float,
    floor_color: tuple[float, float, float],
    wall_color: tuple[float, float, float],
    steps: int,
) -> Path:
    mp4_path = output_dir / f"{case_name}.mp4"
    png_path = output_dir / f"{case_name}.png"

    env_cfg = room.make_room_env_cfg()
    env_cfg.update(
        {
            "envmap_multiplier": envmap_multiplier,
            "envmap_rotation_deg": envmap_rotation_deg,
            "envmap_camera_fov": camera_fov,
            "floor_roughness": floor_roughness,
            "floor_metallic": floor_metallic,
            "floor_color": floor_color,
            "walls": [
                {**wall, "color": wall_color}
                for wall in room.ROOM_ENV_CFG["walls"]
            ],
        }
    )

    env = room.base.build_env(
        video_path=str(mp4_path),
        show_viewer=False,
        envmap_hdr=str(room.DEFAULT_HDR_PATH),
        use_nyx_camera=True,
        extra_env_cfg=env_cfg,
        robot_cfg_overrides=room.XARM7_ROBOT_CFG,
    )
    with torch.no_grad():
        env.reset()
        for _ in range(steps):
            env.scene.step()
    env.scene.stop_recording()

    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(png_path)
            break
    return png_path


def image_stats(path: Path, reference: np.ndarray | None) -> dict[str, float]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    stats = {
        "mean": float(image.mean()),
        "std": float(image.std()),
        "mean_r": float(image[..., 0].mean()),
        "mean_g": float(image[..., 1].mean()),
        "mean_b": float(image[..., 2].mean()),
        "dark_frac": float((image.mean(axis=2) < 35).mean()),
        "bright_frac": float((image.mean(axis=2) > 220).mean()),
    }
    if reference is not None:
        ref_img = Image.fromarray(reference.astype(np.uint8)).resize(
            (image.shape[1], image.shape[0])
        )
        ref = np.asarray(ref_img, dtype=np.float32)
        stats["mae_to_left_png"] = float(np.abs(image - ref).mean())
    return stats


def main(cfg: Config) -> None:
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.single_case:
        backend = gs.cpu if cfg.backend == "cpu" else gs.gpu
        gs.init(backend=backend, precision="32", logging_level="warning")
        render_case(
            output_dir=output_dir,
            case_name=require_single_case_value(cfg.case_name, "case_name"),
            envmap_multiplier=require_single_case_value(cfg.envmap_multiplier, "envmap_multiplier"),
            envmap_rotation_deg=require_single_case_value(cfg.envmap_rotation, "envmap_rotation"),
            camera_fov=require_single_case_value(cfg.camera_fov, "camera_fov"),
            floor_roughness=require_single_case_value(cfg.floor_roughness, "floor_roughness"),
            floor_metallic=require_single_case_value(cfg.floor_metallic, "floor_metallic"),
            floor_color=room.hex_rgb(cfg.floor_color),
            wall_color=room.hex_rgb(cfg.wall_color),
            steps=cfg.steps,
        )
        return

    reference = None
    ref_path = cfg.reference
    if ref_path.exists():
        reference = np.asarray(Image.open(ref_path).convert("RGB"))

    rows = []
    combos = itertools.product(
        parse_floats(cfg.envmap_multipliers),
        parse_floats(cfg.envmap_rotations),
        parse_floats(cfg.camera_fovs),
        parse_floats(cfg.floor_roughnesses),
        parse_floats(cfg.floor_metallics),
    )
    for idx, (multiplier, rotation, fov, roughness, metallic) in enumerate(combos):
        case_name = (
            f"case_{idx:03d}_m{multiplier:.2f}_r{rotation:.0f}_"
            f"fov{fov:.2f}_fr{roughness:.2f}_fm{metallic:.2f}"
        )
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--single-case",
                "--backend",
                cfg.backend,
                "--output-dir",
                str(output_dir),
                "--case-name",
                case_name,
                "--envmap-multiplier",
                str(multiplier),
                "--envmap-rotation",
                str(rotation),
                "--camera-fov",
                str(fov),
                "--floor-roughness",
                str(roughness),
                "--floor-metallic",
                str(metallic),
                "--floor-color",
                cfg.floor_color,
                "--wall-color",
                cfg.wall_color,
                "--steps",
                str(cfg.steps),
            ],
            check=True,
        )
        png_path = output_dir / f"{case_name}.png"
        stats = image_stats(png_path, reference)
        row = {
            "case": case_name,
            "png": str(png_path),
            "envmap_multiplier": multiplier,
            "envmap_rotation_deg": rotation,
            "camera_fov": fov,
            "floor_roughness": roughness,
            "floor_metallic": metallic,
            **stats,
        }
        rows.append(row)
        print(row)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
