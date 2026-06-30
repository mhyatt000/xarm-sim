import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

import genesis as gs
import scripted_grasp_policy_room as room


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


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

    env_cfg = dict(room.ROOM_ENV_CFG)
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
        envmap_hdr=str(Path("lab-hdri.exr")),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/sim2real_static_sweep")
    parser.add_argument("--reference", default="left.png")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--envmap-multipliers", default="0.8,1.2,1.6")
    parser.add_argument("--envmap-rotations", default="0,30,60")
    parser.add_argument("--camera-fovs", default=str(room.ZED_LEFT_FHD_VERTICAL_FOV_DEG))
    parser.add_argument("--floor-roughnesses", default="0.25,0.45")
    parser.add_argument("--floor-metallics", default="0.0,0.3")
    parser.add_argument("--floor-color", default="#6f767b")
    parser.add_argument("--wall-color", default="#aaabab")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--case-name")
    parser.add_argument("--envmap-multiplier", type=float)
    parser.add_argument("--envmap-rotation", type=float)
    parser.add_argument("--camera-fov", type=float)
    parser.add_argument("--floor-roughness", type=float)
    parser.add_argument("--floor-metallic", type=float)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_case:
        backend = gs.cpu if args.backend == "cpu" else gs.gpu
        gs.init(backend=backend, precision="32", logging_level="warning")
        render_case(
            output_dir=output_dir,
            case_name=args.case_name,
            envmap_multiplier=args.envmap_multiplier,
            envmap_rotation_deg=args.envmap_rotation,
            camera_fov=args.camera_fov,
            floor_roughness=args.floor_roughness,
            floor_metallic=args.floor_metallic,
            floor_color=room.hex_rgb(args.floor_color),
            wall_color=room.hex_rgb(args.wall_color),
            steps=args.steps,
        )
        return

    reference = None
    ref_path = Path(args.reference)
    if ref_path.exists():
        reference = np.asarray(Image.open(ref_path).convert("RGB"))

    rows = []
    combos = itertools.product(
        parse_floats(args.envmap_multipliers),
        parse_floats(args.envmap_rotations),
        parse_floats(args.camera_fovs),
        parse_floats(args.floor_roughnesses),
        parse_floats(args.floor_metallics),
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
                args.backend,
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
                args.floor_color,
                "--wall-color",
                args.wall_color,
                "--steps",
                str(args.steps),
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
    main()
