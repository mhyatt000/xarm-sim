from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import tyro

import genesis as gs
from xsim import scripted_grasp_policy as base


def hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


ZED_FHD_RESOLUTION = (1920, 1080)
ZED_LEFT_FHD_FY = 1066.84
ZED_LEFT_FHD_VERTICAL_FOV_DEG = 53.69412375493809
DEFAULT_HDR_PATH = PROJECT_ROOT / "assets" / "lab-hdri.hdr"
DEFAULT_SPLAT_PATH = PROJECT_ROOT / "last.ply"
ROBOT_URDF_PATH = PROJECT_ROOT / "xarm7_standalone.urdf"


ROOM_ENV_CFG = {
    "floor_color": hex_rgb("#535362"),
    "floor_metallic": 0.5,
    "floor_roughness": 0.25,
    "floor_visualization": False,
    "object_color": hex_rgb("#992722"),
    "box_size": (0.0273611977, 0.0273611977, 0.0273611977),
    "envmap_resolution": ZED_FHD_RESOLUTION,
    "envmap_camera_fov": ZED_LEFT_FHD_VERTICAL_FOV_DEG,
    "envmap_camera_pos": (1.1204473972320557, 0.445988267660141, 0.23296283185482025),
    "envmap_camera_lookat": (0.28134316205978394, -0.09777036309242249, 0.24815453682094812),
    "envmap_camera_up": (0.025124434381723404, -0.010843032971024513, 0.9996255040168762),
    "nyx_light_fields": [],
    "walls": [],
}


XARM7_GRIPPER_OPEN_QPOS = 0.85
XARM7_GRIPPER_CLOSED_QPOS = 0.0


XARM7_ROBOT_CFG = {
    "robot_morph": "urdf",
    "robot_file": str(ROBOT_URDF_PATH),
    "robot_fixed": True,
    "merge_fixed_links": False,
    "ee_link_name": "link_tcp",
    "gripper_link_names": ["left_finger", "right_finger"],
    "arm_dof_dim": 7,
    "gripper_dof_dim": 6,
    "default_arm_dof": [math.radians(v) for v in [0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0]],
    "default_gripper_dof": [XARM7_GRIPPER_OPEN_QPOS] * 6,
    "gripper_open_dof": XARM7_GRIPPER_OPEN_QPOS,
    "gripper_close_dof": XARM7_GRIPPER_CLOSED_QPOS,
    "dofs_kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350, 350, 350, 350, 350],
    "dofs_kv": [450, 450, 350, 350, 200, 200, 200, 35, 35, 35, 35, 35, 35],
    "dofs_force_lower": [-50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50],
    "dofs_force_upper": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    "ik_method": "dls_ik",
    "ik_init_at_home": True,
    "ik_max_samples": 50,
    "ik_max_solver_iters": 40,
}


def quat_xyzw_from_rpy_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def make_room_env_cfg(
    splat_uri: str | Path | None = None,
    splat_pos: tuple[float, float, float] | None = None,
    splat_rot_rpy_deg: tuple[float, float, float] | None = None,
    splat_quat: tuple[float, float, float, float] | None = None,
    splat_scale: float | None = None,
) -> dict:
    env_cfg = deepcopy(ROOM_ENV_CFG)
    if splat_uri is None and DEFAULT_SPLAT_PATH.exists():
        splat_uri = DEFAULT_SPLAT_PATH

    if splat_uri is not None:
        splat_cfg = {
            "uri": str(Path(splat_uri).expanduser()),
            "rotation": (0.0, 0.0, -0.70710678, 0.70710678),
        }
        env_cfg["nyx_light_fields"] = [splat_cfg]
    elif any(value is not None for value in (splat_pos, splat_rot_rpy_deg, splat_quat, splat_scale)):
        raise ValueError("Splat transform arguments require --splat-uri or a repo-root last.ply.")
    else:
        return env_cfg

    if splat_pos is not None:
        splat_cfg["position"] = tuple(splat_pos)
    if splat_rot_rpy_deg is not None:
        splat_cfg["rotation"] = quat_xyzw_from_rpy_deg(*splat_rot_rpy_deg)
    if splat_quat is not None:
        splat_cfg["rotation"] = tuple(splat_quat)
    if splat_scale is not None:
        splat_cfg["scale"] = splat_scale
    return env_cfg


class RoomGraspPolicy(base.GraspPolicy):
    pass


@dataclass
class Config:
    """Configuration for the scripted room grasp rollout."""

    video_path: Path = Path("scripted_grasp_room.mp4")  # Output MP4 path.
    steps_per_segment: int = 50  # Simulation steps for each scripted motion segment.
    viewer: bool = False  # Show the Genesis viewer while rendering.
    backend: Literal["cpu", "gpu"] = "gpu"  # Genesis backend to initialize.
    envmap_hdr: Path = DEFAULT_HDR_PATH  # HDR environment map path.
    splat_uri: Path | None = None  # Optional splat path; defaults to repo-root last.ply when present.
    no_envmap: bool = False  # Disable the HDR environment map.
    use_nyx_camera: bool = False  # Render through the Nyx camera.
    splat_pos: tuple[float, float, float] | None = None  # Optional splat XYZ position.
    splat_rot_rpy_deg: tuple[float, float, float] | None = None  # Optional splat RPY rotation in degrees.
    splat_quat: tuple[float, float, float, float] | None = None  # Optional splat quaternion as XYZW.
    splat_scale: float | None = None  # Optional splat scale.


def build_env(
    video_path: str,
    show_viewer: bool,
    envmap_hdr: str | None = None,
    use_nyx_camera: bool = False,
    room_env_cfg: dict | None = None,
):
    return base.build_env(
        video_path=video_path,
        show_viewer=show_viewer,
        envmap_hdr=envmap_hdr,
        use_nyx_camera=use_nyx_camera,
        extra_env_cfg=room_env_cfg or ROOM_ENV_CFG,
        robot_cfg_overrides=XARM7_ROBOT_CFG,
    )


def main(cfg: Config) -> None:
    if cfg.splat_rot_rpy_deg is not None and cfg.splat_quat is not None:
        raise SystemExit("Use either --splat-rot-rpy-deg or --splat-quat, not both.")

    backend = gs.cpu if cfg.backend == "cpu" else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    video_path = str(cfg.video_path)
    envmap_hdr = None if cfg.no_envmap else str(cfg.envmap_hdr)
    room_env_cfg = make_room_env_cfg(
        splat_uri=cfg.splat_uri,
        splat_pos=cfg.splat_pos,
        splat_rot_rpy_deg=cfg.splat_rot_rpy_deg,
        splat_quat=cfg.splat_quat,
        splat_scale=cfg.splat_scale,
    )
    env = build_env(
        video_path=video_path,
        show_viewer=cfg.viewer,
        envmap_hdr=envmap_hdr,
        use_nyx_camera=cfg.use_nyx_camera or envmap_hdr is not None,
        room_env_cfg=room_env_cfg,
    )
    policy = RoomGraspPolicy(env, steps_per_segment=cfg.steps_per_segment)

    total_steps = 1 + (6 - 1) * cfg.steps_per_segment
    with torch.no_grad():
        env.reset()
        for _ in range(total_steps):
            command = policy.step()
            env.robot.go_to_goal(command.pose, open_gripper=command.open_gripper)
            env.scene.step()

    env.scene.stop_recording()
    print(f"Wrote {video_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
