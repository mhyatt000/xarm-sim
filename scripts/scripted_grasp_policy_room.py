import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch

import genesis as gs
import scripted_grasp_policy as base


def hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


ZED_FHD_RESOLUTION = (1920, 1080)
ZED_LEFT_FHD_FY = 1066.84
ZED_LEFT_FHD_VERTICAL_FOV_DEG = 53.69412375493809
LAST_PLY = Path(__file__).with_name("last.ply")


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
    "nyx_light_fields": [
        {
            "uri": str(LAST_PLY),
            "rotation": (0.0, 0.0, -0.70710678, 0.70710678),
        },
    ],
    "walls": [],
}


XARM7_GRIPPER_OPEN_QPOS = 0.85
XARM7_GRIPPER_CLOSED_QPOS = 0.0


XARM7_ROBOT_CFG = {
    "robot_morph": "urdf",
    "robot_file": str(Path(__file__).with_name("xarm7_standalone.urdf")),
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
    splat_pos: list[float] | None = None,
    splat_rot_rpy_deg: list[float] | None = None,
    splat_quat: list[float] | None = None,
    splat_scale: float | None = None,
) -> dict:
    env_cfg = deepcopy(ROOM_ENV_CFG)
    splat_cfg = env_cfg["nyx_light_fields"][0]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default="scripted_grasp_room.mp4")
    parser.add_argument("--steps-per-segment", type=int, default=50)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--envmap-hdr", default=str(Path(__file__).with_name("lab-hdri.exr")))
    parser.add_argument("--no-envmap", action="store_true")
    parser.add_argument("--use-nyx-camera", action="store_true")
    parser.add_argument("--splat-pos", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--splat-rot-rpy-deg", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--splat-quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--splat-scale", type=float)
    args = parser.parse_args()
    if args.splat_rot_rpy_deg is not None and args.splat_quat is not None:
        parser.error("Use either --splat-rot-rpy-deg or --splat-quat, not both.")

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    video_path = str(Path(args.video_path))
    envmap_hdr = None if args.no_envmap else args.envmap_hdr
    room_env_cfg = make_room_env_cfg(
        splat_pos=args.splat_pos,
        splat_rot_rpy_deg=args.splat_rot_rpy_deg,
        splat_quat=args.splat_quat,
        splat_scale=args.splat_scale,
    )
    env = build_env(
        video_path=video_path,
        show_viewer=args.viewer,
        envmap_hdr=envmap_hdr,
        use_nyx_camera=args.use_nyx_camera or envmap_hdr is not None,
        room_env_cfg=room_env_cfg,
    )
    policy = RoomGraspPolicy(env, steps_per_segment=args.steps_per_segment)

    total_steps = 1 + (6 - 1) * args.steps_per_segment
    with torch.no_grad():
        env.reset()
        for _ in range(total_steps):
            command = policy.step()
            env.robot.go_to_goal(command.pose, open_gripper=command.open_gripper)
            env.scene.step()

    env.scene.stop_recording()
    print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
