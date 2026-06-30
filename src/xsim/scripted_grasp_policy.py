import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import genesis as gs
from xsim import grasp_env
from xsim.grasp_env import GraspEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDR_PATH = PROJECT_ROOT / "assets" / "lab-hdri.hdr"


@dataclass
class GraspCommand:
    pose: torch.Tensor
    open_gripper: bool


class GraspPolicy:
    """Scripted privileged grasp policy for the local GraspEnv."""

    def __init__(self, env: GraspEnv, steps_per_segment: int = 50) -> None:
        self.env = env
        self.steps_per_segment = steps_per_segment
        self._commands = self._make_command_generator()

    def step(self, *args, **kwargs) -> GraspCommand:
        """Return the next command, ignoring observations and using env state."""
        return next(self._commands)

    def _make_command_generator(self):
        ee_pose = self.env.robot.ee_pose.clone()
        obj_pos = self.env.object.get_pos().clone()
        quat = ee_pose[:, 3:7].clone()

        current = ee_pose.clone()
        above_10cm = torch.cat([obj_pos + torch.tensor([0.0, 0.0, 0.10], device=self.env.device), quat], dim=-1)
        at_object = torch.cat([obj_pos, quat], dim=-1)
        above_20cm = torch.cat([obj_pos + torch.tensor([0.0, 0.0, 0.20], device=self.env.device), quat], dim=-1)

        waypoints = [
            GraspCommand(current, True),
            GraspCommand(above_10cm, True),
            GraspCommand(at_object, True),
            GraspCommand(at_object, False),
            GraspCommand(above_20cm, False),
            GraspCommand(above_20cm, True),
        ]

        last = waypoints[0]
        yield last
        for target in waypoints[1:]:
            for i in range(1, self.steps_per_segment + 1):
                alpha = i / self.steps_per_segment
                pose = last.pose.lerp(target.pose, alpha)
                yield GraspCommand(pose, target.open_gripper)
            last = target

        while True:
            yield waypoints[-1]


def build_env(
    video_path: str,
    show_viewer: bool,
    envmap_hdr: str | None = None,
    use_nyx_camera: bool = False,
    extra_env_cfg: dict | None = None,
    robot_cfg_overrides: dict | None = None,
) -> GraspEnv:
    grasp_env._ENABLE_MADRONA = False
    env_cfg = {
        "num_envs": 1,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 10.0,
        "ctrl_dt": 0.01,
        "box_size": [0.08, 0.03, 0.06],
        "box_fixed": False,
        "image_resolution": (64, 64),
        "visualize_camera": True,
        "record_video": {"vis_cam": video_path},
        "use_nyx_camera": use_nyx_camera,
    }
    if envmap_hdr:
        env_cfg.update(
            {
                "envmap_hdr": envmap_hdr,
                "envmap_rotation_deg": 30.0,
                "envmap_multiplier": 1.5,
                "envmap_resolution": (1280, 720),
                "envmap_camera_pos": (2.0, -1.5, 2.0),
                "envmap_camera_lookat": (0.0, 0.0, 0.9),
            }
        )
    if use_nyx_camera and not envmap_hdr:
        env_cfg["envmap_resolution"] = (1280, 720)
        env_cfg["envmap_camera_pos"] = (2.0, -1.5, 2.0)
        env_cfg["envmap_camera_lookat"] = (0.0, 0.0, 0.9)
        env_cfg["nyx_lights"] = [
            {
                "type": "directional",
                "dir": (-0.4, -0.4, -0.8),
                "color": (1.0, 1.0, 1.0),
                "intensity": 5.0,
                "shadow": True,
            }
        ]

    if extra_env_cfg:
        env_cfg.update(extra_env_cfg)

    reward_cfg = {"keypoints": 1.0}
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    if robot_cfg_overrides:
        robot_cfg.update(robot_cfg_overrides)

    return GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=show_viewer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default="scripted_grasp.mp4")
    parser.add_argument("--steps-per-segment", type=int, default=50)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--envmap-hdr", default=str(DEFAULT_HDR_PATH))
    parser.add_argument("--no-envmap", action="store_true")
    parser.add_argument("--use-nyx-camera", action="store_true")
    args = parser.parse_args()

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    video_path = str(Path(args.video_path))
    envmap_hdr = None if args.no_envmap else args.envmap_hdr
    env = build_env(
        video_path=video_path,
        show_viewer=args.viewer,
        envmap_hdr=envmap_hdr,
        use_nyx_camera=args.use_nyx_camera or envmap_hdr is not None,
    )
    policy = GraspPolicy(env, steps_per_segment=args.steps_per_segment)

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
