"""LiftBlockEnv: xArm7 red-block pickup env for synthetic MCAP data generation.

A purpose-built Genesis env (reuses ``Manipulator`` from ``grasp_env`` but does not touch
``GraspEnv``) with:

- an explicit **collision-box table** (not just a visual splat), top face at z=0,
- a **0.03175 m red cube** (1.25 in) spawned in a configurable table rectangle,
- three cameras matching the reference MCAP rig — ``side``/``over`` static, ``wrist``
  mounted on the EE — each 640×480, exposing intrinsics (fov→K) and extrinsics
  (base→optical-frame 4×4),
- **physics dt vs record decimation** decoupling.

Frames/axes: Genesis cameras use an OpenGL convention; extrinsics are converted to the
OpenCV optical frame (z forward, x right, y down) to match the ``*_optical_frame`` naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch

import genesis as gs

from xsim.grasp_env import Manipulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_URDF_PATH = PROJECT_ROOT / "xarm7_standalone.urdf"

BLOCK_SIZE = 0.03175  # 1.25 inch cube edge (m)
BLOCK_COLOR = (0.6, 0.15, 0.13)

# OpenGL camera (x right, y up, -z forward) → OpenCV optical (x right, y down, +z forward).
_T_GL_TO_CV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

XARM7_ROBOT_CFG: dict = {
    "robot_morph": "urdf",
    "robot_file": str(ROBOT_URDF_PATH),
    "robot_fixed": True,
    "merge_fixed_links": False,
    # Skip the Nyx material-URDF rewrite (writes to a shared /tmp path); colors are set by
    # apply_robot_visual_surfaces and Genesis resolves relative mesh paths itself.
    "rewrite_robot_visual_urdf": False,
    "ee_link_name": "link_tcp",
    "gripper_link_names": ["left_finger", "right_finger"],
    "arm_dof_dim": 7,
    "gripper_dof_dim": 6,
    "default_arm_dof": [math.radians(v) for v in [0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0]],
    # xArm gripper joint convention (verified by finger separation): 0.0 = open (fingers
    # apart), 0.85 = closed (fingers together).
    "default_gripper_dof": [0.0] * 6,
    "gripper_open_dof": 0.0,
    "gripper_close_dof": 0.85,
    "dofs_kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 350, 350, 350, 350, 350, 350],
    "dofs_kv": [450, 450, 350, 350, 200, 200, 200, 35, 35, 35, 35, 35, 35],
    "dofs_force_lower": [-50] * 13,
    "dofs_force_upper": [50] * 13,
    "ik_method": "dls_ik",
    "ik_init_at_home": True,
    "ik_max_samples": 50,
    "ik_max_solver_iters": 40,
}


@dataclass
class CameraView:
    """Placement for one camera. Static cams use pos/lookat; the wrist cam attaches to a link."""

    name: str
    pos: tuple[float, float, float] | None = None
    lookat: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    attach_link: str | None = None        # e.g. "link_tcp" for the wrist cam
    attach_offset: tuple = field(default=None)  # 4x4 offset_T from link frame to camera


def _look_offset_T(back=0.12, side=0.0) -> np.ndarray:
    """Offset transform mounting the wrist camera on ``link_tcp``.

    The TCP approach axis is +z (points out of the gripper / downward at home). A Genesis
    camera looks along its own −z, so a 180°-about-x rotation aims the camera along +z_tcp
    (down the tool toward the grasp point). The camera is set ``back`` metres up the tool
    axis (−z_tcp) so the fingertips and workspace are in view.
    """
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # 180° about x
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (side, 0.0, -back)
    return T


DEFAULT_CAMERAS: tuple[CameraView, ...] = (
    CameraView("side", pos=(0.45, 0.75, 0.35), lookat=(0.45, 0.0, 0.05)),
    CameraView("over", pos=(0.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.0)),
    CameraView("wrist", attach_link="link_tcp", attach_offset=_look_offset_T()),
)


@dataclass
class TableCfg:
    size: tuple[float, float, float] = (1.2, 1.6, 0.4)  # x, y, z
    top_z: float = 0.0                                  # top face height (robot base sits here)
    color: tuple[float, float, float] = (0.55, 0.55, 0.6)


@dataclass
class LiftEnvCfg:
    res: tuple[int, int] = (640, 480)
    fov_deg: float = 42.0                 # vertical FOV → intrinsics
    physics_dt: float = 0.01              # stable sim step
    record_every: int = 2                 # emit every k-th step → record_dt = physics_dt*k
    rectangle_x: tuple[float, float] = (0.35, 0.58)   # cube spawn range (m)
    rectangle_y: tuple[float, float] = (-0.15, 0.15)
    drop_zone: tuple[float, float, float] = (0.46, 0.0, 0.12)  # central zone (a few in above center)
    table: TableCfg = field(default_factory=TableCfg)
    show_viewer: bool = False


class LiftBlockEnv:
    def __init__(self, cfg: LiftEnvCfg | None = None, robot_cfg: dict | None = None, cameras=DEFAULT_CAMERAS):
        self.cfg = cfg or LiftEnvCfg()
        self.robot_cfg = robot_cfg or XARM7_ROBOT_CFG
        self.camera_views = list(cameras)
        self.device = gs.device
        self.res = self.cfg.res
        self.record_dt = self.cfg.physics_dt * self.cfg.record_every

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.physics_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=self.cfg.show_viewer,
        )

        # ground plane (below the table) for physics stability
        self.scene.add_entity(gs.morphs.Plane(visualization=False, collision=True))

        # explicit collision-box table: top face at cfg.table.top_z
        t = self.cfg.table
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=t.size, pos=(0.45, 0.0, t.top_z - t.size[2] / 2.0), fixed=True),
            surface=gs.surfaces.Plastic(color=t.color, roughness=0.7),
        )

        # robot (base at world origin, on the table top)
        self.robot = Manipulator(num_envs=1, scene=self.scene, args=self.robot_cfg, device=gs.device)

        # red cube (high friction so the gripper can hold it)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), fixed=False),
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Plastic(color=BLOCK_COLOR, roughness=0.6),
        )

        # cameras (rasterizer, headless): gives intrinsics/extrinsics + attach
        self.cams: dict[str, "gs.vis.camera.Camera"] = {}
        for view in self.camera_views:
            cam = self.scene.add_camera(
                res=self.res, fov=self.cfg.fov_deg, GUI=False,
                pos=view.pos or (1.0, 0.0, 0.5), lookat=view.lookat or (0.0, 0.0, 0.0),
            )
            self.cams[view.name] = cam

        self.scene.build(n_envs=1)
        self.robot.set_pd_gains()

        # place static cams + attach wrist cam
        for view in self.camera_views:
            cam = self.cams[view.name]
            if view.attach_link is not None:
                link = self.robot._robot_entity.get_link(view.attach_link)
                cam.attach(link, view.attach_offset)
            else:
                cam.set_pose(pos=view.pos, lookat=view.lookat, up=view.up)

        self.reset()

    # -- lifecycle --
    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.robot.reset(envs_idx=None, skip_forward=True)
        x = float(rng.uniform(*self.cfg.rectangle_x))
        y = float(rng.uniform(*self.cfg.rectangle_y))
        z = self.cfg.table.top_z + BLOCK_SIZE / 2.0
        yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
        pos = torch.tensor([[x, y, z]], device=self.device, dtype=gs.tc_float)
        quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=self.device, dtype=gs.tc_float)
        self.cube.set_pos(pos, skip_forward=True)
        self.cube.set_quat(quat, skip_forward=False)
        self._sync_attached_cams()

    def step(self) -> None:
        self.scene.step()
        self._sync_attached_cams()

    def _sync_attached_cams(self) -> None:
        for view in self.camera_views:
            if view.attach_link is not None:
                self.cams[view.name].move_to_attach()

    # -- observations --
    def render(self) -> dict[str, np.ndarray]:
        out = {}
        for name, cam in self.cams.items():
            rgb = cam.render(rgb=True)[0]
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            out[name] = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        return out

    def intrinsics(self, name: str) -> np.ndarray:
        return np.asarray(self.cams[name].intrinsics, dtype=np.float64)

    def extrinsic_base_cam(self, name: str) -> np.ndarray:
        """4x4 camera(optical)-to-base transform for FrameTransform (base → cam)."""
        cam_to_world_gl = np.asarray(self.cams[name].transform, dtype=np.float64)
        if cam_to_world_gl.ndim == 3:
            cam_to_world_gl = cam_to_world_gl[0]
        return cam_to_world_gl @ _T_GL_TO_CV

    def proprio(self):
        """Return (joint_pos, joint_vel, joint_eff) for the 7 arm joints and the EE pose."""
        ent = self.robot._robot_entity
        pos = np.asarray(ent.get_dofs_position().cpu()).reshape(-1)[:7]
        vel = np.asarray(ent.get_dofs_velocity().cpu()).reshape(-1)[:7]
        force = np.asarray(ent.get_dofs_force().cpu()).reshape(-1)[:7]
        ee = np.asarray(self.robot.ee_pose.cpu()).reshape(-1)  # [x,y,z, qw,qx,qy,qz]
        return pos, vel, force, ee

    def gripper_norm(self) -> float:
        """Normalized gripper opening in [0,1] (1=open, 0=closed), matching bela convention."""
        g = float(np.asarray(self.robot._robot_entity.get_dofs_position().cpu()).reshape(-1)[self.robot._arm_dof_dim])
        close = float(self.robot_cfg["gripper_close_dof"]) or 0.85
        return float(np.clip(1.0 - g / close, 0.0, 1.0))

    def cube_pos(self) -> np.ndarray:
        return np.asarray(self.cube.get_pos().cpu()).reshape(-1)

    def camera_specs(self):
        """Return {name: (width, height, fx, fy, cx, cy)} for the MCAP CameraSpecs."""
        specs = {}
        for name in self.cams:
            K = self.intrinsics(name)
            specs[name] = (self.res[0], self.res[1], float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        return specs
