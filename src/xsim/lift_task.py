"""LiftBlockEnv: xArm7 red-block pickup env for synthetic MCAP data generation.

A purpose-built Genesis env (reuses ``Manipulator`` from ``grasp_env`` but does not touch
``GraspEnv``) with:

- a flat **collision-plane table** at z=0, aligned with the splat's real table top,
- a **0.03175 m red cube** (1.25 in) spawned in a configurable table rectangle,
- three cameras matching the real MCAP rig: ``low``/``side`` static and ``wrist``
  mounted on the EE; each defaults to 640x480.
- **physics dt vs record decimation** decoupling.

Frames/axes: Genesis cameras use an OpenGL convention internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from xsim.grasp_env import Manipulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_URDF_PATH = PROJECT_ROOT / "xarm7_standalone.urdf"
# cleaned copy (scripts/clean_splat.py strips the see-through table mush + baked robot);
# fall back to the raw scan if it hasn't been generated
_CLEAN_SPLAT = PROJECT_ROOT / "assets" / "lab_clean.ply"
DEFAULT_SPLAT_PATH = _CLEAN_SPLAT if _CLEAN_SPLAT.exists() else Path("/data/store/lab.ply")

BLOCK_SIZE = 0.03175  # 1.25 inch cube edge (m)
BLOCK_COLOR = (0.48, 0.05, 0.04)  # saturated red; brighter albedos wash to salmon under the nyx light

# OpenGL camera (x right, y up, -z forward) → OpenCV optical (x right, y down, +z forward).
_T_GL_TO_CV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot_from_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    x, y, z, w = quat_xyzw_from_rpy_deg(roll, pitch, yaw)
    return _quat_wxyz_to_rot((w, x, y, z))


def _c2w_gl_from_view(pos, lookat, up) -> np.ndarray:
    """OpenGL camera-to-world (x right, y up, −z forward) from a pos/lookat/up view."""
    pos = np.asarray(pos, dtype=np.float64)
    forward = np.asarray(lookat, dtype=np.float64) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, np.cross(right, forward), -forward, pos
    return T


def quat_xyzw_from_rpy_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _make_light_field(
    uri: Path,
    position: tuple[float, float, float] | None,
    rotation_xyzw: tuple[float, float, float, float],
    scale: float | tuple[float, float, float] | None,
):
    light_field = nps.LightFieldAsset()
    light_field.type = nps.ELightFieldType.GaussianField
    light_field.uri = str(uri.expanduser())
    # The nyx scene exporter converts every mesh instance from Genesis z-up to Nyx y-up
    # (float3_z_up_to_y_up_a / quaternion_z_up_to_y_up_a, see nyx_scene_exporter.py) but
    # passes LightFieldAssets through raw — so we must apply the same world conversion
    # here or the splat lands in a different frame than the cameras and meshes.
    if position is not None:
        light_field.position = nps.float3_z_up_to_y_up_a(nps.float3(*position))
    light_field.rotation = nps.quaternion_z_up_to_y_up_a(nps.quaternion(*rotation_xyzw))
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = (float(scale), float(scale), float(scale))
        light_field.scale = nps.float3(*scale)
    return light_field


def _as_single_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _quat_wxyz_to_rot(quat) -> np.ndarray:
    w, x, y, z = _as_single_np(quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat_wxyz)
    T[:3, 3] = _as_single_np(pos)
    return T


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
    # low ready pose matching the real demos' start (TCP ≈ (0.34, 0, 0.10), top-down) —
    # IK-solved; the old high home (TCP z=0.29) gave episodes a different opening style
    # than the real recordings (see compare_batches report)
    "default_arm_dof": [math.radians(v) for v in [0.0, -26.2, 0.0, 13.5, 0.0, 25.0, 90.0]],
    # xArm gripper joint convention (verified by finger separation): 0.0 = open (fingers
    # apart), 0.85 = hard fully closed. For a 31.75 mm cube, command a tighter
    # task grasp target instead of the hard stop; this holds the block without
    # driving as deeply through it as full closure.
    "default_gripper_dof": [0.0] * 6,
    "gripper_open_dof": 0.0,
    "gripper_close_dof": 0.85,
    # 0.58 -> recorded norm floor 0.32 (real demos read ~0.37, but rigid sim fingers need
    # the extra squeeze; 0.53 matches the real reading exactly and drops the cube)
    "gripper_grasp_dof": 0.58,
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
    fov_deg: float | None = None          # vertical FOV; falls back to cfg.fov_deg
    attach_link: str | None = None        # e.g. "link_tcp" for the wrist cam
    attach_offset: tuple = field(default=None)  # 4x4 offset_T from link frame to camera


def view_from_c2w_cv(name: str, c2w: np.ndarray | tuple, fov_deg: float | None = None) -> CameraView:
    """CameraView from a calibrated OpenCV camera-to-world (robot-base) pose."""
    T = np.asarray(c2w, dtype=np.float64)
    pos = T[:3, 3]
    return CameraView(
        name,
        pos=tuple(pos),
        lookat=tuple(pos + T[:3, 2]),  # CV optical +z = view direction
        up=tuple(-T[:3, 1]),           # CV optical +y points down
        fov_deg=fov_deg,
    )


def _look_offset_T(back=0.12, side=0.0, lift=0.0, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """Offset transform mounting the wrist camera on ``link_tcp``.

    The TCP approach axis is +z (points out of the gripper / downward at home). A Genesis
    camera looks along its own −z, so a 180°-about-x rotation aims the camera along +z_tcp
    (down the tool toward the grasp point). The camera is set ``back`` metres up the tool
    axis (−z_tcp) so the fingertips and workspace are in view. ``pitch_deg`` tilts the view
    about the camera x-axis; negative values push the gripper toward the bottom of the
    image like the real EE-mounted RealSense. ``yaw_deg`` tilts about the camera y-axis
    (aims a side-mounted camera back toward the tool axis). ``roll_deg`` spins the image
    about the optical axis.
    """
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # 180° about x
    th = math.radians(pitch_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)], [0.0, math.sin(th), math.cos(th)]])
    ps = math.radians(yaw_deg)
    Ry = np.array([[math.cos(ps), 0.0, math.sin(ps)], [0.0, 1.0, 0.0], [-math.sin(ps), 0.0, math.cos(ps)]])
    ph = math.radians(roll_deg)
    Rz = np.array([[math.cos(ph), -math.sin(ph), 0.0], [math.sin(ph), math.cos(ph), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R0 @ Rx @ Ry @ Rz
    T[:3, 3] = (side, lift, -back)  # lift: off the gripper body, along y_tcp
    return T


# Logitech extrinsics from /data/store/opencv_calibrated (dream_sam_roboreg, pose_c2w_cv,
# robot-base frame): low = 1e9c6aae (/dev/video8), side = ad3f052e (/dev/video10). Both
# were solved with approximate intrinsics fx=fy=515, cx=320, cy=240 → vFOV ≈ 49.98°, so
# the sim cameras must render with that same FOV for the extrinsics to be consistent.
LOGITECH_FOV_DEG = math.degrees(2.0 * math.atan(240.0 / 515.0))
# RealSense D435 colour at 640×480: fx ≈ 617 → vFOV ≈ 42.6° (no calibration data; guess).
REALSENSE_FOV_DEG = 42.5

LOW_C2W_CV = (
    (-0.6532074213027954, 0.09281682223081589, -0.7514687180519104, 1.0390468835830688),
    (0.7571737170219421, 0.07631510496139526, -0.6487403512001038, 0.48672395944595337),
    (-0.0028656369540840387, -0.9927542209625244, -0.12012804299592972, 0.23500925302505493),
    (0.0, 0.0, 0.0, 1.0),
)
SIDE_C2W_CV = (
    (-0.9901586174964905, 0.07804439961910248, -0.11616794764995575, 0.4850386679172516),
    (0.13690687716007233, 0.7123172879219055, -0.6883754134178162, 0.6458088159561157),
    (0.02902454137802124, -0.697504997253418, -0.7159919738769531, 0.9215802550315857),
    (0.0, 0.0, 0.0, 1.0),
)

DEFAULT_CAMERAS: tuple[CameraView, ...] = (
    view_from_c2w_cv("low", LOW_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
    view_from_c2w_cv("side", SIDE_C2W_CV, fov_deg=LOGITECH_FOV_DEG),
    # Matched against the real wrist stream (no calibration data): the real RealSense is
    # side-mounted, so the finger axis runs near-horizontal, the fingers enter from the
    # frame bottom with the assembly parked on the right half, and the white housing peeks
    # in at the bottom. Candidate "P1" from the 2026-07-02 iterative mount sweep
    # (outputs/wrist_mount/wrist_mount_final_P.png), verified frame-by-frame by grifflee.
    CameraView(
        "wrist",
        fov_deg=REALSENSE_FOV_DEG,
        attach_link="link_tcp",
        attach_offset=_look_offset_T(back=0.14, side=0.085, lift=-0.03, pitch_deg=-5.0, yaw_deg=25.0, roll_deg=-90.0),
    ),
)


# Splat (lab.ply) → world alignment, solved 2026-07-01 by scripts/align_ransac.py:
# RANSAC geometry on the ZED fused point cloud (human-verified table/robot landmarks,
# checkpoint CP1), closed-form fused→robot solve (table rect center agrees with the
# calibrated-camera IPM measurement to 5 cm, CP2 human-verified), photometric refine,
# then scaled ICP splat→fused (1.1 cm RMS; scale 0.9966 — the splat is metric).
# Semantics: p_world = scale · R(quat) · p_splat + pos.
DEFAULT_SPLAT_POS = (-0.2237, 0.7717, 0.1711)
DEFAULT_SPLAT_QUAT = (-0.501119, 0.487918, -0.50087, 0.509849)  # xyzw
DEFAULT_SPLAT_SCALE = 0.9966


def splat_world_transform(pos=DEFAULT_SPLAT_POS, quat=DEFAULT_SPLAT_QUAT, scale=DEFAULT_SPLAT_SCALE):
    """(4x4 world-from-splat transform incl. scale) for cropping/analysis tooling."""
    x, y, z, w = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = pos
    return T


@dataclass
class TableCfg:
    top_z: float = 0.0                                  # table-top height (robot base sits here)
    # real cart footprint, measured by inverse-perspective-mapping the calibrated
    # cap.npz photos onto the z=0 plane (robot base at one end of the table)
    center_xy: tuple[float, float] = (0.375, 0.01)
    size_xy: tuple[float, float] = (0.93, 0.62)
    color: tuple[float, float, float] = (0.13, 0.14, 0.17)  # dark slate like the real cart


@dataclass
class LiftEnvCfg:
    res: tuple[int, int] = (640, 480)
    fov_deg: float = 42.0                 # fallback vertical FOV → intrinsics
    physics_dt: float = 1.0 / 120.0       # stable sim step; ×record_every → 30 Hz like real
    record_every: int = 4                 # emit every k-th step → record_dt = physics_dt*k
    rectangle_x: tuple[float, float] = (0.35, 0.58)   # cube spawn range (m)
    rectangle_y: tuple[float, float] = (-0.15, 0.15)
    drop_zone: tuple[float, float, float] = (0.46, 0.0, 0.09)  # central zone, low like the real demos
    table: TableCfg = field(default_factory=TableCfg)
    table_mode: Literal["slab", "plane"] = "slab"  # plane = visible infinite tabletop, no finite cart slab
    table_transparent: bool = False        # hide the visual table slab while keeping table collision
    show_viewer: bool = False
    render_backend: Literal["raster", "nyx"] = "raster"
    splat_uri: Path | None = DEFAULT_SPLAT_PATH
    splat_pos: tuple[float, float, float] | None = DEFAULT_SPLAT_POS
    splat_rot_rpy_deg: tuple[float, float, float] | None = None
    splat_quat: tuple[float, float, float, float] = DEFAULT_SPLAT_QUAT
    splat_scale: float | None = DEFAULT_SPLAT_SCALE
    nyx_spp: int = 8
    nyx_light_intensity: float = 2.0  # 5.0 washed out the mesh entities vs the dim splat
    # per-episode camera jitter, applied in reset() around the calibrated nominal poses
    # (the nominals themselves never move); the actual sampled poses are exposed via
    # episode_extrinsics so batch manifests can record them. 0 = off.
    cam_jitter_deg: float = 0.0    # low/side: ± per-axis rpy, in the camera frame (deg)
    cam_jitter_cm: float = 0.0     # low/side: ± per-axis world xyz (cm)
    wrist_jitter_deg: float = 0.0  # wrist mount offset: ± per-axis rpy (deg)
    wrist_jitter_cm: float = 0.0   # wrist mount offset: ± per-axis xyz (cm)


class LiftBlockEnv:
    def __init__(self, cfg: LiftEnvCfg | None = None, robot_cfg: dict | None = None, cameras=DEFAULT_CAMERAS):
        self.cfg = cfg or LiftEnvCfg()
        self.robot_cfg = robot_cfg or XARM7_ROBOT_CFG
        self.camera_views = list(cameras)
        self.device = gs.device
        self.res = self.cfg.res
        self.record_dt = self.cfg.physics_dt * self.cfg.record_every

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.physics_dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=self.cfg.show_viewer,
        )

        # flat table plane: always provides collision at the aligned table-top height
        # (z=0). In plane mode it is also the visible infinite tabletop.
        t = self.cfg.table
        table_surface = gs.surfaces.Plastic(color=t.color, roughness=0.8)
        # The surface must be set even in slab mode: Nyx exports collision-only
        # primitives regardless of visualization=False, so without it the infinite
        # plane renders in Nyx's default light gray and shows up as a bright sheet
        # where the dark room floor belongs. With the dark slate surface it blends
        # into the room floor exactly as in all verified batches.
        self.table = self.scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, t.top_z),
                visualization=self.cfg.table_mode == "plane",
                collision=True,
            ),
            surface=table_surface,
        )
        if self.cfg.table_mode == "slab":
            if not self.cfg.table_transparent:
                # visual-only cart body with the real cart's measured footprint: the real dark
                # metal cart scans as sparse see-through mush in the splat (scripts/clean_splat.py
                # crops that region out), so this box stands in for it — full depth down to the
                # floor so it occludes the under-table region from every camera angle, exactly
                # like the real cart does
                slab_h = 0.72
                self.scene.add_entity(
                    gs.morphs.Box(
                        size=(t.size_xy[0], t.size_xy[1], slab_h),
                        pos=(t.center_xy[0], t.center_xy[1], t.top_z - slab_h / 2.0),
                        fixed=True,
                        visualization=True,
                        collision=False,
                    ),
                    surface=table_surface,
                )
        elif self.cfg.table_mode != "plane":
            raise ValueError(f"unknown table_mode: {self.cfg.table_mode!r}")

        # robot (base at world origin, on the table top)
        self.robot = Manipulator(num_envs=1, scene=self.scene, args=self.robot_cfg, device=gs.device)

        # red cube (high friction so the gripper can hold it)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), fixed=False),
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Plastic(color=BLOCK_COLOR, roughness=0.6),
        )

        self.cams = {}
        self._manual_attached_cams = []
        self._rig_attached_camera_names = set()
        self._add_cameras()

        self.scene.build(n_envs=1)
        self.robot.set_pd_gains()

        # place static cams + attach wrist cam; keep the nominal poses that per-episode
        # jitter centers on, and the attach machinery so reset() can re-pose everything
        self._nominal_c2w_gl = {
            v.name: _c2w_gl_from_view(v.pos, v.lookat, v.up) for v in self.camera_views if v.attach_link is None
        }
        self._attach_links = {}
        self._attach_offsets = {}
        self.episode_extrinsics: dict[str, np.ndarray] = {}
        for view in self.camera_views:
            cam = self.cams[view.name]
            if view.attach_link is not None:
                link = self.robot._robot_entity.get_link(view.attach_link)
                self._attach_links[view.name] = link
                self._attach_offsets[view.name] = np.asarray(view.attach_offset, dtype=np.float64)
                if hasattr(cam, "attach"):
                    cam.attach(link, view.attach_offset)
                    self._rig_attached_camera_names.add(view.name)
                else:
                    self._manual_attached_cams.append((view.name, cam, link))
            elif hasattr(cam, "set_pose"):
                cam.set_pose(pos=view.pos, lookat=view.lookat, up=view.up)

        self.reset()

    def _splat_light_fields(self):
        if self.cfg.splat_uri is None:
            return ()
        splat_uri = Path(self.cfg.splat_uri).expanduser()
        if not splat_uri.exists():
            raise FileNotFoundError(f"splat file does not exist: {splat_uri}")
        rotation = self.cfg.splat_quat
        if self.cfg.splat_rot_rpy_deg is not None:
            rotation = quat_xyzw_from_rpy_deg(*self.cfg.splat_rot_rpy_deg)
        return (_make_light_field(splat_uri, self.cfg.splat_pos, rotation, self.cfg.splat_scale),)

    def _add_cameras(self) -> None:
        if self.cfg.render_backend == "nyx":
            lights = [
                {
                    "type": "directional",
                    "dir": (-0.4, -0.4, -0.8),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": self.cfg.nyx_light_intensity,
                    "shadow": True,
                }
            ]
            light_fields = self._splat_light_fields()
            for view in self.camera_views:
                self.cams[view.name] = self.scene.add_sensor(
                    NyxCameraOptions(
                        res=self.res,
                        fov=view.fov_deg or self.cfg.fov_deg,
                        pos=view.pos or (1.0, 0.0, 0.5),
                        lookat=view.lookat or (0.0, 0.0, 0.0),
                        up=view.up,
                        near=0.02,
                        far=50.0,
                        spp=self.cfg.nyx_spp,
                        render_mode=npr.ERenderMode.FastPathTracer,
                        lights=lights,
                        light_fields=light_fields,
                    )
                )
            return

        for view in self.camera_views:
            self.cams[view.name] = self.scene.add_camera(
                res=self.res, fov=view.fov_deg or self.cfg.fov_deg, GUI=False,
                pos=view.pos or (1.0, 0.0, 0.5), lookat=view.lookat or (0.0, 0.0, 0.0),
                near=0.02, far=50.0,  # default near=0.1 clips the wrist cam's own gripper
            )

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
        # after the cube draws, so a given seed reproduces the same cube placement as
        # jitter-free runs
        self._randomize_cameras(rng)
        self._sync_attached_cams()

    def _randomize_cameras(self, rng: np.random.Generator) -> None:
        """Per-episode camera jitter around the nominal poses; records actual extrinsics.

        Static cams get an orientation delta (± cam_jitter_deg per rpy axis, applied in
        the camera frame) and a world-frame position delta (± cam_jitter_cm per axis).
        The wrist mount offset gets the same treatment in its own frame when wrist jitter
        is enabled. ``episode_extrinsics`` then holds camera(optical CV)→world for static
        cams — the same convention as LOW_C2W_CV/SIDE_C2W_CV — and link_tcp→camera
        (optical CV) under ``wrist_mount``.
        """
        self.episode_extrinsics = {}
        for view in self.camera_views:
            cam = self.cams[view.name]
            if view.attach_link is None:
                c2w = self._nominal_c2w_gl[view.name].copy()
                d_rpy = rng.uniform(-1.0, 1.0, 3) * self.cfg.cam_jitter_deg
                d_xyz = rng.uniform(-1.0, 1.0, 3) * (self.cfg.cam_jitter_cm / 100.0)
                c2w[:3, :3] = c2w[:3, :3] @ _rot_from_rpy_deg(*d_rpy)
                c2w[:3, 3] += d_xyz
                pos = tuple(c2w[:3, 3])
                lookat = tuple(c2w[:3, 3] - c2w[:3, 2])
                up = tuple(c2w[:3, 1])
                if hasattr(cam, "set_pose"):
                    cam.set_pose(pos=pos, lookat=lookat, up=up)
                else:
                    cam.update_camera_pose(pos=pos, lookat=lookat, up=up)
                self.episode_extrinsics[view.name] = c2w @ _T_GL_TO_CV
            else:
                offset = np.asarray(view.attach_offset, dtype=np.float64).copy()
                if self.cfg.wrist_jitter_deg or self.cfg.wrist_jitter_cm:
                    delta = np.eye(4)
                    delta[:3, :3] = _rot_from_rpy_deg(*(rng.uniform(-1.0, 1.0, 3) * self.cfg.wrist_jitter_deg))
                    delta[:3, 3] = rng.uniform(-1.0, 1.0, 3) * (self.cfg.wrist_jitter_cm / 100.0)
                    offset = offset @ delta
                    if view.name in self._rig_attached_camera_names:
                        cam.attach(self._attach_links[view.name], offset)
                self._attach_offsets[view.name] = offset
                self.episode_extrinsics[f"{view.name}_mount"] = offset @ _T_GL_TO_CV

    def step(self) -> None:
        self.scene.step()
        self._sync_attached_cams()

    def _sync_attached_cams(self) -> None:
        for view in self.camera_views:
            if view.name in self._rig_attached_camera_names:
                self.cams[view.name].move_to_attach()
        for name, cam, link in self._manual_attached_cams:
            link_T = _pose_to_T(link.get_pos(), link.get_quat())
            cam_T = link_T @ self._attach_offsets[name]
            pos = cam_T[:3, 3]
            lookat = pos - cam_T[:3, 2]
            up = cam_T[:3, 1]
            cam.update_camera_pose(pos=tuple(pos), lookat=tuple(lookat), up=tuple(up))

    # -- observations --
    def render(self) -> dict[str, np.ndarray]:
        out = {}
        for name, cam in self.cams.items():
            if hasattr(cam, "render"):
                rgb = cam.render(rgb=True)[0]
            else:
                rgb = cam.read(envs_idx=0).rgb
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            else:
                rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            out[name] = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        return out

    def intrinsics(self, name: str) -> np.ndarray:
        cam = self.cams[name]
        if hasattr(cam, "intrinsics"):
            return np.asarray(cam.intrinsics, dtype=np.float64)
        # nyx sensors don't expose K; derive it from the view's vertical FOV
        view = next(v for v in self.camera_views if v.name == name)
        w, h = self.res
        fy = (h / 2.0) / math.tan(math.radians(view.fov_deg or self.cfg.fov_deg) / 2.0)
        return np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])

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
