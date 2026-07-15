"""Camera placement specs and splat asset descriptions.

Placement is a model concern (robosuite declares static cams in the arena XML
and eye-in-hand cams in the robot XML); rendering config — resolution, backend,
spp, lights — is an env/renderer concern and deliberately absent here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# OpenGL camera (x right, y up, -z forward) -> OpenCV optical (x right, y down, +z forward).
T_GL_TO_CV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


@dataclass(frozen=True)
class CameraSpec:
    """Placement for one camera. Static cams use pos/lookat; attached cams ride
    a robot link via a 4x4 link->camera offset transform (OpenGL convention).
    Frozen: specs are descriptions — replace them (Arena.set_camera), don't mutate."""

    name: str
    pos: tuple[float, float, float] | None = None
    lookat: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float | None = None  # vertical FOV; None -> env default
    attach_link: str | None = None
    attach_offset: tuple[tuple[float, ...], ...] | None = None  # nested 4x4


@dataclass(frozen=True)
class SplatAsset:
    """A Gaussian-splat scan plus its solved world alignment
    (p_world = scale * R(quat) * p_splat + pos).

    A renderer asset — nyx loads it as a LightFieldAsset riding the camera
    sensors — never scene geometry; whether it renders is NyxConfig's call.
    """

    uri: Path
    pos: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    scale: float = 1.0


def view_from_c2w_cv(name: str, c2w, fov_deg: float | None = None) -> CameraSpec:
    """CameraSpec from a calibrated OpenCV camera-to-world (robot-base) pose."""
    T = np.asarray(c2w, dtype=np.float64)
    pos = T[:3, 3]
    return CameraSpec(
        name,
        pos=tuple(pos),
        lookat=tuple(pos + T[:3, 2]),  # CV optical +z = view direction
        up=tuple(-T[:3, 1]),  # CV optical +y points down
        fov_deg=fov_deg,
    )


def look_offset_T(
    back: float = 0.12,
    side: float = 0.0,
    lift: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    """Link->camera offset for a tool-mounted camera looking along the TCP +z
    (approach) axis, set ``back`` metres up the tool so the fingers stay in view;
    pitch/yaw/roll tilt the view in the camera frame."""
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # 180 deg about x
    th = math.radians(pitch_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)], [0.0, math.sin(th), math.cos(th)]])
    ps = math.radians(yaw_deg)
    Ry = np.array([[math.cos(ps), 0.0, math.sin(ps)], [0.0, 1.0, 0.0], [-math.sin(ps), 0.0, math.cos(ps)]])
    ph = math.radians(roll_deg)
    Rz = np.array([[math.cos(ph), -math.sin(ph), 0.0], [math.sin(ph), math.cos(ph), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R0 @ Rx @ Ry @ Rz
    T[:3, 3] = (side, lift, -back)
    return tuple(tuple(float(v) for v in row) for row in T)


def _as_single_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def quat_wxyz_to_rot(quat) -> np.ndarray:
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


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(quat_wxyz)
    T[:3, 3] = _as_single_np(pos)
    return T
