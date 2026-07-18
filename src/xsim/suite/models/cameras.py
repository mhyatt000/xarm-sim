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

    @classmethod
    def from_c2w_cv(cls, name: str, c2w, fov_deg: float | None = None) -> CameraSpec:
        """CameraSpec from a calibrated OpenCV camera-to-world (robot-base) pose."""
        T = np.asarray(c2w, dtype=np.float64)
        pos = T[:3, 3]
        return cls(
            name,
            pos=tuple(pos),
            lookat=tuple(pos + T[:3, 2]),  # CV optical +z = view direction
            up=tuple(-T[:3, 1]),  # CV optical +y points down
            fov_deg=fov_deg,
        )

    def c2w_cv(self) -> np.ndarray:
        """4x4 OpenCV camera-to-world rebuilt from pos/lookat/up."""
        if self.pos is None or self.lookat is None:
            raise ValueError(f"camera {self.name!r} has no world pose (attached)")
        return invert_rigid(viewmats_cv(self.pos, self.lookat, self.up))[0]


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
    return CameraSpec.from_c2w_cv(name, c2w, fov_deg)


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


def rot_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rots_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """(B, 4) wxyz -> (B, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product; q1 is (4,), q2 is (N, 4), both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def viewmats_cv(pos, lookat, up) -> np.ndarray:
    """Batched world->camera in the OpenCV optical frame (x right, y down,
    +z forward): (B, 3) or (3,) pos/lookat/up -> (B, 4, 4)."""
    pos, lookat, up = np.atleast_2d(pos, lookat, up)
    z = lookat - pos
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(z, np.broadcast_to(up, z.shape))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(z, x)
    T = np.tile(np.eye(4), (len(z), 1, 1))
    T[:, :3, :3] = np.stack([x, y, z], axis=-2)
    T[:, :3, 3] = -(T[:, :3, :3] @ pos[..., None])[..., 0]
    return T


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """(B, 4, 4) rigid transforms -> batched inverse."""
    R = T[:, :3, :3]
    out = np.tile(np.eye(4), (len(T), 1, 1))
    out[:, :3, :3] = R.transpose(0, 2, 1)
    out[:, :3, 3] = -(R.transpose(0, 2, 1) @ T[:, :3, 3, None])[..., 0]
    return out
