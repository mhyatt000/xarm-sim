"""Write simulated episodes as Foxglove-SDK protobuf MCAP.

Matches the on-disk reference recordings (`/data/fast/episodes/*_base.mcap`, written by
`foxglove-sdk-python`), which carry three `foxglove.RawImage` camera streams
(`/cam/{side,wrist,over}/image_raw`, encoding ``yuv422_yuy2``, 640×480, step 1280), and
extends them with the requested ``{camera intrinsics/extrinsics, joints}``:

- ``/cam/<name>/image_raw``    → ``foxglove.RawImage``          (YUYV422)
- ``/cam/<name>/calibration``  → ``foxglove.CameraCalibration`` (intrinsics K)
- ``/tf``                      → ``foxglove.FrameTransforms``   (extrinsics base→cam)
- ``/xarm/joint_states``       → ``foxglove.JointStates``       (7 arm joints)

This module is **sim-agnostic**: it turns per-step numpy buffers + camera specs into that
MCAP. Rig meaning: ``wrist`` is EE-mounted, ``over`` overhead, ``side`` a side view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import foxglove
from foxglove.channels import (
    CameraCalibrationChannel,
    FrameTransformsChannel,
    JointStatesChannel,
    PoseInFrameChannel,
    RawImageChannel,
)
from foxglove.messages import (
    CameraCalibration,
    FrameTransform,
    FrameTransforms,
    JointState,
    JointStates,
    Pose,
    PoseInFrame,
    Quaternion,
    RawImage,
    Timestamp,
    Vector3,
)
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2

CAMERAS: tuple[str, ...] = ("side", "wrist", "over")
YUYV_ENCODING = "yuv422_yuy2"
JOINT_NAMES: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
BASE_FRAME = "base_link"


def camera_frame(name: str) -> str:
    # Match the reference `_base.mcap` RawImage frame_ids (e.g. "side_optical_frame").
    return f"{name}_optical_frame"


def rgb_to_yuyv(rgb: np.ndarray) -> np.ndarray:
    """RGB ``uint8[H,W,3]`` → packed YUYV422 ``uint8[H,W,2]`` (yuy2).

    Layout that ``cv2.cvtColor(..., COLOR_YUV2RGB_YUY2)`` decodes: channel 0 = per-pixel
    luma Y; channel 1 = shared chroma, U on even columns and V on odd columns (one chroma
    sample per horizontal pixel pair). Real-frame round-trip MAE ≈ 5 (inherent 4:2:2 loss).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected [H,W,3] rgb, got {rgb.shape}")
    h, w = rgb.shape[:2]
    if w % 2:
        raise ValueError(f"YUYV needs even width, got {w}")
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV)
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = y
    u_pair = ((u[:, 0::2].astype(np.uint16) + u[:, 1::2]) // 2).astype(np.uint8)
    v_pair = ((v[:, 0::2].astype(np.uint16) + v[:, 1::2]) // 2).astype(np.uint8)
    out[:, 0::2, 1] = u_pair
    out[:, 1::2, 1] = v_pair
    return out


def _timestamp(stamp_ns: int) -> Timestamp:
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return Timestamp(sec=sec, nsec=nsec)


# -- /xgym/gripper: replicate the established `xclients.Gripper` protobuf schema so the
#    gripper channel matches the existing toolchain (xclients/messages/gripper.py). --
_GRIPPER_TYPE = "xclients.Gripper"


def _gripper_file_descriptor() -> "descriptor_pb2.FileDescriptorProto":
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = "xclients/gripper.proto"
    proto.package = "xclients"
    proto.syntax = "proto2"
    proto.dependency.append("google/protobuf/timestamp.proto")
    msg = proto.message_type.add()
    msg.name = "Gripper"
    f = msg.field.add()
    f.name, f.number = "timestamp", 1
    f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REQUIRED
    f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    f.type_name = ".google.protobuf.Timestamp"
    for number, name in ((2, "rad"), (3, "norm"), (4, "raw")):
        f = msg.field.add()
        f.name, f.number = name, number
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    return proto


def _gripper_descriptor_set() -> bytes:
    ds = descriptor_pb2.FileDescriptorSet()
    ds.file.add().ParseFromString(timestamp_pb2.DESCRIPTOR.serialized_pb)
    ds.file.add().CopyFrom(_gripper_file_descriptor())
    return ds.SerializeToString()


def _gripper_message_cls():
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(_gripper_file_descriptor())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_GRIPPER_TYPE))


GRIPPER_SCHEMA = foxglove.Schema(name=_GRIPPER_TYPE, encoding="protobuf", data=_gripper_descriptor_set())
_GRIPPER_CLS = _gripper_message_cls()


def encode_gripper(stamp_ns: int, norm: float) -> bytes:
    """Serialize a gripper sample. ``norm`` in [0,1] (1=open, 0=closed), matching bela."""
    msg = _GRIPPER_CLS(rad=norm * 0.85, norm=norm, raw=norm * 850.0)
    msg.timestamp.seconds = int(stamp_ns) // 1_000_000_000
    msg.timestamp.nanos = int(stamp_ns) % 1_000_000_000
    return msg.SerializeToString()


@dataclass
class CameraSpec:
    """Static intrinsics for one camera (extrinsics are supplied per step)."""

    name: str
    width: int = 640
    height: int = 480
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion_model: str = "plumb_bob"
    D: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def frame_id(self) -> str:
        return camera_frame(self.name)

    def K(self) -> list[float]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    def P(self) -> list[float]:
        return [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]


def mat4_to_translation_quat(T: np.ndarray) -> tuple[Vector3, Quaternion]:
    """4×4 homogeneous transform → (translation, quaternion xyzw)."""
    T = np.asarray(T, dtype=np.float64)
    t = T[:3, 3]
    q = _rotmat_to_quat_xyzw(T[:3, :3])
    return Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])), Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


class EpisodeMcapWriter:
    """Streams one episode to an MCAP. Use as a context manager and call ``log_step``."""

    def __init__(
        self,
        path: str | Path,
        cameras: dict[str, CameraSpec],
        *,
        joint_names: tuple[str, ...] = JOINT_NAMES,
        base_frame: str = BASE_FRAME,
        allow_overwrite: bool = True,
    ) -> None:
        self.path = str(path)
        self.cameras = cameras
        self.joint_names = tuple(joint_names)
        self.base_frame = base_frame
        self._allow_overwrite = allow_overwrite
        self._mcap = None
        self._img: dict[str, RawImageChannel] = {}
        self._calib: dict[str, CameraCalibrationChannel] = {}
        self._tf: FrameTransformsChannel | None = None
        self._joints: JointStatesChannel | None = None
        self._robot_states: PoseInFrameChannel | None = None
        self._gripper = None
        self._calib_logged = False

    def __enter__(self) -> "EpisodeMcapWriter":
        self._mcap = foxglove.open_mcap(self.path, allow_overwrite=self._allow_overwrite)
        for name in self.cameras:
            self._img[name] = RawImageChannel(f"/cam/{name}/image_raw")
            self._calib[name] = CameraCalibrationChannel(f"/cam/{name}/calibration")
        self._tf = FrameTransformsChannel("/tf")
        self._joints = JointStatesChannel("/xarm/joint_states")
        self._robot_states = PoseInFrameChannel("/xarm/robot_states")
        self._gripper = foxglove.Channel("/xgym/gripper", schema=GRIPPER_SCHEMA, message_encoding="protobuf")
        return self

    def __exit__(self, *exc) -> None:
        if self._mcap is not None:
            self._mcap.__exit__(*exc)
            self._mcap = None

    def log_step(
        self,
        stamp_ns: int,
        images: dict[str, np.ndarray],
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_eff: np.ndarray,
        extrinsics: dict[str, np.ndarray],
        ee_pose: np.ndarray,
        gripper_norm: float,
    ) -> None:
        """Log one recorded timestep.

        ``images[name]`` = RGB ``uint8[H,W,3]``; ``extrinsics[name]`` = 4×4 ``T_base_cam``
        (camera pose in the base frame). Joints are length-7 arrays (rad / rad·s / N·m).
        ``ee_pose`` = ``[x,y,z, qw,qx,qy,qz]`` (TCP pose in the base frame, metres);
        ``gripper_norm`` in [0,1] (1=open, 0=closed). These complete the proprio needed to
        derive actions downstream (crossformer sets ``action = proprio.copy()``).
        """
        ts = _timestamp(stamp_ns)

        for name, spec in self.cameras.items():
            rgb = images[name]
            yuyv = rgb_to_yuyv(rgb)
            self._img[name].log(
                RawImage(
                    timestamp=ts,
                    frame_id=spec.frame_id,
                    width=spec.width,
                    height=spec.height,
                    encoding=YUYV_ENCODING,
                    step=spec.width * 2,
                    data=yuyv.tobytes(),
                ),
                log_time=int(stamp_ns),
            )
            # Intrinsics are static — log once (cheap and Foxglove-friendly).
            if not self._calib_logged:
                self._calib[name].log(
                    CameraCalibration(
                        timestamp=ts,
                        frame_id=spec.frame_id,
                        width=spec.width,
                        height=spec.height,
                        distortion_model=spec.distortion_model,
                        D=list(spec.D),
                        K=spec.K(),
                        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        P=spec.P(),
                    ),
                    log_time=int(stamp_ns),
                )

        transforms = []
        for name in self.cameras:
            translation, rotation = mat4_to_translation_quat(extrinsics[name])
            transforms.append(
                FrameTransform(
                    timestamp=ts,
                    parent_frame_id=self.base_frame,
                    child_frame_id=camera_frame(name),
                    translation=translation,
                    rotation=rotation,
                )
            )
        self._tf.log(FrameTransforms(transforms=transforms), log_time=int(stamp_ns))

        joints = [
            JointState(name=n, position=float(p), velocity=float(v), effort=float(e))
            for n, p, v, e in zip(self.joint_names, joint_pos, joint_vel, joint_eff)
        ]
        self._joints.log(JointStates(timestamp=ts, joints=joints), log_time=int(stamp_ns))

        # EE (TCP) pose in the base frame → /xarm/robot_states
        px, py, pz, qw, qx, qy, qz = (float(v) for v in ee_pose)
        self._robot_states.log(
            PoseInFrame(
                timestamp=ts,
                frame_id=self.base_frame,
                pose=Pose(position=Vector3(x=px, y=py, z=pz), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
            ),
            log_time=int(stamp_ns),
        )

        # normalized gripper → /xgym/gripper
        self._gripper.log(encode_gripper(int(stamp_ns), float(gripper_norm)), log_time=int(stamp_ns))

        self._calib_logged = True
