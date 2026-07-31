"""Viser-based live viewer for policy inputs and outputs.

``PolicyViewer`` owns a viser server and displays camera images and joint
values as GUI panels, plus a URDF robot in the 3D scene: solid at the
measured joints, translucent blue ghost at the next commanded pose, and a
translucent red ghost at the final pose of the policy's action chunk
(meaningful here because the DAgger policy outputs absolute joint targets).
``ViserWrappedPolicy`` wraps a policy and pushes each step's inputs and
outputs to a viewer before returning — display errors are logged, never
raised, so visualization can't break serving.
"""

from __future__ import annotations

from functools import partial
import logging
from pathlib import Path

import numpy as np
import viser
from webpolicy.base_policy import BasePolicy

log = logging.getLogger(__name__)

Images = np.ndarray | list[np.ndarray] | dict[str, np.ndarray]

# translucent light-blue ghost = next commanded pose; red = final pose of
# the current action chunk
_GHOST_COLOR: tuple[int, int, int] = (120, 190, 255)
_FINAL_COLOR: tuple[int, int, int] = (255, 110, 110)
_GHOST_OPACITY: float = 0.25


class UrdfRobot:
    """Render a URDF in a viser scene at settable joint configs.

    A self-contained stand-in for ``viser.extras.ViserUrdf`` that renders every
    visual mesh via ``add_mesh_simple`` so the caller controls ``color`` and
    ``opacity`` — needed for the translucent ghost, which ViserUrdf can't do
    (it forwards neither, and mesh handles can't set opacity after creation).
    FK wiring mirrors ViserUrdf: a frame per link, meshes as static children of
    their link frame, and ``update_cfg`` moving only the link frames.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: str | Path,
        *,
        root_node_name: str = "/robot",
        color: tuple[int, int, int] | None = None,
        opacity: float | None = None,
    ) -> None:
        import yourdfpy

        path = Path(urdf_path)
        self._urdf = yourdfpy.URDF.load(
            str(path), filename_handler=partial(yourdfpy.filename_handler_magic, dir=str(path.parent))
        )
        self._server = server
        self._root = root_node_name

        # a frame per link (moved by update_cfg); meshes hang statically off them
        self._link_frames: dict[str, viser.SceneNodeHandle] = {}
        for joint in self._urdf.joint_map.values():
            self._link_frames[joint.child] = server.scene.add_frame(self._name(joint.child), show_axes=False)

        self._meshes: list[viser.SceneNodeHandle] = []
        for geom_name, mesh in self._urdf.scene.geometry.items():
            parent = self._urdf.scene.graph.transforms.parents[geom_name]
            m = mesh.copy()
            m.apply_transform(self._urdf.get_transform(geom_name, parent))
            name = self._name(geom_name)
            if color is None and opacity is None:
                self._meshes.append(server.scene.add_mesh_trimesh(name, m))
            else:
                self._meshes.append(
                    server.scene.add_mesh_simple(
                        name,
                        m.vertices,
                        m.faces,
                        color=color if color is not None else (200, 200, 200),
                        opacity=1.0 if opacity is None else opacity,
                    )
                )

        self._visible = True

    @property
    def num_actuated(self) -> int:
        return len(self._urdf.actuated_joint_names)

    def set_visible(self, visible: bool) -> None:
        if visible == self._visible:
            return
        self._visible = visible
        for m in self._meshes:
            m.visible = visible

    def _name(self, frame_name: str) -> str:
        """Scene-node path from base_frame down to ``frame_name`` (see ViserUrdf)."""
        base = self._urdf.scene.graph.base_frame
        frames: list[str] = []
        while frame_name != base:
            frames.append(frame_name)
            frame_name = self._urdf.scene.graph.transforms.parents[frame_name]
        if self._root != "/":
            frames.append(self._root)
        return "/".join(frames[::-1])

    def _pad_cfg(self, cfg: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.num_actuated, dtype=np.float64)
        cfg = np.asarray(cfg, dtype=np.float64).reshape(-1)
        n = min(cfg.shape[0], vec.shape[0])
        vec[:n] = cfg[:n]
        return vec

    def update_cfg(self, cfg: np.ndarray) -> None:
        """Set actuated joints, padding/truncating ``cfg`` to the URDF's DOF count."""
        import viser.transforms as vtf

        self._urdf.update_cfg(self._pad_cfg(cfg))
        for joint in self._urdf.joint_map.values():
            T = self._urdf.get_transform(joint.child, joint.parent)
            frame = self._link_frames[joint.child]
            frame.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
            frame.position = T[:3, 3]


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    scaled = img * 255 if img.max() <= 1.5 else img
    return scaled.clip(0, 255).astype(np.uint8)


def _named_images(images: Images) -> dict[str, np.ndarray]:
    """Normalize ndarray | list | dict to {name: HWC uint8}, squeezing lead dims."""
    if isinstance(images, dict):
        named = images
    elif isinstance(images, (list, tuple)):
        named = {f"cam{i}": img for i, img in enumerate(images)}
    else:
        arr = np.asarray(images)
        while arr.ndim > 4:
            arr = arr[0]
        named = {"cam0": arr} if arr.ndim == 3 else {f"cam{i}": v for i, v in enumerate(arr)}
    out = {}
    for name, img in named.items():
        img = np.asarray(img)
        while img.ndim > 3:
            img = img[0]
        out[name] = _to_uint8(img)
    return out


class PolicyViewer:
    """Manage a viser server displaying images, joints, and the URDF robot.

    All ``show_*`` methods are idempotent per key: GUI handles are created
    once and updated in place.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        urdf_path: str | Path | None = None,
        show_ghost: bool = True,
    ) -> None:
        self.server = viser.ViserServer(host=host, port=port)
        self._images: dict[str, viser.GuiImageHandle] = {}
        self._numbers: dict[str, viser.GuiNumberHandle] = {}
        self._folders: dict[str, viser.GuiFolderHandle] = {}

        # solid robot at the current joints + translucent ghosts: blue at the
        # next commanded pose, red at the final pose of the action chunk
        # (created only when a URDF is given)
        self._robot: UrdfRobot | None = None
        self._ghost: UrdfRobot | None = None
        self._ghost_final: UrdfRobot | None = None
        if urdf_path is not None:
            self._robot = UrdfRobot(self.server, urdf_path, root_node_name="/robot")
            if show_ghost:
                self._ghost = UrdfRobot(
                    self.server,
                    urdf_path,
                    root_node_name="/ghost",
                    color=_GHOST_COLOR,
                    opacity=_GHOST_OPACITY,
                )
                self._ghost_final = UrdfRobot(
                    self.server,
                    urdf_path,
                    root_node_name="/ghost_final",
                    color=_FINAL_COLOR,
                    opacity=_GHOST_OPACITY,
                )
                self._ghost_final.set_visible(False)

    def show_robot(self, joints: np.ndarray, *, ghost: bool = False) -> None:
        """Set the solid (current) or ghost (commanded) robot's joint config."""
        robot = self._ghost if ghost else self._robot
        if robot is None:
            return
        vec = np.asarray(joints, dtype=np.float32)
        while vec.ndim > 1:
            vec = vec[0]
        robot.update_cfg(vec)

    def show_plan(self, plan: np.ndarray) -> None:
        """Ghost an action chunk: blue at its next action (row 0), red at its
        final action (row -1, hidden when the plan is a single action)."""
        plan = np.asarray(plan, dtype=np.float32)
        while plan.ndim > 2:
            plan = plan[0]
        if plan.ndim == 1:
            plan = plan[None]
        if self._ghost is not None:
            self._ghost.update_cfg(plan[0])
        if self._ghost_final is not None:
            self._ghost_final.set_visible(len(plan) > 1)
            if len(plan) > 1:
                self._ghost_final.update_cfg(plan[-1])

    def show_images(self, images: Images, *, prefix: str = "input") -> None:
        for name, img in _named_images(images).items():
            key = f"{prefix}/{name}"
            if key in self._images:
                self._images[key].image = img
            else:
                with self._folder(prefix):
                    self._images[key] = self.server.gui.add_image(img, label=name)

    def show_joints(self, joints: np.ndarray, *, prefix: str = "input", label: str = "j") -> None:
        """Display a 1D vector as read-only GUI numbers (extra lead dims take [0])."""
        vec = np.asarray(joints, dtype=np.float32)
        while vec.ndim > 1:
            vec = vec[0]
        for i, v in enumerate(vec):
            key = f"{prefix}/{label}{i}"
            if key in self._numbers:
                self._numbers[key].value = float(v)
            else:
                with self._folder(prefix):
                    self._numbers[key] = self.server.gui.add_number(f"{label}{i}", float(v), disabled=True)

    def _folder(self, prefix: str) -> viser.GuiFolderHandle:
        if prefix not in self._folders:
            self._folders[prefix] = self.server.gui.add_folder(prefix)
        return self._folders[prefix]


class ViserWrappedPolicy(BasePolicy):
    """Plot each step's inputs (camera images, measured joints) and output
    (commanded joints as numbers, blue ghost at the next action, red ghost
    at the final action of the chunk) to a PolicyViewer, then return the
    inner policy's result.

    Defaults match the scripts/serve_dagger.py contract: per-view image keys,
    ``robot0_joint_pos`` (7,) + ``robot0_gripper_norm`` (1,) proprio, and a
    ``result["action"]`` of absolute joint targets + gripper. Raw client obs
    go under the ``input`` folder; when the inner policy has a
    ``_preprocess``, the frames and proprio vector the net actually consumes
    are also shown, under ``net``.
    """

    def __init__(
        self,
        inner: BasePolicy,
        viewer: PolicyViewer,
        *,
        views: tuple[str, ...] = ("low", "side", "wrist"),
        joint_key: str = "robot0_joint_pos",
        gripper_key: str = "robot0_gripper_norm",
    ) -> None:
        self.inner = inner
        self.viewer = viewer
        self.views = views
        self.joint_key = joint_key
        self.gripper_key = gripper_key

    def reset(self, payload: dict | None = None) -> None:
        self.inner.reset(payload)

    def step(self, obs: dict) -> dict:
        try:
            self._show_inputs(obs)
        except Exception:
            log.exception("viser input display failed")
        result = self.inner.step(obs)
        try:
            self._show_outputs(result)
        except Exception:
            log.exception("viser output display failed")
        return result

    def _show_inputs(self, obs: dict) -> None:
        images = {name: obs[name] for name in self.views if name in obs}
        if images:
            self.viewer.show_images(images)
        prep = getattr(self.inner, "_preprocess", None)
        if prep is not None:
            # what the net actually sees: resized/cropped frames in training
            # view order + the concatenated proprio vector
            prop, rgb = prep(obs)
            frames = np.asarray(rgb)
            while frames.ndim > 4:
                frames = frames[0]  # (V, 3, H, W)
            self.viewer.show_images(
                {name: f.transpose(1, 2, 0) for name, f in zip(self.views, frames)},
                prefix="net",
            )
            self.viewer.show_joints(prop, prefix="net", label="p")
        joints = obs.get(self.joint_key)
        if joints is None:
            return
        gripper = obs.get(self.gripper_key)
        if gripper is not None:  # [j0..j6, gripper] = the URDF's actuated joints
            joints = np.concatenate([np.atleast_1d(np.squeeze(joints)), np.atleast_1d(np.squeeze(gripper))])
        self.viewer.show_joints(joints)
        self.viewer.show_robot(joints)

    def _show_outputs(self, result: dict) -> None:
        action = result.get("action") if isinstance(result, dict) else None
        if action is None:
            return
        # the full remaining horizon when the inner policy exposes it (see
        # DaggerPolicy.horizon) — in rtc/replan modes the returned action is
        # a single row of a chunk the wrapper would otherwise never see
        plan = getattr(self.inner, "horizon", None)
        if plan is None:
            plan = np.asarray(action)
        self.viewer.show_plan(plan)
        self.viewer.show_joints(plan, prefix="output")
