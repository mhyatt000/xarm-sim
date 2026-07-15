"""Environment layer that owns robots, cameras, and the action interface."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Literal

import gymnasium as gym
import numpy as np

from xsim.suite.environments.base import GenesisEnv
from xsim.suite.models.cameras import CameraSpec, pose_to_T
from xsim.suite.models.robots import create_robot_model
from xsim.suite.renderers import NyxConfig
from xsim.suite.robots import Robot


class RobotEnv(GenesisEnv):
    """Env with robots: owns Robot instances, defines the action interface from
    their controllers, and fans actions out. Envs never touch actuators —
    all control flows through robot.control().

    Cameras are declared on the models (Arena.cameras, RobotModel.cameras);
    this layer selects which to instantiate (``camera_names``; None = all
    declared), at what resolution, and on which render backend. Frames are
    rendered on demand via :meth:`render_views` — never on the physics path.
    """

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        camera_names: list[str] | None = None,
        camera_res: tuple[int, int] = (640, 480),
        fov_deg: float = 42.0,
        render_backend: Literal["raster", "nyx"] = "raster",
        renderer_config: NyxConfig | None = None,
        **kwargs,
    ):
        names = [robots] if isinstance(robots, str) else list(robots)
        self.robots = [Robot(create_robot_model(name)) for name in names]
        self.camera_names = camera_names
        self.camera_res = camera_res
        self.fov_deg = fov_deg
        self.render_backend = render_backend
        self.renderer_config = renderer_config
        self.cams: dict[str, object] = {}
        self._camera_specs: dict[str, CameraSpec] = {}
        self._camera_owner: dict[str, object] = {}  # spec name -> robot entity (None = static)
        self._rig_attached: set[str] = set()
        self._manual_attached: list[tuple[object, object, np.ndarray]] = []
        super().__init__(**kwargs)

    # -- cameras -----------------------------------------------------------------
    def _declared_cameras(self) -> list[tuple[CameraSpec, object]]:
        """(spec, owning robot entity) pairs; arena cams have no owner."""
        declared = [(spec, None) for spec in self.model.arena.cameras]
        for robot in self.robots:
            declared += [(spec, robot.model.entity) for spec in robot.model.cameras]
        return declared

    def _setup_cameras(self) -> None:
        declared = self._declared_cameras()
        if self.camera_names is not None:
            by_name = {spec.name: (spec, owner) for spec, owner in declared}
            unknown = [n for n in self.camera_names if n not in by_name]
            if unknown:
                raise ValueError(f"unknown cameras {unknown}; declared: {sorted(by_name)}")
            declared = [by_name[n] for n in self.camera_names]
        for spec, owner in declared:
            if spec.attach_link is not None and owner is None:
                raise ValueError(f"arena camera {spec.name!r} cannot attach to a link")
            self._camera_specs[spec.name] = spec
            self._camera_owner[spec.name] = owner
        if not self._camera_specs:
            return

        if self.render_backend == "nyx":
            from xsim.suite.renderers.nyx import build_lights, camera_options, make_light_field

            cfg = self.renderer_config or NyxConfig()
            splat = cfg.splat or getattr(self.model.arena, "splat", None)
            light_fields = (make_light_field(splat),) if (cfg.use_splat and splat) else ()
            lights = build_lights(cfg)
            for i, spec in enumerate(self._camera_specs.values()):
                # lights/splat ride the FIRST sensor only: the exporter
                # concatenates them across all sensors (see renderers.nyx)
                self.cams[spec.name] = self.scene.add_sensor(
                    camera_options(
                        spec, self.camera_res, spec.fov_deg or self.fov_deg, cfg.spp,
                        lights if i == 0 else [], light_fields if i == 0 else (),
                    )
                )
        else:
            for spec in self._camera_specs.values():
                self.cams[spec.name] = self.scene.add_camera(
                    res=self.camera_res, fov=spec.fov_deg or self.fov_deg, GUI=False,
                    pos=spec.pos or (1.0, 0.0, 0.5), lookat=spec.lookat or (0.0, 0.0, 0.0),
                    near=0.02, far=50.0,  # default near clips the wrist cam's own gripper
                )

    def _bind_cameras(self) -> None:
        """Post-build: pose static cams (incl. their up vector) and attach mounted ones."""
        for name, spec in self._camera_specs.items():
            cam = self.cams[name]
            if spec.attach_link is not None:
                link = self._camera_owner[name].get_link(spec.attach_link)
                offset = np.asarray(spec.attach_offset, dtype=np.float64)
                if hasattr(cam, "attach"):
                    cam.attach(link, offset)
                    self._rig_attached.add(name)
                else:
                    self._manual_attached.append((cam, link, offset))
            elif hasattr(cam, "set_pose"):
                cam.set_pose(pos=spec.pos, lookat=spec.lookat, up=spec.up)
            else:
                cam.update_camera_pose(pos=spec.pos, lookat=spec.lookat, up=spec.up)
        self._sync_attached_cams()

    def _sync_attached_cams(self) -> None:
        for name in self._rig_attached:
            self.cams[name].move_to_attach()
        for cam, link, offset in self._manual_attached:
            T = pose_to_T(link.get_pos(), link.get_quat()) @ offset
            pos = tuple(T[:3, 3])
            cam.update_camera_pose(pos=pos, lookat=tuple(T[:3, 3] - T[:3, 2]), up=tuple(T[:3, 1]))

    def _post_sim_step(self) -> None:
        self._sync_attached_cams()

    def render_views(self) -> dict[str, np.ndarray]:
        """Named RGB frames from every instantiated camera (on demand — rendering
        is never part of the step loop)."""
        out = {}
        for name, cam in self.cams.items():
            rgb = cam.render(rgb=True)[0] if hasattr(cam, "render") else cam.read(envs_idx=0).rgb
            rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            out[name] = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        return out

    def render(self):
        views = self.render_views()
        if not views:
            return None
        return np.concatenate([views[k] for k in sorted(views)], axis=1)

    def intrinsics(self, name: str) -> np.ndarray:
        cam = self.cams[name]
        if hasattr(cam, "intrinsics"):
            return np.asarray(cam.intrinsics, dtype=np.float64)
        # nyx sensors don't expose K; derive it from the spec's vertical FOV
        w, h = self.camera_res
        fov = self._camera_specs[name].fov_deg or self.fov_deg
        fy = (h / 2.0) / math.tan(math.radians(fov) / 2.0)
        return np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])

    # -- robots ------------------------------------------------------------------
    def _setup_references(self) -> None:
        super()._setup_references()
        for robot in self.robots:
            robot.setup()
        self._bind_cameras()
        lows, highs = zip(*(robot.action_limits for robot in self.robots))
        self.action_space = gym.spaces.Box(
            np.concatenate(lows).astype(np.float32),
            np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )

    @property
    def action_dim(self) -> int:
        return sum(r.action_dim for r in self.robots)

    def _setup_observables(self) -> OrderedDict[str, Callable[[], np.ndarray]]:
        observables = super()._setup_observables()
        for i, robot in enumerate(self.robots):
            pf = f"robot{i}_"
            observables[pf + "joint_pos"] = lambda robot=robot: robot.joint_positions
            observables[pf + "joint_vel"] = lambda robot=robot: robot.joint_velocities
            observables[pf + "eef_pos"] = lambda robot=robot: robot.ee_pos
            observables[pf + "eef_quat"] = lambda robot=robot: robot.ee_quat
            observables[pf + "gripper_norm"] = lambda robot=robot: [robot.gripper_norm]
        return observables

    def _pre_action(self, action) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(
                f"Action has dimension {a.shape[0]}, expected {self.action_dim}"
            )
        offset = 0
        for robot in self.robots:
            robot.control(a[offset : offset + robot.action_dim])
            offset += robot.action_dim

    def _reset_internal(self) -> None:
        super()._reset_internal()
        for robot in self.robots:
            robot.reset()
        self._sync_attached_cams()  # wrist cam follows the freshly reset arm
