"""Environment layer that owns robots, cameras, and the action interface."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Literal

import gymnasium as gym
import numpy as np

from xsim.suite.environments.base import GenesisEnv
from xsim.suite.models.cameras import CameraSpec
from xsim.suite.models.robots import create_robot_model
from xsim.suite.renderers import BatchConfig, NyxConfig
from xsim.suite.robots import Robot


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


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
        render_backend: Literal["raster", "nyx", "batch"] = "raster",
        renderer_config: NyxConfig | BatchConfig | None = None,
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
        # live splat backgrounds (BatchConfig.splat_bg): renderer, per-cam bound
        # poses, and the per-env frames composited in render_views
        self._splat_bg = None
        self._static_cam_poses: dict[str, tuple] = {}  # name -> (pos, lookat, up)
        self._attached_rigs: dict[str, tuple] = {}  # name -> (link, offset_T)
        self._splat_bg_frames: dict[str, np.ndarray] = {}
        super().__init__(**kwargs)

    # -- cameras -----------------------------------------------------------------
    def _scene_renderer(self):
        if self.render_backend == "batch":
            import genesis as gs

            cfg = self.renderer_config or BatchConfig()
            return gs.options.renderers.BatchRenderer(use_rasterizer=cfg.use_rasterizer)
        return None

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
                        owner=self._camera_owner[spec.name],
                    )
                )
        else:
            for spec in self._camera_specs.values():
                self.cams[spec.name] = self.scene.add_camera(
                    res=self.camera_res, fov=spec.fov_deg or self.fov_deg, GUI=False,
                    pos=spec.pos or (1.0, 0.0, 0.5), lookat=spec.lookat or (0.0, 0.0, 0.0),
                    near=0.02, far=50.0,  # default near clips the wrist cam's own gripper
                    # batch (madrona) cameras render every env; raster is single-env
                    **({} if self.render_backend == "batch" else {"env_idx": 0}),
                )
            if self.render_backend == "batch":
                # madrona takes no lights from the scene; unlit frames are near-black
                cfg = self.renderer_config or BatchConfig()
                for light in cfg.lights:
                    self.scene.add_light(
                        pos=light.pos, dir=light.dir, color=light.color,
                        directional=True, castshadow=light.castshadow,
                        cutoff=45.0, intensity=light.intensity,
                    )

    def _bind_cameras(self) -> None:
        """Post-build: pose static cams (incl. their up vector) and attach mounted ones."""
        jitter = self._static_cam_jitter()
        for name, spec in self._camera_specs.items():
            cam = self.cams[name]
            if spec.attach_link is not None:
                # nyx sensors have no attach(); they mount via options at creation
                # (entity_idx/link_idx_local/offset_T) and re-pose per env inside
                # every render, so there is nothing to bind or sync for them
                if hasattr(cam, "attach"):
                    link = self._camera_owner[name].get_link(spec.attach_link)
                    offset = np.asarray(spec.attach_offset, dtype=np.float64)
                    cam.attach(link, offset)
                    self._rig_attached.add(name)
                    self._attached_rigs[name] = (link, offset)
            elif hasattr(cam, "set_pose"):
                pos, lookat = jitter(spec) if jitter else (spec.pos, spec.lookat)
                cam.set_pose(pos=pos, lookat=lookat, up=spec.up)
                self._static_cam_poses[name] = (
                    pos if pos is not None else (1.0, 0.0, 0.5),  # add_camera defaults
                    lookat if lookat is not None else (0.0, 0.0, 0.0),
                    spec.up,
                )
            else:
                cam.update_camera_pose(pos=spec.pos, lookat=spec.lookat, up=spec.up)
        self._sync_attached_cams()

    def _static_cam_jitter(self) -> Callable | None:
        """Per-env pose jitter for static cams — batch backend only, where each
        camera holds one pose per env. Returns spec -> ((n_envs, 3) pos,
        (n_envs, 3) lookat), or None when jitter is off."""
        if self.render_backend != "batch":
            return None
        cfg = self.renderer_config or BatchConfig()
        if not (cfg.cam_pos_noise or cfg.cam_lookat_noise):
            return None
        rng = np.random.default_rng(cfg.cam_noise_seed)

        def jitter(spec: CameraSpec):
            # None pos/lookat keep the pose given at add_camera; only jitter
            # what the spec actually pins down
            shape = (self.n_envs, 3)
            pos, lookat = spec.pos, spec.lookat
            if pos is not None:
                pos = np.asarray(pos) + rng.uniform(
                    -cfg.cam_pos_noise, cfg.cam_pos_noise, shape
                )
            if lookat is not None:
                lookat = np.asarray(lookat) + rng.uniform(
                    -cfg.cam_lookat_noise, cfg.cam_lookat_noise, shape
                )
            return pos, lookat

        return jitter

    def _setup_splat_bg(self) -> None:
        cfg = self.renderer_config
        if not (self.render_backend == "batch" and getattr(cfg, "splat_bg", False)):
            return
        splat = getattr(self.model.arena, "splat", None)
        if splat is None:
            return
        from xsim.suite.renderers.splat_bg import SplatBackground

        self._splat_bg = SplatBackground(splat, chunk=cfg.splat_chunk)

    def _render_splat_bg(self, envs_idx=None) -> None:
        """Refresh per-env splat background frames for every camera. Static cams
        render once (their per-env poses are fixed at build); attached cams
        re-render the reset envs from their current link pose — the backdrop
        drifts as the arm moves, but wrist frames rarely show background."""
        if self._splat_bg is None:
            return
        from xsim.suite.models.cameras import T_GL_TO_CV
        from xsim.suite.renderers.splat_bg import (
            invert_rigid, rots_from_quat_wxyz, viewmats_cv,
        )

        idx = np.arange(self.n_envs) if envs_idx is None else np.atleast_1d(envs_idx)
        for name, spec in self._camera_specs.items():
            K = self.intrinsics(name)
            if spec.attach_link is None:
                pose = self._static_cam_poses.get(name)
                if pose is None or name in self._splat_bg_frames:
                    continue
                # unjittered specs give one shared pose -> a (1, H, W, 3) frame
                # that broadcasts across envs in the composite
                self._splat_bg_frames[name] = self._splat_bg.render(
                    viewmats_cv(*pose), K, self.camera_res
                )
            else:
                rig = self._attached_rigs.get(name)
                if rig is None:
                    continue
                link, offset = rig
                link_T = np.tile(np.eye(4), (len(idx), 1, 1))
                link_T[:, :3, :3] = rots_from_quat_wxyz(
                    np.atleast_2d(_np(link.get_quat()))[idx]
                )
                link_T[:, :3, 3] = np.atleast_2d(_np(link.get_pos()))[idx]
                # attach offsets are OpenGL link->cam transforms; gsplat wants
                # OpenCV world->cam
                vm = invert_rigid(link_T @ (offset @ T_GL_TO_CV))
                frames = self._splat_bg.render(vm, K, self.camera_res)
                buf = self._splat_bg_frames.get(name)
                if buf is None:
                    w, h = self.camera_res
                    buf = np.zeros((self.n_envs, h, w, 3), dtype=np.uint8)
                    self._splat_bg_frames[name] = buf
                buf[idx] = frames

    def _sync_attached_cams(self) -> None:
        for name in self._rig_attached:
            self.cams[name].move_to_attach()

    def _post_sim_step(self) -> None:
        self._sync_attached_cams()

    def render_views(self, all_envs: bool = False) -> dict[str, np.ndarray]:
        """Named RGB frames from every instantiated camera (on demand — rendering
        is never part of the step loop). ``all_envs=True`` returns (n_envs, H, W, 3)
        stacks instead of env 0's frame — nyx and batch backends only, which
        render every env anyway; raster cameras are built on env 0."""
        out = {}
        for name, cam in self.cams.items():
            bg = self._splat_bg_frames.get(name)
            if hasattr(cam, "render"):  # raster or batch camera
                if all_envs and self.render_backend != "batch":
                    raise ValueError(
                        "all_envs rendering requires render_backend='nyx' or "
                        "'batch'; raster cameras are single-env"
                    )
                if bg is not None:
                    # splat backdrop wherever madrona rendered no geometry
                    rgb_t, _, seg_t, _ = cam.render(rgb=True, segmentation=True)
                    rgb = np.where(
                        (_np(seg_t) == 0)[..., None], bg, _np(rgb_t)[..., :3]
                    )
                else:
                    rgb = cam.render(rgb=True)[0]
            else:
                rgb = cam.read().rgb if all_envs else cam.read(envs_idx=0).rgb
            rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            if not all_envs and rgb.ndim == 4:
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
        self._setup_splat_bg()
        lows, highs = zip(*(robot.action_limits for robot in self.robots))
        self.single_action_space = gym.spaces.Box(
            np.concatenate(lows).astype(np.float32),
            np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.n_envs
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
            observables[pf + "gripper_norm"] = lambda robot=robot: robot.gripper_norm[:, None]
        return observables

    def _pre_action(self, action) -> None:
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.n_envs, self.action_dim):
            raise ValueError(
                f"Action has shape {a.shape}, expected {(self.n_envs, self.action_dim)}"
            )
        offset = 0
        for robot in self.robots:
            robot.control(a[:, offset : offset + robot.action_dim])
            offset += robot.action_dim

    def _reset_internal(self, envs_idx=None) -> None:
        super()._reset_internal(envs_idx)
        for robot in self.robots:
            robot.reset(envs_idx)
        self._sync_attached_cams()  # wrist cam follows the freshly reset arm
        self._render_splat_bg(envs_idx)
