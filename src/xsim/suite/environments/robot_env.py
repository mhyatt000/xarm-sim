"""Environment layer that owns robots, cameras, and the action interface."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Literal

import gymnasium as gym
import numpy as np

from xsim.suite.environments.base import GenesisEnv
from xsim.suite.models.cam_space import CamSampler
from xsim.suite.models.cameras import (
    CameraSpec,
    T_GL_TO_CV,
    invert_rigid,
    rots_from_quat_wxyz,
    viewmats_cv,
)
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
        render_backend: Literal["raster", "nyx", "batch"] = "batch",
        renderer_config: NyxConfig | BatchConfig | None = None,
        init_tcp_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        | None = None,
        **kwargs,
    ):
        names = [robots] if isinstance(robots, str) else list(robots)
        self.robots = [Robot(create_robot_model(name)) for name in names]
        # ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)): reset seats robot 0's arm
        # at IK(uniform TCP in this box, home orientation); None = home pose
        self.init_tcp_box = init_tcp_box
        self._home_ee_quat: np.ndarray | None = None
        self.camera_names = camera_names
        self.camera_res = camera_res
        self.fov_deg = fov_deg
        self.render_backend = render_backend
        self.renderer_config = renderer_config
        self.cams: dict[str, object] = {}
        self._camera_specs: dict[str, CameraSpec | CamSampler] = {}
        self._camera_owner: dict[str, object] = {}  # spec name -> robot entity (None = static)
        self._rig_attached: set[str] = set()  # spec-mounted cams using genesis attach()
        # live splat backgrounds (Arena.splat_bg): renderer, per-cam bound
        # poses, and the per-env frames composited in render_views
        self._splat_bg = None
        # name -> per-env ((n, 3) pos, (n, 3) lookat, (n, 3) up); n = n_envs on
        # batch, 1 elsewhere (raster cams hold env 0's pose)
        self._static_cam_poses: dict[str, tuple] = {}
        # name -> (link, GL link->cam offset): (4, 4) for spec mounts, (n, 4, 4)
        # for sampled mounts (genesis attach() takes one offset, so per-env
        # offsets are re-posed manually in _sync_attached_cams)
        self._attached_rigs: dict[str, tuple] = {}
        self._splat_bg_frames: dict[str, np.ndarray] = {}
        self._splat_steps = 0  # global step count driving splat_resplat_every
        self._render_stale = False  # reset moved cameras/state without advancing scene.t
        super().__init__(**kwargs)

    # -- cameras -----------------------------------------------------------------
    def _scene_renderer(self):
        if self.render_backend == "batch":
            import genesis as gs

            cfg = self.renderer_config or BatchConfig()
            return gs.options.renderers.BatchRenderer(use_rasterizer=cfg.use_rasterizer)
        return None

    def _declared_cameras(self) -> list[tuple[CameraSpec | CamSampler, object]]:
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
            sampled = [s.name for s in self._camera_specs.values() if isinstance(s, CamSampler)]
            if sampled:
                raise ValueError(
                    f"cameras {sampled} are CamSamplers, unsupported on the nyx "
                    "backend: sensors take their pose at sensor creation"
                )
            from xsim.suite.renderers.nyx import build_lights, camera_options, make_light_field

            cfg = self.renderer_config or NyxConfig()
            splat = cfg.splat or self.model.arena.splat
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
                # samplers have no fixed pose; real poses land at _bind_cameras
                self.cams[spec.name] = self.scene.add_camera(
                    res=self.camera_res, fov=spec.fov_deg or self.fov_deg, GUI=False,
                    pos=getattr(spec, "pos", None) or (1.0, 0.0, 0.5),
                    lookat=getattr(spec, "lookat", None) or (0.0, 0.0, 0.0),
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
        self._resolve_static_poses()
        for name, spec in self._camera_specs.items():
            cam = self.cams[name]
            if spec.attach_link is not None:
                # nyx sensors have no attach(); they mount via options at creation
                # (entity_idx/link_idx_local/offset_T) and re-pose per env inside
                # every render, so there is nothing to bind or sync for them
                if hasattr(cam, "attach"):
                    link = self._camera_owner[name].get_link(spec.attach_link)
                    if isinstance(spec, CamSampler):
                        n = self.n_envs if self.render_backend == "batch" else 1
                        self._attached_rigs[name] = (link, self._sample_rig_offsets(spec, n))
                    else:
                        offset = np.asarray(spec.attach_offset, dtype=np.float64)
                        cam.attach(link, offset)
                        self._rig_attached.add(name)
                        self._attached_rigs[name] = (link, offset)
            elif hasattr(cam, "set_pose"):
                self._push_cam_pose(name, *self._static_cam_poses[name])
            else:
                cam.update_camera_pose(pos=spec.pos, lookat=spec.lookat, up=spec.up)
        self._sync_attached_cams()

    def _resolve_static_poses(self) -> None:
        """Per-env pose arrays for every static camera — the only static-cam
        representation: fixed specs tile their single pose, samplers draw one
        per env."""
        n = self.n_envs if self.render_backend == "batch" else 1
        for name, spec in self._camera_specs.items():
            if spec.attach_link is not None:
                continue
            if isinstance(spec, CamSampler):
                self._static_cam_poses[name] = spec.sample(self.np_random, n)
            else:
                self._static_cam_poses[name] = tuple(
                    np.tile(np.asarray(v, dtype=np.float64), (n, 1))
                    for v in (
                        spec.pos or (1.0, 0.0, 0.5),  # add_camera defaults
                        spec.lookat or (0.0, 0.0, 0.0),
                        spec.up,
                    )
                )

    def _push_cam_pose(self, name: str, pos, lookat, up) -> None:
        # always full arrays, never envs_idx: genesis set_pose(envs_idx=...)
        # has an inverted membership check; raster cams take a single pose
        if self.render_backend == "batch":
            self.cams[name].set_pose(pos=pos, lookat=lookat, up=up)
        else:
            self.cams[name].set_pose(pos=pos[0], lookat=lookat[0], up=up[0])

    def _sample_rig_offsets(self, spec: CamSampler, n: int) -> np.ndarray:
        """(n, 4, 4) link->camera offsets in the OpenGL convention genesis
        attach offsets use, from link-frame sampled poses."""
        pos, lookat, up = spec.sample(self.np_random, n)
        return invert_rigid(viewmats_cv(pos, lookat, up)) @ T_GL_TO_CV

    def _link_T(self, link, idx=None) -> np.ndarray:
        quat = np.atleast_2d(_np(link.get_quat()))
        pos = np.atleast_2d(_np(link.get_pos()))
        if idx is not None:
            quat, pos = quat[idx], pos[idx]
        T = np.tile(np.eye(4), (len(quat), 1, 1))
        T[:, :3, :3] = rots_from_quat_wxyz(quat)
        T[:, :3, 3] = pos
        return T

    def _resample_cameras(self, envs_idx=None) -> None:
        idx = np.arange(self.n_envs) if envs_idx is None else np.atleast_1d(envs_idx)
        for name, spec in self._camera_specs.items():
            if not (isinstance(spec, CamSampler) and spec.resample_on_reset):
                continue
            if spec.attach_link is None:
                pos, lookat, up = self._static_cam_poses[name]
                rows = idx[idx < len(pos)]  # non-batch cams hold env 0's pose only
                if not len(rows):
                    continue
                pos[rows], lookat[rows], up[rows] = spec.sample(self.np_random, len(rows))
                self._push_cam_pose(name, pos, lookat, up)
            else:
                link, offset = self._attached_rigs[name]
                rows = idx[idx < len(offset)]
                if len(rows):
                    offset[rows] = self._sample_rig_offsets(spec, len(rows))

    def _setup_splat_bg(self) -> None:
        arena = self.model.arena
        if not (self.render_backend == "batch" and arena.splat_bg and arena.splat is not None):
            return
        from xsim.suite.renderers.splat_bg import SplatBackground

        cfg = self.renderer_config or BatchConfig()
        self._splat_bg = SplatBackground(
            arena.splat, chunk=cfg.splat_chunk, prune_opacity=cfg.splat_prune_opacity
        )

    def _render_splat_bg(self, envs_idx=None, force: bool = False) -> None:
        """Refresh per-env splat background frames for every camera. Fixed static
        cams render once — a (1, H, W, 3) frame that broadcasts across envs —
        unless ``force``; sampled static cams keep an (n_envs, H, W, 3) buffer
        whose reset rows are refreshed; attached cams re-render the reset envs
        from their current link pose — the backdrop drifts as the arm moves, but
        wrist frames rarely show background."""
        if self._splat_bg is None:
            return
        idx = np.arange(self.n_envs) if envs_idx is None else np.atleast_1d(envs_idx)
        for name, spec in self._camera_specs.items():
            K = self.intrinsics(name)
            if spec.attach_link is None:
                pos, lookat, up = self._static_cam_poses[name]
                fixed = not isinstance(spec, CamSampler) or not spec.resample_on_reset
                if fixed and name in self._splat_bg_frames and not force:
                    continue
                if not isinstance(spec, CamSampler):
                    self._splat_bg_frames[name] = self._splat_bg.render(
                        viewmats_cv(pos[:1], lookat[:1], up[:1]), K, self.camera_res
                    )
                    continue
                buf = self._splat_bg_frames.get(name)
                # first render fills every env: build-time poses are all live
                rows = idx if buf is not None else np.arange(len(pos))
                frames = self._splat_bg.render(
                    viewmats_cv(pos[rows], lookat[rows], up[rows]), K, self.camera_res
                )
                if buf is None:
                    self._splat_bg_frames[name] = frames
                else:
                    buf[rows] = frames
            else:
                rig = self._attached_rigs.get(name)
                if rig is None:
                    continue
                link, offset = rig
                off = offset[idx] if offset.ndim == 3 else offset
                # attach offsets are OpenGL link->cam transforms; gsplat wants
                # OpenCV world->cam
                vm = invert_rigid(self._link_T(link, idx) @ (off @ T_GL_TO_CV))
                frames = self._splat_bg.render(vm, K, self.camera_res)
                buf = self._splat_bg_frames.get(name)
                if buf is None:
                    w, h = self.camera_res
                    buf = np.zeros((self.n_envs, h, w, 3), dtype=np.uint8)
                    self._splat_bg_frames[name] = buf
                buf[idx] = frames
        # release the rasterization peak back to the driver: taichi and madrona
        # allocate outside torch, and the cached chunk buffers would otherwise
        # hold the peak for the whole rollout (4 ranks x 44GB L40S died on this)
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sync_attached_cams(self) -> None:
        for name in self._rig_attached:
            self.cams[name].move_to_attach()
        for name, (link, offset) in self._attached_rigs.items():
            if offset.ndim == 2:  # attach()ed above
                continue
            c2w = self._link_T(link)[: len(offset)] @ offset  # OpenGL cam-to-world
            pos = c2w[:, :3, 3]
            self._push_cam_pose(name, pos, pos - c2w[:, :3, 2], c2w[:, :3, 1])

    def _post_sim_step(self) -> None:
        self._sync_attached_cams()

    def render_views(self, all_envs: bool = False) -> dict[str, np.ndarray]:
        """Named RGB frames from every instantiated camera (on demand — rendering
        is never part of the step loop). ``all_envs=True`` returns (n_envs, H, W, 3)
        stacks instead of env 0's frame — nyx and batch backends only, which
        render every env anyway; raster cameras are built on env 0."""
        out = {}
        # the batch renderer caches frames until scene.t advances, so a reset
        # (new camera poses, new state, same t) must force one fresh pass; the
        # cache is shared across cameras, so only the first render needs it
        force = self._render_stale and self.render_backend == "batch"
        self._render_stale = False
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
                    rgb_t, _, seg_t, _ = cam.render(
                        rgb=True, segmentation=True, force_render=force
                    )
                    rgb = np.where(
                        (_np(seg_t) == 0)[..., None], bg, _np(rgb_t)[..., :3]
                    )
                else:
                    rgb = cam.render(rgb=True, force_render=force)[0]
                force = False
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
        if self.init_tcp_box is not None:
            self._randomize_init_tcp(envs_idx)
        self._resample_cameras(envs_idx)
        self._sync_attached_cams()  # wrist cam follows the freshly reset arm
        self._render_splat_bg(envs_idx)
        self._render_stale = True

    def _randomize_init_tcp(self, envs_idx=None) -> None:
        """Seat robot 0's arm at IK(uniform TCP in ``init_tcp_box``), keeping the
        home orientation. IK is solved for the full batch (Genesis IK is
        all-envs); only ``envs_idx`` rows are applied."""
        import torch

        robot = self.robots[0]
        if self._home_ee_quat is None:  # robots were just reset -> EE is at home
            i = 0 if envs_idx is None else int(np.atleast_1d(envs_idx)[0])
            self._home_ee_quat = np.asarray(robot.ee_quat, dtype=np.float64)[i].copy()
        lo, hi = np.array(self.init_tcp_box, dtype=np.float64).T  # (3,), (3,)
        pos = self.np_random.uniform(lo, hi, size=(self.n_envs, 3))
        pose = np.concatenate([pos, np.tile(self._home_ee_quat, (self.n_envs, 1))], axis=1)
        q = robot.ik(torch.as_tensor(pose, dtype=torch.float32))
        robot.set_arm_qpos(q, envs_idx)

    def step(self, action):
        out = super().step(action)
        every = self.model.arena.splat_resplat_every
        if self._splat_bg is not None and every > 0:
            self._splat_steps += 1
            if self._splat_steps % every == 0:
                self._render_splat_bg(force=True)
        return out
