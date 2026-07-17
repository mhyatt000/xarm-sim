"""Madrona batch renderer config.

Renderer-only concerns for render_backend="batch": rasterizer vs raytracer,
and the light rig (madrona takes no lights from the scene; unlit frames are
near-black). Lighting is part of the observation domain for image policies —
freeze it per experiment, or vary it deliberately as domain randomization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchLight:
    dir: tuple[float, float, float]
    intensity: float
    castshadow: bool = False
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pos: tuple[float, float, float] = (0.0, 0.0, 3.0)  # unused for directional


@dataclass
class BatchConfig:
    # raytracer by default: 273k vs 102k env-cam frames/s at B=256x3cam@64px on a
    # 5090, with real shadows; the rasterizer is the compatibility path
    use_rasterizer: bool = False
    # key/fill pair matching the nyx DEFAULT_LIGHT_DIR look; x0.85 intensity from
    # the visual match against the nyx render (see gs-madrona notes: ambient is
    # hardcoded 0.05, so the pair carries almost all illumination)
    lights: tuple[BatchLight, ...] = (
        BatchLight(dir=(-0.4, -0.4, -0.8), intensity=1.7, castshadow=True),
        BatchLight(dir=(0.5, 0.3, -0.6), intensity=0.85),
    )
    # per-env static-camera jitter (domain randomization): each env's static
    # cams get a uniform offset in [-noise, noise] per axis added to the spec's
    # pos/lookat, sampled once at build. Batch cams hold one pose per env, so
    # this is free; attached cams ride their link and are untouched. Background
    # plates baked at the default pose (make_plates.py) misalign under jitter.
    cam_pos_noise: float = 0.0  # metres
    cam_lookat_noise: float = 0.0  # metres
    cam_noise_seed: int | None = None
    # live splat backgrounds: rasterize the arena splat with gsplat at reset and
    # composite it wherever segmentation reads background — per-env, so it
    # follows jittered static cams (baked plates can't). Wrist cams get their
    # reset-pose view; it drifts as the arm moves, but wrist frames rarely show
    # background pixels.
    splat_bg: bool = False
    splat_chunk: int = 128  # cameras per gsplat rasterization call
