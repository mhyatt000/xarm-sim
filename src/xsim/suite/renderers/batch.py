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
    # splat-background perf knobs (whether/when to composite is arena policy:
    # Arena.splat_bg / Arena.splat_resplat_every)
    splat_chunk: int = 1024  # cameras per gsplat rasterization call
    # drop gaussians below this opacity at load: render cost is linear in
    # count, and <=0.15 keeps plates visually intact (see gsplat_plates.py)
    splat_prune_opacity: float = 0.15
