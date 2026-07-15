"""Nyx (path-traced) renderer config and asset builders.

Renderer-only concerns: spp, lights, the shadow split, and turning a SplatAsset
into a nyx LightFieldAsset.

The gs_nyx imports MUST stay at module top: nyx's memory globals are only wired
up when its libraries load before ``gs.init()``, and importing them any later
asserts ("Memory Globals not initialized") or segfaults on the first
LightFieldAsset use. Import this module (or ``xsim.suite``) before initializing
Genesis — the same constraint ``xsim.task_env`` satisfies the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# gs_nyx must load BEFORE gs.init(): imported any later, nyx's memory globals
# never initialize and the first LightFieldAsset.uri assignment asserts/segfaults.
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from xsim.suite.models.cameras import CameraSpec, SplatAsset

DEFAULT_LIGHT_DIR = (-0.4, -0.4, -0.8)


@dataclass
class NyxConfig:
    spp: int = 8
    light_dir: tuple[float, float, float] = DEFAULT_LIGHT_DIR
    light_intensity: float = 2.0  # 5.0 washes out mesh entities vs the dim splat
    # fraction of the light that casts shadows; the rest becomes a coincident
    # shadowless fill so total illumination is unchanged
    shadow_strength: float = 1.0
    use_splat: bool = True
    splat: SplatAsset | None = None  # None -> the arena's splat, if it has one


def make_light_field(splat: SplatAsset):
    """SplatAsset -> nyx LightFieldAsset, in the renderer's y-up world."""
    uri = Path(splat.uri).expanduser()
    if not uri.exists():
        raise FileNotFoundError(f"splat file does not exist: {uri}")
    light_field = nps.LightFieldAsset()
    light_field.type = nps.ELightFieldType.GaussianField
    light_field.uri = str(uri)
    # The nyx exporter converts mesh instances from Genesis z-up to nyx y-up but
    # passes LightFieldAssets through raw, so the same conversion applies here.
    light_field.position = nps.float3_z_up_to_y_up_a(nps.float3(*splat.pos))
    light_field.rotation = nps.quaternion_z_up_to_y_up_a(nps.quaternion(*splat.quat_xyzw))
    light_field.scale = nps.float3(splat.scale, splat.scale, splat.scale)
    return light_field


def build_lights(cfg: NyxConfig) -> list[dict]:
    """Directional key/fill pair implementing the shadow dial.

    The exporter concatenates lights across ALL sensors, so callers must attach
    these to the first sensor only; the x3 keeps the approved look from when the
    one light was baked once per camera (lights sum linearly).
    """
    intensity = cfg.light_intensity * 3.0
    base = {"color": (1.0, 1.0, 1.0), "type": "directional", "dir": tuple(cfg.light_dir)}
    s = float(np.clip(cfg.shadow_strength, 0.0, 1.0))
    lights = []
    if s > 0.0:
        lights.append(dict(base, intensity=intensity * s, shadow=True))
    if s < 1.0:
        lights.append(dict(base, intensity=intensity * (1.0 - s), shadow=False))
    return lights


def camera_options(
    spec: CameraSpec,
    res: tuple[int, int],
    fov_deg: float,
    spp: int,
    lights: list[dict],
    light_fields: tuple,
):
    return NyxCameraOptions(
        res=res,
        fov=fov_deg,
        pos=spec.pos or (1.0, 0.0, 0.5),
        lookat=spec.lookat or (0.0, 0.0, 0.0),
        up=spec.up,
        near=0.02,
        far=50.0,
        spp=spp,
        render_mode=npr.ERenderMode.FastPathTracer,
        lights=lights,
        light_fields=light_fields,
    )
