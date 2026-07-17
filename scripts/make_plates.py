"""Render splat-only background plates for the arena's static cameras.

One nyx render per static camera pose with ONLY the arena's gaussian splat in
view (no meshes): the plates composite behind madrona batch frames wherever
the segmentation mask says background (see ImageObsWrapper), giving image
policies the lab backdrop at batch-render throughput. Wrist/attached cameras
move, so they get no plates.

    uv run python scripts/make_plates.py                 # assets/plates/<cam>.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import xsim.suite  # noqa: F401  # gs_nyx must import before gs.init
from xsim.suite.models.arenas import TableArena
from xsim.suite.renderers import NyxConfig
from xsim.suite.renderers.nyx import build_lights, camera_options, make_light_field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    out: Path = PROJECT_ROOT / "assets" / "plates"
    res: tuple[int, int] = (640, 480)
    spp: int = 64
    fov_deg: float = 42.0  # fallback for specs without their own


def main(cfg: Config) -> None:
    import cv2
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    arena = TableArena()
    if arena.splat is None:
        raise SystemExit("arena has no splat asset; nothing to render")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 120),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )
    # nyx needs an entity to export; park a pebble far underground, out of view
    scene.add_entity(gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.0, -5.0)))

    ncfg = NyxConfig(spp=cfg.spp)
    light_fields = (make_light_field(arena.splat),)
    lights = build_lights(ncfg)
    sensors = {}
    static = [s for s in arena.cameras if s.attach_link is None]
    for i, spec in enumerate(static):
        sensors[spec.name] = scene.add_sensor(
            camera_options(spec, cfg.res, spec.fov_deg or cfg.fov_deg, cfg.spp,
                           lights if i == 0 else [], light_fields if i == 0 else ())
        )
    scene.build()
    scene.step()

    cfg.out.mkdir(parents=True, exist_ok=True)
    for name, sensor in sensors.items():
        rgb = sensor.read().rgb
        rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        while rgb.ndim > 3:
            rgb = rgb[0]
        path = cfg.out / f"{name}.png"
        cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., :3][..., ::-1]))
        print(f"{name}: {rgb.shape} -> {path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
