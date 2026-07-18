"""Base arena model."""

from __future__ import annotations

import genesis as gs

from xsim.suite.models.cam_space import CamSampler
from xsim.suite.models.cameras import CameraSpec, SplatAsset


class Arena:
    """Base workspace model; subclasses add their fixtures to a scene and may
    declare workspace cameras (robosuite keeps these in the arena XML) — fixed
    placements (CameraSpec) or pose distributions (CamSampler)."""

    cameras: tuple[CameraSpec | CamSampler, ...] = ()
    # scanned-scene splat and its compositing policy: the arena owns what the
    # background looks like; BatchConfig keeps only renderer perf knobs
    splat: SplatAsset | None = None
    splat_bg: bool = False
    # re-rasterize backgrounds every N policy steps (0 = reset only); needed
    # because attached/wrist cams drift mid-episode
    splat_resplat_every: int = 0

    def add_to(self, scene: gs.Scene) -> None:
        raise NotImplementedError

    def set_camera(self, spec: CameraSpec | CamSampler) -> None:
        """Find-or-replace a camera (fixed spec or sampler) by name (robosuite's
        Arena.set_camera); task envs may nudge cameras in _load_model before the
        scene exists."""
        self.cameras = (*(c for c in self.cameras if c.name != spec.name), spec)
