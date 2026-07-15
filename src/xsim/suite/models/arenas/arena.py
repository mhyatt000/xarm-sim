"""Base arena model."""

from __future__ import annotations

import genesis as gs

from xsim.suite.models.cameras import CameraSpec


class Arena:
    """Base workspace model; subclasses add their fixtures to a scene and may
    declare static workspace cameras (robosuite keeps these in the arena XML)."""

    cameras: tuple[CameraSpec, ...] = ()

    def add_to(self, scene: gs.Scene) -> None:
        raise NotImplementedError

    def set_camera(self, spec: CameraSpec) -> None:
        """Find-or-replace a camera spec by name (robosuite's Arena.set_camera);
        task envs may nudge cameras in _load_model before the scene exists."""
        self.cameras = (*(c for c in self.cameras if c.name != spec.name), spec)
