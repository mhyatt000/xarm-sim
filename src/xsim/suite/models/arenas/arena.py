"""Base arena model."""

from __future__ import annotations

import genesis as gs


class Arena:
    """Base workspace model; subclasses add their fixtures to a scene."""

    def add_to(self, scene: gs.Scene) -> None:
        raise NotImplementedError
