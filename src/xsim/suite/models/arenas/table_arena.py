"""Table workspace arena."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs

from xsim.suite.models.arenas.arena import Arena


@dataclass
class TableArena(Arena):
    """Flat table: infinite collision plane plus an optional visual slab."""

    # Robot base sits on a 1 cm mounting plate, so table top is 1 cm below the base origin.
    top_z: float = -0.01
    center_xy: tuple[float, float] = (0.375, 0.01)
    # Real cart 3ft x 2ft top.
    size_xy: tuple[float, float] = (0.9144, 0.6096)
    color: tuple[float, float, float] = (0.13, 0.14, 0.17)
    slab: bool = True
    slab_height: float = 0.72

    def add_to(self, scene: gs.Scene) -> None:
        surface = gs.surfaces.Plastic(color=self.color, roughness=0.8)
        scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, self.top_z),
                visualization=not self.slab,
                collision=True,
            ),
            surface=surface,
        )
        if self.slab:
            # Stands in for the real cart body; occludes the under-table region.
            scene.add_entity(
                gs.morphs.Box(
                    size=(self.size_xy[0], self.size_xy[1], self.slab_height),
                    pos=(
                        self.center_xy[0],
                        self.center_xy[1],
                        self.top_z - self.slab_height / 2,
                    ),
                    fixed=True,
                    visualization=True,
                    collision=False,
                ),
                surface=surface,
            )
