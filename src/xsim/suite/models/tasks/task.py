"""Task composition of arena, robots, and objects."""

from __future__ import annotations

import genesis as gs

from xsim.suite.models.arenas.arena import Arena
from xsim.suite.models.objects.objects import GenesisObject


class Task:
    """Arena + robot model(s) + free object(s) composed into one world model.

    Genesis loads URDF/MJCF directly into entities, so composition is
    entity-level (no XML merging): add_to() populates the scene and binds
    each RobotModel's entity for the runtime Robot to pick up in setup().
    """

    def __init__(
        self,
        arena: Arena,
        robot_models,
        objects: GenesisObject | list[GenesisObject] | None = None,
    ) -> None:
        self.arena = arena
        self.robot_models: list = (
            list(robot_models)
            if isinstance(robot_models, (list, tuple))
            else [robot_models]
        )
        if objects is None:
            self.objects: list[GenesisObject] = []
        elif isinstance(objects, (list, tuple)):
            self.objects = list(objects)
        else:
            self.objects = [objects]

    def add_to(self, scene: gs.Scene) -> None:
        self.arena.add_to(scene)
        for model in self.robot_models:
            model.entity = scene.add_entity(
                material=gs.materials.Rigid(),
                morph=model.make_morph(),
            )
        for obj in self.objects:
            obj.add_to(scene)
