"""Environment classes and the name-based registry."""

from __future__ import annotations

from xsim.suite.environments.base import REGISTERED_ENVS, GenesisEnv, make
from xsim.suite.environments.robot_env import RobotEnv
from xsim.suite.environments.manipulation import (
    Lift,
    LiftEZ,
    LiftRelease,
    ManipulationEnv,
    PlaceObj,
    PlaceObjaverse,
    PlaceObjBin,
    PlaceObjPlate,
    Stack,
    StackRGY,
)

__all__ = [
    "REGISTERED_ENVS",
    "GenesisEnv",
    "Lift",
    "LiftEZ",
    "LiftRelease",
    "ManipulationEnv",
    "PlaceObj",
    "PlaceObjBin",
    "PlaceObjPlate",
    "PlaceObjaverse",
    "RobotEnv",
    "Stack",
    "StackRGY",
    "make",
]
