"""Environment classes and the name-based registry."""

from __future__ import annotations

from xsim.suite.environments.base import REGISTERED_ENVS, GenesisEnv, make
from xsim.suite.environments.robot_env import RobotEnv
from xsim.suite.environments.manipulation import Lift, LiftEZ, ManipulationEnv

__all__ = [
    "REGISTERED_ENVS",
    "GenesisEnv",
    "Lift",
    "LiftEZ",
    "ManipulationEnv",
    "RobotEnv",
    "make",
]
