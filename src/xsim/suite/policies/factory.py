"""Env -> scripted expert wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xsim.suite.environments.manipulation.lift import Lift, LiftRelease
from xsim.suite.environments.manipulation.stack import Stack
from xsim.suite.policies.dual import ActiveArmView, DualArmPolicy
from xsim.suite.policies.lift_expert import LiftExpertPolicy
from xsim.suite.policies.stack_expert import StackExpertPolicy

if TYPE_CHECKING:
    from xsim.suite.environments.robot_env import RobotEnv
    from xsim.suite.robots.robot import Robot


def expert_core(env: RobotEnv, robot: Robot | ActiveArmView):
    """The env's FSM expert, driving one arm (a Robot or an ActiveArmView)."""
    if isinstance(env, Stack):
        return StackExpertPolicy(env, robot=robot)
    if isinstance(env, Lift):
        return LiftExpertPolicy(
            env, robot=robot, recycle=not isinstance(env, LiftRelease)
        )
    raise ValueError(f"no scripted expert for {type(env).__name__}")


def expert_for(env: RobotEnv):
    """Scripted expert for ``env``: the FSM core on ``robots[0]``, or the core
    behind the nearest-arm composer when the env has several robots."""
    if len(env.robots) > 1:
        return DualArmPolicy(env, make_core=expert_core)
    return expert_core(env, env.robots[0])
