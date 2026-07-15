"""Gym-style policy protocol.

The env owns ``step(action)``; a policy is just ``obs -> action``. Policies
never apply actions, mutate the robot, or decide termination — ``terminated``/
``truncated`` come from the env, and a finished scripted policy simply keeps
emitting its final hold action.

The action is a vector in the env's action space. For the EE-pose envs in this
repo that is ``[1, 8]`` = ``[x, y, z, qw, qx, qy, qz, g]``: an absolute TCP
pose plus a gripper channel, ``g > 0.5`` meaning open.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

Action = torch.Tensor  # [1, D] vector in the env's action space

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0


@runtime_checkable
class Policy(Protocol):
    """Minimal gym-style policy: ``reset`` per episode, ``act`` per step."""

    def reset(self, obs: Any = None) -> None:
        """Start a new episode. ``obs`` is what ``env.reset()`` returned."""

    def act(self, obs: Any = None) -> Action:
        """Map one observation to one action. Must not mutate the env."""
