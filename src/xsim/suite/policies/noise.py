"""Policy wrappers that perturb a child policy's actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class NoisyPolicyWrapper:
    """With probability ``prob`` per act(), adds zero-mean gaussian noise to the
    wrapped policy's action; otherwise the action passes through untouched.

    ``sigma`` broadcasts against the action, so per-dimension scales work — e.g.
    ``sigma=[0.02] * 7 + [0.0]`` jitters the arm joints of the canonical
    ``[j0..j6, g]`` action while leaving the gripper channel deterministic
    (a scalar sigma also perturbs ``g`` and can flip it across the 0.5
    open/close threshold).
    """

    def __init__(
        self,
        policy,
        sigma: float | Sequence[float] = 0.01,
        prob: float = 0.5,
        seed: int | None = None,
    ):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.policy = policy
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def reset(self, obs=None) -> None:
        self.policy.reset(obs)

    def act(self, obs=None) -> np.ndarray:
        action = np.asarray(self.policy.act(obs))
        if self.rng.random() >= self.prob:
            return action
        noisy = action + self.rng.normal(0.0, self.sigma, size=action.shape)
        dtype = action.dtype if np.issubdtype(action.dtype, np.floating) else np.float64
        return noisy.astype(dtype)
