"""Object placement sampling for episode resets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class UniformRandomSampler:
    """Uniform (x, y, yaw) draw inside a world-frame rectangle."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)

    def sample(
        self, rng: np.random.Generator, n: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, yaw) arrays of shape (n,)."""
        return (
            rng.uniform(*self.x_range, n),
            rng.uniform(*self.y_range, n),
            rng.uniform(*self.yaw_range, n),
        )
