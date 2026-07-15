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

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        return (
            float(rng.uniform(*self.x_range)),
            float(rng.uniform(*self.y_range)),
            float(rng.uniform(*self.yaw_range)),
        )
