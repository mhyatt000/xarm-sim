"""Camera-pose samplers: distributions over placements, drawn at env reset.

A CameraSpec (cameras.py) is one fixed placement; a CamSampler describes where
a camera *may* be, and the env draws a batch of poses from it. Placement stays
a model concern — rendering config is deliberately absent, same as cameras.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CamSampler:
    """Base: a camera whose pose is sampled rather than fixed. Frozen: samplers
    are descriptions — replace them (Arena.set_camera), don't mutate."""

    name: str
    fov_deg: float | None = None  # vertical FOV; None -> env default
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    attach_link: str | None = None
    resample_on_reset: bool = True

    def sample(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(n, 3) pos, (n, 3) lookat, (n, 3) up. World frame when attach_link is
        None; link frame otherwise."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ShellLookatSampler(CamSampler):
    """Positions in a chopped-sphere shell around the origin, lookats in a box.

    The position volume is the sphere |p| <= radius intersected with the x and z
    slabs; inner_scale carves out the same shape shrunk toward the origin, so
    cameras keep their distance from the workspace. Camera/lookat pairs whose
    view ray sits below min_elevation_deg are rejected (near-horizontal views
    see mostly table edge and background).
    """

    radius: float
    x_range: tuple[float, float]
    z_range: tuple[float, float]
    inner_scale: float | None = 0.5
    lookat_lo: tuple[float, float, float]
    lookat_hi: tuple[float, float, float]
    min_elevation_deg: float = 8.0

    def _inside(self, p: np.ndarray, scale: float) -> np.ndarray:
        r = scale * self.radius
        x_lo, x_hi = (scale * b for b in self.x_range)
        z_lo, z_hi = (scale * b for b in self.z_range)
        return (
            (np.linalg.norm(p, axis=1) <= r)
            & (p[:, 0] >= x_lo)
            & (p[:, 0] <= x_hi)
            & (p[:, 2] >= z_lo)
            & (p[:, 2] <= z_hi)
        )

    def sample(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Proposal box is the volume's bounding box, so rejection stays cheap.
        lo = np.array([max(-self.radius, self.x_range[0]), -self.radius, max(-self.radius, self.z_range[0])])
        hi = np.array([min(self.radius, self.x_range[1]), self.radius, min(self.radius, self.z_range[1])])
        la_lo = np.asarray(self.lookat_lo, dtype=np.float64)
        la_hi = np.asarray(self.lookat_hi, dtype=np.float64)

        pos = np.empty((n, 3))
        lookat = np.empty((n, 3))
        filled = 0
        while filled < n:
            m = max(2 * (n - filled), 256)
            p = rng.uniform(lo, hi, size=(m, 3))
            la = rng.uniform(la_lo, la_hi, size=(m, 3))
            keep = self._inside(p, 1.0)
            if self.inner_scale is not None:
                keep &= ~self._inside(p, self.inner_scale)
            d = p - la
            elev = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
            keep &= elev >= self.min_elevation_deg
            p, la = p[keep], la[keep]
            take = min(len(p), n - filled)
            pos[filled : filled + take] = p[:take]
            lookat[filled : filled + take] = la[:take]
            filled += take
        up = np.tile(np.asarray(self.up, dtype=np.float64), (n, 1))
        return pos, lookat, up
