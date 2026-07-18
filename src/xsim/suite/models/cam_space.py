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
class MountSampler(CamSampler):
    """Randomized rigid mount, sampled in the attach link's frame: positions
    uniform in a spherical shell about ``apex`` restricted to a cone around
    ``axis``, lookats uniform in a ball. The spec, verbatim (derivation and
    plot: scripts/cam_wrist.py):

        consider robot points eef gripper finger tip right and left. the
        finger tip, NOT the knuckles. consider also TCP. line A is the line
        from right to left tip. line B is from EEF to TCP. C intersects A,B
        and is perpendicular to both (orthogonal). D is colinear to C but
        intersects EEF. 11cm along D (away from EEF) is the cam pos center.
        create a cone by rotating a line 30 degrees from D (centerline) but
        still intersecting with D at EEF. the space the camera can occupy is
        from 10cm to 12cm within that cone. the lookat is sampled from sphere
        8cm diameter centered at TCP.

    ``apex`` is EEF and ``axis`` is D; the shell floor is dropped to 9 cm so
    the sampled space contains the committed physical bracket (r = 9.6 cm,
    27 deg off-axis — inside the cone but under the verbatim 10 cm floor)."""

    apex: tuple[float, float, float]
    axis: tuple[float, float, float]  # cone centerline (line D), need not be unit
    half_angle_deg: float = 30.0
    r_range: tuple[float, float] = (0.09, 0.12)
    lookat_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lookat_radius: float = 0.04

    def sample(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.asarray(self.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        cos_t = rng.uniform(np.cos(np.radians(self.half_angle_deg)), 1.0, n)
        phi = rng.uniform(0.0, 2 * np.pi, n)
        sin_t = np.sqrt(1.0 - cos_t**2)
        u = np.array([0.0, 0.0, 1.0])
        u = u - (u @ axis) * axis
        if np.linalg.norm(u) < 1e-9:  # axis parallel to z: any perpendicular works
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        dirs = (
            cos_t[:, None] * axis
            + sin_t[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
        )
        r_lo, r_hi = self.r_range
        r = np.cbrt(rng.uniform(r_lo**3, r_hi**3, n))  # uniform in shell volume
        pos = np.asarray(self.apex) + r[:, None] * dirs

        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        lookat = np.asarray(self.lookat_center) + self.lookat_radius * np.cbrt(
            rng.uniform(size=(n, 1))
        ) * d
        return pos, lookat, np.tile(np.asarray(self.up, dtype=np.float64), (n, 1))


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
