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
        orthogonal to D at center mark, there will be diameter=8cm circle.
        make elipsoid with this circle and 2cm radius along D. the lookat
        area is oblong . begin with a ellipse centered at TCP . diameter is
        8cm colinear to A, and radius along C equal to cam_center-EEF
        distance. take the half of this ellipse closest to cam_center, and
        union with 8cm circle at TCP

    ``apex`` is EEF and ``axis`` is D. The committed physical bracket lies
    OUTSIDE this ellipsoid (2.5 cm behind the center along D vs the 2 cm
    semi-axis) — the sampled space deliberately does not contain it. The
    lookat region is planar (the A-C plane through ``lookat_center``); the
    circle sits inside the full ellipse, so the union is the ellipse's
    toward-camera half plus the circle's far half."""

    apex: tuple[float, float, float]
    axis: tuple[float, float, float]  # centerline (line D), need not be unit
    center_r: float = 0.11  # cam pos center along D; also the lookat ellipse C semi-axis
    pos_r_across: float = 0.04  # ellipsoid semi-axis orthogonal to D (8 cm circle)
    pos_r_along: float = 0.02  # ellipsoid semi-axis along D
    lookat_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lookat_across: tuple[float, float, float] = (0.0, 1.0, 0.0)  # line A direction
    lookat_radius: float = 0.04

    def sample(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.asarray(self.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        u = np.array([0.0, 0.0, 1.0])
        u = u - (u @ axis) * axis
        if np.linalg.norm(u) < 1e-9:  # axis parallel to z: any perpendicular works
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        # uniform in the ellipsoid: unit ball scaled per semi-axis
        b = rng.normal(size=(n, 3))
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        b *= np.cbrt(rng.uniform(size=(n, 1)))
        center = np.asarray(self.apex) + self.center_r * axis
        pos = center + np.outer(self.pos_r_along * b[:, 0], axis) + self.pos_r_across * (
            np.outer(b[:, 1], u) + np.outer(b[:, 2], v)
        )

        across = np.asarray(self.lookat_across, dtype=np.float64)
        across = across - (across @ axis) * axis
        across /= np.linalg.norm(across)
        rl, a = self.lookat_radius, self.center_r
        s = np.empty(n)
        t = np.empty(n)
        filled = 0
        while filled < n:
            m = max(2 * (n - filled), 256)
            u = rng.uniform([-rl, -rl], [a, rl], size=(m, 2))
            keep = np.where(
                u[:, 0] >= 0,
                (u[:, 0] / a) ** 2 + (u[:, 1] / rl) ** 2 <= 1.0,
                u[:, 0] ** 2 + u[:, 1] ** 2 <= rl**2,
            )
            u = u[keep]
            take = min(len(u), n - filled)
            s[filled : filled + take] = u[:take, 0]
            t[filled : filled + take] = u[:take, 1]
            filled += take
        lookat = np.asarray(self.lookat_center) + s[:, None] * axis + t[:, None] * across
        return pos, lookat, np.tile(np.asarray(self.up, dtype=np.float64), (n, 1))


@dataclass(frozen=True, kw_only=True)
class BallLookatSampler(CamSampler):
    """Positions uniform in a solid ball around ``center``, lookats in a box.

    Small-perturbation counterpart of :class:`ShellLookatSampler`: instead of
    covering a workspace-sized shell, the camera stays within ``radius`` of a
    known-good placement. Near-horizontal view rays are rejected the same way.
    """

    center: tuple[float, float, float]
    radius: float
    lookat_lo: tuple[float, float, float]
    lookat_hi: tuple[float, float, float]
    min_elevation_deg: float = 8.0

    def sample(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.asarray(self.center, dtype=np.float64)
        la_lo = np.asarray(self.lookat_lo, dtype=np.float64)
        la_hi = np.asarray(self.lookat_hi, dtype=np.float64)

        pos = np.empty((n, 3))
        lookat = np.empty((n, 3))
        filled = 0
        while filled < n:
            m = max(2 * (n - filled), 256)
            # uniform in the ball: unit direction times cbrt-distributed radius
            b = rng.normal(size=(m, 3))
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            b *= np.cbrt(rng.uniform(size=(m, 1)))
            p = center + self.radius * b
            la = rng.uniform(la_lo, la_hi, size=(m, 3))
            d = p - la
            elev = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
            keep = elev >= self.min_elevation_deg
            p, la = p[keep], la[keep]
            take = min(len(p), n - filled)
            pos[filled : filled + take] = p[:take]
            lookat[filled : filled + take] = la[:take]
            filled += take
        up = np.tile(np.asarray(self.up, dtype=np.float64), (n, 1))
        return pos, lookat, up


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
