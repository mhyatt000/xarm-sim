"""Gsplat-rendered arena-splat backgrounds for batch (madrona) frames.

Madrona has no native gaussian splats; batch frames come back with black
background pixels (segmentation 0). This module rasterizes the arena's
aligned splat from arbitrary camera poses so RobotEnv can composite a live
backdrop per env at reset — unlike baked plates (make_plates.py), the
background follows per-env jittered camera poses.

View-dependent color is dropped (SH degree 0 only): rotating higher-order
SH under the splat->world alignment isn't worth it for a backdrop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xsim.suite.models.cameras import (  # noqa: F401 — re-exported for scripts
    SplatAsset,
    invert_rigid,
    quat_mul_wxyz,
    rot_from_quat_xyzw,
    rots_from_quat_wxyz,
    viewmats_cv,
)

SH_C0 = 0.28209479177387814

_PLY_DTYPES = {
    b"float": "<f4",
    b"float32": "<f4",
    b"double": "<f8",
    b"float64": "<f8",
    b"int": "<i4",
    b"int32": "<i4",
    b"uint": "<u4",
    b"uint32": "<u4",
    b"short": "<i2",
    b"int16": "<i2",
    b"ushort": "<u2",
    b"uint16": "<u2",
    b"char": "i1",
    b"int8": "i1",
    b"uchar": "u1",
    b"uint8": "u1",
}


def read_ply_vertices(path: Path) -> np.ndarray:
    """Read the vertex element of a binary little-endian PLY as a structured array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        count = 0
        fields: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == b"end_header":
                break
            if tokens[0] == b"format":
                fmt = tokens[1]
            elif tokens[0] == b"element":
                in_vertex = tokens[1] == b"vertex"
                if in_vertex:
                    count = int(tokens[2])
            elif tokens[0] == b"property" and in_vertex:
                if tokens[1] == b"list":
                    raise ValueError("list properties on vertices are not supported")
                fields.append((tokens[-1].decode(), _PLY_DTYPES[tokens[1]]))
        if fmt != b"binary_little_endian":
            raise ValueError(f"only binary_little_endian PLY is supported, got {fmt!r}")
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


def load_world_splat(
    asset: SplatAsset, ply: Path | None = None, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    """PLY gaussians transformed by the asset's solved splat->world alignment."""
    v = read_ply_vertices(ply or Path(asset.uri).expanduser())
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    Ra = rot_from_quat_xyzw(asset.quat_xyzw)
    means = asset.scale * means @ Ra.T + np.asarray(asset.pos)

    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    qx, qy, qz, qw = asset.quat_xyzw
    quats = quat_mul_wxyz(np.array([qw, qx, qy, qz]), quats)

    scales = asset.scale * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    colors = np.clip(
        0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1), 0.0, 1.0
    )
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return dict(means=to(means), quats=to(quats), scales=to(scales),
                opacities=to(opacities), colors=to(colors))


class SplatBackground:
    """Chunked gsplat rasterizer over the arena splat, returning uint8 frames."""

    def __init__(
        self,
        asset: SplatAsset,
        ply: Path | None = None,
        device: str = "cuda",
        chunk: int = 128,
        prune_opacity: float = 0.0,
    ):
        self.device = device
        self.chunk = chunk
        self.splat = load_world_splat(asset, ply, device)
        if prune_opacity > 0:
            keep = self.splat["opacities"] >= prune_opacity
            self.splat = {k: v[keep] for k, v in self.splat.items()}

    def render(self, viewmats: np.ndarray, K: np.ndarray, res: tuple[int, int]) -> np.ndarray:
        """(C, 4, 4) OpenCV world->cam viewmats + (3, 3) or (C, 3, 3) intrinsics
        -> (C, H, W, 3) uint8 frames at ``res`` = (W, H)."""
        import gsplat

        w, h = res
        vm = torch.from_numpy(np.asarray(viewmats)).float().to(self.device)
        Ks = torch.from_numpy(np.broadcast_to(K, (len(vm), 3, 3)).copy()).float().to(self.device)
        out = np.empty((len(vm), h, w, 3), dtype=np.uint8)
        for s in range(0, len(vm), self.chunk):
            e = s + self.chunk
            rgb, _, _ = gsplat.rasterization(
                self.splat["means"], self.splat["quats"], self.splat["scales"],
                self.splat["opacities"], self.splat["colors"], vm[s:e], Ks[s:e], w, h,
            )
            out[s:e] = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        return out
