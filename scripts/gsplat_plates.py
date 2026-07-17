"""Benchmark gsplat as a background-plate generator for jittered static cams.

Loads the arena's aligned splat, samples n_views camera poses around the
static cam specs (same uniform jitter as BatchConfig cam_pos_noise /
cam_lookat_noise), and rasterizes them in chunks with gsplat, timing
throughput. RGB+ED mode so the depth channel needed for compositing is
priced in. The first len(static cams) views are noise-free spec poses —
compare those frames against assets/plates/<cam>.png to check alignment.

Splat loading, alignment, and camera math live in xsim.suite.renderers.splat_bg
(the env's live-background path, BatchConfig.splat_bg); this script only adds
the pose sampling and the timing loop.

At 2048 envs, gsplat (~0.475 s/step at 320x240 unpruned; ~0.156 s/step with
the default 64x64 / chunk 2048 / prune 0.15) vs madrona (~1.5 it/s) means
regenerating plates every step is a 1.7x -> 1.23x slowdown (serial costs add).

    uv pip install -e ../gsplat
    uv run python scripts/gsplat_plates.py --n-views 2048
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from xsim.suite.models.arenas.table_arena import TableArena
from xsim.suite.renderers.splat_bg import load_world_splat, viewmats_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    ply: Path | None = None  # None -> the arena splat, with its solved alignment
    res: tuple[int, int] = (64, 64)  # (W, H)
    n_views: int = 2048  # plates per step; first len(static cams) are noise-free
    steps: int = 100  # timing loop iterations over the full view set
    cam_pos_noise: float = 0.8  # metres, matches BatchConfig
    cam_lookat_noise: float = 0.03
    seed: int = 0
    chunk: int = 2048  # cameras per rasterization call
    fov_deg: float | None = None  # None -> each spec's own (Logitech calib)
    out: Path = PROJECT_ROOT / "outputs" / "gsplat_plates"
    n_save: int = 6  # sample frames written for eyeballing
    radius_clip: float = 0.0  # px; >0 skips subpixel splats for speed
    render_mode: str = "RGB+ED"  # "RGB" drops the depth channel compositing needs
    packed: bool = True  # gsplat's sparse pipeline (its default)
    inference: bool = True  # torch.inference_mode() around the timing loop
    prune_opacity: float = 0.15  # drop gaussians below this opacity after loading


def main(cfg: Config) -> None:
    import cv2
    import gsplat

    if not torch.cuda.is_available():
        raise SystemExit("gsplat needs CUDA")
    device = "cuda"
    W, H = cfg.res

    arena = TableArena()
    if arena.splat is None and cfg.ply is None:
        raise SystemExit("arena has no splat asset and no --ply given")
    t0 = time.perf_counter()
    splat = load_world_splat(arena.splat, cfg.ply, device)
    n_gauss = splat["means"].shape[0]
    print(f"loaded {n_gauss:,} gaussians in {time.perf_counter() - t0:.1f}s")
    if cfg.prune_opacity > 0:
        keep = splat["opacities"] >= cfg.prune_opacity
        splat = {k: v[keep] for k, v in splat.items()}
        n_gauss = splat["means"].shape[0]
        print(f"pruned to {n_gauss:,} gaussians (opacity >= {cfg.prune_opacity})")

    static = [s for s in arena.cameras if s.attach_link is None]
    rng = np.random.default_rng(cfg.seed)
    poses, lookats, ups, Ks, names = [], [], [], [], []
    for i in range(cfg.n_views):
        spec = static[i % len(static)]
        pos, lookat = np.asarray(spec.pos), np.asarray(spec.lookat)
        if i >= len(static):  # first pass over specs stays noise-free for eyeballing
            pos = pos + rng.uniform(-cfg.cam_pos_noise, cfg.cam_pos_noise, 3)
            lookat = lookat + rng.uniform(-cfg.cam_lookat_noise, cfg.cam_lookat_noise, 3)
        poses.append(pos)
        lookats.append(lookat)
        ups.append(spec.up)
        fov = cfg.fov_deg or spec.fov_deg
        fy = (H / 2.0) / np.tan(np.radians(fov) / 2.0)
        Ks.append(np.array([[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]))
        names.append(f"{i:04d}_{spec.name}" + ("" if i >= len(static) else "_exact"))
    vm_np = viewmats_cv(np.stack(poses), np.stack(lookats), np.stack(ups))
    viewmats = torch.from_numpy(vm_np).float().to(device)
    Ks = torch.from_numpy(np.stack(Ks)).float().to(device)

    def render(vm: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        rgbd, _, _ = gsplat.rasterization(
            splat["means"], splat["quats"], splat["scales"], splat["opacities"],
            splat["colors"], vm, K, W, H,
            render_mode=cfg.render_mode, radius_clip=cfg.radius_clip,
            packed=cfg.packed,
        )
        return rgbd  # (C, H, W, 3|4), last channel expected depth in RGB+ED

    # warmup: first call pays kernel compile / autotune
    _ = render(viewmats[:1], Ks[:1])
    torch.cuda.synchronize()

    frames = []
    t0 = time.perf_counter()
    with torch.inference_mode() if cfg.inference else nullcontext():
        for step in range(cfg.steps):
            for s in range(0, cfg.n_views, cfg.chunk):
                rgbd = render(viewmats[s : s + cfg.chunk], Ks[s : s + cfg.chunk])
                if step == 0 and s == 0:
                    frames = rgbd[: cfg.n_save].clamp(0, 1).cpu()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total = cfg.steps * cfg.n_views
    print(
        f"{cfg.steps} steps x {cfg.n_views} plates @ {W}x{H} (chunk {cfg.chunk}, "
        f"{cfg.render_mode}, packed={cfg.packed}, inference={cfg.inference}, "
        f"radius_clip={cfg.radius_clip}): "
        f"{dt:.2f}s total, {dt / cfg.steps:.3f} s/step, "
        f"{1e3 * dt / total:.3f} ms/plate, {total / dt:.0f} plates/s"
    )
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    cfg.out.mkdir(parents=True, exist_ok=True)
    for name, rgbd in zip(names, frames):
        rgb = (rgbd[..., :3].numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(cfg.out / f"{name}.png"), rgb[..., ::-1])
    print(f"wrote {len(frames)} sample frames -> {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
