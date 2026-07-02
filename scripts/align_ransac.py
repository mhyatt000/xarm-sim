"""RANSAC alignment of the lab scan to the robot-base frame.

Works from the ZED fused point cloud (`/data/store/fused_point_cloud.ply`) — metric,
real surface points — instead of interpreting the gaussian splat, which reconstructs the
dark tabletop see-through and explodes view-dependent colors at in-scene camera angles.

Stage 1 (this file, ``--stage 1``): robustly extract the scene geometry —
gravity + floor plane (RANSAC on surface normals + plane offset), the tabletop plane and
its rectangle, the robot column, and the two camera-pole tips — and write a labeled
top-down + elevation image for **human verification** (checkpoint CP1) plus a
``landmarks.json`` for stage 2.

    uv run python scripts/align_ransac.py --stage 1
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "align_ransac"

# real tabletop rectangle measured in ROBOT coordinates by inverse-perspective-mapping
# the calibrated cap.npz photos onto z=0 (see plan); used only to *identify* the table
# among furniture in stage 1 — the solve happens in stage 2
IPM_RECT_DIMS = (0.93, 0.62)
# calibrated camera heights above the tabletop (robot frame z)
CAM_HEIGHTS = {"low": 0.235, "side": 0.92}


@dataclass
class Cfg:
    stage: int = 1
    src: Path = Path("/data/store/fused_point_cloud.ply")
    out_dir: Path = OUT_DIR
    voxel: float = 0.01          # subsample for speed (m)


# ---------------------------------------------------------------- loading

def load_fused(src: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (P, N, C) float arrays; caches the parsed ASCII ply as npz."""
    cache = cache_dir / "fused_cache.npz"
    if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
        d = np.load(cache)
        return d["P"], d["N"], d["C"]
    rows = np.loadtxt(src, skiprows=14)
    P, N, C = rows[:, :3], rows[:, 3:6], rows[:, 6:9] / 255.0
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, P=P.astype(np.float32), N=N.astype(np.float32), C=C.astype(np.float32))
    return P, N, C


def voxel_downsample(P, N, C, voxel):
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[idx], N[idx], C[idx]


# ---------------------------------------------------------------- stage 1 geometry

def _normal_modes(N: np.ndarray, k=3, iters=500, tol_deg=6.0, rng=None) -> list[np.ndarray]:
    """Top-k dominant (unsigned) normal directions via greedy RANSAC modes."""
    rng = rng or np.random.default_rng(0)
    cos_tol = np.cos(np.radians(tol_deg))
    sub = N[rng.choice(len(N), min(len(N), 60000), replace=False)]
    modes = []
    for _ in range(k):
        best_cnt, best = 0, None
        for _ in range(iters):
            cand = sub[rng.integers(len(sub))]
            cnt = (np.abs(sub @ cand) > cos_tol).sum()
            if cnt > best_cnt:
                best_cnt, best = cnt, cand
        aligned = sub[np.abs(sub @ best) > cos_tol]
        signs = np.sign(aligned @ best)
        d = (aligned * signs[:, None]).mean(axis=0)
        d /= np.linalg.norm(d)
        modes.append(d)
        sub = sub[np.abs(sub @ d) < cos_tol]  # remove this mode, find the next
        if len(sub) < 1000:
            break
    return modes


def ransac_gravity(P: np.ndarray, N: np.ndarray, rng=None) -> np.ndarray:
    """Gravity = the normal mode whose two strongest parallel planes are a ceiling
    height apart (2.1-3.0 m). Wall-normal modes pair at room width/length instead."""
    best = None
    for d in _normal_modes(N, rng=rng):
        hs, _ = plane_offsets(P, N, d)
        lo, hi = np.percentile(hs, [0.5, 99.5])
        if hi - lo < 1.5:
            continue
        o1, n1 = strongest_offset(hs, lo, (lo + hi) / 2)
        o2, n2 = strongest_offset(hs, (lo + hi) / 2, hi)
        sep = abs(o2 - o1)
        score = min(n1, n2)
        print(f"  normal mode {d.round(3)}: planes at {o1:.2f}/{o2:.2f} (sep {sep:.2f} m, n {n1}/{n2})")
        if 2.1 < sep < 3.0 and (best is None or score > best[0]):
            best = (score, d)
    if best is None:
        raise SystemExit("no normal mode with ceiling-height plane separation found")
    return best[1]


def plane_offsets(P, N, g, cos_tol=0.85):
    """Heights (P·g) of points whose normal is along ±g (horizontal surfaces)."""
    m = np.abs(N @ g) > cos_tol
    return (P[m] @ g), m


def strongest_offset(heights, lo, hi, bin_w=0.01):
    hist, edges = np.histogram(heights, bins=int((hi - lo) / bin_w), range=(lo, hi))
    j = np.argmax(hist)
    c = (edges[j] + edges[j + 1]) / 2
    sel = np.abs(heights - c) < bin_w * 1.5
    return float(np.median(heights[sel])), int(hist[j])


def find_clusters_2d(xy, cell=0.03, min_pts=40):
    """Connected-component clusters on a 2D occupancy grid; returns list of point-index arrays."""
    if len(xy) == 0:
        return []
    lo = xy.min(axis=0) - cell
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 3
    grid = np.zeros(shape, np.uint8)
    grid[ij[:, 0], ij[:, 1]] = 1
    n_lbl, lbl = cv2.connectedComponents(grid, connectivity=8)
    pt_lbl = lbl[ij[:, 0], ij[:, 1]]
    clusters = []
    for k in range(1, n_lbl):
        idx = np.where(pt_lbl == k)[0]
        if len(idx) >= min_pts:
            clusters.append(idx)
    return sorted(clusters, key=len, reverse=True)


def stage1(cfg: Cfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    P, N, C = load_fused(cfg.src, cfg.out_dir)
    P, N, C = voxel_downsample(P, N, C, cfg.voxel)
    print(f"{len(P)} points after {cfg.voxel:.0e} voxel downsample")

    # gravity + floor
    g = ransac_gravity(P, N)
    h_all = P @ g
    # floor = strongest horizontal plane in the lower half of the height range; make g point UP
    heights, horiz_mask = plane_offsets(P, N, g)
    lo, hi = np.percentile(h_all, [1, 99])
    f1, n1 = strongest_offset(heights, lo, (lo + hi) / 2)
    f2, n2 = strongest_offset(heights, (lo + hi) / 2, hi)
    floor_h = f1 if n1 >= n2 else f2
    # ceiling should be ~2.3-2.8 above the floor along +up; flip g if needed
    if (f2 if floor_h == f1 else f1) < floor_h:
        g, h_all, heights, floor_h = -g, -h_all, -heights, -floor_h
    print(f"gravity(up) = {g.round(4)}, floor at g·p = {floor_h:.3f}")

    # basis for horizontal coordinates
    e1 = np.cross(g, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(g, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    uv = np.stack([P @ e1, P @ e2], axis=1)
    h = h_all - floor_h  # height above floor, up-positive

    # tabletop: upward-facing horizontal surfaces 0.5-0.95 above the floor
    up_face = (N @ g) > 0.8
    band = up_face & (h > 0.50) & (h < 0.95)
    cand_planes = []
    bh = h[band]
    hist, edges = np.histogram(bh, bins=45, range=(0.5, 0.95))
    for j in np.argsort(hist)[-6:]:
        if hist[j] < 100:
            continue
        c = (edges[j] + edges[j + 1]) / 2
        cand_planes.append(c)
    table = None
    for hc in sorted(set(np.round(cand_planes, 2))):
        sel = band & (np.abs(h - hc) < 0.02)
        for idx in find_clusters_2d(uv[sel], min_pts=150):
            pts = uv[sel][idx].astype(np.float32)
            rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,l),angle)
            dims = sorted(rect[1], reverse=True)
            score = abs(dims[0] - IPM_RECT_DIMS[0]) + abs(dims[1] - IPM_RECT_DIMS[1])
            if table is None or score < table["score"]:
                table = {"score": score, "h": float(np.median(h[sel][idx])), "rect": rect,
                         "dims": dims, "n": len(idx)}
    r = table["rect"]
    print(f"table candidate: h={table['h']:.3f} above floor, rect dims {table['dims'][0]:.2f}x{table['dims'][1]:.2f} "
          f"(target {IPM_RECT_DIMS[0]}x{IPM_RECT_DIMS[1]}), center uv=({r[0][0]:.2f},{r[0][1]:.2f}), n={table['n']}")

    # robot column: points rising above the table inside its footprint
    box = cv2.boxPoints(r)  # 4x2
    inside = cv2.pointPolygonTest
    ht = table["h"]
    above = (h > ht + 0.06) & (h < ht + 0.95)
    in_rect = np.array([cv2.pointPolygonTest(box, (float(u), float(v)), True) > -0.05
                        for u, v in uv[above]])
    robot_uv, robot_h = uv[above][in_rect], h[above][in_rect]
    clusters = find_clusters_2d(robot_uv, min_pts=100)
    robot = None
    if clusters:
        idx = clusters[0]
        robot = {"uv": robot_uv[idx].mean(axis=0).tolist(),
                 "top": float(np.percentile(robot_h[idx], 98)), "n": len(idx)}
        print(f"robot candidate: uv=({robot['uv'][0]:.2f},{robot['uv'][1]:.2f}), "
              f"top {robot['top']-ht:.2f} above table, n={robot['n']}")

    # pole candidates: thin tall clusters near (but maybe off) the table
    near = (h > ht + 0.03) & (h < ht + 1.15)
    d_to_rect = np.array([cv2.pointPolygonTest(box, (float(u), float(v)), True) for u, v in uv[near]])
    ring = (d_to_rect > -0.75)  # within 0.75 m of the rectangle (inside or out)
    if robot is not None:
        away = np.linalg.norm(uv[near] - np.array(robot["uv"]), axis=1) > 0.25
        ring = ring & away
    poles = []
    for idx in find_clusters_2d(uv[near][ring], cell=0.025, min_pts=25):
        pu = uv[near][ring][idx]
        ph = h[near][ring][idx]
        extent = pu.max(axis=0) - pu.min(axis=0)
        if max(extent) < 0.30 and (ph.max() - ph.min()) > 0.10:
            poles.append({"uv": pu.mean(axis=0).tolist(),
                          "top_above_table": float(np.percentile(ph, 99) - ht), "n": len(idx)})
    poles = sorted(poles, key=lambda p: -p["n"])[:6]
    for i, p in enumerate(poles):
        print(f"pole cand {i}: uv=({p['uv'][0]:.2f},{p['uv'][1]:.2f}), top {p['top_above_table']:.2f} above table, n={p['n']}")

    # ------------------------------------------------------------ CP1 artifact
    res = 0.005
    u0, v0 = uv.min(axis=0) - 0.2
    u1, v1 = uv.max(axis=0) + 0.2
    W, H = int((u1 - u0) / res), int((v1 - v0) / res)
    img = np.zeros((H, W, 3), np.float32)
    zb = np.full((H, W), -10.0, np.float32)
    px = ((uv[:, 0] - u0) / res).astype(int)
    py = ((uv[:, 1] - v0) / res).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (h < 2.0)
    order = np.argsort(h[ok])
    xo, yo, co = px[ok][order], py[ok][order], C[ok][order]
    img[yo, xo] = co  # higher points drawn last (painter's algo by height)
    img = (img * 255).astype(np.uint8)

    def to_px(u, v):
        return int((u - u0) / res), int((v - v0) / res)

    # metric grid every 0.5 m
    for gu in np.arange(np.ceil(u0 * 2) / 2, u1, 0.5):
        x, _ = to_px(gu, 0)
        cv2.line(img, (x, 0), (x, H - 1), (60, 60, 60), 1)
    for gv in np.arange(np.ceil(v0 * 2) / 2, v1, 0.5):
        _, y = to_px(0, gv)
        cv2.line(img, (0, y), (W - 1, y), (60, 60, 60), 1)

    box_px = np.array([to_px(u, v) for u, v in box])
    cv2.polylines(img, [box_px], True, (0, 255, 0), 2)
    cv2.putText(img, "TABLE?", tuple(box_px.min(axis=0) - [0, 8]), 0, 0.7, (0, 255, 0), 2)
    if robot is not None:
        cv2.circle(img, to_px(*robot["uv"]), 12, (0, 0, 255), 2)
        cv2.putText(img, "ROBOT?", to_px(robot["uv"][0] + 0.06, robot["uv"][1]), 0, 0.7, (0, 0, 255), 2)
    for i, p in enumerate(poles):
        cv2.circle(img, to_px(*p["uv"]), 8, (0, 255, 255), 2)
        cv2.putText(img, f"P{i}:{p['top_above_table']:.2f}", to_px(p["uv"][0] + 0.04, p["uv"][1] + 0.04),
                    0, 0.5, (0, 255, 255), 1)
    cv2.putText(img, "grid 0.5 m | green=table? red=robot? yellow=pole cands (top height above table)",
                (12, H - 12), 0, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(cfg.out_dir / "cp1_topdown.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # elevation view along e2 (u = e1, vertical = h) for the table area, same markers
    sel = (np.abs(uv[:, 1] - r[0][1]) < 1.2) & (h > -0.05) & (h < 2.2)
    eu, eh, ec = uv[sel][:, 0], h[sel], C[sel]
    EW = int((u1 - u0) / res)
    EH = int(2.3 / res)
    ev = np.zeros((EH, EW, 3), np.float32)
    exi = ((eu - u0) / res).astype(int)
    eyi = ((2.2 - eh) / res).astype(int)
    ok = (exi >= 0) & (exi < EW) & (eyi >= 0) & (eyi < EH)
    ev[eyi[ok], exi[ok]] = ec[ok]
    ev = (ev * 255).astype(np.uint8)
    ty = int((2.2 - ht) / res)
    cv2.line(ev, (0, ty), (EW - 1, ty), (0, 255, 0), 1)
    fy = int(2.2 / res)
    cv2.line(ev, (0, fy - 1), (EW - 1, fy - 1), (255, 128, 0), 1)
    cv2.putText(ev, f"green = table plane ({ht:.2f} above floor) | orange = floor", (12, 24), 0, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(cfg.out_dir / "cp1_elevation.png"), cv2.cvtColor(ev, cv2.COLOR_RGB2BGR))

    landmarks = {
        "gravity_up": g.tolist(), "floor_offset": floor_h, "basis_e1": e1.tolist(), "basis_e2": e2.tolist(),
        "table_h_above_floor": table["h"], "table_rect_center_uv": [float(r[0][0]), float(r[0][1])],
        "table_rect_dims": [float(d) for m in [0] for d in table["dims"]], "table_rect_angle_deg": float(r[2]),
        "table_box_uv": box.tolist(), "robot": robot, "poles": poles,
    }
    (cfg.out_dir / "landmarks.json").write_text(json.dumps(landmarks, indent=2))
    print(f"\nwrote {cfg.out_dir}/cp1_topdown.png, cp1_elevation.png, landmarks.json")


def main(cfg: Cfg) -> None:
    if cfg.stage == 1:
        stage1(cfg)
    else:
        raise SystemExit(f"stage {cfg.stage} not implemented yet (stage 2/3 come after CP1 review)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
