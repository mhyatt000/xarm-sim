"""Diagnostic: LiftExpertPolicy success rate over the table.

Two modes (share one batched-rollout harness; one grid cell / sample per env):

* grid (default): whole-table (x, y) heatmap. For K repeats with random cube
  yaws, place every env's cube at its grid (x, y), roll out the scripted
  expert, latch per-cell success, average over repeats -> rate in [0, 1].
  Also reports the reachable-clip ceiling (drop far-reach + base keep-out).

* prod (--prod): the CLEAN TEACHER CEILING on the production spawn. Sample
  cube (x, y) uniformly from the training/eval distribution with random yaw,
  one independent sample per env, roll out once -> a single SR with 95% CI.

CPU is fine (this env's torch/driver mismatch forces a CPU fallback even with
CUDA_VISIBLE_DEVICES=0; the physics still runs, just slower).

Run:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diag_expert_spawn_heatmap.py
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diag_expert_spawn_heatmap.py --prod
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# tuned spawn box the expert was swept on (sanity-check region)
BOX_X = (0.35, 0.58)
BOX_Y = (-0.15, 0.15)
# production training/eval spawn distribution
PROD_X = (0.20, 0.40)
PROD_Y = (-0.288, 0.288)
# reachable-clip: cells kept must NOT be far-reach or base keep-out
FAR_X = 0.60           # drop x > FAR_X
BASE_X = 0.15          # base keep-out: x < BASE_X AND |y| < BASE_Y
BASE_Y = 0.12


def build_env(n_envs, horizon, control_freq, noslip_iterations):
    import genesis as gs

    from xsim.suite import make
    from xsim.suite.policies import LiftExpertPolicy

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7", camera_names=[], camera_res=(640, 480),
        render_backend="raster", n_envs=n_envs, horizon=horizon,
        control_freq=control_freq, noslip_iterations=noslip_iterations,
        randomize_cameras=False,
    )
    base = env  # make() returns the raw (unwrapped) Lift env
    # keep succeeded envs' cubes put: no termination-driven reset mid-episode
    base._check_terminated = lambda: np.zeros(base.n_envs, dtype=bool)
    expert = LiftExpertPolicy(base, cartesian=False)
    return env, base, expert


def rollout(env, base, expert, xs, ys, yaws, steps, tag=""):
    """Place cubes at (xs, ys, yaws), roll the expert, return latched success."""
    z_rest = base.arena.top_z + base.cube.top_offset
    env.reset()
    base.cube.set_pose(xs, ys, z_rest, yaws, envs_idx=None)
    base._success_hold[:] = 0
    expert.reset()
    ever = np.zeros(base.n_envs, dtype=bool)
    for t in range(steps):
        action = expert.act()
        env.step(action)
        ever |= base._check_success()
        if ever.all():
            print(f"  {tag}: all envs succeeded by step {t}")
            break
    return ever


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def run_prod(args):
    n = args.prod_envs
    env, base, expert = build_env(
        n, args.horizon, args.control_freq, args.noslip_iterations
    )
    rng = np.random.default_rng(args.seed)
    xs = rng.uniform(*PROD_X, size=n)
    ys = rng.uniform(*PROD_Y, size=n)
    yaws = rng.uniform(-np.pi, np.pi, size=n)
    print(f"PROD ceiling: n={n} envs, x~U{PROD_X}, y~U{PROD_Y}, random yaw, "
          f"{args.steps} steps")
    ever = rollout(env, base, expert, xs, ys, yaws, args.steps, tag="prod")
    k = int(ever.sum())
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    lo, hi = wilson_ci(k, n)
    print("\n===== CLEAN TEACHER CEILING (production spawn) =====")
    print(f"successes      : {k} / {n}")
    print(f"success rate   : {p:.4f}")
    print(f"95% CI (normal): [{p - 1.96 * se:.4f}, {p + 1.96 * se:.4f}]")
    print(f"95% CI (Wilson): [{lo:.4f}, {hi:.4f}]")


def run_grid(args):
    n_envs = args.nx * args.ny
    env, base, expert = build_env(
        n_envs, args.horizon, args.control_freq, args.noslip_iterations
    )

    cx, cy = base.arena.center_xy
    sx, sy = base.arena.size_xy
    m = args.margin
    x_lo, x_hi = cx - sx / 2 + m, cx + sx / 2 - m
    y_lo, y_hi = cy - sy / 2 + m, cy + sy / 2 - m
    xs_axis = np.linspace(x_lo, x_hi, args.nx)
    ys_axis = np.linspace(y_lo, y_hi, args.ny)
    XX, YY = np.meshgrid(xs_axis, ys_axis)  # (ny, nx)
    cell_x = XX.ravel()
    cell_y = YY.ravel()

    print(f"table: center={cx:.4f},{cy:.4f} size={sx:.4f}x{sy:.4f} "
          f"top_z={base.arena.top_z:.4f}")
    print(f"grid x in [{x_lo:.4f},{x_hi:.4f}] ({args.nx})  "
          f"y in [{y_lo:.4f},{y_hi:.4f}] ({args.ny})")
    print(f"n_envs={n_envs}  repeats={args.repeats}  steps/rollout={args.steps}")

    rng = np.random.default_rng(args.seed)
    per_repeat = np.zeros((args.repeats, n_envs), dtype=np.float64)
    for k in range(args.repeats):
        yaws = rng.uniform(-np.pi, np.pi, size=n_envs)
        ever = rollout(env, base, expert, cell_x, cell_y, yaws, args.steps,
                       tag=f"repeat {k}")
        per_repeat[k] = ever.astype(np.float64)
        print(f"  repeat {k}: mean SR so far = {ever.mean():.4f}")

    success_rate = per_repeat.mean(axis=0)
    grid = success_rate.reshape(args.ny, args.nx)

    in_box = ((cell_x >= BOX_X[0]) & (cell_x <= BOX_X[1])
              & (cell_y >= BOX_Y[0]) & (cell_y <= BOX_Y[1]))
    # reachable-clip mask: keep cells that are neither far-reach nor base keep-out
    far = cell_x > FAR_X
    base_ko = (cell_x < BASE_X) & (np.abs(cell_y) < BASE_Y)
    reachable = ~far & ~base_ko

    overall = success_rate.mean()
    inside = success_rate[in_box].mean() if in_box.any() else float("nan")
    outside = success_rate[~in_box].mean() if (~in_box).any() else float("nan")
    clip = success_rate[reachable].mean() if reachable.any() else float("nan")

    order = np.argsort(success_rate, kind="stable")
    worst = order[:10]

    print("\n===== GRID RESULTS =====")
    print(f"overall mean SR       : {overall:.4f}  (n={n_envs} cells)")
    print(f"inside-box SR         : {inside:.4f}  (n={int(in_box.sum())})")
    print(f"outside-box SR        : {outside:.4f}  (n={int((~in_box).sum())})")
    print(f"reachable-clip SR     : {clip:.4f}  (n={int(reachable.sum())}; "
          f"dropped x>{FAR_X} and base keep-out x<{BASE_X}&|y|<{BASE_Y})")
    print("worst 10 cells (x, y, SR):")
    for i in worst:
        print(f"  x={cell_x[i]:+.4f}  y={cell_y[i]:+.4f}  SR={success_rate[i]:.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "expert_spawn_heatmap.csv"
    np.savetxt(csv_path, np.column_stack([cell_x, cell_y, success_rate]),
               delimiter=",", header="x,y,success_rate", comments="", fmt="%.6f")
    print(f"\nwrote {csv_path.resolve()}")

    _plot(args, grid, xs_axis, ys_axis, x_lo, x_hi, y_lo, y_hi, cx, cy, sx, sy,
          overall, inside, outside, clip)


def _plot(args, grid, xs_axis, ys_axis, x_lo, x_hi, y_lo, y_hi,
          cx, cy, sx, sy, overall, inside, outside, clip):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dx = (xs_axis[1] - xs_axis[0]) / 2 if args.nx > 1 else 0.02
    dy = (ys_axis[1] - ys_axis[0]) / 2 if args.ny > 1 else 0.02
    extent = [x_lo - dx, x_hi + dx, y_lo - dy, y_hi + dy]

    fig, axp = plt.subplots(figsize=(10, 7))
    im = axp.imshow(grid, origin="lower", extent=extent, aspect="equal",
                    cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=axp, fraction=0.035, pad=0.02)
    cbar.set_label("success rate")

    axp.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False,
                            edgecolor="white", lw=1.5, ls="--", label="table"))
    axp.add_patch(Rectangle((BOX_X[0], BOX_Y[0]), BOX_X[1] - BOX_X[0],
                            BOX_Y[1] - BOX_Y[0], fill=False, edgecolor="red",
                            lw=2.0, label="tuned spawn box"))
    axp.plot(0.0, 0.0, marker="*", ms=18, color="orange", mec="black",
             label="robot base")

    axp.set_xlabel("x (m)")
    axp.set_ylabel("y (m)")
    axp.set_title(
        "LiftExpertPolicy success rate vs cube spawn (x,y)\n"
        f"overall SR={overall:.3f}  in-box={inside:.3f}  out-box={outside:.3f}  "
        f"reach-clip={clip:.3f}  (K={args.repeats}, {args.nx}x{args.ny} grid)"
    )
    axp.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.tight_layout()

    png_path = args.outdir / "expert_spawn_heatmap.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path.resolve()}")

    if args.extra_copy is not None:
        import shutil

        args.extra_copy.mkdir(parents=True, exist_ok=True)
        dst = args.extra_copy / "expert_spawn_heatmap.png"
        shutil.copy(png_path, dst)
        print(f"copied PNG -> {dst.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true",
                    help="clean-teacher-ceiling on the production spawn (with CI)")
    ap.add_argument("--prod-envs", type=int, default=2048)
    ap.add_argument("--nx", type=int, default=24, help="grid cells along x")
    ap.add_argument("--ny", type=int, default=18, help="grid cells along y")
    ap.add_argument("--repeats", type=int, default=5, help="random-yaw repeats (K)")
    ap.add_argument("--steps", type=int, default=300, help="control steps per rollout")
    ap.add_argument("--margin", type=float, default=0.03, help="table-edge margin, m")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--control-freq", type=float, default=30.0)
    ap.add_argument("--noslip-iterations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/diag"))
    ap.add_argument("--extra-copy", type=Path, default=None,
                    help="optional second dir to copy the PNG into")
    args = ap.parse_args()

    if args.prod:
        run_prod(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
