"""Diagnostic: is the Lift SUCCESS GATE discarding near-successful lifts?

The current gate (src/xsim/suite/environments/manipulation/lift.py
``_raw_success`` + ``_check_success``) latches success only when, for
``success_hold_ticks`` CONSECUTIVE control steps:

  high  : cube_z            >  arena.top_z + lift_height        (lift_height=0.05)
  slow  : ||cube_vel - ee_vel|| < success_max_speed             (0.10 m/s)
  near  : ||cube_xy - ee_xy||    < success_eef_xy_radius        (0.08 m)
  touch : cube-robot contact                                    (required)

If this gate is too strict, a policy that lifts and holds the cube but, say,
carries it at 0.12 m/s relative to the hand reads as a FAILURE and the reported
success rate understates the true skill. This script rolls the near-oracle
expert, logs the raw per-step signals, and RE-SCORES every episode under a grid
of looser gates to see how many FAILs flip to SUCCESS.

Grid: rel_vel_max in {0.05, 0.10, 0.20, inf}, eef_xy_radius in {0.05, 0.08,
0.12}, hold_ticks in {1, 3, 5}, contact_required in {True, False}. (lift_height
stays fixed — it defines "a lift", not the tightness of the catch.)

Outputs (outputs/diag/): gate_sweep.csv, gate_sweep.png, gate_sweep.txt.

Tiny CPU smoke (safe on a shared box, no GPU):
    uv run python scripts/diag_gate_sweep.py --smoke

Full run later on a free GPU (pure physics, no cameras):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diag_gate_sweep.py \
        --backend gpu --n-envs 512 --n-episodes 4 --spawn production
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SPAWNS = {
    "production": dict(
        x_range=(0.20, 0.40),
        y_range=(-0.288, 0.288),
        init_tcp_box=((0.10, 0.40), (-0.3048, 0.3048), (-0.01, 0.30)),
    ),
    "box": dict(x_range=(0.35, 0.58), y_range=(-0.15, 0.15), init_tcp_box=None),
    "table": dict(x_range=(0.30, 0.58), y_range=(-0.288, 0.288), init_tcp_box=None),
}

# gate grid (current gate = 0.10 / 0.08 / hold 1 / contact True)
REL_VELS = [0.05, 0.10, 0.20, np.inf]
XY_RADII = [0.05, 0.08, 0.12]
HOLDS = [1, 3, 5]
CONTACTS = [True, False]
CUR = dict(rel_vel=0.10, xy_radius=0.08, hold=1, contact=True)


def any_run_ge(raw: np.ndarray, hold: int) -> np.ndarray:
    """Per-episode: did the instantaneous condition hold for >= `hold`
    CONSECUTIVE steps? Replicates lift.py's reset-on-False hold counter.
    raw: (E, T) bool -> (E,) bool."""
    E, T = raw.shape
    cnt = np.zeros(E, dtype=np.int64)
    hit = np.zeros(E, dtype=bool)
    for t in range(T):
        cnt = np.where(raw[:, t], cnt + 1, 0)
        hit |= cnt >= hold
    return hit


def collect(env, expert, steps, n_episodes, seed):
    """Roll the expert; per control step per env log cube/ee pos+vel, contact,
    and the env's own _check_success() latch (ground truth for the current gate).
    Each env runs one full episode per batch (no auto-reset). Returns stacked
    arrays over (E = n_envs * n_episodes, T = steps)."""
    n = env.n_envs
    cube_p, cube_v, ee_p, ee_v, touch, gt = [], [], [], [], [], []
    for b in range(n_episodes):
        env.reset(seed=seed + b)
        env._success_hold[:] = 0
        expert.reset()
        cp, cv, ep, ev, tc, g = [], [], [], [], [], []
        for _ in range(steps):
            a = np.asarray(expert.act(), dtype=np.float32)
            env.step(a)
            cp.append(np.asarray(env.cube.get_pos(), dtype=np.float64))
            cv.append(np.asarray(env.cube.get_vel(), dtype=np.float64))
            ep.append(np.asarray(env.robots[0].ee_pos, dtype=np.float64))
            ev.append(np.asarray(env.robots[0].ee_vel, dtype=np.float64))
            tc.append(np.asarray(env._robot_contact(), dtype=bool))
            g.append(np.asarray(env._check_success(), dtype=bool))
        # (T, n, ...) -> (n, T, ...)
        cube_p.append(np.transpose(np.stack(cp), (1, 0, 2)))
        cube_v.append(np.transpose(np.stack(cv), (1, 0, 2)))
        ee_p.append(np.transpose(np.stack(ep), (1, 0, 2)))
        ee_v.append(np.transpose(np.stack(ev), (1, 0, 2)))
        touch.append(np.transpose(np.stack(tc), (1, 0)))
        gt.append(np.transpose(np.stack(g), (1, 0)))
    return (np.concatenate(cube_p), np.concatenate(cube_v),
            np.concatenate(ee_p), np.concatenate(ee_v),
            np.concatenate(touch), np.concatenate(gt))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spawn", choices=sorted(SPAWNS), default="production")
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--n-episodes", type=int, default=4,
                    help="env-batch rollouts; episodes = n_envs * n_episodes")
    ap.add_argument("--steps", type=int, default=200, help="control steps / episode")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--control-freq", type=float, default=30.0)
    ap.add_argument("--noslip-iterations", type=int, default=10)
    ap.add_argument("--lift-height", type=float, default=0.05,
                    help="fixed height gate (defines 'a lift', not swept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/diag"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sanity run (few envs / steps / episodes)")
    args = ap.parse_args()

    if args.smoke:
        args.backend = "cpu"
        args.n_envs = min(args.n_envs, 8)
        args.steps = min(args.steps, 40)
        args.n_episodes = 1

    import genesis as gs

    from xsim.suite import make
    from xsim.suite.policies import LiftExpertPolicy

    sp = SPAWNS[args.spawn]
    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu,
            precision="32", logging_level="warning")
    env = make(
        "Lift", robots="XArm7", camera_names=[], render_backend="raster",
        x_range=sp["x_range"], y_range=sp["y_range"],
        init_tcp_box=sp["init_tcp_box"],
        n_envs=args.n_envs, horizon=args.horizon,
        control_freq=args.control_freq, noslip_iterations=args.noslip_iterations,
        randomize_cameras=False,
        lift_height=args.lift_height,
    )
    # no mid-episode reset: let each env run its whole episode so re-scoring sees
    # the full trajectory (matches the heatmap diagnostic's convention)
    env._check_terminated = lambda: np.zeros(env.n_envs, dtype=bool)
    expert = LiftExpertPolicy(env, cartesian=False)

    top_z = float(env.arena.top_z)
    # confirm we replicate lift.py's live gate constants
    print(f"spawn={args.spawn}  backend={args.backend}  n_envs={args.n_envs}")
    print(f"gate constants: top_z={top_z:.4f} lift_height={args.lift_height} "
          f"success_max_speed={env.success_max_speed} "
          f"eef_xy_radius={env.success_eef_xy_radius} "
          f"hold_ticks={env.success_hold_ticks}")

    cube_p, cube_v, ee_p, ee_v, touch, gt = collect(
        env, expert, args.steps, args.n_episodes, args.seed)
    E, T = touch.shape
    print(f"collected {E} episodes x {T} steps")

    # precompute the per-step scalar signals (E, T)
    high = cube_p[:, :, 2] > (top_z + args.lift_height)
    rel_speed = np.linalg.norm(cube_v - ee_v, axis=-1)
    xy_gap = np.linalg.norm(cube_p[:, :, :2] - ee_p[:, :, :2], axis=-1)
    gt_success = gt.any(axis=1)   # env's own current-gate latch, per episode

    def score(rel_vel, xy_radius, hold, contact) -> np.ndarray:
        raw = high & (rel_speed < rel_vel) & (xy_gap < xy_radius)
        if contact:
            raw = raw & touch
        return any_run_ge(raw, hold)

    # current-gate re-score (validates the recompute against env truth)
    cur_success = score(CUR["rel_vel"], CUR["xy_radius"], CUR["hold"], CUR["contact"])
    cur_sr = float(cur_success.mean())
    gt_sr = float(gt_success.mean())

    # full grid
    rows = []
    best = {"sr": -1.0}
    loosest = None
    for rv in REL_VELS:
        for xr in XY_RADII:
            for h in HOLDS:
                for c in CONTACTS:
                    succ = score(rv, xr, h, c)
                    sr = float(succ.mean())
                    flips = int((succ & ~cur_success).sum())
                    row = dict(rel_vel_max=rv, eef_xy_radius=xr, hold_ticks=h,
                               contact_required=c, success_rate=sr,
                               flips_from_current=flips, n_episodes=E)
                    rows.append(row)
                    if sr > best["sr"]:
                        best = {"sr": sr, **row}
                    if (rv == REL_VELS[-1] and xr == XY_RADII[-1]
                            and h == HOLDS[0] and c is False):
                        loosest = {"sr": sr, "flips": flips}

    gap = best["sr"] - cur_sr
    meaningful = gap > 0.03

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "gate_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "=== success-gate sweep (near-oracle expert) ===",
        f"spawn={args.spawn}  backend={args.backend}  episodes={E}  steps={T}",
        f"env current-gate SR (env._check_success)  = {gt_sr:.4f}",
        f"recomputed current-gate SR (should match)  = {cur_sr:.4f}  "
        f"(delta={abs(gt_sr - cur_sr):.4f})",
        f"loosest gate (rel_vel=inf, r=0.12, hold=1, no-contact) SR = "
        f"{(loosest or {}).get('sr', float('nan')):.4f}  "
        f"flips={( loosest or {}).get('flips', 0)}",
        f"best gate in grid SR = {best['sr']:.4f}  at rel_vel={best['rel_vel_max']} "
        f"r={best['eef_xy_radius']} hold={best['hold_ticks']} "
        f"contact={best['contact_required']}  (flips={best['flips_from_current']})",
        f"max gain over current gate = {gap*100:.2f}%  "
        f"({'MEANINGFUL (>3%)' if meaningful else 'not meaningful (<=3%)'})",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    txt = args.outdir / "gate_sweep.txt"
    txt.write_text(summary + "\n")
    print(f"\nwrote {csv_path.resolve()}")
    print(f"wrote {txt.resolve()}")

    # --- figure: SR sensitivity to each knob --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = rows

    def marginal(key, values):
        """mean SR over grid rows holding `key`==v (others swept)."""
        return [np.mean([r["success_rate"] for r in arr if r[key] == v])
                for v in values]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    labels_rv = [("inf" if np.isinf(v) else f"{v:g}") for v in REL_VELS]
    axes[0].plot(labels_rv, marginal("rel_vel_max", REL_VELS), "-o", color="navy")
    axes[0].set_title("SR vs rel_vel_max")
    axes[0].set_xlabel("rel_vel_max (m/s)")

    axes[1].plot([f"{v:g}" for v in XY_RADII],
                 marginal("eef_xy_radius", XY_RADII), "-o", color="darkgreen")
    axes[1].set_title("SR vs eef_xy_radius")
    axes[1].set_xlabel("eef_xy_radius (m)")

    axes[2].plot([str(v) for v in HOLDS], marginal("hold_ticks", HOLDS),
                 "-o", color="darkorange")
    axes[2].set_title("SR vs hold_ticks")
    axes[2].set_xlabel("hold_ticks")

    axes[3].plot(["True", "False"], marginal("contact_required", [True, False]),
                 "-o", color="firebrick")
    axes[3].set_title("SR vs contact_required")
    axes[3].set_xlabel("contact_required")

    for ax in axes:
        ax.set_ylabel("mean success rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].axhline(cur_sr, color="grey", ls="--", label=f"current={cur_sr:.3f}")
    axes[0].legend(fontsize=8)

    fig.suptitle(
        f"gate sweep  |  spawn={args.spawn}  episodes={E}  |  "
        f"current SR={cur_sr:.3f}  best SR={best['sr']:.3f}  "
        f"gain={gap*100:.1f}% ({'meaningful' if meaningful else 'minor'})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = args.outdir / "gate_sweep.png"
    fig.savefig(png, dpi=140)
    print(f"wrote {png.resolve()}")


if __name__ == "__main__":
    main()
