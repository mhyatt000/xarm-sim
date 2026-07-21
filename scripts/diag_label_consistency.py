"""Diagnostic: is the expert's action label a well-defined (near-unimodal)
function of the STUDENT's no-velocity state observation, or is it multimodal?

Motivation. The state student regresses the teacher's 8-dim [j0..j6, g] label
from a flat privileged obs with velocity REMOVED (sim PD velocity profiles don't
transfer to the real arm, so the user dropped every ``*_vel`` key). If two
different teacher labels sit at the same no-vel obs, no MSE-trained net can fit
both — it averages them, and the student caps below the teacher. This script
measures that label inconsistency directly.

It rolls ``LiftExpertPolicy`` (reactive FSM, label is a function of live state)
on the production spawn, collecting (obs, action) at every control step, where
``obs`` is EXACTLY the no-vel student state vector, then computes:

  A. k-NN label disagreement: among each anchor's k obs-nearest neighbors, the
     action spread (mean pairwise L2 + per-dim std) as a function of obs-radius.
     Large spread among obs-near points == multimodal labels.
  B. k-NN BC floor: leave-one-out k-NN regression residual RMSE. This is the
     irreducible MSE a perfect net still cannot beat given these labels — i.e.
     how much student error is unavoidable (bad labels) vs fixable (bad net).
  C. cube_xy binning: within-bin action variance heatmap over the spawn box.

Outputs (outputs/diag/): label_consistency.png, label_consistency.txt.

Tiny CPU smoke (safe on a shared box, no GPU):
    uv run python scripts/diag_label_consistency.py --smoke

Full run later on a free GPU (does NOT need cameras; pure physics):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diag_label_consistency.py \
        --backend gpu --n-envs 512 --n-pairs 150000 --k 8 --spawn production
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# --- spawn presets ---------------------------------------------------------
# production == scripts/simpledagger.py Config defaults (the domain the student
# actually trains on). box == the tuned expert sanity box. table == a wide sweep.
SPAWNS = {
    "production": dict(
        x_range=(0.20, 0.40),
        y_range=(-0.288, 0.288),
        init_tcp_box=((0.10, 0.40), (-0.3048, 0.3048), (-0.01, 0.30)),
    ),
    "box": dict(
        x_range=(0.35, 0.58),
        y_range=(-0.15, 0.15),
        init_tcp_box=None,
    ),
    "table": dict(
        x_range=(0.30, 0.58),
        y_range=(-0.288, 0.288),
        init_tcp_box=None,
    ),
}


def novel_keys(space_keys) -> list[str]:
    """The student's no-velocity state layout: every observable key except the
    velocity ones (GymWrapper concatenates in sorted-key order). If the exact
    trainer filter differs this is the stated fallback: all non-vel keys."""
    return [k for k in sorted(space_keys) if "vel" not in k]


def flat_novel(obs: dict, keys: list[str], n_envs: int) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).reshape(n_envs, -1) for k in keys],
        axis=-1,
    )


def collect(env, expert, keys, n_pairs, steps, n_envs, seed):
    """Roll the expert, recording (no-vel obs, teacher action, cube_xy) at every
    control step, obs and action both read from the SAME pre-step state."""
    obs_rows, act_rows, xy_rows = [], [], []
    collected, ep = 0, 0
    while collected < n_pairs:
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        for _ in range(steps):
            a = np.asarray(expert.act(), dtype=np.float32)
            obs_rows.append(flat_novel(obs, keys, n_envs))
            act_rows.append(a)
            xy_rows.append(np.asarray(obs["cube_pos"], dtype=np.float32)[:, :2])
            obs, _, _, _, _ = env.step(a)
            collected += n_envs
            if collected >= n_pairs:
                break
        ep += 1
    obs = np.concatenate(obs_rows)[:n_pairs]
    act = np.concatenate(act_rows)[:n_pairs]
    xy = np.concatenate(xy_rows)[:n_pairs]
    return obs, act, xy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=120_000,
                    help="target (obs, action) pairs to collect")
    ap.add_argument("--k", type=int, default=8, help="neighbors for kNN metrics")
    ap.add_argument("--spawn", choices=sorted(SPAWNS), default="production")
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200, help="control steps / episode")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--control-freq", type=float, default=30.0)
    ap.add_argument("--noslip-iterations", type=int, default=10)
    ap.add_argument("--anchors", type=int, default=4000,
                    help="anchor points for the kNN-disagreement metric")
    ap.add_argument("--eval-cap", type=int, default=20_000,
                    help="points scored for the BC-floor (neighbors use full set)")
    ap.add_argument("--grid", type=int, default=12, help="cube_xy bins per axis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/diag"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sanity run (few envs / steps / pairs)")
    args = ap.parse_args()

    if args.smoke:
        args.backend = "cpu"
        args.n_envs = min(args.n_envs, 8)
        args.steps = min(args.steps, 30)
        args.n_pairs = min(args.n_pairs, 800)
        args.anchors = min(args.anchors, 200)
        args.eval_cap = min(args.eval_cap, 400)
        args.grid = min(args.grid, 6)

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
    )
    # no mid-episode reset on success: keep collecting the held-lift states too
    env._check_terminated = lambda: np.zeros(env.n_envs, dtype=bool)
    expert = LiftExpertPolicy(env, cartesian=False)

    keys = novel_keys(env.single_observation_space.spaces)
    all_keys = sorted(env.single_observation_space.spaces)
    dropped = [k for k in all_keys if k not in keys]
    print(f"spawn={args.spawn}  backend={args.backend}  n_envs={args.n_envs}")
    print(f"no-vel student obs keys ({len(keys)}): {keys}")
    print(f"dropped velocity keys: {dropped}")

    obs, act, xy = collect(env, expert, keys, args.n_pairs, args.steps,
                           args.n_envs, args.seed)
    M, D = obs.shape
    A = act.shape[1]
    print(f"collected {M} pairs  obs_dim={D}  act_dim={A}")

    # standardize obs (kNN is scale-sensitive); label std is the reference scale
    mu, sd = obs.mean(0), obs.std(0) + 1e-8
    Z = (obs - mu) / sd
    act_mu = act.mean(0)
    act_std = act.std(0) + 1e-12          # per-dim label scale
    global_pair_l2 = float(np.sqrt(2.0) * np.linalg.norm(act_std))  # E||a-a'|| ref

    from sklearn.neighbors import NearestNeighbors

    k = max(2, args.k)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)

    rng = np.random.default_rng(args.seed)

    # --- Metric A: kNN label disagreement vs obs-radius ---------------------
    a_idx = rng.choice(M, size=min(args.anchors, M), replace=False)
    dist_a, nbr_a = nn.kneighbors(Z[a_idx])          # (n_a, k+1) includes self
    nbr_a = nbr_a[:, 1:]                              # drop self column
    obs_radius = dist_a[:, 1:].mean(1)               # mean neighbor obs-distance
    nbr_acts = act[nbr_a]                             # (n_a, k, A)
    local_std = nbr_acts.std(axis=1)                 # (n_a, A) per-dim spread
    # mean pairwise L2 among the k neighbor actions (unbiased-ish via 2*trace(cov))
    local_pair_l2 = np.sqrt(2.0) * np.linalg.norm(local_std, axis=1)  # (n_a,)
    spread_ratio = float(local_pair_l2.mean() / (global_pair_l2 + 1e-12))

    # --- Metric B: leave-one-out kNN BC floor -------------------------------
    e_idx = rng.choice(M, size=min(args.eval_cap, M), replace=False)
    _, nbr_e = nn.kneighbors(Z[e_idx])
    nbr_e = nbr_e[:, 1:]                              # drop self -> LOO
    pred = act[nbr_e].mean(axis=1)                   # (n_e, A)
    resid = act[e_idx] - pred
    rmse_dim = np.sqrt((resid ** 2).mean(axis=0))    # (A,)
    rmse_all = float(np.sqrt((resid ** 2).mean()))
    # fraction of label std left unexplained by obs-neighbors (~ sqrt(1-R^2))
    unexplained = float(np.sqrt((resid ** 2).mean() / (act_std ** 2).mean()))
    rmse_dim_norm = rmse_dim / act_std

    # --- Metric C: within-bin action variance over cube_xy ------------------
    xr, yr = sp["x_range"], sp["y_range"]
    gx = np.clip(((xy[:, 0] - xr[0]) / (xr[1] - xr[0]) * args.grid).astype(int),
                 0, args.grid - 1)
    gy = np.clip(((xy[:, 1] - yr[0]) / (yr[1] - yr[0]) * args.grid).astype(int),
                 0, args.grid - 1)
    var_grid = np.full((args.grid, args.grid), np.nan)
    cnt_grid = np.zeros((args.grid, args.grid))
    for ix in range(args.grid):
        for iy in range(args.grid):
            m = (gx == ix) & (gy == iy)
            cnt_grid[iy, ix] = m.sum()
            if m.sum() >= 2:
                # mean per-dim variance of the standardized labels in the bin
                v = (act[m] / act_std).var(axis=0).mean()
                var_grid[iy, ix] = v

    # --- verdict ------------------------------------------------------------
    if unexplained < 0.15:
        verdict = "labels look UNIMODAL (near-deterministic in the no-vel obs)"
    elif unexplained < 0.35:
        verdict = "labels look MILDLY MULTIMODAL"
    else:
        verdict = "labels look STRONGLY MULTIMODAL (MSE cannot fit them)"

    dim_names = [f"j{i}" for i in range(A - 1)] + ["grip"]
    lines = [
        "=== label consistency (expert -> no-vel student obs) ===",
        f"spawn={args.spawn}  backend={args.backend}  n_envs={args.n_envs}  "
        f"pairs={M}  k={k}",
        f"no-vel obs keys ({len(keys)}): {keys}",
        f"dropped vel keys: {dropped}",
        "",
        f"[A] kNN action-spread / global-spread ratio = {spread_ratio:.3f} "
        "(0=unimodal, 1=obs tells you nothing)",
        f"[B] BC-floor overall RMSE            = {rmse_all:.4f}",
        f"[B] BC-floor unexplained label-std   = {unexplained:.3f} "
        "(~sqrt(1-R^2); irreducible fraction)",
        "[B] per-dim BC-floor RMSE (raw / normalized-by-label-std):",
    ]
    for nm, r, rn in zip(dim_names, rmse_dim, rmse_dim_norm):
        lines.append(f"      {nm:>5}: {r:.4f}  ({rn:.3f})")
    lines += [
        f"[C] median within-bin normalized action var = "
        f"{np.nanmedian(var_grid):.3f}  (over {int((cnt_grid>=2).sum())} bins)",
        "",
        f"VERDICT: {verdict}",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    args.outdir.mkdir(parents=True, exist_ok=True)
    txt = args.outdir / "label_consistency.txt"
    txt.write_text(summary + "\n")
    print(f"\nwrote {txt.resolve()}")

    # --- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # A: obs-radius vs action spread (binned mean) + global reference
    ax = axes[0]
    order = np.argsort(obs_radius)
    r_sorted = obs_radius[order]
    s_sorted = local_pair_l2[order]
    nb = min(20, max(1, len(order) // 10))
    edges = np.linspace(r_sorted[0], r_sorted[-1], nb + 1)
    bidx = np.clip(np.digitize(r_sorted, edges) - 1, 0, nb - 1)
    bx = 0.5 * (edges[:-1] + edges[1:])
    by = np.array([s_sorted[bidx == b].mean() if (bidx == b).any() else np.nan
                   for b in range(nb)])
    ax.scatter(obs_radius, local_pair_l2, s=6, alpha=0.15, color="steelblue")
    ax.plot(bx, by, "-o", color="navy", lw=2, label="binned mean")
    ax.axhline(global_pair_l2, color="red", ls="--",
               label=f"global spread={global_pair_l2:.3f}")
    ax.set_xlabel("mean obs-distance to k neighbors (std units)")
    ax.set_ylabel("neighbor action spread  E||a-a'|| (rad)")
    ax.set_title(f"[A] label disagreement vs obs-radius\nratio={spread_ratio:.3f}")
    ax.legend(fontsize=8)

    # B: per-dim BC-floor RMSE (raw) with normalized annotation
    ax = axes[1]
    xb = np.arange(A)
    ax.bar(xb, rmse_dim, color="mediumseagreen")
    for xi, rn in zip(xb, rmse_dim_norm):
        ax.text(xi, rmse_dim[xi], f"{rn:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xb)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("BC-floor RMSE (rad)")
    ax.set_title(f"[B] kNN BC floor  overall={rmse_all:.3f}\n"
                 f"unexplained label-std={unexplained:.3f}  (bars annotated / label std)")

    # C: within-bin action variance heatmap over cube_xy
    ax = axes[2]
    im = ax.imshow(var_grid, origin="lower",
                   extent=[xr[0], xr[1], yr[0], yr[1]], aspect="auto",
                   cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="norm action var")
    ax.set_xlabel("cube x (m)")
    ax.set_ylabel("cube y (m)")
    ax.set_title("[C] within-bin action variance vs cube_xy")

    fig.suptitle(f"label consistency  |  spawn={args.spawn}  pairs={M}  k={k}  "
                 f"|  {verdict}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = args.outdir / "label_consistency.png"
    fig.savefig(png, dpi=140)
    print(f"wrote {png.resolve()}")


if __name__ == "__main__":
    main()
