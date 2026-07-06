"""Verify a sim MCAP batch against the real lift recordings.

Two layers, matching what is actually meaningful to compare (the real episodes were
recorded on the old static-camera rig, so pixels are not comparable — labels are):

1. **Format (hard gate)** — every sim file must match the real reference on the six
   core topics/schemas, include the sim-only calibration topics, record at ~30 Hz, and
   carry a sane frame count.
2. **Label distributions** — joint positions, TCP height profile, gripper timing and
   episode length, sim batch vs a stratified sample of the 409 real episodes. Rendered
   to one report PNG; printed summary alongside.

    uv run python scripts/compare_batches.py --sim-dir outputs/sim_mcap/pilot_v1
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_mcap import (  # noqa: E402
    _str,
    parse_gripper,
    parse_joint_states,
    parse_pose,
    read_records,
    scan,
    compare_topic_layout,
)

REAL_DIR = Path("/data/store/mcaps/single/lift")
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")

# series colors: validated reference palette, fixed slot order (1=real, 2=sim)
C_REAL, C_SIM = "#2a78d6", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e2"


@dataclass
class Cfg:
    sim_dir: Path = PROJECT_ROOT / "outputs" / "sim_mcap" / "pilot_v1"
    real_dir: Path = REAL_DIR
    max_real: int = 60          # stratified sample across recording sessions
    out_dir: Path = PROJECT_ROOT / "outputs" / "batch_report"
    reference: Path | None = None  # defaults to first sampled real file
    task: str = "lift"          # protocol whose frame-count gate applies (lift|stack)


def parse_episode(path: Path) -> dict | None:
    """Proprio series for one episode: joints [T,7], tcp_z [T] (m), grip [T], rate."""
    chans, times, msgs = {}, {}, {}
    want = {"/xarm/joint_states", "/xarm/robot_states", "/xgym/gripper"}
    for op, body in read_records(str(path)):
        if op == 0x04:
            (cid,) = struct.unpack_from("<H", body, 0)
            topic, _ = _str(body, 4)
            chans[cid] = topic
        elif op == 0x05:
            (cid,) = struct.unpack_from("<H", body, 0)
            topic = chans.get(cid)
            if topic in want:
                (_, _, log_t, _) = struct.unpack_from("<HIQQ", body, 0)
                times.setdefault(topic, []).append(log_t)
                msgs.setdefault(topic, []).append(body[22:])
    if set(msgs) != want:
        return None
    joints = np.array([[j["position"] for j in parse_joint_states(m)] for m in msgs["/xarm/joint_states"]])
    tcp_z = np.array([parse_pose(m)[0][2] for m in msgs["/xarm/robot_states"]]) / 1000.0  # mm -> m
    grip = np.array([parse_gripper(m)["norm"] for m in msgs["/xgym/gripper"]])
    ts = np.array(times["/xarm/joint_states"], dtype=np.float64)
    dur = (ts[-1] - ts[0]) / 1e9
    rate = (len(ts) - 1) / dur if dur > 0 else 0.0
    return {"joints": joints, "tcp_z": tcp_z, "grip": grip, "rate": rate,
            "duration": dur, "frames": len(ts)}


def sample_real(real_dir: Path, k: int) -> list[Path]:
    files = sorted(real_dir.glob("*.mcap"))
    by_session: dict[str, list[Path]] = {}
    for f in files:
        by_session.setdefault(f.name.split("_episode")[0], []).append(f)
    rng = np.random.default_rng(7)
    picked: list[Path] = []
    sessions = list(by_session.values())
    while len(picked) < min(k, len(files)):
        for group in sessions:
            if group and len(picked) < k:
                picked.append(group.pop(rng.integers(len(group))))
    return picked


def check_format(sim_files: list[Path], reference: Path, task: str = "lift") -> list[str]:
    ref_chans, ref_schemas, _, _ = scan(str(reference))
    problems = []
    for f in sim_files:
        chans, schemas, counts, _ = scan(str(f))
        for problem in compare_topic_layout(chans, schemas, counts, ref_chans, ref_schemas, task=task):
            problems.append(f"{f.name}: {problem}")
    return problems


def resample(y: np.ndarray, n: int = 100) -> np.ndarray:
    x = np.linspace(0, 1, len(y))
    return np.interp(np.linspace(0, 1, n), x, y)


def band(ax, eps: list[np.ndarray], color: str, label: str):
    Y = np.stack([resample(e) for e in eps])
    t = np.linspace(0, 1, Y.shape[1])
    lo, med, hi = np.percentile(Y, [10, 50, 90], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(t, med, color=color, linewidth=2, label=label)


def main(cfg: Cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    sim_files = sorted(cfg.sim_dir.glob("*.mcap"))
    if not sim_files:
        raise SystemExit(f"no mcaps in {cfg.sim_dir}")
    real_files = sample_real(cfg.real_dir, cfg.max_real)
    reference = cfg.reference or real_files[0]

    print(f"sim: {len(sim_files)} files, real sample: {len(real_files)} of "
          f"{len(list(cfg.real_dir.glob('*.mcap')))}, reference: {reference.name}")

    problems = check_format(sim_files, reference, task=cfg.task)
    if problems:
        print("\nFORMAT: FAIL")
        for p in problems:
            print("  " + p)
    else:
        print("FORMAT: PASS (core topics match the real layout; calibration topics present; frame counts sane)")

    sim = [e for f in sim_files if (e := parse_episode(f))]
    real = [e for f in real_files if (e := parse_episode(f))]
    print(f"parsed {len(sim)} sim / {len(real)} real episodes")

    def stats(eps, key):
        v = np.array([e[key] for e in eps])
        return f"{v.mean():.2f}±{v.std():.2f}"
    print(f"rate Hz     sim {stats(sim,'rate')}   real {stats(real,'rate')}")
    print(f"duration s  sim {stats(sim,'duration')}   real {stats(real,'duration')}")
    print(f"frames      sim {stats(sim,'frames')}   real {stats(real,'frames')}")

    # ---------------- report figure
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.28)

    sim_j = np.concatenate([e["joints"] for e in sim])
    real_j = np.concatenate([e["joints"] for e in real])
    for j in range(7):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        rng_ = (min(real_j[:, j].min(), sim_j[:, j].min()), max(real_j[:, j].max(), sim_j[:, j].max()))
        ax.hist(real_j[:, j], bins=40, range=rng_, density=True, histtype="stepfilled",
                color=C_REAL, alpha=0.30, edgecolor=C_REAL, linewidth=1.2)
        ax.hist(sim_j[:, j], bins=40, range=rng_, density=True, histtype="stepfilled",
                color=C_SIM, alpha=0.30, edgecolor=C_SIM, linewidth=1.2)
        ax.set_title(JOINT_NAMES[j], fontsize=9, color=INK)
        ax.set_xlabel("joint position (rad)")
        ax.set_ylabel("probability density (1/rad)")
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0.05, 0.75, "joint position probability densities (x: rad, y: 1/rad)", fontsize=10, color=INK, weight="bold")
    ax.text(0.05, 0.55, "real (sampled episodes)", fontsize=10, color=C_REAL, weight="bold")
    ax.text(0.05, 0.40, "sim (this batch)", fontsize=10, color=C_SIM, weight="bold")
    ax.text(0.05, 0.12, "sim inside the real support =\nplausible label coverage", fontsize=8.5, color=MUTED)

    ax = fig.add_subplot(gs[2, 0:2])
    band(ax, [e["tcp_z"] for e in real], C_REAL, "real")
    band(ax, [e["tcp_z"] for e in sim], C_SIM, "sim")
    ax.set_title("TCP height over normalized episode time", fontsize=10, color=INK)
    ax.set_xlabel("normalized episode progress (unitless, 0-1)"); ax.set_ylabel("TCP z height (m)")
    ax.legend(frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[2, 2:4])
    band(ax, [e["grip"] for e in real], C_REAL, "real")
    band(ax, [e["grip"] for e in sim], C_SIM, "sim")
    ax.set_title("gripper norm (1 = open) over normalized episode time", fontsize=10, color=INK)
    ax.set_xlabel("normalized episode progress (unitless, 0-1)"); ax.set_ylabel("gripper opening norm (unitless)")
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle(f"sim batch '{cfg.sim_dir.name}' ({len(sim)} eps) vs real lift sample ({len(real)} of 409 eps) — "
                 f"median band = 10–90 pct", fontsize=11, color=INK, y=0.995)
    out = cfg.out_dir / f"report_{cfg.sim_dir.name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nreport -> {out}")

    fig2 = plt.figure(figsize=(13, 7.8))
    gs2 = fig2.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    for j in range(7):
        ax = fig2.add_subplot(gs2[j // 4, j % 4])
        band(ax, [e["joints"][:, j] for e in real], C_REAL, "real")
        band(ax, [e["joints"][:, j] for e in sim], C_SIM, "sim")
        ax.set_title(f"{JOINT_NAMES[j]} position over normalized episode time", fontsize=9, color=INK)
        ax.set_xlabel("normalized episode progress (unitless, 0-1)")
        ax.set_ylabel("joint position (rad)")
        if j == 0:
            ax.legend(frameon=False, loc="best")
    ax = fig2.add_subplot(gs2[1, 3]); ax.axis("off")
    ax.text(0.05, 0.75, "joint trajectories", fontsize=10, color=INK, weight="bold")
    ax.text(0.05, 0.55, "solid line = median episode", fontsize=9, color=MUTED)
    ax.text(0.05, 0.40, "shaded band = 10-90 pct", fontsize=9, color=MUTED)
    ax.text(0.05, 0.22, "x-axis is normalized episode progress (unitless, 0-1),\nso different-duration demos align\nfrom start (0) to end (1).", fontsize=8.5, color=MUTED)
    fig2.suptitle(f"joint trajectories for sim batch {cfg.sim_dir.name} vs real lift sample",
                  fontsize=11, color=INK, y=0.995)
    joint_out = cfg.out_dir / f"report_{cfg.sim_dir.name}_joint_time.png"
    fig2.savefig(joint_out, dpi=130, bbox_inches="tight")
    print(f"joint-time report -> {joint_out}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
