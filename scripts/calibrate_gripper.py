"""Calibrate a ported gripper's GripperModel values on the Lift env.

Three probes, results printed for freezing into the gripper dataclass:

  width:  hold the arm at home, sweep the gripper channel 1 -> 0, report the
          finger-link separation and gripper_norm at each setpoint (pick
          grasp_dofs / max_open_width; check the dof->width map is monotone).
  grasp:  run the reactive lift expert with the holding band widened to
          (0.01, 0.99), log the gripper_norm plateau closed on air vs seated
          on the cube (pick hold_norm_lo/hi), and report lift success.
  render: save a frame at the home pose (mount/TCP eyeball check).

    uv run python scripts/calibrate_gripper.py --robot XArm7Rethink
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import xsim.suite as suite
from xsim.suite.policies import LiftExpertPolicy

APPROACH, DESCEND, CLOSE, LIFT, RELEASE = range(5)


@dataclass
class Config:
    robot: str = "XArm7Rethink"
    n_envs: int = 8
    seed: int = 0
    grasp_ticks: int = 200
    out: Path = Path("outputs/calibrate_gripper")


def finger_sep(robot) -> np.ndarray:
    la, lb = robot.gripper.finger_link_names
    a = np.asarray(robot.gripper_entity.get_link(la).get_pos().detach().cpu())
    b = np.asarray(robot.gripper_entity.get_link(lb).get_pos().detach().cpu())
    return np.linalg.norm(a - b, axis=-1)


def main(cfg: Config) -> None:
    env = suite.make(
        "Lift", robots=[cfg.robot], horizon=10_000, n_envs=cfg.n_envs,
        render_backend="raster",
    )
    env.reset(seed=cfg.seed)
    r = env.robots[0]

    cfg.out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    frame = env.render()
    imageio.imwrite(cfg.out / f"{cfg.robot}_home.png", frame)
    print(f"render -> {cfg.out / f'{cfg.robot}_home.png'}")
    print(f"home ee_pos {r.ee_pos[0].round(4)}  ee_quat {r.ee_quat[0].round(4)}")

    print("\nwidth map (arm held at home):")
    hold = np.asarray(r.joint_positions)
    print(f"{'a':>5} {'sep_mm':>8} {'norm':>6}")
    for a in np.linspace(1.0, 0.0, 11):
        act = np.concatenate([hold, np.full((cfg.n_envs, 1), a)], axis=1)
        for _ in range(20):
            env.step(act.astype(np.float32))
        print(f"{a:5.2f} {1000 * finger_sep(r)[0]:8.2f} {r.gripper_norm[0]:6.3f}")

    print("\ngrasp probe (lift expert, holding band widened):")
    r.gripper.hold_norm_lo, r.gripper.hold_norm_hi = 0.01, 0.99
    obs, _ = env.reset(seed=cfg.seed)
    policy = LiftExpertPolicy(env)
    policy.reset(obs)
    air, seated = [], []
    success = np.zeros(cfg.n_envs, dtype=bool)
    for _ in range(cfg.grasp_ticks):
        act = np.asarray(policy.act(obs))
        obs, _, _, _, info = env.step(act)
        success |= info["success"]
        norm = r.gripper_norm
        closing = policy.phase == CLOSE
        cube = np.asarray(env.cube.get_pos())
        near = np.linalg.norm(cube - r.ee_pos, axis=1) < 0.04
        air.extend(norm[closing & ~near & (policy._close_ticks > 15)])
        seated.extend(norm[(policy.phase == LIFT) | (closing & near & (policy._close_ticks > 15))])
    for name, vals in [("closed on air", air), ("seated on cube", seated)]:
        v = np.asarray(vals)
        print(
            f"  {name:15s} n={len(v):5d} "
            + (f"norm p10/p50/p90 = {np.percentile(v, [10, 50, 90]).round(3)}" if len(v) else "-")
        )
    print(f"  lift success: {int(success.sum())}/{cfg.n_envs}")


if __name__ == "__main__":
    main(tyro.cli(Config))
