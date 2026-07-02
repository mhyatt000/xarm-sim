"""Blink test: pose the sim at REAL joint configs and flip sim ↔ real photos.

The strongest cheap accuracy check for the sim rig. Two modes, both run by default:

- **cap**: for each of the 24 calibration captures in ``cap.npz`` (real photos from both
  calibrated Logitech cameras + the exact joint config), pose the sim identically and
  render through the same cameras. Writes ``real | sim | blend`` panels, an alternating
  sim/real flip GIF per (config, camera), and edge-chamfer metrics. The robot silhouette
  bypasses the splat entirely, so its chamfer isolates {URDF + calibration + camera
  model}; the background chamfer scores splat placement.
- **wrist**: the wrist RealSense mount is a guess (no calibration), but it is EE-mounted
  and unchanged since the May recordings — so we pose the sim at joint configs taken from
  a real May episode and compare wrist views side by side.

    uv run python scripts/blink_test.py
    uv run python scripts/blink_test.py --mode cap --configs 0 5 10
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import genesis as gs  # noqa: E402

from validate_mcap import parse_gripper, parse_joint_states, parse_rawimage, read_records, _str  # noqa: E402
from xsim.lift_task import DEFAULT_CAMERAS, LiftBlockEnv, LiftEnvCfg  # noqa: E402

CAP_NPZ = Path("/data/store/opencv_calibrated/cap.npz")
REAL_EPISODE = Path("/data/store/mcaps/single/lift/2026-05-18_1300_episode_000003.mcap")
SERIAL_BY_CAM = {"low": "1e9c6aae", "side": "ad3f052e"}
K = np.array([[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]])
W2C = {
    "low": np.array([
        [-0.6532072424888611, 0.7571735978126526, -0.0028656029608100653, 0.31085190176963806],
        [0.09281685203313828, 0.07631509751081467, -0.992754340171814, 0.09972100704908371],
        [-0.7514686584472656, -0.6487403512001038, -0.12012804299592972, 1.1247998476028442],
        [0.0, 0.0, 0.0, 1.0]]),
    "side": np.array([
        [-0.9901586174964905, 0.13690684735774994, 0.029024558141827583, 0.3651010990142822],
        [0.07804442942142487, 0.7123172283172607, -0.6975049376487732, 0.14493148028850555],
        [-0.11616790294647217, -0.6883753538131714, -0.7159919142723083, 1.160748839378357],
        [0.0, 0.0, 0.0, 1.0]]),
}
GRIPPER_CLOSE_DOF = 0.85


@dataclass
class Cfg:
    mode: Literal["both", "cap", "wrist"] = "both"
    configs: list[int] = field(default_factory=lambda: list(range(24)))
    wrist_fracs: tuple[float, ...] = (0.05, 0.30, 0.50, 0.70, 0.90)
    real_episode: Path = REAL_EPISODE
    out_dir: Path = PROJECT_ROOT / "outputs" / "blink_test"
    nyx_spp: int = 8


def pose_robot(env: LiftBlockEnv, arm_q, gripper_dof: float = 0.0) -> None:
    qpos = torch.zeros(13, dtype=torch.float32, device=env.device)
    qpos[:7] = torch.tensor(np.asarray(arm_q, dtype=np.float32))
    qpos[7:] = float(gripper_dof)
    ent = env.robot._robot_entity
    ent.set_qpos(qpos, zero_velocity=True, skip_forward=False)
    ent.control_dofs_position(qpos.reshape(1, -1))
    env.step()  # rasteriser/nyx visual state refreshes on scene.step


def park_cube(env: LiftBlockEnv) -> None:
    """The cap photos show a bare table; drop the cube far below the scene."""
    pos = torch.tensor([[0.45, 0.0, -5.0]], device=env.device, dtype=gs.tc_float)
    env.cube.set_pos(pos, skip_forward=True)


def project(pw: np.ndarray, cam: str) -> tuple[int, int]:
    pc = W2C[cam][:3, :3] @ pw + W2C[cam][:3, 3]
    uv = K[:2, :2] @ (pc[:2] / pc[2]) + K[:2, 2]
    return int(uv[0]), int(uv[1])


def robot_roi(env: LiftBlockEnv, cam: str, margin: int = 170) -> tuple[int, int, int, int]:
    """Image box around the robot: projected base + TCP, padded."""
    ee = np.asarray(env.robot.ee_pose.cpu()).reshape(-1)[:3]
    pts = [project(np.zeros(3), cam), project(np.array([0.0, 0.0, 0.3]), cam), project(ee, cam)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (max(0, min(xs) - margin), max(0, min(ys) - margin),
            min(639, max(xs) + margin), min(479, max(ys) + margin))


def chamfer(sim: np.ndarray, real: np.ndarray, box=None, invert_box=False) -> float:
    """Median distance (px) from sim edges to nearest real edge, within/outside a box."""
    def edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(g, 60, 140)
    es, er = edges(sim), edges(real)
    mask = np.zeros(es.shape, bool)
    if box is not None:
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        if invert_box:
            mask = ~mask
    else:
        mask[:] = True
    dist_real = cv2.distanceTransform(255 - er, cv2.DIST_L2, 3)
    sel = (es > 0) & mask
    if sel.sum() < 50:
        return float("nan")
    return float(np.median(dist_real[sel]))


def flip_gif(path: Path, real: np.ndarray, sim: np.ndarray, n_cycles: int = 3) -> None:
    frames = []
    for _ in range(n_cycles):
        frames += [real, sim]
    imageio.mimsave(path, frames, duration=0.7, loop=0)


def run_cap(env: LiftBlockEnv, cfg: Cfg) -> None:
    d = np.load(CAP_NPZ, allow_pickle=True)
    q = d["q"]
    rows = []
    for qi in cfg.configs:
        pose_robot(env, q[qi, :7], gripper_dof=0.0)
        imgs = env.render()
        for cam, serial in SERIAL_BY_CAM.items():
            real = d[f"img__camera_logitech_{serial}"][qi]
            sim = imgs[cam]
            blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            panel = np.concatenate([real, sim, blend], axis=1)
            cv2.imwrite(str(cfg.out_dir / f"cap_{qi:02d}_{cam}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            flip_gif(cfg.out_dir / f"flip_{qi:02d}_{cam}.gif", real, sim)
            box = robot_roi(env, cam)
            rows.append((qi, cam,
                         chamfer(sim, real, box),
                         chamfer(sim, real, box, invert_box=True)))
        print(f"config {qi:2d}: " + "  ".join(
            f"{cam} robot={r:5.1f}px bg={b:5.1f}px" for (i, cam, r, b) in rows[-2:]))
    arr = np.array([(r[2], r[3]) for r in rows], dtype=float)
    with open(cfg.out_dir / "chamfer.csv", "w") as f:
        f.write("config,camera,robot_px,background_px\n")
        for qi, cam, r, b in rows:
            f.write(f"{qi},{cam},{r:.2f},{b:.2f}\n")
    print(f"\nMEDIAN over {len(rows)} views: robot {np.nanmedian(arr[:,0]):.1f} px, "
          f"background {np.nanmedian(arr[:,1]):.1f} px  (target: robot < ~4 px)")
    print(f"panels + flip GIFs -> {cfg.out_dir}")


def load_real_wrist_frames(path: Path, fracs) -> list[dict]:
    chans, msgs = {}, {}
    for op, body in read_records(str(path)):
        if op == 0x04:
            (cid,) = struct.unpack_from("<H", body, 0)
            topic, _ = _str(body, 4)
            chans[cid] = topic
        elif op == 0x05:
            (cid,) = struct.unpack_from("<H", body, 0)
            msgs.setdefault(cid, []).append(body[22:])
    by_topic = {t: msgs[cid] for cid, t in chans.items() if cid in msgs}
    wrist = by_topic["/camera/camera/color/image_raw/compressed"]
    joints = by_topic["/xarm/joint_states"]
    grip = by_topic["/xgym/gripper"]
    n = min(len(wrist), len(joints), len(grip))
    out = []
    for fr in fracs:
        i = int((n - 1) * fr)
        ri = parse_rawimage(wrist[i])
        img = np.frombuffer(ri["data"], np.uint8).reshape(ri["height"], ri["width"], 3)
        js = parse_joint_states(joints[i])
        out.append({
            "idx": i,
            "image": img,
            "q": [j["position"] for j in js],
            "gripper_norm": parse_gripper(grip[i])["norm"],
        })
    return out


def run_wrist(env: LiftBlockEnv, cfg: Cfg) -> None:
    frames = load_real_wrist_frames(cfg.real_episode, cfg.wrist_fracs)
    for fr in frames:
        gd = (1.0 - float(fr["gripper_norm"])) * GRIPPER_CLOSE_DOF
        pose_robot(env, fr["q"], gripper_dof=gd)
        # the real frames have the cube in/near the gripper — put the sim cube on the
        # table under the TCP so the views are structurally comparable
        ee = np.asarray(env.robot.ee_pose.cpu()).reshape(-1)[:3]
        cube_pos = torch.tensor([[float(ee[0]), float(ee[1]), 0.016]],
                                device=env.device, dtype=gs.tc_float)
        env.cube.set_pos(cube_pos, skip_forward=True)
        env.step()
        sim = env.render()["wrist"]
        panel = np.concatenate([fr["image"], sim], axis=1)
        cv2.putText(panel, f"REAL wrist (May ep, frame {fr['idx']})", (12, 470), 0, 0.6, (0, 255, 0), 2)
        cv2.putText(panel, "SIM wrist (same joints)", (652, 470), 0, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(cfg.out_dir / f"wrist_{fr['idx']:04d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"wrist frame {fr['idx']}: gripper_norm={fr['gripper_norm']:.2f}")
    print(f"wrist panels -> {cfg.out_dir}")


def main(cfg: Cfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = LiftBlockEnv(LiftEnvCfg(render_backend="nyx", nyx_spp=cfg.nyx_spp), cameras=DEFAULT_CAMERAS)
    if cfg.mode in ("both", "cap"):
        park_cube(env)  # the cap photos show a bare table
        run_cap(env, cfg)
    if cfg.mode in ("both", "wrist"):
        run_wrist(env, cfg)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
