"""Roll a DAgger student checkpoint once and write an all-envs grid video.

Shares simpledagger's Config (the play/eval fields apply; training fields are
ignored). Green border = success tick, red = timeout; prints a per-env
spawn/outcome table.

    uv run python scripts/play.py --play outputs/dagger/<exp>/best.pt --policy image --n-envs 16
"""

from __future__ import annotations

import numpy as np
import torch
import tyro

from simpledagger import Config, build_env, build_student
from xsim.algo import flat, image_proprio_keys
from xsim.utils.video import VideoSink, tile_grid


def play(cfg: Config) -> None:
    env = build_env(cfg, render=True)
    B = cfg.n_envs
    image = cfg.policy == "image"
    device = torch.device("cuda" if cfg.backend == "gpu" else "cpu")
    student = build_student(cfg, env, device)
    student.load_state_dict(torch.load(cfg.play, map_location=device))
    student.eval()
    state_keys = sorted(env.unwrapped.single_observation_space.spaces)
    proprio_keys = image_proprio_keys(state_keys)

    obs, _ = env.reset(seed=cfg.eval_seed)
    spawn = np.asarray(env.unwrapped.cube.get_pos(), dtype=np.float64).copy()
    q = np.asarray(env.unwrapped.cube.get_quat(), dtype=np.float64)
    spawn_yaw = 2.0 * np.arctan2(q[:, 3], q[:, 0])
    status = np.zeros(B, dtype=np.int64)  # 0 live, 1 success, 2 fail
    ep_len = np.zeros(B, dtype=np.int64)
    live = np.ones(B, dtype=bool)
    out = cfg.play_video or cfg.play.parent / "rollout.mp4"
    sink = VideoSink(out, 1.0 / env.unwrapped.control_dt)

    def snap(obs) -> None:
        if image:  # reuse the policy's own frames (image_hw px, upscaled)
            rgb = obs["rgb"].transpose(0, 1, 3, 4, 2)  # (B, V, H, W, 3)
            views = [rgb[:, i] for i in range(rgb.shape[1])]
        else:
            d = env.unwrapped.render_views(all_envs=True)
            views = [d[k] for k in sorted(d)]
        sink.add(np.concatenate(
            [tile_grid(v, cfg.video_max_width, status, upscale=True) for v in views],
            axis=1))

    snap(obs)
    flow = cfg.loss == "flow"
    tick, plan = 0, None
    while live.any():
        so = (flat(obs, proprio_keys), obs["rgb"]) if image else obs
        if flow:
            if tick % cfg.replan == 0:
                plan = student.act(so)
            a = plan[:, tick % cfg.replan]
        else:
            a = student.act(so)
        tick += 1
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated | truncated
        status[live & done & info["success"]] = 1
        status[live & done & ~info["success"]] = 2
        ep_len += live
        live &= ~done
        snap(obs)

    sink.close()

    print(f"\ncheckpoint: {cfg.play}   success {int((status == 1).sum())}/{B}")
    print(f"{'env':>3} {'outcome':>8} {'len':>4} {'cube_x':>7} {'cube_y':>7} {'yaw_deg':>8}")
    for i in range(B):
        print(f"{i:>3} {'success' if status[i] == 1 else 'FAIL':>8} {ep_len[i]:>4} "
              f"{spawn[i, 0]:>7.3f} {spawn[i, 1]:>7.3f} {np.degrees(spawn_yaw[i]):>8.1f}")
    print(f"video -> {out}")


def main(cfg: Config) -> None:
    if cfg.play is None:
        raise SystemExit("play.py needs --play <checkpoint.pt>")
    play(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
