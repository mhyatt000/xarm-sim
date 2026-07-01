# xarm-sim

A Genesis-based grasp simulator with a sim-to-real rendering pipeline. The
distinctive piece is not the grasping itself but the rendering path: the
simulated scene is rendered through **Nyx** (a path tracer) using an HDR
environment map plus an optional **Gaussian-splat "light field"**, so a sim
scene can be made to match a real room captured with a **ZED stereo camera**.

The package layer (`xsim`) is robot-agnostic; the xArm7 wiring lives in
`scripts/`.

## What's in the box

| Dependency | Role |
| --- | --- |
| `genesis-world[render]` | Physics simulation, scene graph, rasterizer cameras, video recording |
| `gs-nyx`, `gs-nyx-plugin` | Nyx path tracer: HDR env maps + Gaussian-splat light fields |
| `gs-madrona` (optional) | Batched GPU rendering on CUDA (auto-detected; falls back to rasterizer) |
| `torch`, `tensordict` | Batched env state and observations |
| `tyro` | Typed CLI for the scripts |
| `av`, `imageio`, `opencv-python`, `Pillow` | Image / video I/O for recording and the sweep |

## Layout

```
xarm-sim/
├── src/xsim/                          # installed package (uv build backend, module "xsim")
│   ├── grasp_env.py                   # core: GraspEnv + Manipulator
│   └── scripted_grasp_policy.py       # scripted waypoint policy + build_env() defaults (Franka)
├── scripts/
│   ├── scripted_grasp_policy_room.py  # swaps in xArm7 + ZED-matched "room" env (tyro CLI)
│   └── sim2real_static_sweep.py       # param sweep to match a render to a real photo
├── xarm7_standalone.urdf              # the xArm7 robot
├── assets/                            # robot meshes (STL / OBJ / GLB)
├── pyproject.toml / uv.lock           # uv-managed, Python 3.11–3.12
└── main.py                            # placeholder entrypoint
```

## Core concepts (`src/xsim/grasp_env.py`)

### `GraspEnv`

A vectorized (`num_envs`-batched) RL-style environment.

- **Scene build** — floor (BSDF surface), optional walls (boxes), the robot, and
  a box object to grasp.
- **Cameras** — a stereo left/right pair (`RasterizerCameraOptions`, or
  `BatchRendererCameraOptions` when Madrona + CUDA are available). Optionally a
  Nyx camera (`NyxCameraOptions`) that renders with an HDR env map and
  Gaussian-splat light fields for photoreal / sim2real output.
- **Recording** — the `record_video` config maps a camera attribute name to an
  MP4 path; recording is wired up before `scene.build()`.
- **Loop** — standard `reset()` / `step(actions)` / `get_observations()`.
  Observations are end-effector-vs-object pose deltas. Reward is
  `_reward_keypoints` (exp of negative keypoint distance between gripper and
  object). `grasp_and_lift_demo()` runs a hard-coded grasp → lift → place → home.

### `Manipulator`

Robot wrapper, driven entirely by a `robot_cfg` dict (so it is robot-agnostic).

- Loads the robot from **URDF or MJCF** (`robot_morph`).
- Two IK backends: Genesis IK (`gs_ik`) and damped-least-squares (`dls_ik`).
- Sets PD gains / force limits, gripper open/close, and `go_to_goal(pose)`.
- Exposes `ee_pose`, finger poses, and `center_finger_pose`.

## Scripts

### `scripts/scripted_grasp_policy_room.py`

Wires the generic env to **xArm7** (7 arm DOF + 6 gripper DOF) and a "room"
environment whose camera intrinsics come from a **ZED FHD left lens**
(`fy = 1066.84`, vertical FOV ≈ 53.69°). A Gaussian splat (`last.ply`) can be
loaded as the scene background. Typed CLI via `tyro`.

```bash
uv run python scripts/scripted_grasp_policy_room.py --help
uv run python scripts/scripted_grasp_policy_room.py --backend gpu --splat-uri last.ply
```

### `scripts/sim2real_static_sweep.py`

Renderer-calibration harness. Sweeps env-map multiplier/rotation, camera FOV,
and floor roughness/metallic, renders each case through Nyx, and computes image
statistics plus MAE against a real reference image (`left.png`) — writing a
`summary.csv`. Use it to tune the renderer so sim images match real ZED
captures.

```bash
uv run python scripts/sim2real_static_sweep.py --output-dir outputs/sweep
```

## Required external assets (not in git)

These are referenced by the scripts but `.gitignore`d, so supply them yourself:

- `assets/lab-hdri.hdr` — HDR environment map.
- `last.ply` (repo root) — Gaussian splat / room scan used as the background.
- `left.png` (repo root) — real reference image for the sim2real sweep.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned via
`.python-version`; uv manages it.

```bash
uv sync          # create .venv and install locked dependencies
uv run python -m xsim.scripted_grasp_policy --help
```

GPU rendering (Nyx / Madrona) needs an NVIDIA GPU with a working CUDA driver.
