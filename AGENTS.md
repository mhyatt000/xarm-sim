# Agent Instructions — xarm-sim sim-data pipeline (handoff 2026-07-02)

You are continuing a verified pipeline that generates photoreal xArm7 block-lift
episodes as Foxglove MCAP for training (crossformer). Everything through pilot
verification is DONE and committed on branch `synthetic-lift-mcap`. Your job is the
scale-up and its verification. Read this whole file before running anything.

## 1. Non-negotiable constraints (from grifflee)

1. **Table rendering: `table_mode="slab"` (the default). NEVER pass
   `--env.table-mode plane`** for testing or generation.
2. **Never move, jitter, or "improve" the camera poses.** `low`/`side` extrinsics in
   `src/xsim/lift_task.py` (`LOW_C2W_CV`, `SIDE_C2W_CV`, `LOGITECH_FOV_DEG`) come from
   the real calibration (`/data/store/opencv_calibrated`). Label correctness depends on
   them exactly as they are.
3. **Do not re-tune the splat alignment by eye.** `DEFAULT_SPLAT_POS/QUAT/SCALE` were
   solved numerically and human-verified. If the room is rescanned or cameras
   recalibrated, re-solve with `scripts/align_ransac.py` (stage 1 → grifflee confirms
   landmarks → solve), then regenerate `assets/lab_clean.ply` via `scripts/clean_splat.py`.
4. **Do not remove the two conversion calls in `_make_light_field`**
   (`float3_z_up_to_y_up_a`, `quaternion_z_up_to_y_up_a`). They work around a gs-nyx bug:
   the scene exporter converts meshes/cameras from Genesis z-up to Nyx y-up but passes
   light fields through raw. Removing them silently misplaces the splat background.
5. **Keep `gripper_grasp_dof: 0.58`.** 0.53 matches the real recorded close depth
   (norm 0.37) but rigid sim fingers drop the cube at that width — tested, 0/3.
6. **Human checkpoints**: before building on any new visual/scene interpretation, and
   after any batch, show grifflee a labeled artifact (image/report) and wait for his
   read. He catches visual errors reliably; this protocol caught every real bug so far.

## 2. Current state (what is already verified)

- **Geometry**: sim robot rendered through the calibrated cameras matches real photos at
  **median 1.9 px** edge error over all 24 calibration configs (`scripts/blink_test.py`;
  artifacts in `outputs/blink_test/`, incl. sim↔real flip GIFs).
- **Wrist camera**: uncalibrated by nature (robot never sees itself); mount is a guess
  validated against real May episode frames at matched joints (~40 px mid-grasp).
- **Demonstration style**: scripted policy tuned to the real demos — low IK-solved ready
  pose (TCP ≈ (0.34, 0, 0.10)), lift 0.09 m, grasp completing ~55–60% of episode,
  release ~85%, tempo jitter 0.85–1.30× → 6.6–9.8 s episodes.
  Verified: `outputs/batch_report/report_pilot_v2.png` — sim tracks the real median TCP
  and gripper profiles inside the real 10–90% band.
- **Pilot v2**: `outputs/sim_mcap/pilot_v2/` — 10/10 success, format gate PASS,
  `manifest.json` records git SHA, config, splat md5, per-episode seeds/stats.
- **Docs**: `docs/TRAINING_HANDOFF.md` = consumer-facing (calibrated-vs-guessed
  inventory, eval caveats). Read it once.

## 3. Repo navigation and daily operations

Prefer the source over the README when in doubt; the README still describes the older
generic grasp demo in places. The current lift-data path is:

- `src/xsim/lift_task.py` -- the active xArm7 lift environment: robot config, calibrated
  `low`/`side` cameras, guessed wrist camera, table/slab rendering, Nyx splat setup,
  spawn/drop ranges, and camera-jitter toggles.
- `src/xsim/scripted_lift_policy.py` -- scripted waypoint policy used for these demos.
- `src/xsim/mcap_writer.py` -- Foxglove MCAP topic/schema writer matching the real lift
  logs.
- `scripts/generate_lift_dataset.py` -- main entrypoint for simulation. `generate`
  writes MCAP; `preview` writes milestone PNGs; `video` writes an MP4 contact sheet.
- `scripts/compare_batches.py` -- batch format gate plus label-distribution report
  against `/data/store/mcaps/single/lift`.
- `scripts/validate_mcap.py` -- inspect one MCAP and optionally compare its topic/schema
  set against a reference real episode.
- `scripts/blink_test.py` and `scripts/verify_splat_alignment.py` -- calibration and
  alignment diagnostics. Use these before trusting visual/camera changes.
- `scripts/align_ransac.py` and `scripts/clean_splat.py` -- only for rescan/recalibration
  workflows.
- `scripts/scripted_grasp_policy_room.py`, `src/xsim/grasp_env.py`, and
  `src/xsim/scripted_grasp_policy.py` -- older/generic grasp environment, not the current
  lift MCAP production path.
- `outputs/` -- generated artifacts, gitignored. Do not commit MCAPs, videos, reports, or
  large assets.

Common starting point:

```bash
cd ~/repo/xarm-sim
uv sync
```

Fast local sanity checks:

```bash
# Fast raster video, no MCAP, useful for policy/physics checks.
uv run python scripts/generate_lift_dataset.py \
    --mode video --backend gpu \
    --video-path outputs/sim_preview/raster_check.mp4 --seed 0

# Photoreal Nyx video, no MCAP, useful before/after a batch.
uv run python scripts/generate_lift_dataset.py \
    --mode video --backend gpu --env.render-backend nyx \
    --video-path outputs/sim_preview/nyx_check.mp4 --seed 0

# Milestone PNGs at reset/waypoints/settled.
uv run python scripts/generate_lift_dataset.py \
    --mode preview --backend gpu \
    --preview-dir outputs/sim_preview/lift_env --seed 0
```

Small MCAP smoke test:

```bash
uv run python scripts/generate_lift_dataset.py \
    --n-episodes 3 --backend gpu --env.render-backend nyx \
    --out-dir outputs/sim_mcap/smoke --seed 300

uv run python scripts/compare_batches.py --sim-dir outputs/sim_mcap/smoke
uv run python scripts/validate_mcap.py \
    outputs/sim_mcap/smoke/episode_000000.mcap \
    --reference /data/store/mcaps/single/lift/2026-05-25_1457_episode_000002.mcap
```

Diagnostic commands:

```bash
# Robot/camera calibration blink checks against cap.npz.
uv run python scripts/blink_test.py --mode cap --configs 0 5 10

# Wrist sanity panels against a real May episode.
uv run python scripts/blink_test.py --mode wrist

# Splat-room alignment panels: real | sim | blend plus an oblique overview.
uv run python scripts/verify_splat_alignment.py --tag check
```

Simulation and toggle notes:

- `--mode generate|preview|video`: `generate` writes one `episode_*.mcap` per rollout and
  `manifest.json`; `preview` writes PNGs; `video` writes an MP4 contact sheet. Preview
  and video do not write MCAP.
- `--backend gpu|cpu`: Genesis backend. Use `gpu` on mayo/RTX for normal work.
- `--env.render-backend raster|nyx`: `raster` is fast and has no photoreal splat room;
  `nyx` is the production photoreal path and reads the splat. Deliverable MCAPs should
  use `nyx` unless grifflee explicitly asks for a raster-only physics pilot.
- `--env.nyx-spp N`: Nyx samples per pixel. Default is 8; increasing it costs time.
- `--env.splat-uri PATH`: defaults to `assets/lab_clean.ply` if present, else
  `/data/store/lab.ply`. Do not swap splats for production without rerunning alignment
  checks.
- `--env.splat-pos`, `--env.splat-quat`, `--env.splat-rot-rpy-deg`, `--env.splat-scale`:
  experiment-only alignment overrides. Do not bake changes without the RANSAC/human
  checkpoint workflow in section 1.
- `--env.rectangle-x LO HI` / `--env.rectangle-y LO HI`: cube spawn rectangle. Any
  widening needs a raster pilot, a Nyx pilot, `compare_batches.py`, and grifflee's read.
- `--env.drop-zone X Y Z`: final delivery target; changing this changes label dynamics.
- `--steps-per-segment`, `--hold-steps`, `--grasp-tcp-offset`: policy/tempo/contact
  controls. These affect the real-vs-sim distribution report.
- `--save-failures`: keeps failed MCAP rollouts instead of deleting them. Use only for
  debugging; production batches should be success-gated.
- `--env.cam-jitter-deg`, `--env.cam-jitter-cm`, `--env.wrist-jitter-deg`,
  `--env.wrist-jitter-cm`: per-episode camera jitter, recorded in `manifest.json`.
  Defaults are 0. Do not use for the verified baseline unless requested.
- `--env.table-transparent`: hides the visual table slab but keeps table collision.
  Debugging only.
- `--env.table-mode plane`: the CLI exposes it, but section 1 prohibits it for this
  pipeline. Keep the default `slab`.
- `--env.show-viewer`: opens the Genesis viewer and requires a GUI-capable session.
- `--env.res WIDTH HEIGHT`, `--env.physics-dt`, `--env.record-every`: low-level capture
  shape/rate knobs. The baseline is 640x480, 1/120 s physics, `record_every=4` -> 30 Hz.

Tyro maps dataclass fields to CLI flags with hyphens and nested dataclasses with dots,
for example `LiftEnvCfg.render_backend` becomes `--env.render-backend` and
`cam_jitter_cm` becomes `--env.cam-jitter-cm`.

## 4. Your tasks, in order

### Task A — Scale-up batch

Ask grifflee for N if not already given (100 ≈ 2 h, 200 ≈ 4 h, 400 ≈ 8 h at ~70 s/ep on
the RTX 5090; run in background / overnight):

```bash
cd ~/repo/xarm-sim
uv run python scripts/generate_lift_dataset.py \
    --n-episodes N --env.render-backend nyx \
    --out-dir outputs/sim_mcap/batch_v1 --seed 1000
```

- Seed 1000 avoids overlap with pilot seeds (100–109, 200–209). Any fresh range works;
  never reuse a range already in a kept batch (manifest.json lists them).
- Success-gated: failed grasps are dropped automatically. Expect ≥ 90% kept (pilots ran
  10/10). If the success rate drops noticeably below that, STOP and investigate before
  burning GPU time — nothing about a bigger batch should change the physics.
- Episodes are ~100 MB each; check disk headroom for N > 200.

### Task B — Verify the batch

```bash
uv run python scripts/compare_batches.py --sim-dir outputs/sim_mcap/batch_v1
```

Pass criteria:
- prints `FORMAT: PASS` (topic/schema/count match vs a real reference; frame counts
  150–320; exits non-zero on failure — treat any failure as a hard stop).
- rate ≈ 30.00 Hz, durations spread roughly 6.5–10 s.
- In `outputs/batch_report/report_batch_v1.png`: sim TCP/gripper median inside the real
  band (like report_pilot_v2); joint histograms overlapping the real support.
- Also eyeball ~3 episodes visually: `--mode video` on the generator config renders an
  mp4 contact sheet without writing MCAP. Confirm the cube reads RED (not salmon — the
  albedo was just darkened and hasn't been reviewed in a full batch), robot on the
  table, background room correct.

**Show grifflee the report PNG + one video before calling it done.**

### Task C (optional, flagged improvement) — widen cube spawn coverage

The reports show sim joint1 narrower than real (cube spawn area is small/symmetric).
If grifflee wants better coverage:
1. Edit `LiftEnvCfg.rectangle_x`/`rectangle_y` in `src/xsim/lift_task.py` modestly
   (e.g., y ±0.15 → ±0.22; keep everything on the real table: world x ∈ [−0.09, 0.84],
   y ∈ [−0.30, 0.32] minus margins for the gripper).
2. Run a 10-episode RASTER pilot first (fast): success rate must stay high — spawns too
   close to the table edge or robot base will fail IK/grasps.
3. Then a 10-episode nyx pilot + `compare_batches.py` + grifflee's read, THEN a batch.

### Task D — Publish

After grifflee approves the batch: copy per `docs/TRAINING_HANDOFF.md` §"Where the data
is" (shared location `/data/store/mcaps/sim/lift` pending mhyatt's confirmation).
Commit any code changes with clear messages on `synthetic-lift-mcap` and push to origin
(grifflee's fork). Do not commit `outputs/` (gitignored) or the `.ply` assets.

## 5. Tool map

| Tool | Purpose |
| --- | --- |
| `scripts/generate_lift_dataset.py` | episodes → MCAP (`generate`), preview PNGs, video mp4; writes manifest.json |
| `scripts/compare_batches.py` | batch format gate + label-distribution report vs real MCAPs |
| `scripts/blink_test.py` | accuracy check: sim vs real photos at calibration configs (cap mode) and vs May wrist frames (wrist mode) |
| `scripts/verify_splat_alignment.py` | nyx render vs calibration photos at the solved splat transform |
| `scripts/align_ransac.py` | re-solve scene alignment from the ZED fused cloud (only after rescan/recalibration) |
| `scripts/clean_splat.py` | regenerate `assets/lab_clean.ply` (crop baked robot/table, cull floaters, flatten SH) |
| `scripts/validate_mcap.py` | single-file MCAP inspector/differ |

## 6. Environment gotchas (will bite you otherwise)

- Renders only refresh on `scene.step()` — `set_qpos` + render without stepping shows
  stale frames. (See `pose_robot` in blink_test.py for the correct pattern.)
- Genesis `add_camera` default `near=0.1` clips the wrist camera's own gripper; the env
  passes `near=0.02`.
- Everything runs headless via EGL on mayo; view images via scp or `mpv --vo=x11`.
- Real MCAPs: `/xarm/robot_states` TCP position is in **millimetres**; gripper norm
  1=open, 0=closed; wrist topic name says "compressed" but payloads are raw rgb8.
- The real episodes (409 files) are from the OLD camera rig — never pixel-compare the
  static cameras against them; `cap.npz` is the current-rig image ground truth.
