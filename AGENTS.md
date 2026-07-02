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

## 3. Your tasks, in order

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

## 4. Tool map

| Tool | Purpose |
| --- | --- |
| `scripts/generate_lift_dataset.py` | episodes → MCAP (`generate`), preview PNGs, video mp4; writes manifest.json |
| `scripts/compare_batches.py` | batch format gate + label-distribution report vs real MCAPs |
| `scripts/blink_test.py` | accuracy check: sim vs real photos at calibration configs (cap mode) and vs May wrist frames (wrist mode) |
| `scripts/verify_splat_alignment.py` | nyx render vs calibration photos at the solved splat transform |
| `scripts/align_ransac.py` | re-solve scene alignment from the ZED fused cloud (only after rescan/recalibration) |
| `scripts/clean_splat.py` | regenerate `assets/lab_clean.ply` (crop baked robot/table, cull floaters, flatten SH) |
| `scripts/validate_mcap.py` | single-file MCAP inspector/differ |

## 5. Environment gotchas (will bite you otherwise)

- Renders only refresh on `scene.step()` — `set_qpos` + render without stepping shows
  stale frames. (See `pose_robot` in blink_test.py for the correct pattern.)
- Genesis `add_camera` default `near=0.1` clips the wrist camera's own gripper; the env
  passes `near=0.02`.
- Everything runs headless via EGL on mayo; view images via scp or `mpv --vo=x11`.
- Real MCAPs: `/xarm/robot_states` TCP position is in **millimetres**; gripper norm
  1=open, 0=closed; wrist topic name says "compressed" but payloads are raw rgb8.
- The real episodes (409 files) are from the OLD camera rig — never pixel-compare the
  static cameras against them; `cap.npz` is the current-rig image ground truth.
