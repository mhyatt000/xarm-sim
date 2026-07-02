# Agent Handoff — state as of 2026-07-02

## Where things stand (all verified, all committed on `synthetic-lift-mcap`)

Pipeline generates photoreal xArm7 block-lift episodes as Foxglove MCAP, format-identical
to `/data/store/mcaps/single/lift`. Everything through pilot verification is DONE:

- **Alignment**: splat/world/cameras solved and verified — robot silhouette matches real
  calibration photos at median 1.9 px (`scripts/blink_test.py`, `outputs/blink_test/`).
  Constants: `DEFAULT_SPLAT_POS/QUAT/SCALE` in `src/xsim/lift_task.py`. Do not re-tune by
  eye; re-solve with `scripts/align_ransac.py` only after a rescan/recalibration.
- **gs-nyx gotcha (upstream bug, worked around)**: the scene exporter converts meshes and
  cameras z-up→y-up but passes light fields raw; `_make_light_field` applies the same
  conversion. Do not remove those two conversion calls.
- **Splat asset**: use `assets/lab_clean.ply` (regenerate with `scripts/clean_splat.py`;
  it crops the baked robot/table volume, culls giant floaters, flattens SH).
- **Demo style**: scripted policy tuned to match the real demos' profiles (low IK-solved
  ready pose, 0.09 m lift, grasp ~55-60% progress, tempo jitter). Verified in
  `outputs/batch_report/report_pilot_v2.png` (sim tracks real median TCP/gripper).
- **Pilot v2**: 10/10 success, format gate PASS — `outputs/sim_mcap/pilot_v2/` + manifest.
- **Gripper**: `gripper_grasp_dof: 0.58`. 0.53 matches the real recorded close depth but
  the rigid sim fingers drop the cube (tested 0/3) — don't "fix" it back.

## Hard constraints (from grifflee)

- **Table: `table_mode="slab"` (default). Never `plane` for testing or generation.**
- Cameras must stay at the calibrated extrinsics (`view_from_c2w_cv` values) — never
  jitter them; label correctness depends on them.
- Human visual checkpoints: show grifflee labeled artifacts before building on any new
  scene/vision interpretation.

## Next steps (in order)

1. **Scale-up batch** (grifflee picks N; ~70 s/episode on the 5090):
   `uv run python scripts/generate_lift_dataset.py --n-episodes N --env.render-backend nyx --out-dir outputs/sim_mcap/batch_v1 --seed 1000`
2. **Verify it**: `uv run python scripts/compare_batches.py --sim-dir outputs/sim_mcap/batch_v1`
   (hard format gate + distribution report PNG; grifflee reviews).
3. Optional coverage improvement flagged in the report: sim joint1 is narrower than real —
   widen `LiftEnvCfg.rectangle_y` (and maybe `rectangle_x`) modestly, re-run a 10-ep pilot
   + report before adopting.
4. Copy to shared store once mhyatt confirms: see `docs/TRAINING_HANDOFF.md` (also has
   the calibrated-vs-guessed inventory and the eval caveats — read before training).

## Known residual approximations

Wrist camera mount is a validated guess (~40 px vs real May frames); intrinsics are the
calibration's fx=fy=515 model; recorded gripper close floor 0.32 vs real 0.37; lighting
tuned by eye (cube color recently darkened — check it reads red, not salmon, in batch_v1).
