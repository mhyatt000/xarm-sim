# Grasp Teleport Investigation

Date: 2026-07-08
Branch: eval-suite
Baseline commit before this investigation: `7548fa0 Add weld-free eval grid harness`

## Status

The weld-free no-slip carry problem is solved separately from this issue. With
`TaskEnvCfg.noslip_iterations=10`, setpoint `0.58`, and the existing scripted
lift path, the cube passes the strict carry-slip gate. The remaining issue is a
visible grab-time pop: when the fingers close, the cube appears to jump from the
inner metal/inside part of the gripper into the soft pad/contact location.

This investigation did not find a complete fix. It did rule out several plausible
causes and left default-off diagnostic knobs in `scripts/test_friction_grasp.py`
so the next pass can reproduce the same tests without rebuilding the tooling.

Generated MP4s under `outputs/` are ignored by git and are not part of this
commit. The paths below refer to local artifacts produced during this session.

## Important Distinction

There are two different phenomena:

1. Carry slip: cube motion relative to the TCP after the grasp is established and
   during lift/transport. This is what `slip_mm` and `slip_deg` measure. The
   Genesis `noslip_iterations=10` flag fixes this; the cube remains stable in
   the gripper frame.
2. Grab teleport/pop: a visual contact-settling event during or immediately after
   close, before the carry-slip reference is established. This is not solved by
   no-slip post-processing and is not well represented by the existing slip gate.

The current success metric is therefore correct for carry stability, but it is
not a visual-quality metric for the initial contact acquisition.

## Diagnostic Knobs Added

All defaults preserve the previous behavior.

- `--video-every-steps 1`: captures every physics step at 30 fps, making a 120 Hz
  slow-motion inspection video.
- `--carry-hold-s`: holds the cube before release so carry slip is visually
  inspectable.
- `--pre-lift-hold-s`: holds the fully closed gripper before lift. This tests
  whether extra contact-settle time before vertical motion removes the pop.
- `--lift-slowdown`: multiplies only the lift segment duration. This tests
  whether initial lift acceleration is causing the pop.
- `--ramp-close`: ramps the finger DOF target across the close segment instead of
  jumping directly from open to the grasp target. This tests whether the close
  command discontinuity is causing the pop.
- `--env.robot-decompose-robot-error-threshold 0.0`: forwards an opt-in Genesis
  URDF import setting that forces robot mesh convex decomposition. This tests
  whether coarse gripper collision hulls are causing the pop.

The helper `_go_to_goal_with_finger()` in `scripts/test_friction_grasp.py` is
intentionally diagnostic. It duplicates the relevant `Manipulator.go_to_goal()`
IK call so the script can command an intermediate finger DOF without changing the
main robot control API.

## Tests Run

All runs used seed `50000`, setpoint `0.58`, `--env.noslip-iterations 10`, and
slow-motion capture unless noted.

### Baseline no-slip diagnostic

Video:
`outputs/friction_grasp/diagnostic_noslip/noweld_sp58_ep00.mp4`

Representative metrics from the same configuration:

- `slip_mm=0.25`
- `slip_deg=0.08-0.09`
- `close_step_mm=0.19@396`
- `early_lift_step_mm=1.72@474`
- User visual result: carry slip looked gone, but grab teleport remained.

Interpretation: no-slip fixes friction creep during carry; it does not address
contact acquisition at close.

### Candidate 1: closed pre-lift hold

Command shape:

```bash
.venv/bin/python scripts/test_friction_grasp.py \
  --backend gpu --setpoints 0.58 --n-episodes 1 --videos-per-setpoint 1 \
  --video-dir outputs/friction_grasp/fix1_pre_lift_hold \
  --video-every-steps 1 --video-fps 30 --carry-hold-s 3 \
  --pre-lift-hold-s 0.5 --env.noslip-iterations 10
```

Video:
`outputs/friction_grasp/fix1_pre_lift_hold/noweld_sp58_ep00.mp4`

Metrics:

- `lifted=True delivered=True pass=True`
- `slip_mm=0.25`
- `close_step_mm=0.19@396`
- `pre_lift_hold_step_mm=0.14@514`
- `early_lift_step_mm=1.72@534`
- User visual result: teleport was visibly the same.

Theory tested: the contact solver might need time to settle at full close before
lift.

Failure conclusion: extra time after full close does not remove the visual pop.
The pop either already occurred before the hold, or the stable closed equilibrium
is the visually popped position.

### Candidate 2: slower lift segment

Command shape:

```bash
.venv/bin/python scripts/test_friction_grasp.py \
  --backend gpu --setpoints 0.58 --n-episodes 1 --videos-per-setpoint 1 \
  --video-dir outputs/friction_grasp/fix2_slow_lift \
  --video-every-steps 1 --video-fps 30 --carry-hold-s 3 \
  --lift-slowdown 4.0 --env.noslip-iterations 10
```

Video:
`outputs/friction_grasp/fix2_slow_lift/noweld_sp58_ep00.mp4`

Metrics:

- `lifted=True delivered=True pass=True`
- `slip_mm=0.44`
- `slip_deg=1.19`
- `close_step_mm=0.19@396`
- `early_lift_step_mm=0.39@474`
- User visual result: teleport was exactly the same.

Theory tested: the cube might be popping because the lift segment accelerates too
quickly after close.

Failure conclusion: slower lift reduces the measured early-lift per-step motion,
but the visual pop remains unchanged. Therefore the current `early_lift_step_mm`
metric is not a reliable proxy for the visual artifact. The artifact is probably
at close/contact seating, not caused by vertical lift acceleration.

### Candidate 3: force Genesis robot collision decomposition

Command shape:

```bash
.venv/bin/python scripts/test_friction_grasp.py \
  --backend gpu --setpoints 0.58 --n-episodes 1 --videos-per-setpoint 1 \
  --video-dir outputs/friction_grasp/fix3_decomposed_robot_collision \
  --video-every-steps 1 --video-fps 30 --carry-hold-s 3 \
  --env.noslip-iterations 10 \
  --env.robot-decompose-robot-error-threshold 0.0
```

Video:
`outputs/friction_grasp/fix3_decomposed_robot_collision/noweld_sp58_ep00.mp4`

Metrics:

- `lifted=True delivered=True pass=True`
- `slip_mm=0.25`
- `slip_deg=0.05`
- `close_step_mm=0.26@394`
- `early_lift_step_mm=1.71@474`
- User visual result: teleport was exactly the same.

Theory tested: Genesis' default robot import might be using coarse convex hulls
for detailed gripper meshes, producing an invisible collision surface that pushes
the cube into a visually different pad location.

Failure conclusion: broad robot mesh convex decomposition did not change the
artifact. This does not fully clear gripper collision geometry as a cause, but it
rules out the simple explanation that the default whole-mesh hull is solely
responsible. It also showed that full robot CoACD is expensive and noisy: the run
spent substantial time decomposing meshes and produced many self-collision filter
pairs. This is not a good production path unless narrowed to dedicated gripper
collision proxies.

### Candidate 4: ramp close only

Command shape:

```bash
.venv/bin/python scripts/test_friction_grasp.py \
  --backend gpu --setpoints 0.58 --n-episodes 1 --videos-per-setpoint 1 \
  --video-dir outputs/friction_grasp/fix4_ramped_close \
  --video-every-steps 1 --video-fps 30 --carry-hold-s 3 \
  --ramp-close --env.noslip-iterations 10
```

Video:
`outputs/friction_grasp/fix4_ramped_close/noweld_sp58_ep00.mp4`

Metrics:

- `lifted=False delivered=False pass=False`
- `slip_mm=166.27`
- `slip_deg=5.53`
- `close_step_mm=0.00@-1`
- `early_lift_step_mm=0.61@474`

Theory tested: the instantaneous command jump from open gripper to `0.58` grasp
DOF might be the source of the contact impulse.

Failure conclusion: ramping the close target alone removes the immediate squeeze
impulse but fails the grasp. The fingers do not build enough normal force soon
enough for the existing trajectory.

### Candidate 4b: ramp close, then hold fully closed

Command shape:

```bash
.venv/bin/python scripts/test_friction_grasp.py \
  --backend gpu --setpoints 0.58 --n-episodes 1 --videos-per-setpoint 1 \
  --video-dir outputs/friction_grasp/fix4b_ramped_close_hold \
  --video-every-steps 1 --video-fps 30 --carry-hold-s 3 \
  --ramp-close --pre-lift-hold-s 0.5 --env.noslip-iterations 10
```

Video:
`outputs/friction_grasp/fix4b_ramped_close_hold/noweld_sp58_ep00.mp4`

Metrics:

- `lifted=True delivered=True pass=True`
- `slip_mm=0.24`
- `slip_deg=0.05`
- `pre_lift_hold_step_mm=0.18@462`
- `early_lift_step_mm=1.72@534`
- User visual result: still minor teleporting.

Theory tested: the close target can be made visually smoother if followed by a
short full-close hold to restore normal force before lift.

Failure conclusion: this is the first candidate that may have improved the
severity from obvious to minor, but it is still not a real fix. It also changes
the contact-acquisition dynamics in a way that may not match the real demo
protocol. Do not move this into the generator or eval defaults without more
validation.

## Current Best Theory

The artifact is most likely contact acquisition/seating at the gripper pads, not
carry slip. The cube is being moved by the contact solver into the first stable
closed-gripper equilibrium. Visually, that looks like the cube starts against or
near an inner metal/inside portion of the gripper, then quickly snaps into the
soft pad region.

Important details supporting this theory:

- `noslip_iterations` fixes tangential creep after contact is established, but
  does not change the grab pop.
- Holding fully closed before lift does not help, so the artifact is not caused
  by insufficient settling after close.
- Slowing lift does not help visually, so the pop is not caused by upward
  acceleration.
- Full robot collision decomposition does not help, so a single coarse default
  robot hull is not the whole explanation.
- Ramped close alone fails to grasp; ramped close plus hold restores success but
  still has minor teleporting. This suggests the successful grasp still requires
  the cube to seat into a contact equilibrium, and that seating step remains
  visible.

The old welded data generation path is still valid: generator defaults are not
changed by this investigation, and the weld path locks the cube after the grasp
point. The teleport artifact matters for weld-free eval video quality and for
confidence that eval physics is physically plausible. It is separate from the
MCAP data contract and from the CrossFormer policy adapter.

## What Not To Chase Next

- Do not spend more time on `noslip_iterations` for this artifact. It solved a
  different problem.
- Do not assume early-lift motion metrics prove visual quality. Candidate 2 made
  `early_lift_step_mm` much smaller while the visible pop stayed the same.
- Do not use full-robot CoACD as a production fix. It is expensive and did not
  improve the artifact.
- Do not adopt pure ramped close. It fails the grasp.

## Recommended Next Steps

1. Instrument the close phase directly:
   - log actual finger DOFs, commanded finger DOF, cube pose, TCP pose, left/right
     finger link poses, and cube pose in each finger frame on every physics step;
   - if Genesis exposes contacts, log contact link names, contact normals,
     penetration depth, normal force, and friction force;
   - produce a close-up video centered on the gripper with frame labels for close
     progress, finger DOF, and contact pair names.
2. Test gripper/cube alignment before changing physics:
   - sweep `grasp_tcp_offset` around the current `0.018` m;
   - sweep small lateral offsets in the finger closing direction and orthogonal
     direction;
   - test whether the cube is initially contacting knuckle/inner metal geometry
     before the pad.
3. Build dedicated gripper collision proxies if instrumentation confirms wrong
   first contact:
   - keep the visual STL/OBJ meshes unchanged;
   - replace or augment finger collision with simple pad-aligned boxes/capsules;
   - consider disabling collision on non-pad gripper parts near the cube if they
     are creating the first contact;
   - validate with the same `test_friction_grasp.py` video gates and 40/40 slip
     sweep before touching eval defaults.
4. If a visual-only mitigation is needed before a geometry fix, use ramped close
   plus full-close hold only as a diagnostic/eval-only option. It is not accepted
   yet: the user still saw minor teleporting, and it changes the contact timing.

## Files Touched By This Diagnostic Commit

- `scripts/test_friction_grasp.py`: adds default-off diagnostic knobs and prints
  close/pre-lift/early-lift motion metrics.
- `src/xsim/task_env.py`: adds default-off
  `TaskEnvCfg.robot_decompose_robot_error_threshold`.
- `src/xsim/grasp_env.py`: forwards the optional URDF decomposition setting into
  `gs.morphs.URDF()`.
- `docs/GRASP_TELEPORT_INVESTIGATION.md`: this handoff note.
