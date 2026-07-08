# Inference contract for the 0707 lift checkpoint

Date: 2026-07-08. Findings from inspecting `/data/store/weights/0707_iconic-spaceship-1191`
(500-episode lift model, trained by mhyatt on the `xarm_sim` synthetic dataset) and the
public `mhyatt000/crossformer` repo. Written for wiring `xsim.eval_policy.RemotePolicy`
to a served model.

## UPDATE (later 2026-07-08): questions answered from the fork

The architecture code landed in **grifflee's fork, `~/repo/crossformer`, branch `dev`**
(pio/multiview/xstate all present). Answers to the open questions, from that code:

1. **Serving code**: run `scripts/server.py` from `~/repo/crossformer` (dev). Still needs
   the TASKS patch below.
2. **View order = [low, side, wrist]**. `scripts/data/make/mcap_robot_sim.py` orders views
   by *sorted image-topic name* (`/cam/low/... < /cam/side/... < /camera/camera/...`), and
   `fix_views` (grain/restructure.py) ranks by calibration validity with a *stable* sort —
   sim episodes have all three cameras valid, so the sorted-topic order survives. Roles:
   primary=worm=low, side=side, left_wrist=wrist (`_XGYM_IMG`, embody.py:615).
3. **dof_ids / chunk_steps required**: `PerceiverIOHead` subclasses `XFlowHead`
   (heads/pio.py:30), so every step payload needs `dof_ids` + `chunk_steps`, and actions
   return RAW (unnormalized) in those dof slots. The `single` embodiment
   (ARM_7DOF, GRIPPER, CART_POS, CART_ORI, KP3DC×3 views) = 140 dofs total; the sim client
   should request just arm+gripper — `Embodiment.REGISTRY["single"]` docstring puts
   ARM_7DOF+GRIPPER at ids (1..8); verify at runtime against `embody.py`'s vocab.
   `chunk_steps = np.arange(chunk)` (server warmup uses this; head max_horizon=50).
4. **kp3d keypoints are optional at inference**: built at dataset-build time from GT
   calibration (`robot_keypoints_in_cameras`); at serve time `ObsPaddingWrapper` zero-fills
   missing keys and `XStateEncoder` trained with `input_drop_prob=0.25` — omit them from
   the client and let padding handle it (compute FK keypoints later only if performance
   demands).
5. Action semantics confirmed: `action = proprio.copy()` shifted by horizon
   (grain/restructure.py `multiarray_transforms`) — absolute next-step targets.

**Serving gap still open**: `crossformer/run/server.py` `TASKS` has no `xarm_sim` entry
(dev included) — add e.g. `"lift_sim": {"text": "pick up the red block", "dataset_name":
"xarm_sim"}` on the fork, or the stats lookup fails. Also decide v1 `Policy` vs
`PolicyV2`/`GrainlikeWrapper` serving path — the exact wire-format the client must send
differs (v1 expects example-batch-shaped obs; GrainlikeWrapper accepts raw-arec-shaped
samples and replays the train transforms). Finalize `ObsSpec`/`ActionSpec` defaults in
`xsim.eval_policy` only after picking the serving path.

## Blocker (RESOLVED by the fork — kept for history): the training code was not public

`params/config.json` instantiates modules by import path, and this checkpoint needs:

- `crossformer.model.components.multiview.StackedVitTokenizer`
- `crossformer.model.components.xstate.XStateEncoder`
- `crossformer.model.components.heads.pio.PerceiverIOHead` (flow matching, 50 steps)

None of these exist on ANY public branch of mhyatt000/crossformer (checked main, dev,
2025dec12, debug-arec, wip-mano + GitHub code search: 0 hits). The 0707 run was trained
from unpushed code. **The checkpoint can only be served from that codebase.** Options:

1. mhyatt pushes/shares the training commit; we run `scripts/server.py` there, OR
2. mhyatt serves it on the training machine and we just connect (webpolicy websocket) —
   nothing crossformer-related needs to run on mayo either way.

Also note for whoever serves it: `crossformer/run/server.py` `TASKS["lift"]` maps to
dataset `xgym_lift_single`, but this checkpoint's `dataset_statistics.json` has only
`xarm_sim` — the served policy needs its dataset name pointed at `xarm_sim` or the
proprio-normalization lookup fails.

## Observation contract (from params/example_batch.msgpack)

Batch = `{observation, task}`; `task` is empty (single-task model, no text/goal).

| key | shape (B,T,...) | dtype | notes |
|---|---|---|---|
| `image` | (1,1,**3**,64,64,3) | uint8 | 3 views STACKED on one key, 64x64. View order unconfirmed — ask mhyatt (dataset builder not public) |
| `view_mask` | (1,1,3) | bool | which views present |
| `proprio_joints` | (1,1,7) | f32 | radians |
| `proprio_gripper` | (1,1,1) | f32 | norm, 1=open (training floor 0.32 at close) |
| `proprio_position` | (1,1,3) | f32 | TCP xyz in **meters** (dataset builder converts the MCAP mm) |
| `proprio_orientation` | (1,1,3) | f32 | TCP euler rpy, radians |
| `proprio_kp3dw_robot` | (1,1,14,3) | f32 | 14 robot keypoints, world frame — sim can likely zero-pad (server has ObsPaddingWrapper; XStateEncoder skip_missing=True) |
| `proprio_kp3dc_robot` | (1,1,3,14,3) | f32 | same keypoints per camera frame |
| `state/base`+`state/id`+`state/view`+`mask/state/base` | (1,1,140) | — | XStateEncoder packing; presumably built server-side from proprio parts |

Server v1 (`Policy.preprocess`) normalizes `proprio_*` with `dataset_statistics[ds]["proprio"]`
and resizes `image_*` keys — note the resize dict is built from keys starting `image_`, so the
stacked `image` key may NOT be auto-resized: client should send 64x64 (or confirm wrapper
behavior in the v2 path).

## Action contract

`dataset_statistics.json` action parts: `joints(7)`, `gripper(1)`, `position(3)`,
`orientation(3)`, `kp3dw_robot(14,3)`, `kp3dc_robot(14,3)`. Training doc says action =
next-step proprio (absolute). PerceiverIO/XFlow-style heads want `dof_ids` and
`chunk_steps` in the step payload and return `{"actions": ...}` (chunk, ensembled) in RAW
dof space (no unnormalization). Which `dof_ids` correspond to which action part — ask
mhyatt (max_dofs=140 slotting).

For the sim adapter the natural drive is `joints(7)` absolute + `gripper(1)` norm
(threshold → open / close-to-0.58), i.e. `ActionSpec(mode="joint_abs")`.

## What our side already has

- `scripts/eval_grid.py` verified end-to-end with the scripted weld-free baseline:
  9/9 lift, 9/9 stack on 3x3 grids; resume verified. Full-grid remote eval is
  `--policy remote --host <machine> --port <port> --env.render-backend nyx`.
- `xsim.eval_policy.ObsSpec/ActionSpec` are configurable but their DEFAULTS predate this
  contract (per-camera image keys, flat proprio). Update them to the table above once
  mhyatt confirms view order + dof_ids; do not guess the view order.

## Open questions for mhyatt

1. Which commit/repo serves `0707_iconic-spaceship-1191`? (or: can you serve it and give us host:port?)
2. View order + camera mapping of the stacked `image` key (low/side/wrist → indices 0/1/2?).
3. `dof_ids`/`chunk_steps` payload the server expects, and the returned action layout.
4. Can the sim client omit/zero the kp3d keypoints, or must we compute them (we can — the
   URDF FK is available — but need the 14-keypoint definition)?
