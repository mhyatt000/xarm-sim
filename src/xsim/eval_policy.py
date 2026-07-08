"""Policy adapters for the grid-evaluation harness.

These adapters present one uniform interface so the eval harness can drive either

- a **remote served crossformer** over the webpolicy websocket (:class:`RemotePolicy`), or
- one of the repo's **scripted waypoint policies** (:class:`ScriptedEvalPolicy`), used as
  the weld-free baseline,

without the harness caring which is running. The single interface is :class:`EvalPolicy`:
``reset()`` prepares for a fresh episode, ``act(env)`` observes the env and *applies* one
control-tick action to it, and ``done`` reports whether the policy's trajectory has ended.

Having ``act(env)`` apply the action directly (rather than return one) keeps all
gripper/weld handling inside the adapter: the harness never calls ``env.grasp_lock()`` for
eval, so the scripted baseline runs **weld-free** and relies on finger friction to hold the
cube (per the Phase-0 feasibility test). Both adapters honor a configurable finger
``close_setpoint`` by mutating ``env.robot._gripper_grasp_dof`` (see the notes on each).

Import stays light on purpose: only numpy/torch at module top level. ``webpolicy`` is
imported lazily inside :meth:`RemotePolicy.__init__` (it is not a dependency of this repo),
and the scripted policies are imported lazily inside :meth:`ScriptedEvalPolicy.reset` so
that importing this module never pulls in Genesis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class EvalPolicy(Protocol):
    """Uniform driver interface the eval harness calls."""

    def reset(self) -> None:
        """Prepare for a fresh episode (call *after* the cube has been placed)."""

    def act(self, env) -> None:
        """Observe ``env``, compute one action, and apply it as a single control tick."""

    @property
    def done(self) -> bool:
        """True once the policy's trajectory has ended (remote policies are never done)."""


# The set of proprio channels ObsSpec knows how to assemble. "ee_pose_mm" scales only the
# position (x,y,z) by 1000; the quaternion is left untouched — this matches the real
# /xarm/robot_states TCP pose, which is reported in millimeters.
_PROPRIO_PARTS = (
    "joint_pos", "joint_vel", "joint_eff", "ee_pos", "ee_pose", "ee_pose_mm", "gripper_norm"
)
# CrossFormer xflow DOF ids: j0..j6 + gripper. See grifflee/crossformer:crossformer/embody.py.
_DEFAULT_DOF_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
_DEFAULT_CHUNK_STEPS = tuple(float(i) for i in range(20))


@dataclass
class ObsSpec:
    """Configurable mapping from the sim observation to CrossFormer's webpolicy payload.

    Defaults match ``grifflee/crossformer`` for xgym lift/stack inference:
    ``{"observation": {"image_primary", "image_side", "image_left_wrist", "proprio_single"},
    "dof_ids", "chunk_steps"}``. The proprio vector follows the dataset standardization
    order: TCP xyz in metres, seven arm joints, then gripper norm.
    """

    image_keys: dict = field(
        default_factory=lambda: {
            "low": "image_primary",
            "side": "image_side",
            "wrist": "image_left_wrist",
        }
    )
    proprio_key: str = "proprio_single"
    proprio_parts: tuple[str, ...] = ("ee_pos", "joint_pos", "gripper_norm")
    payload_key: str | None = "observation"
    include_xflow_control: bool = True
    dof_ids: tuple[int, ...] = _DEFAULT_DOF_IDS
    chunk_steps: tuple[float, ...] = _DEFAULT_CHUNK_STEPS

    def build(self, env) -> dict:
        """Render cameras and assemble a CrossFormer webpolicy payload."""
        frames = env.render()
        obs: dict = {}
        for cam_name, obs_key in self.image_keys.items():
            obs[obs_key] = np.ascontiguousarray(frames[cam_name]).astype(np.uint8)
        obs[self.proprio_key] = self._proprio_vector(env)

        payload = obs if self.payload_key is None else {self.payload_key: obs}
        if self.include_xflow_control:
            payload["dof_ids"] = np.asarray(self.dof_ids, dtype=np.int32)
            payload["chunk_steps"] = np.asarray(self.chunk_steps, dtype=np.float32)
        return payload

    def _proprio_vector(self, env) -> np.ndarray:
        joint_pos, joint_vel, joint_eff, ee_pose = env.proprio()
        pieces = []
        for part in self.proprio_parts:
            if part == "joint_pos":
                pieces.append(np.asarray(joint_pos, dtype=np.float64).reshape(-1))
            elif part == "joint_vel":
                pieces.append(np.asarray(joint_vel, dtype=np.float64).reshape(-1))
            elif part == "joint_eff":
                pieces.append(np.asarray(joint_eff, dtype=np.float64).reshape(-1))
            elif part == "ee_pos":
                pieces.append(np.asarray(ee_pose, dtype=np.float64).reshape(-1)[:3])
            elif part == "ee_pose":
                pieces.append(np.asarray(ee_pose, dtype=np.float64).reshape(-1))
            elif part == "ee_pose_mm":
                ee = np.asarray(ee_pose, dtype=np.float64).reshape(-1).copy()
                ee[:3] *= 1000.0  # position m -> mm; quat untouched
                pieces.append(ee)
            elif part == "gripper_norm":
                pieces.append(np.asarray([env.gripper_norm()], dtype=np.float64))
            else:
                raise ValueError(f"unknown proprio part {part!r}; supported: {_PROPRIO_PARTS}")
        return np.concatenate(pieces).astype(np.float32)


@dataclass
class ActionSpec:
    """How to interpret the action the server returns and apply it to the env."""

    mode: Literal["ee_abs", "ee_delta", "joint_abs"] = "joint_abs"
    # key into the returned dict holding the action array/dict; None means the return IS it
    key: str | None = "actions"
    fallback_keys: tuple[str, ...] = (
        "action", "joint_action", "joint_actions", "proprio_single", "vector", "qpos", "joints",
    )
    # If the server returns an action chunk shaped [H, D], execute this row.
    chunk_index: int = 0
    # scale applied to the ee position channels in "ee_abs" (mm->m by default; 1.0 if the
    # server already emits meters). Deltas ("ee_delta") are passed through unscaled.
    ee_pos_scale: float = 0.001
    # which element of the flattened action vector carries the gripper norm; None-able if
    # the server returns the gripper on a separate dict key (gripper_key)
    gripper_index: int | None = -1
    gripper_key: str | None = None
    gripper_open_threshold: float = 0.5  # gripper norm > threshold -> open (1=open)
    close_setpoint: float | None = None  # finger dof when closed; None -> robot's grasp dof


class RemotePolicy:
    """Drives a crossformer served over the webpolicy websocket.

    ``close_setpoint`` handling: :meth:`Manipulator.go_to_goal` / ``apply_action`` hard-code
    the closed finger dof to ``_gripper_grasp_dof``, so to honor a custom setpoint we mutate
    ``env.robot._gripper_grasp_dof`` **once** on the first :meth:`act` after a reset (reset
    itself has no env handle), not per step. ``joint_abs`` reads the same attribute for its
    closed finger target, so all three modes stay consistent.
    """

    def __init__(
        self,
        host: str,
        port: int,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        control_every: int = 4,
        client_factory: Callable[[str, int], object] | None = None,
    ):
        if client_factory is None:
            try:
                from webpolicy.client import Client
            except ImportError as exc:  # webpolicy is not a dependency of xarm-sim
                raise ImportError(
                    "webpolicy is required for RemotePolicy; add it to xarm-sim deps: "
                    "uv add git+https://github.com/mhyatt000/webpolicy"
                ) from exc
            client_factory = Client
        self._client = client_factory(host, port)
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        # the harness steps this policy every ``control_every``-th physics step (the
        # position controller holds the last target in between), so a served model runs
        # at the 30 Hz training cadence (record_every=4 physics steps) rather than 120 Hz
        self.control_every = control_every
        self._setpoint_applied = False

    def reset(self) -> None:
        self._client.reset()
        self._setpoint_applied = False  # re-apply close_setpoint on the next act

    def act(self, env) -> None:
        spec = self.action_spec
        if spec.close_setpoint is not None and not self._setpoint_applied:
            env.robot._gripper_grasp_dof = spec.close_setpoint
            self._setpoint_applied = True

        result = self._client.step(self.obs_spec.build(env))
        action = self._action_vector(result)
        is_open = self._gripper_open(result, action)

        if spec.mode == "ee_abs":
            pos = action[:3] * spec.ee_pos_scale
            quat = action[3:7]
            pose = torch.as_tensor(
                np.concatenate([pos, quat])[None, :], dtype=torch.float32, device=env.device
            )
            env.robot.go_to_goal(pose, open_gripper=is_open)
        elif spec.mode == "ee_delta":
            delta = torch.as_tensor(action[:6][None, :], dtype=torch.float32, device=env.device)
            env.robot.apply_action(delta, open_gripper=is_open)
        elif spec.mode == "joint_abs":
            robot = env.robot
            q_pos = robot._robot_entity.get_qpos()
            q_pos = (q_pos.unsqueeze(0) if q_pos.ndim == 1 else q_pos).clone()
            q_pos[:, robot._arm_dof_idx] = torch.as_tensor(
                action[:7], dtype=q_pos.dtype, device=q_pos.device
            )
            finger = robot._gripper_open_dof if is_open else robot._gripper_grasp_dof
            q_pos[:, robot._fingers_dof] = finger
            robot._robot_entity.control_dofs_position(position=q_pos)
        else:
            raise ValueError(f"unknown action mode {spec.mode!r}")

    @property
    def done(self) -> bool:
        return False

    def _action_vector(self, result) -> np.ndarray:
        if self.action_spec.mode == "joint_abs":
            return self._joint_action_vector(result)
        return self._row_vector(self._action_payload(result))

    def _action_payload(self, result):
        spec = self.action_spec
        if not isinstance(result, dict) or spec.key is None:
            return result
        if spec.key in result:
            return result[spec.key]
        if "joints" in result or "gripper" in result:
            return result
        for key in spec.fallback_keys:
            if key in result:
                return result[key]
        raise KeyError(
            f"action key {spec.key!r} not found; tried fallbacks {spec.fallback_keys}; "
            f"available keys={sorted(result)}"
        )

    def _row_vector(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim >= 2:
            rows = arr.reshape(-1, arr.shape[-1])
            idx = self.action_spec.chunk_index
            if not -len(rows) <= idx < len(rows):
                raise IndexError(f"chunk_index {idx} out of range for action chunk with {len(rows)} rows")
            return rows[idx].reshape(-1)
        return arr.reshape(-1)

    def _joint_action_vector(self, result) -> np.ndarray:
        payload = self._action_payload(result)
        if isinstance(payload, dict):
            if "joints" in payload:
                joints = self._row_vector(payload["joints"])
            else:
                joints = np.asarray([self._row_vector(payload[f"joint{i}"])[0] for i in range(1, 8)])
            if "gripper" in payload:
                gripper = self._row_vector(payload["gripper"])
            elif self.action_spec.gripper_key and self.action_spec.gripper_key in payload:
                gripper = self._row_vector(payload[self.action_spec.gripper_key])
            else:
                gripper = np.asarray([1.0])
            return np.concatenate([joints[:7], [float(gripper[0])]])

        slots = self._row_vector(payload)
        dof_ids = result.get("dof_ids") if isinstance(result, dict) else None
        if dof_ids is not None:
            ids = np.asarray(dof_ids).reshape(-1)
            ordered = np.full(8, np.nan, dtype=np.float64)
            for slot, dof_id in enumerate(ids[: len(slots)]):
                did = int(dof_id)
                if 1 <= did <= 7:
                    ordered[did - 1] = slots[slot]
                elif did == 8:
                    ordered[7] = slots[slot]
            if np.isfinite(ordered).all():
                return ordered

        if slots.size < 8:
            raise ValueError(f"joint_abs action needs 8 values, got shape {np.asarray(payload).shape}")
        return slots[:8]

    def _gripper_open(self, result, action: np.ndarray) -> bool:
        spec = self.action_spec
        if spec.gripper_key is not None and isinstance(result, dict):
            payload = self._action_payload(result)
            if isinstance(payload, dict) and spec.gripper_key in payload:
                g = float(self._row_vector(payload[spec.gripper_key])[0])
            elif spec.gripper_key in result:
                g = float(self._row_vector(result[spec.gripper_key])[0])
            else:
                g = float(action[spec.gripper_index]) if spec.gripper_index is not None else 1.0
        elif spec.gripper_index is not None:
            g = float(action[spec.gripper_index])
        else:
            return True  # no gripper channel configured -> leave the gripper open
        return g > spec.gripper_open_threshold


class ScriptedEvalPolicy:
    """Weld-free wrapper around the repo's scripted lift/stack policies.

    Runs the scripted waypoint trajectory with **no** ``env.grasp_lock()`` call — friction
    holds the cube (Phase-0 feasibility test) — behind the same interface as
    :class:`RemotePolicy`. ``close_setpoint`` is applied by setting
    ``env.robot._gripper_grasp_dof`` in :meth:`reset`, the same knob ``go_to_goal`` reads.

    Contract: the scripted policy caches the cube pose when its own ``reset()`` runs, and the
    harness moves the cube *after* ``env.reset()``. So the inner policy is (re-)constructed
    inside :meth:`reset`, and :meth:`reset` must be called **after** the cube is placed —
    once per episode.
    """

    def __init__(
        self,
        env,
        task: Literal["lift", "stack"],
        steps_per_segment: int = 108,
        grasp_tcp_offset: float = 0.018,
        close_setpoint: float | None = None,
        tail_steps: int = 36,  # ~0.3 s at 120 Hz, matching the generator's release_tail_s=0.3
    ):
        self._env = env
        self.task = task
        self.steps_per_segment = steps_per_segment
        self.grasp_tcp_offset = grasp_tcp_offset
        self.close_setpoint = close_setpoint
        self.tail_steps = tail_steps
        # the scripted waypoint trajectory is authored at the physics rate, so the harness
        # steps it every physics step (unlike RemotePolicy's decimated 30 Hz cadence)
        self.control_every = 1
        self._policy = None
        self._step_count = 0
        self._end_step = 0

    def reset(self) -> None:
        env = self._env
        if self.close_setpoint is not None:
            env.robot._gripper_grasp_dof = self.close_setpoint
        # lazy import keeps this module free of Genesis until an episode actually runs
        if self.task == "lift":
            from xsim.scripted_lift_policy import ScriptedLiftPolicy as Policy
        elif self.task == "stack":
            from xsim.scripted_stack_policy import ScriptedStackPolicy as Policy
        else:
            raise ValueError(f"unknown task {self.task!r}; expected 'lift' or 'stack'")
        # constructed here (not in __init__) so it reads the freshly placed cube pose
        self._policy = Policy(
            env, steps_per_segment=self.steps_per_segment, grasp_tcp_offset=self.grasp_tcp_offset
        )
        self._policy.reset()
        self._step_count = 0
        self._end_step = self._policy.release_step + self.tail_steps

    def act(self, env) -> None:
        cmd = self._policy.step()
        env.robot.go_to_goal(cmd.pose, open_gripper=cmd.open_gripper)
        self._step_count += 1

    @property
    def done(self) -> bool:
        return self._policy is not None and self._step_count > self._end_step
