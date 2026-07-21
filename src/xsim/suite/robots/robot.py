"""Runtime robot that composes part controllers over a Genesis entity."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from xsim.suite.controllers import Controller, GripperController, JointPositionController
from xsim.suite.models.grippers import GripperModel, gripper_factory
from xsim.suite.models.robots import RobotModel


def _quat_error(cur: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """World-frame angle-axis error (n, 3) of the relative rotation
    ``R_cur R_tgt^-1`` between current and target quats (wxyz). Its derivative
    w.r.t. q is the world-frame angular Jacobian (rows 3:6 of get_jacobian), so
    driving it to zero aligns the EE orientation. Antipode-folded (w>=0) for the
    shortest arc."""
    cw, cx, cy, cz = cur[:, 0], cur[:, 1], cur[:, 2], cur[:, 3]
    # conj(tgt): negate the vector part
    tw, tx, ty, tz = tgt[:, 0], -tgt[:, 1], -tgt[:, 2], -tgt[:, 3]
    # Hamilton product q_err = cur * conj(tgt)
    ew = cw * tw - cx * tx - cy * ty - cz * tz
    ex = cw * tx + cx * tw + cy * tz - cz * ty
    ey = cw * ty - cx * tz + cy * tw + cz * tx
    ez = cw * tz + cx * ty - cy * tx + cz * tw
    v = torch.stack([ex, ey, ez], dim=-1)
    sign = torch.where(ew < 0.0, -1.0, 1.0)     # antipode: shortest arc
    ew = ew * sign
    v = v * sign.unsqueeze(-1)
    vn = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vn.squeeze(-1), ew.clamp(-1.0, 1.0))
    axis = v / vn.clamp_min(1e-8)
    return axis * angle.unsqueeze(-1)


class Robot:
    """Runtime robot: binds the entity created from its RobotModel and composes
    part controllers (arm + gripper from the gripper factory). RobotEnv calls
    control(action); the robot fans the action out to its parts."""

    def __init__(self, robot_model: RobotModel):
        self.model = robot_model
        self.gripper: GripperModel | None = (
            gripper_factory(robot_model.gripper_name) if robot_model.gripper_name else None
        )
        self.entity = None
        self.arm_controller: JointPositionController | None = None
        self.gripper_controller: GripperController | None = None
        self._controllers: list[Controller] = []

    def setup(self) -> None:
        if self.model.entity is None:
            raise RuntimeError(
                f"RobotModel {self.model.name!r} has no bound entity; "
                "was the Task added to a scene?"
            )
        self.entity = self.model.entity
        self._arm_idx = torch.arange(self.model.arm_dofs, device=gs.device)
        self.arm_controller = JointPositionController(
            self.entity,
            self._arm_idx,
            self.model.arm_kp,
            self.model.arm_kv,
            self.model.arm_force_limit,
        )
        if self.gripper is not None:
            self._finger_idx = torch.arange(
                self.model.arm_dofs,
                self.model.arm_dofs + self.gripper.n_dofs,
                device=gs.device,
            )
            self.gripper_controller = GripperController(
                self.entity, self._finger_idx, self.gripper
            )
        self._controllers = [
            c for c in (self.arm_controller, self.gripper_controller) if c is not None
        ]
        for c in self._controllers:
            c.setup()
        self._ee_link = self.entity.get_link(self.model.ee_link_name)
        init = list(self.model.default_arm_qpos) + (
            list(self.gripper.default_dofs) if self.gripper else []
        )
        self._init_qpos = torch.tensor(init, device=gs.device, dtype=gs.tc_float)

    @property
    def action_dim(self) -> int:
        return sum(c.action_dim for c in self._controllers)

    @property
    def action_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.arm_controller.joint_limits
        if self.gripper_controller is not None:
            lo = np.concatenate([lo, np.array([0.0], dtype=np.float64)])
            hi = np.concatenate([hi, np.array([1.0], dtype=np.float64)])
        return lo, hi

    def control(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.action_dim:
            raise ValueError(
                f"action has shape {a.shape}, expected (n_envs, {self.action_dim})"
            )
        offset = 0
        for c in self._controllers:
            c.run(a[:, offset : offset + c.action_dim])
            offset += c.action_dim

    def reset(self, envs_idx=None) -> None:
        self.entity.set_qpos(
            self._init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=False
        )
        for c in self._controllers:
            c.reset()

    def set_arm_qpos(self, q: np.ndarray, envs_idx=None) -> None:
        """Seat the arm at ``q`` (n_envs, arm_dofs) with the gripper at its
        defaults; ``envs_idx`` selects which of the n_envs rows are applied."""
        qpos = self._init_qpos.unsqueeze(0).repeat(q.shape[0], 1).clone()
        qpos[:, : self.model.arm_dofs] = torch.as_tensor(
            q, device=gs.device, dtype=gs.tc_float
        )
        if envs_idx is not None:
            qpos = qpos[torch.as_tensor(np.atleast_1d(envs_idx), device=gs.device)]
        self.entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=False)
        for c in self._controllers:
            c.reset()

    def ik(self, pose: torch.Tensor, from_current: bool = False) -> np.ndarray:
        """Arm joint targets (n_envs, 7) for EE poses [n_envs, 7] = [x,y,z,qw,qx,qy,qz].

        ``from_current`` seeds the solve at each env's live qpos, returning the
        branch nearest the arm's actual configuration — required when the
        result is used as a per-tick regression label (home seeding can jump
        branches, making labels discontinuous in state; see img-v8b).

        Dispatches to ``ik_softcost`` when the robot model selects the
        ``"softcost"`` backend; otherwise the byte-identical Genesis path below.
        """
        if self.model.ik_backend == "softcost":
            return self.ik_softcost(pose, from_current=from_current)
        pose = pose.to(gs.device)
        init_qpos = (
            None
            if from_current
            else self._init_qpos.unsqueeze(0).expand(pose.shape[0], -1)
            if self.model.ik_init_at_home
            else None
        )
        q = self.entity.inverse_kinematics(
            link=self._ee_link,
            pos=pose[:, :3],
            quat=pose[:, 3:7],
            init_qpos=init_qpos,
            max_samples=self.model.ik_max_samples,
            max_solver_iters=self.model.ik_max_solver_iters,
            damping=self.model.ik_damping,
            dofs_idx_local=self._arm_idx,
        )
        return np.asarray(q[:, self._arm_idx].detach().cpu(), dtype=np.float64)

    def ik_softcost(self, pose: torch.Tensor, from_current: bool = True) -> np.ndarray:
        """Batched weighted soft-cost IK (Gauss-Newton / Levenberg-Marquardt).

        Minimises  sum_i || w_i * cost_i(q) ||^2  over the arm joints, stacking
        the cost blocks into one residual r(q) and taking LM steps
        ``dq = -(J^T J + lambda I)^-1 J^T r`` (J = d r / d q, per-block). Cost
        blocks (weights on ``self.model``):
          - pose position   : w_pos * (FK_pos(q) - target_pos)        [3]
          - pose orientation: w_rot * angleaxis(R_cur R_tgt^-1)       [3]
          - home rest-pose  : w_home * (q - q_home)                   [n_arm]
          - joint limits     : w_limit * signed_relu_barrier(q)       [n_arm]
          - manipulability   : optional nullspace ascent of sqrt(det(J J^T))
            added to the step (w_manip>0; approximate, off by default).
        The home block replaces a hard nullspace projection: it is the "stay near
        the rest pose" preference balanced softly in the same least-squares, so
        EE-pose -> joint-target is a near-single-valued continuous function.

        Contract matches ``ik``: EE poses [n_envs, 7] = [x,y,z,qw,qx,qy,qz] ->
        arm joint targets (n_envs, n_arm) np.float64. Fully batched over n_envs.

        The Genesis Jacobian (``get_jacobian`` -> (n_envs, 6, n_dofs), rows
        [linear(3), angular(3)] in world frame) is evaluated at the live sim
        state, so each iteration seats the iterate via ``set_qpos`` and reads the
        EE link pose; the original qpos/qvel are restored on exit (transparent to
        the caller's physics).
        """
        m = self.model
        pose = pose.to(device=gs.device, dtype=gs.tc_float)
        B = pose.shape[0]
        arm = self._arm_idx
        n = int(arm.shape[0])
        tgt_pos = pose[:, :3]
        tgt_quat = pose[:, 3:7]
        tgt_quat = tgt_quat / tgt_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        lo, hi = self.entity.get_dofs_limit(arm)
        lo = lo.to(device=gs.device, dtype=gs.tc_float).reshape(-1)
        hi = hi.to(device=gs.device, dtype=gs.tc_float).reshape(-1)
        q_home = self._init_qpos[:n].to(device=gs.device, dtype=gs.tc_float)

        # save full state to restore on exit (IK must not perturb the sim)
        q_full0 = self.entity.get_qpos().clone()
        v_full0 = self.entity.get_dofs_velocity().clone()

        q = (q_full0[:, arm].clone() if from_current
             else q_home.unsqueeze(0).expand(B, -1).contiguous())
        qf = q_full0.clone()

        wp, wr = float(m.ik_w_pos), float(m.ik_w_rot)
        wh, wl, wm = float(m.ik_w_home), float(m.ik_w_limit), float(m.ik_w_manip)
        lam = float(m.ik_sc_damping)
        eye = torch.eye(n, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        margin = 0.05      # rad, joint-limit barrier onset before the hard limit
        max_step = 0.5     # per-iter |dq| clamp (rad), mirrors Genesis max_step_size

        for _ in range(int(m.ik_iters)):
            qf[:, arm] = q
            self.entity.set_qpos(qf, zero_velocity=False, skip_forward=False)
            J = self.entity.get_jacobian(self._ee_link)[:, :, arm]  # (B,6,n)
            Jp, Jr = J[:, :3, :], J[:, 3:, :]
            cur_pos = self._ee_link.get_pos().reshape(B, 3)
            cur_quat = self._ee_link.get_quat().reshape(B, 4)

            e_pos = cur_pos - tgt_pos                        # (B,3)
            e_rot = _quat_error(cur_quat, tgt_quat)          # (B,3) world angle-axis
            over = (q - (hi - margin)).clamp(min=0.0)        # (B,n) past upper margin
            under = ((lo + margin) - q).clamp(min=0.0)       # (B,n) past lower margin
            active = ((over > 0) | (under > 0)).to(gs.tc_float)

            # normal equations: H = sum_k Jk^T Jk, g = sum_k Jk^T rk (weights^2)
            H = (wp * wp) * (Jp.transpose(1, 2) @ Jp)
            H = H + (wr * wr) * (Jr.transpose(1, 2) @ Jr)
            H = H + (wh * wh) * eye
            H = H + (wl * wl) * torch.diag_embed(active)
            g = (wp * wp) * torch.einsum("bij,bi->bj", Jp, e_pos)
            g = g + (wr * wr) * torch.einsum("bij,bi->bj", Jr, e_rot)
            g = g + (wh * wh) * (q - q_home)
            g = g + (wl * wl) * (over - under)

            dq = -torch.linalg.solve(H + lam * eye, g.unsqueeze(-1)).squeeze(-1)
            if wm > 0.0:  # approximate manipulability ascent in the pose nullspace
                dq = dq + wm * self._manip_step(q, qf, arm, J)
            dq = dq.clamp(-max_step, max_step)
            q = (q + dq).clamp(lo, hi)

        # restore the sim exactly as we found it
        self.entity.set_qpos(q_full0, zero_velocity=False, skip_forward=False)
        self.entity.set_dofs_velocity(v_full0, skip_forward=False)
        return np.asarray(q.detach().cpu(), dtype=np.float64)

    def _manip_step(self, q, qf, arm, J):
        """Approximate manipulability-ascent step: nullspace-projected numerical
        gradient of log sqrt(det(J J^T)). Perturbs each arm joint, re-seats the
        sim and re-reads the Jacobian — O(n_arm) extra evaluations per iter, off
        by default (w_manip=0). Approximate: finite differences, first-order
        nullspace projection. Leaves ``qf`` re-seated at ``q`` on return."""
        n = int(arm.shape[0])
        B = q.shape[0]

        def logmu(Jm):  # log sqrt(det(J J^T)) with a floor for near-singular J
            A = Jm @ Jm.transpose(1, 2)
            det = torch.linalg.det(A).clamp_min(1e-12)
            return 0.5 * torch.log(det)

        base = logmu(J)
        eps = 1e-3
        grad = torch.zeros(B, n, device=gs.device, dtype=gs.tc_float)
        for j in range(n):
            qf[:, arm[j]] = q[:, j] + eps
            self.entity.set_qpos(qf, zero_velocity=False, skip_forward=False)
            Jj = self.entity.get_jacobian(self._ee_link)[:, :, arm]
            grad[:, j] = (logmu(Jj) - base) / eps
            qf[:, arm[j]] = q[:, j]
        # project onto the nullspace of the pose task: N = I - J^+ J
        eye = torch.eye(n, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        Jpinv = torch.linalg.pinv(J)
        N = eye - Jpinv @ J
        return torch.einsum("bij,bj->bi", N, grad)

    @property
    def joint_positions(self) -> np.ndarray:
        q = np.asarray(self.entity.get_dofs_position().detach().cpu())
        return q[:, : self.model.arm_dofs]

    @property
    def joint_velocities(self) -> np.ndarray:
        v = np.asarray(self.entity.get_dofs_velocity().detach().cpu())
        return v[:, : self.model.arm_dofs]

    @property
    def ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_pos().detach().cpu())

    @property
    def ee_quat(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_quat().detach().cpu())

    @property
    def ee_vel(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_vel().detach().cpu())

    @property
    def gripper_norm(self) -> np.ndarray:
        q = np.asarray(self.entity.get_dofs_position().detach().cpu())
        if self.gripper is None:
            return np.ones(q.shape[0], dtype=np.float64)
        g = q[:, self.model.arm_dofs]
        return np.clip(1.0 - g / self.gripper.close_dof, 0.0, 1.0)
