"""Drive the rigid MANO hands from MANO-style 3D keypoints (teleop).

The hand is assets/mano/mano_hand_planar.urdf (20 dof; per finger MCP flex+abd,
PIP, DIP). Joint angles are fit so the hand reaches the 21 MANO/OpenPose
keypoints (wrist + [mcp,pip,dip,tip] per finger); that fit IS the retargeting --
per-finger Gauss-Newton IK on <=4 dofs. The rig (rest joints + axes) is parsed
straight from the URDF; joint ids are the kp indices themselves.

World frame: kp3d is PnP-lifted into the camera frame and mapped through the
teleop-camera extrinsic (eye (0,0,0.36) looking down -x, retarget.py's default),
so hands live at their TRUE world positions -- no recentering. --record renders
through the suite's madrona batch raytracer with the lab gsplat composited as
background, from a camera at that same physical pose, so renders are directly
comparable to the raw webcam video.

Control split:
  wrist   6-dof actuated base chain (3 prismatic + 3 revolute, prepended by the suite's
          build_floating_urdf) position-controlled with stiffness+damping toward the
          tracked palm pose (pos + intrinsic-XYZ euler, 2pi-unwrapped)
  fingers control_dofs_position to the IK'd angles

  python scripts/teleop_hand.py selftest          # numeric IK round-trip, no sim
  python scripts/teleop_hand.py demo --record out.mp4
  python scripts/teleop_hand.py live --vis        # webcam -> WiLoR server -> both hands
  python scripts/teleop_hand.py live --no-sim --dump-kp caps.npz   # tracking debug
  python scripts/teleop_hand.py live --replay caps.npz --vis       # offline sim debug
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MANO_DIR = REPO / "assets" / "mano"
MANO_LEFT_DIR = REPO / "assets" / "mano_left"
# idle poses in front of the teleop camera (which looks down -x from the origin)
HAND_POS = (-0.45, 0.0, 0.32)
LIVE_POS = {"right": (-0.45, -0.15, 0.32), "left": (-0.45, 0.15, 0.32)}

# the physical teleop camera: eye at (0,0,0.36) looking down world -x, level
# (matches retarget.py's DEFAULT_WORLD_FROM_CAM_FLU); fov from fx=515 @ 640x480
CAM_POS, CAM_LOOKAT, CAM_UP = (0.0, 0.0, 0.36), (-1.0, 0.0, 0.36), (0.0, 0.0, 1.0)
CAM_RES = (640, 480)
CAM_FOV = math.degrees(2 * math.atan((CAM_RES[1] / 2) / 515.0))

# camera FLU -> world (retarget.py default)
DEFAULT_WORLD_FROM_CAM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.36],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# opencv RDF camera coords -> ROS FLU
FLU2RDF = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Joint ids ARE the 21 MANO/OpenPose kp indices; every finger drives MCP(flex+abd),
# PIP(flex), DIP(flex). The last chain entry is the fingertip (undriven target).
FINGERS = {
    "thumb":  {"chain": [0, 1, 2, 3, 4],     "drive": {1: 2, 2: 1, 3: 1}},
    "index":  {"chain": [0, 5, 6, 7, 8],     "drive": {5: 2, 6: 1, 7: 1}},
    "middle": {"chain": [0, 9, 10, 11, 12],  "drive": {9: 2, 10: 1, 11: 1}},
    "ring":   {"chain": [0, 13, 14, 15, 16], "drive": {13: 2, 14: 1, 15: 1}},
    "pinky":  {"chain": [0, 17, 18, 19, 20], "drive": {17: 2, 18: 1, 19: 1}},
}
KP_TO_JOINT = list(range(21))
# kp joint id -> URDF link number (mind the ring/pinky link-number swap)
KP_LINK = {
    1: 13, 2: 14, 3: 15,
    5: 1, 6: 2, 7: 3,
    9: 4, 10: 5, 11: 6,
    13: 10, 14: 11, 15: 12,
    17: 7, 18: 8, 19: 9,
}
TIP_LINK = {4: "link_15_r_thumb3", 8: "link_03_r_index3", 12: "link_06_r_middle3",
            16: "link_12_r_ring3", 20: "link_09_r_pinky3"}


def _obj_vertices(path):
    vs = [l.split()[1:4] for l in Path(path).read_text().splitlines() if l.startswith("v ")]
    return np.asarray(vs, dtype=float)


_RIG = None


def mano_rig():
    """(joints, flex, abd), each (21,3) indexed by kp id, parsed straight from the
    right-hand URDF: all origins have rpy=0, so rest joint positions are cumulative
    origin sums and the explicit <axis> vectors are the rest-frame flex/abd axes.
    Tips (no URDF frames) come from each distal mesh's farthest vertex."""
    global _RIG
    if _RIG is not None:
        return _RIG
    import xml.etree.ElementTree as ET

    root = ET.parse(MANO_DIR / "mano_hand_planar.urdf").getroot()
    origin, axis = {}, {}
    for j in root.findall("joint"):
        origin[j.get("name")] = np.fromstring(j.find("origin").get("xyz"), sep=" ")
        a = np.fromstring(j.find("axis").get("xyz"), sep=" ")
        axis[j.get("name")] = a / np.linalg.norm(a)

    joints, flex, abd = (np.zeros((21, 3)) for _ in range(3))
    for spec in FINGERS.values():
        p = np.zeros(3)
        for jid in spec["chain"][1:-1]:  # driven joints; the last chain entry is the tip
            name = f"link_{KP_LINK[jid]:02d}"
            p = p + origin[f"{name}_flex"]
            joints[jid] = p
            flex[jid] = axis[f"{name}_flex"]
            if spec["drive"].get(jid) == 2:
                abd[jid] = axis[f"{name}_abd"]
        tip_id = spec["chain"][-1]
        v = _obj_vertices(MANO_DIR / "links_planar" / f"{TIP_LINK[tip_id]}.obj")
        joints[tip_id] = p + v[np.argmax(np.linalg.norm(v, axis=1))]
    _RIG = (joints, flex, abd)
    return _RIG


# --------------------------------------------------------------------------- FK / IK
def _rot(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _fk_clean(name, angles, joints, flex, abd):
    """Pose one finger's joint positions from its driven angles. `angles` maps
    joint_id -> (flex,) or (flex, abd); rotations accumulate down the chain, each
    applied about the joint's own axes carried into the accumulated frame. The MCP
    composes flex -> abd, matching the URDF's flex-parent-of-abd joint chain."""
    chain = FINGERS[name]["chain"]
    out = {chain[0]: joints[chain[0]].copy()}
    R = np.eye(3)
    for i in range(1, len(chain)):
        j, jp = chain[i], chain[i - 1]
        if j in angles:
            a = angles[j]
            Rj = _rot(R @ flex[j], a[0])
            if len(a) > 1:
                Rj = Rj @ _rot(R @ abd[j], a[1])
            R = Rj @ R
        out[j] = out[jp] + R @ (joints[j] - joints[jp])
    return out


def ik_finger(name, target, joints, flex, abd, iters=40, x0=None, tol=1e-3):
    """Gauss-Newton fit of a finger's driven angles so its FK joint positions match
    `target` (dict joint_id -> xyz). This is the 'best fit to reach kp3d'. `x0`
    warm-starts from the previous frame's solution; `tol` (m) is an early exit."""
    spec = FINGERS[name]
    dofs = [(j, d) for j, d in spec["drive"].items()]  # (joint, n_dof)
    x = np.zeros(sum(d for _, d in dofs)) if x0 is None else np.asarray(x0, dtype=float).copy()

    def unpack(x):
        angles, k = {}, 0
        for j, d in dofs:
            angles[j] = x[k : k + d]
            k += d
        return angles

    tj = [j for j in spec["chain"][1:] if j in target]

    def resid(x):
        p = _fk_clean(name, unpack(x), joints, flex, abd)
        return np.concatenate([p[j] - target[j] for j in tj])

    for _ in range(iters):
        r = resid(x)
        if np.linalg.norm(r) < tol:
            break
        J = np.zeros((len(r), len(x)))
        for k in range(len(x)):
            dx = np.zeros_like(x)
            dx[k] = 1e-4
            J[:, k] = (resid(x + dx) - r) / 1e-4
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + np.clip(step, -0.5, 0.5)
    return unpack(x), float(np.linalg.norm(resid(x))), x


def retarget(kp21, joints, flex, abd, warm=None):
    """21 MANO keypoints -> {joint_id: angles} for all fingers. `warm` (dict, mutated)
    carries each finger's packed solution across frames for warm-started IK."""
    target = {jid: kp21[k] for k, jid in enumerate(KP_TO_JOINT)}
    out, err = {}, 0.0
    for name in FINGERS:
        x0 = warm.get(name) if warm is not None else None
        ang, e, x = ik_finger(name, target, joints, flex, abd, x0=x0)
        if warm is not None:
            warm[name] = x
        out.update(ang)
        err = max(err, e)
    return out, err


# --------------------------------------------------------------------------- selftest
def selftest():
    joints, flex, abd = mano_rig()
    rng = np.random.default_rng(1)
    worst = 0.0
    for name in FINGERS:
        gt = {j: rng.uniform(-0.2, 0.8, size=d) for j, d in FINGERS[name]["drive"].items()}
        posed = _fk_clean(name, gt, joints, flex, abd)
        ang, err, _ = ik_finger(name, posed, joints, flex, abd)
        derr = max(np.abs(np.concatenate([ang[j] - gt[j] for j in gt])))
        worst = max(worst, err)
        print(f"  {name:7s} kp_fit_err={err * 1e3:6.3f} mm   max_angle_err={np.degrees(derr):5.2f} deg")
    print(f"worst keypoint fit error: {worst * 1e3:.3f} mm")

    # decompose round-trip, right + mirrored-left rest rigs: a rigidly-moved rest hand
    # must come back as exactly the rest keypoints and exactly the applied rotation
    for side, mirror in (("right", 1.0), ("left", -1.0)):
        rest = joints[KP_TO_JOINT] * np.array([mirror, 1.0, 1.0])
        o_rest, R_rest = palm_frame(rest)
        Rw = _rot(np.array([0.3, -0.5, 0.8]), 0.7)
        moved = rest @ Rw.T + np.array([0.4, -0.1, 0.35])
        local, _o_in, quat = decompose(moved, o_rest, R_rest)
        kp_err = np.abs(local - rest).max()
        q_err = np.abs(quat - _R_to_quat(Rw)).max()
        print(f"  decompose {side:5s} kp_err={kp_err:.2e} m   quat_err={q_err:.2e}")
        assert kp_err < 1e-9 and q_err < 1e-9, f"decompose round-trip failed for {side}"


# --------------------------------------------------------------------------- frames
def palm_frame(kp):
    """World->local rotation and origin from the 21 keypoints. Deterministic (no SVD
    sign ambiguity): fingers axis from wrist->MCPs, side axis from index->pinky MCP,
    so the same convention holds for a left hand and never flips frame-to-frame."""
    wrist = kp[0]
    mcp = kp[[5, 9, 13, 17]]  # index/middle/ring/pinky MCP
    fdir = mcp.mean(0) - wrist
    fdir /= np.linalg.norm(fdir) + 1e-12
    side = kp[5] - kp[17]  # index MCP -> pinky MCP
    side = side - side.dot(fdir) * fdir
    side /= np.linalg.norm(side) + 1e-12
    palm_n = np.cross(fdir, side)
    R = np.stack([side, palm_n, fdir])  # rows = local basis in world coords (world->local)
    return wrist, R


def decompose(kp, o_rest, R_rest):
    """One world-frame kp21 -> (finger kp in the rig's rest frame, wrist pos, wrist quat).
    The quat is the world rotation taking the rest-built hand to the tracked pose
    (identity when the tracked palm frame matches the rig's rest palm frame)."""
    o_in, R_in = palm_frame(kp)
    local = ((kp - o_in) @ R_in.T) @ R_rest + o_rest
    return local, o_in, _R_to_quat(R_in.T @ R_rest)


def make_kp_sequence(joints, flex, abd, n):
    """Author a MANO-like kp3d trajectory (open -> fist) plus a wrist wobble. Keypoints
    are already in the rest/wrist-local frame (FK from rest joints), so no palm-frame
    alignment is needed -- the wrist target pose is returned separately."""
    seq = []
    for t in range(n):
        phase = 0.5 - 0.5 * np.cos(2 * np.pi * t / n)  # 0..1..0
        kp = np.zeros((21, 3))
        for name in FINGERS:
            ang = {}
            for j, d in FINGERS[name]["drive"].items():
                base = 1.3 if j != list(FINGERS[name]["drive"])[0] else 0.9
                ang[j] = np.array([base * phase] + ([0.0] if d > 1 else []))
            posed = _fk_clean(name, ang, joints, flex, abd)
            for k, jid in enumerate(KP_TO_JOINT):
                kp[k] = posed.get(jid, joints[jid])
        R = _rot(np.array([0, 0, 1.0]), 0.4 * np.sin(2 * np.pi * t / n))
        trans = np.array([0.04 * np.sin(2 * np.pi * t / n), 0.0, 0.03 * phase])
        seq.append((kp, np.array(HAND_POS) + trans, _R_to_quat(R)))
    return seq


def kp_sequence_from_file(path, steps, joints):
    """Load real (T,21,3) MANO/OpenPose keypoints (world frame). Each frame yields
    (local_kp, wrist_pos, wrist_quat): fingers expressed in the REST frame the IK
    expects, the wrist pose driving the free base. Recentered so the first wrist sits
    at the demo build pose (demo is a sandbox; live/replay use true world coords)."""
    kp_all = np.load(path)
    kp_all = kp_all["kp"] if hasattr(kp_all, "files") else kp_all
    kp_all = np.asarray(kp_all, dtype=np.float64)
    assert kp_all.ndim == 3 and kp_all.shape[1:] == (21, 3), f"expected (T,21,3), got {kp_all.shape}"
    if steps and steps < len(kp_all):
        kp_all = kp_all[:: max(1, len(kp_all) // steps)][:steps]
    kp_all = kp_all - (kp_all[0, 0] - np.array(HAND_POS))

    o_rest, R_rest = palm_frame(joints[KP_TO_JOINT])  # rest hand's own palm frame, in rig coords
    return [decompose(kp, o_rest, R_rest) for kp in kp_all]


class SimHand:
    """One hand in the scene: rig data, rigid entity, per-frame command state.

    The left hand uses assets mirrored across x=0 (generated once). Finger IK always
    runs on the right-hand rig: left keypoints are expressed in the left rest frame,
    reflected across x, solved, and the angles applied verbatim -- valid because the
    mirrored URDF's joint axes are the reflection-conjugated axes
    ((x,y,z) -> (x,-y,-z)), so angle values transfer."""

    def __init__(self, side, pos, ema_n=4):
        self.side, self.pos = side, np.asarray(pos, dtype=float)
        self.ik_joints, self.flex, self.abd = mano_rig()
        self.out = MANO_DIR
        if side == "left":
            from xsim.suite.models.robots.mano import mirror_mano_assets

            mirror_mano_assets(MANO_DIR, MANO_LEFT_DIR)  # idempotent
            self.out = MANO_LEFT_DIR
        mirror = np.array([1.0 if side == "right" else -1.0, 1.0, 1.0])
        rest_kp = self.ik_joints[KP_TO_JOINT] * mirror
        self.o_rest, self.R_rest = palm_frame(rest_kp)
        self._rig_spans = np.linalg.norm(rest_kp[[5, 9, 13, 17]] - rest_kp[0], axis=1)  # wrist->MCP
        self.alpha = 2.0 / (ema_n + 1.0)
        self.warm, self.ema, self.misses = {}, None, 0
        self.scale = None  # observed-hand / rig palm-size ratio, estimated online
        self._prev_quat = None  # last commanded wrist quat (hand-swap gate)
        self.targets = None  # (finger qpos targets, 6-dof base qpos targets)

    def add_to(self, scene):
        import genesis as gs

        from xsim.suite.models.robots.mano import build_floating_urdf

        # 6-dof actuated base chain (base_0..5) prepended to the hand URDF: the global
        # pose is an ordinary position-controlled joint group with stiffness+damping,
        # not a kinematic teleport (which would zero dof velocities and needs rate
        # limits to stay stable). Morph at the origin so base_0..2 ARE world coords.
        self.entity = scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1.0),
            morph=gs.morphs.URDF(file=str(build_floating_urdf(self.out / "mano_hand_planar.urdf")), fixed=True, convexify=False),
        )
        return self.entity

    def setup(self):
        """Post-build: dof maps and PD gains (call after scene.build)."""
        rigid = self.rigid = self.entity
        self.name_to_dofs = {j.name: list(j.dofs_idx_local) for j in rigid.joints}
        self.base_dofs = [self.name_to_dofs[f"base_{i}"][0] for i in range(6)]
        self.finger_dofs = [d for d in range(rigid.n_dofs) if d not in self.base_dofs]
        self.fmap = {d: i for i, d in enumerate(self.finger_dofs)}
        rigid.set_dofs_kp(np.full(len(self.finger_dofs), 14.0), self.finger_dofs)
        rigid.set_dofs_kv(np.full(len(self.finger_dofs), 1.4), self.finger_dofs)
        # the suite ManoR gains: gravity-compensated, so they shape tracking, not droop
        rigid.set_dofs_kp(np.array([400.0, 400.0, 400.0, 30.0, 30.0, 30.0]), self.base_dofs)
        rigid.set_dofs_kv(np.array([40.0, 40.0, 40.0, 2.5, 2.5, 2.5]), self.base_dofs)
        lo, hi = rigid.get_dofs_limit(self.finger_dofs)
        self.f_lo, self.f_hi = np.asarray(lo.cpu()), np.asarray(hi.cpu())
        # start at the idle pose, targets held there until the hand is first tracked
        base0 = np.concatenate([self.pos, np.zeros(3)])
        rigid.set_dofs_position(base0, self.base_dofs)
        self._prev_base = base0.copy()
        self.targets = (np.zeros(len(self.finger_dofs)), base0)

    def set_local(self, kp_local, tgt_pos, tgt_quat):
        """Finger kp already in this rig's rest frame + explicit wrist pose -> targets."""
        kp_ik = kp_local * np.array([-1.0, 1.0, 1.0]) if self.side == "left" else kp_local
        ang, err = retarget(kp_ik, self.ik_joints, self.flex, self.abd, warm=self.warm)
        ftgt = np.zeros(len(self.finger_dofs))
        for jid, a in ang.items():
            base = f"link_{KP_LINK[jid]:02d}"
            for nm, val in ((f"{base}_flex", a[0]), (f"{base}_abd", a[1] if len(a) > 1 else None)):
                if val is not None and nm in self.name_to_dofs:
                    ftgt[self.fmap[self.name_to_dofs[nm][0]]] = val
        # noisy keypoints can walk the unconstrained IK far past anatomy; the joint
        # limits are the contract with the solver
        ftgt = np.clip(ftgt, self.f_lo, self.f_hi)
        # base chain composes Rx(q3) Ry(q4) Rz(q5): intrinsic-XYZ euler of the wrist
        # rotation, unwrapped toward the previous target for 2pi continuity
        eul = _euler_xyz(_quat_to_R(np.asarray(tgt_quat, dtype=float)), self._prev_base[3:])
        base_tgt = np.concatenate([np.asarray(tgt_pos, dtype=float), eul])
        self._prev_base = base_tgt.copy()
        self.targets = (ftgt, base_tgt)
        return err

    def command(self, kp):
        """World-frame kp21 (None on a missed detection) -> targets. Holds the last
        targets through dropouts and misdetections."""
        if kp is None:
            return self._miss()
        fresh = self.ema is None
        # jump gate widens with consecutive misses: the hand really does move during a
        # dropout, and a fixed gate would reject the re-acquisition until the ema reset
        if not fresh and np.linalg.norm(kp[0] - self.ema[0]) > 0.25 + 0.02 * self.misses:
            return self._miss()  # implausible wrist jump: mislabeled/duplicate detection
        self.ema = kp if fresh else self.alpha * kp + (1 - self.alpha) * self.ema
        local, o_in, quat = decompose(self.ema, self.o_rest, self.R_rest)
        # hand-scale normalization: the tracked hand's palm span vs the rig's (MANO
        # template palms run close to 1.0, WiLoR shape variation still shows up)
        s = float(np.median(np.linalg.norm(local[[5, 9, 13, 17]] - local[0], axis=1) / self._rig_spans))
        self.scale = s if self.scale is None else 0.05 * s + 0.95 * self.scale
        local = local[0] + (local - local[0]) / self.scale
        if not fresh and self._prev_quat is not None:
            dq = min(1.0, abs(float(np.dot(quat, self._prev_quat))))
            if 2 * np.arccos(dq) > 1.2:
                return self._miss()  # palm flipped ~a hand-swap in one frame: hold instead
        self._prev_quat = quat
        self.misses = 0
        return self.set_local(local, o_in, quat)

    def _miss(self):
        """Missed/rejected frame: hold last targets; a short dropout drops the EMA/jump
        gate so the hand can re-enter anywhere."""
        self.misses += 1
        if self.misses > 8:
            self.ema = None

    def drive(self):
        """Position-control fingers AND the actuated base toward the current targets;
        returns the wrist position error. No-op untargeted."""
        if self.targets is None:
            return 0.0
        ftgt, base_tgt = self.targets
        self.rigid.control_dofs_position(ftgt, self.finger_dofs)
        self.rigid.control_dofs_position(base_tgt, self.base_dofs)
        cur = np.asarray(self.rigid.get_dofs_position(self.base_dofs).cpu())
        return float(np.linalg.norm(base_tgt[:3] - cur[:3]))

    def teleport(self, pos, quat):
        """Place the base immediately (demo start pose)."""
        base = np.concatenate([np.asarray(pos, dtype=float), _euler_xyz(_quat_to_R(np.asarray(quat, dtype=float)), np.zeros(3))])
        self.rigid.set_dofs_position(base, self.base_dofs)
        self._prev_base = base.copy()
        self.targets = (self.targets[0] if self.targets else np.zeros(len(self.finger_dofs)), base)

    def finger_torque(self):
        """Summed |control torque| on the finger dofs (rises on contact); read post-step."""
        return float(np.abs(np.asarray(self.rigid.get_dofs_control_force(self.finger_dofs).cpu())).sum())


# --------------------------------------------------------------------------- render
def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


class MadronaRecorder:
    """Frames from a madrona batch camera at the physical teleop-camera pose, with the
    lab gsplat composited where seg==0 -- the same render stack scripts/suite.py uses,
    so recordings are directly comparable to the raw webcam video."""

    def __init__(self, cam, path, splat_bg=True):
        self.cam, self.path, self.frames, self._first = cam, Path(path).resolve(), [], True
        cam.set_pose(pos=CAM_POS, lookat=CAM_LOOKAT, up=CAM_UP)
        self.bg = None
        if splat_bg:
            from xsim.suite.models.arenas.table_arena import DEFAULT_SPLAT
            from xsim.suite.models.cameras import viewmats_cv
            from xsim.suite.renderers.splat_bg import SplatBackground

            f = 0.5 * CAM_RES[1] / math.tan(math.radians(CAM_FOV) / 2)
            K = np.array([[f, 0.0, CAM_RES[0] / 2], [0.0, f, CAM_RES[1] / 2], [0.0, 0.0, 1.0]])
            self.bg = SplatBackground(DEFAULT_SPLAT, prune_opacity=0.15).render(
                viewmats_cv(CAM_POS, CAM_LOOKAT, CAM_UP), K, CAM_RES
            )[0]

    def capture(self):
        rgb_t, _, seg_t, _ = self.cam.render(rgb=True, segmentation=True, force_render=self._first)
        self._first = False
        rgb = _np(rgb_t)[..., :3]
        frame = np.where((_np(seg_t) == 0)[..., None], self.bg, rgb) if self.bg is not None else rgb
        self.frames.append(frame.astype(np.uint8))

    def save(self, fps=20):
        import cv2

        self.path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, CAM_RES)
        for f in self.frames:
            vw.write(f[..., ::-1])
        vw.release()
        print(f"wrote {self.path} ({len(self.frames)} frames)")


def build_scene(hands, vis, record, splat_bg=True):
    """One scene holding all `hands` (rigid, no MPM). With `record`, adds the madrona
    batch raytracer + a camera at the physical teleop-cam pose + the suite light pair."""
    import genesis as gs

    mid = np.mean([h.pos for h in hands], axis=0)
    view = (mid[0] + 0.55, mid[1] - 0.55, mid[2] + 0.25)

    gs.init(backend=gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False, enable_adjacent_collision=False, constraint_timeconst=0.02
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=view, camera_lookat=tuple(mid), camera_fov=45, max_FPS=40
        ),
        show_viewer=vis,
        **({"renderer": gs.options.renderers.BatchRenderer(use_rasterizer=False)} if record else {}),
    )
    scene.add_entity(gs.morphs.Plane())
    cam = None
    if record:
        cam = scene.add_camera(
            res=CAM_RES, fov=CAM_FOV, GUI=False, pos=CAM_POS, lookat=CAM_LOOKAT, near=0.02, far=50.0
        )
        # madrona takes no scene ambience (hardcoded 0.05); the suite's light pair
        for d, i, sh in (((-0.4, -0.4, -0.8), 1.7, True), ((0.5, 0.3, -0.6), 0.85, False)):
            scene.add_light(pos=(0, 0, 3), dir=d, color=(1, 1, 1), directional=True, castshadow=sh, cutoff=45.0, intensity=i)
    for h in hands:
        h.add_to(scene)
    scene.build(n_envs=0)
    for h in hands:
        h.setup()
    rec = MadronaRecorder(cam, record, splat_bg) if record else None
    return scene, rec


def demo(record, steps, vis, kp_file=None, splat_bg=True):
    joints, flex, abd = mano_rig()
    hand = SimHand("right", HAND_POS)
    scene, rec = build_scene([hand], vis, record, splat_bg)

    seq = kp_sequence_from_file(kp_file, steps, joints) if kp_file else make_kp_sequence(joints, flex, abd, steps)
    hand.teleport(seq[0][1], seq[0][2])

    def run_once():
        ts, trs = [], []
        for kp_local, pos, quat in seq:
            hand.set_local(kp_local, pos, quat)
            trs.append(hand.drive())
            scene.step()
            ts.append(hand.finger_torque())
            if rec:
                rec.capture()
        return float(np.mean(ts)), float(np.mean(trs) * 1e3)

    if vis and not rec:
        # live viewer: loop the trajectory until the window is closed
        print("live viewer running -- close the window to stop")
        while scene.viewer.is_alive():
            mt, mtr = run_once()
            print(f"  cycle: mean |finger torque| {mt:.3f}   wrist err {mtr:.1f} mm")
    else:
        mt, mtr = run_once()
        if rec:
            rec.save()
        print(f"mean |finger control torque|: {mt:.4f}   mean wrist track err: {mtr:.1f} mm")


# --------------------------------------------------------------------------- live
def lift_hand_pnp(kp2d, kp3d_rel, K):
    """Place WiLoR's wrist-relative hand shape into the camera frame via PnP (the
    single-view metric lift; same math as xclients' lift_hand_pnp)."""
    import cv2

    obj = np.ascontiguousarray(kp3d_rel, dtype=np.float64)
    img = np.ascontiguousarray(kp2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, dtype=np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return (R @ obj.T).T + tvec.reshape(3)


def _probe_cameras(n=10):
    """Try cv2 indices 0..n-1 and report which ones deliver frames (`live --cap -1`)."""
    import cv2

    for i in range(n):
        cap = cv2.VideoCapture(i)
        ok, f = cap.read() if cap.isOpened() else (False, None)
        print(f"cap {i}: opened={cap.isOpened()} read={ok}" + (f" shape={f.shape} mean_px={f.mean():.1f}" if ok else ""))
        cap.release()


class MjpegPreview:
    """Latest-frame MJPEG stream at http://<host>:<port>/ in any browser. cv2.imshow is
    unavailable here (opencv-python-headless shadows the GUI build) and a browser
    stream also works over ssh."""

    def __init__(self, port):
        import http.server
        import threading

        self._lock = threading.Lock()
        self._jpg = None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            jpg = outer._jpg
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"camera preview: http://localhost:{port}/")

    def push(self, frame_bgr):
        import cv2

        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()


def _annotate(frame, dets):
    """Draw WiLoR 2D keypoints: green = right hand, red = left."""
    import cv2

    for d in dets:
        kp2d = np.asarray(d["keypoints_2d"], dtype=int)
        col = (0, 255, 0) if d["is_right"] else (0, 0, 255)
        for x, y in kp2d:
            cv2.circle(frame, (int(x), int(y)), 3, col, -1)
        cv2.putText(frame, "R" if d["is_right"] else "L", tuple(kp2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    return frame


def _dets_to_world(dets, K, world_from_cam_rdf):
    """WiLoR detections -> {'left'/'right': {kp2d, cam, world}} (first detection per side)."""
    got = {}
    for d in dets:
        side = "right" if d["is_right"] else "left"
        if side in got:
            continue
        kp_cam = lift_hand_pnp(d["keypoints_2d"], d["keypoints_3d"], K)
        if kp_cam[0, 2] <= 0.0:  # PnP placed the hand behind the camera
            continue
        kp_h = np.concatenate([kp_cam, np.ones((21, 1))], axis=1)
        got[side] = {
            "kp2d": np.asarray(d["keypoints_2d"], dtype=np.float64),
            "cam": kp_cam,
            "world": (kp_h @ world_from_cam_rdf.T)[:, :3],
        }
    return got


def _build_hands(a):
    sides = {"lr": ["right", "left"], "r": ["right"], "l": ["left"]}[a.hands]
    hands = {s: SimHand(s, LIVE_POS[s], ema_n=a.ema) for s in sides}
    scene, rec = build_scene(list(hands.values()), a.vis, a.record, not a.no_splat_bg)
    return hands, scene, rec


def _world_from_cam(a):
    """4x4 opencv-camera->world from the --extr/--yaw flags."""
    extr = np.asarray(np.loadtxt(a.extr)) if a.extr else DEFAULT_WORLD_FROM_CAM_FLU
    if a.yaw:
        c, s = np.cos(np.radians(a.yaw)), np.sin(np.radians(a.yaw))
        extr = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]) @ extr
    return extr @ FLU2RDF


def replay(a):
    """Feed a `--dump-kp` recording through the full sim pipeline: live viewer, no
    camera / webpolicy / WiLoR. The offline debug mode for the sim side."""
    data = np.load(a.replay)
    hands, scene, rec = _build_hands(a)
    # prefer camera-frame kp (re-mapped through the current --extr/--yaw) so extrinsic
    # experiments don't need a fresh capture; fall back to baked-in world kp
    wfc = _world_from_cam(a)
    frames = {}
    for s in hands:
        if f"{s}_cam" in data.files and len(data[f"{s}_cam"]):
            kp = data[f"{s}_cam"]
            kp_h = np.concatenate([kp, np.ones((*kp.shape[:2], 1))], axis=2)
            frames[s] = (kp_h @ wfc.T)[..., :3]
        elif s in data.files and len(data[s]):
            frames[s] = data[s]
    if not frames:
        raise RuntimeError(f"{a.replay} has no frames for hands {list(hands)} (files: {data.files})")
    T = max(len(v) for v in frames.values())
    print(f"replaying {T} frames from {a.replay} ({', '.join(frames)})")

    def run_once():
        for t in range(T):
            for s, h in hands.items():
                kp = frames[s][t] if s in frames and t < len(frames[s]) else None
                h.command(None if kp is None or np.isnan(kp).any() else kp)
            for _ in range(a.sim_steps):
                for h in hands.values():
                    h.drive()
                scene.step()
            if rec:
                rec.capture()

    if a.vis and not rec:
        print("live viewer running -- close the window to stop")
        while scene.viewer.is_alive():
            run_once()
            for h in hands.values():
                h.ema = None
    else:
        run_once()
        if rec:
            rec.save()


def live(a):
    """Webcam -> WiLoR server -> PnP lift -> world frame -> drive both sim hands."""
    import cv2
    from webpolicy.client import Client

    if a.cap < 0:
        return _probe_cameras()
    if a.replay:
        return replay(a)

    cap = cv2.VideoCapture(a.cap)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always consume the freshest frame
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read from camera {a.cap} (run --cap -1 to probe indices)")
    ih, iw = frame.shape[:2]
    K = np.array([[a.fx, 0.0, iw / 2.0], [0.0, a.fy, ih / 2.0], [0.0, 0.0, 1.0]])
    world_from_cam_rdf = _world_from_cam(a)

    # build the (slow) scene before opening the websocket so it cannot go stale
    hands, scene, rec = (None, None, None) if a.no_sim else _build_hands(a)
    client = Client(a.host, a.port)
    preview = MjpegPreview(a.preview_port) if a.preview_port else None
    print(f"camera {a.cap} ({iw}x{ih}) -> WiLoR at {a.host}:{a.port}; ctrl-c (or close the viewer) to stop")

    dump = video = None
    if a.no_sim and a.dump_kp:
        # everything needed to audit retargeting offline: raw rgb video + 2d/cam/world kp
        dump = {f"{s}{k}": [] for s in ("right", "left") for k in ("", "_cam", "_2d")}
        dump["t"] = []
        vpath = str(Path(a.dump_kp).resolve().with_suffix(".mp4"))
        video = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 20, (iw, ih))
    step = 0
    try:
        while (scene.viewer.is_alive() if scene and a.vis else True) and (not a.steps or step < a.steps):
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                continue
            dets = (client.step({"image": frame, "type": "image"}) or {}).get("hands") or []
            if preview:
                preview.push(_annotate(frame.copy(), dets))
            got = _dets_to_world(dets, K, world_from_cam_rdf)
            if dump is not None:
                video.write(frame)
                dump["t"].append(time.monotonic())
                for s in ("right", "left"):
                    r = got.get(s)
                    dump[s].append(r["world"] if r else np.full((21, 3), np.nan))
                    dump[f"{s}_cam"].append(r["cam"] if r else np.full((21, 3), np.nan))
                    dump[f"{s}_2d"].append(r["kp2d"] if r else np.full((21, 2), np.nan))
            if a.no_sim:
                if step % 10 == 0:
                    msg = f"frame {step:5d}  mean_px={frame.mean():5.1f}  hands={len(dets)}"
                    for s, r in got.items():
                        msg += f"  {s[0].upper()}@world{np.round(r['world'][0], 2)}"
                    print(msg)
            else:
                for side, hand in hands.items():
                    r = got.get(side)
                    hand.command(r["world"] if r else None)
                # several sim steps per camera frame: one 3 ms step per ~100 ms WiLoR
                # round-trip would leave the hands in ~30x slow motion
                for _ in range(a.sim_steps):
                    errs = {s: h.drive() for s, h in hands.items()}
                    scene.step()
                if rec:
                    rec.capture()
                if step % 30 == 0:
                    stat = "  ".join(
                        f"{s}:{'miss' if h.misses else f'{e * 1e3:5.1f}mm'}"
                        for (s, h), e in zip(hands.items(), errs.values())
                    )
                    print(f"frame {step:6d}  {stat}")
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if dump is not None and step:
            video.release()
            out = Path(a.dump_kp).resolve()
            np.savez(out, **{s: np.asarray(v) for s, v in dump.items()})
            print(f"wrote {out} + {out.with_suffix('.mp4').name} ({step} frames)")
        if rec and rec.frames:
            rec.save()


def _R_to_quat(R):
    """Robust rotation matrix -> (w,x,y,z); branches on the largest diagonal term so
    rotations near 180 deg (real tracked palms hit these) don't divide by ~0."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1.0 + t)
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k]))
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _euler_xyz(R, prev):
    """Intrinsic-XYZ euler with Rx(a)Ry(b)Rz(c) = R, matching the base chain's revolute
    order; a and c are 2pi-unwrapped toward `prev` so the +-pi wrap of a top-down yaw
    doesn't spin the wrist the long way. Singular at |b|=pi/2 (rare for tracked palms)."""
    b = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    a = np.arctan2(-R[1, 2], R[2, 2])
    c = np.arctan2(-R[0, 1], R[0, 0])
    eul = np.array([a, b, c])
    for i in (0, 2):
        eul[i] += 2 * np.pi * np.round((prev[i] - eul[i]) / (2 * np.pi))
    return eul


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("demo")
    d.add_argument("--record", type=str, default=None)
    d.add_argument("--steps", type=int, default=200)
    d.add_argument("--vis", action="store_true")
    d.add_argument("--kp", type=str, default=None, help="(T,21,3) npz/npy of world-frame MANO keypoints; omit for synthetic")
    d.add_argument("--no-splat-bg", action="store_true")
    l = sub.add_parser("live")
    l.add_argument("--host", type=str, default="localhost", help="WiLoR webpolicy server")
    l.add_argument("--port", type=int, default=8084)
    l.add_argument("--cap", type=int, default=0, help="cv2 camera index; -1 probes 0..9 and exits")
    l.add_argument("--no-sim", action="store_true", help="debug tracking only: skip genesis, preview + stats")
    l.add_argument("--dump-kp", type=str, default=None, help="with --no-sim: record kp + raw video to this npz/.mp4")
    l.add_argument("--replay", type=str, default=None, help="drive the sim from a --dump-kp npz; no camera/WiLoR")
    l.add_argument("--sim-steps", type=int, default=8, help="sim steps per camera/replay frame")
    l.add_argument("--preview-port", type=int, default=8090, help="MJPEG preview of the annotated camera; 0 disables")
    l.add_argument("--fx", type=float, default=515.0)
    l.add_argument("--fy", type=float, default=515.0)
    l.add_argument("--extr", type=str, default=None, help="4x4 camera-FLU->world txt; default matches retarget.py")
    l.add_argument("--yaw", type=float, default=0.0, help="rotate the extrinsic about world z (deg); 180 flips left/right feel")
    l.add_argument("--hands", choices=["lr", "l", "r"], default="lr")
    l.add_argument("--ema", type=int, default=4, help="EMA horizon on world kp3d; 1 disables")
    l.add_argument("--vis", action="store_true")
    l.add_argument("--record", type=str, default=None, help="mp4 rendered via madrona+gsplat from the teleop-cam pose")
    l.add_argument("--no-splat-bg", action="store_true")
    l.add_argument("--steps", type=int, default=0, help="0 = run until stopped")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "live":
        live(a)
    else:
        a.hands = "r"
        demo(a.record, a.steps, a.vis, a.kp, not a.no_splat_bg)


if __name__ == "__main__":
    main()
