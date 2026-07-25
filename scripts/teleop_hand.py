"""Drive the NIMBLE hybrid hand from MANO-style 3D keypoints (teleop).

Confirms the design discussion: the NIMBLE URDF skeleton is NIMBLE's anatomical
bone tree, not MANO's, so MANO pose params cannot be copied onto the joints. What
works -- and what this does -- is fit joint angles so the hand reaches the 21
MANO/OpenPose keypoints (wrist + [mcp,pip,dip,tip] per finger). That fit IS the
retargeting; it is per-finger IK, which is small because each finger drives <=4
joints.

Because the 21 keypoints carry no CMC point for index/middle/ring, those `met`
joints are locked (see LOCKED); the fit would otherwise be under-determined.

Control split:
  wrist   6-DOF free base, set kinematically from the tracked palm transform
  fingers control_dofs_position to the IK'd angles
Sensing:
  get_dofs_control_force -> per-joint torque the controller exerts (rises on contact)

  python scripts/teleop_hand.py selftest          # numeric IK round-trip, no sim
  python scripts/teleop_hand.py demo --record out.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NIMBLE = REPO / "extra" / "NIMBLE_model"
OUT = NIMBLE / "output" / "genesis"
HAND_POS = (0.0, 0.0, 0.30)

# NIMBLE joint ids per finger: [met, pro, int, dis, tip]; thumb has no `int`.
# Driven joints and their DOF (flex only, or flex+abd). `met` on the non-thumb/pinky
# fingers is locked -- no keypoint constrains it.
FINGERS = {
    "thumb":  {"chain": [0, 1, 2, 3, 4],      "drive": {1: 2, 2: 2, 3: 1}},
    "index":  {"chain": [0, 5, 6, 7, 8, 9],   "drive": {6: 2, 7: 1, 8: 1}, "lock": [5]},
    "middle": {"chain": [0, 10, 11, 12, 13, 14], "drive": {11: 2, 12: 1, 13: 1}, "lock": [10]},
    "ring":   {"chain": [0, 15, 16, 17, 18, 19], "drive": {16: 2, 17: 1, 18: 1}, "lock": [15]},
    "pinky":  {"chain": [0, 20, 21, 22, 23, 24], "drive": {21: 2, 22: 1, 23: 1}, "lock": [20]},
}
# MANO/OpenPose 21-kp order -> the NIMBLE joint id each keypoint targets.
KP_TO_JOINT = [
    0,                 # 0 wrist
    1, 2, 3, 4,        # thumb  cmc, mcp, ip, tip
    6, 7, 8, 9,        # index  mcp(pro2), pip(int2), dip(dis2), tip
    11, 12, 13, 14,    # middle
    16, 17, 18, 19,    # ring
    21, 22, 23, 24,    # pinky
]


def _load_nimble_tables():
    from nimble_genesis import shim_pytorch3d  # reuse the same shim

    shim_pytorch3d()
    sys.path.insert(0, str(NIMBLE))
    from utils import JOINT_PARENT_ID_DICT, BONE_ID_JOINT_DICT

    joint_to_bone = {j: b for b, j in BONE_ID_JOINT_DICT.items()}
    return JOINT_PARENT_ID_DICT, joint_to_bone


def rest_rig():
    """Rest joint positions (25,3) and per-joint flex/abd axes from the built rig."""
    from nimble_genesis import joint_frames, shim_pytorch3d

    shim_pytorch3d()  # joint_frames imports utils, which imports pytorch3d
    sys.path.insert(0, str(NIMBLE))  # so `from utils import ...` inside joint_frames resolves
    rig = np.load(OUT / "rig.npz")
    joints = rig["joints"]
    flex, abd = joint_frames(joints)  # (25,3) each
    return joints, flex, abd


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
    applied about the joint's own axes carried into the accumulated frame."""
    chain = FINGERS[name]["chain"]
    out = {chain[0]: joints[chain[0]].copy()}
    R = np.eye(3)
    for i in range(1, len(chain)):
        j, jp = chain[i], chain[i - 1]
        if j in angles:
            a = angles[j]
            Rj = _rot(R @ flex[j], a[0])
            if len(a) > 1:
                Rj = _rot(R @ abd[j], a[1]) @ Rj
            R = Rj @ R
        out[j] = out[jp] + R @ (joints[j] - joints[jp])
    return out


def ik_finger(name, target, joints, flex, abd, iters=40):
    """Gauss-Newton fit of a finger's driven angles so its FK joint positions match
    `target` (dict joint_id -> xyz). This is the 'best fit to reach kp3d'."""
    spec = FINGERS[name]
    dofs = [(j, d) for j, d in spec["drive"].items()]  # (joint, n_dof)
    x = np.zeros(sum(d for _, d in dofs))

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
        J = np.zeros((len(r), len(x)))
        for k in range(len(x)):
            dx = np.zeros_like(x)
            dx[k] = 1e-4
            J[:, k] = (resid(x + dx) - r) / 1e-4
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + np.clip(step, -0.5, 0.5)
    return unpack(x), float(np.linalg.norm(resid(x)))


def retarget(kp21, joints, flex, abd):
    """21 MANO keypoints -> {joint_id: angles} for all fingers, plus wrist pose."""
    target = {jid: kp21[k] for k, jid in enumerate(KP_TO_JOINT)}
    out, err = {}, 0.0
    for name in FINGERS:
        ang, e = ik_finger(name, target, joints, flex, abd)
        out.update(ang)
        err = max(err, e)
    return out, err


# --------------------------------------------------------------------------- selftest
def selftest():
    joints, flex, abd = rest_rig()
    rng = np.random.default_rng(0)
    worst = 0.0
    for name in FINGERS:
        gt = {}
        for j, d in FINGERS[name]["drive"].items():
            gt[j] = rng.uniform(-0.4, 0.9, size=d)
        posed = _fk_clean(name, gt, joints, flex, abd)
        kp = np.zeros((21, 3))
        for k, jid in enumerate(KP_TO_JOINT):
            kp[k] = posed.get(jid, joints[jid])
        ang, err = ik_finger(name, {j: posed[j] for j in posed}, joints, flex, abd)
        derr = max(np.abs(np.concatenate([ang[j] - gt[j] for j in gt])))
        worst = max(worst, err)
        print(f"  {name:7s} kp_fit_err={err * 1e3:6.3f} mm   max_angle_err={np.degrees(derr):5.2f} deg")
    print(f"worst keypoint fit error: {worst * 1e3:.3f} mm")


# --------------------------------------------------------------------------- demo
def palm_frame(kp):
    """World->local rotation and origin from the 21 keypoints, matching the rest-frame
    convention used to build the rig (fingers +z, palm normal +y). Used to express
    world-frame MANO keypoints in the wrist-local frame the finger IK expects."""
    wrist = kp[0]
    mcp = kp[[5, 9, 13, 17]]  # index/middle/ring/pinky MCP
    fdir = mcp.mean(0) - wrist
    fdir /= np.linalg.norm(fdir) + 1e-12
    _, _, vh = np.linalg.svd(mcp - mcp.mean(0))
    palm_n = vh[2] - vh[2].dot(fdir) * fdir
    palm_n /= np.linalg.norm(palm_n) + 1e-12
    side = np.cross(palm_n, fdir)
    R = np.stack([side, palm_n, fdir])  # rows = local basis in world coords (world->local)
    return wrist, R


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
    expects, the wrist pose driving the free base.

    The finger IK compares against the rig's rest joints in their original frame, so
    input keypoints are mapped input-palm-frame -> rest-palm-frame rather than into an
    ad-hoc frame; otherwise a constant rotation offset shows up as bogus finger angles."""
    kp_all = np.load(path)
    kp_all = kp_all["kp"] if hasattr(kp_all, "files") else kp_all
    kp_all = np.asarray(kp_all, dtype=np.float64)
    assert kp_all.ndim == 3 and kp_all.shape[1:] == (21, 3), f"expected (T,21,3), got {kp_all.shape}"
    if steps and steps < len(kp_all):
        kp_all = kp_all[:: max(1, len(kp_all) // steps)][:steps]
    # recenter so the first frame's wrist sits at the hand's build pose: the MPM flesh
    # lives in a world-fixed grid, so absolute teleop coords would drive it out of grid.
    kp_all = kp_all - (kp_all[0, 0] - np.array(HAND_POS))

    rest_kp = joints[KP_TO_JOINT]
    o_rest, R_rest = palm_frame(rest_kp)  # rest hand's own palm frame, in rig coords
    seq = []
    for kp in kp_all:
        o_in, R_in = palm_frame(kp)
        # input world -> input-palm-frame -> back out through the rest palm frame
        local = ((kp - o_in) @ R_in.T) @ R_rest + o_rest
        seq.append((local, o_in, _R_to_quat(R_in.T)))
    return seq


def strip_visual_urdf(src: Path) -> Path:
    """Write a sibling URDF with all <visual> tags removed. The hybrid renders the
    rigid bones (surface hardcoded in HybridEntity), and under fast motion the rigid
    bones lead the compliant MPM skin and poke through it. Collision geoms stay, so
    the coupling is unchanged; the bones simply stop rendering and only the skin shows."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(src)
    for link in tree.getroot().findall("link"):
        for vis in link.findall("visual"):
            link.remove(vis)
    dst = src.with_name(src.stem + "_novis.urdf")
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    return dst


def demo(record, steps, vis, kp_file=None):
    import genesis as gs
    from genesis.utils import geom as gu

    joints, flex, abd = rest_rig()
    parents, joint_to_bone = _load_nimble_tables()
    rig = np.load(OUT / "rig.npz")
    skin_v, skin_sw, kept = rig["skin_v"], rig["skin_sw"], list(rig["bones_kept"])

    gs.init(backend=gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=20),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False, enable_adjacent_collision=False, constraint_timeconst=0.02
        ),
        mpm_options=gs.options.MPMOptions(
            # world-fixed grid; pad must cover the wrist's translation workspace
            lower_bound=skin_v.min(0) + np.array(HAND_POS) - 0.15,
            upper_bound=skin_v.max(0) + np.array(HAND_POS) + 0.15,
            gravity=(0, 0, 0),
            particle_size=0.0035,
            grid_density=128,
            enable_CPIC=True,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.55), camera_lookat=(0.0, 0.0, 0.32), camera_fov=45, max_FPS=40
        ),
        show_viewer=vis,
    )
    scene.add_entity(gs.morphs.Plane())
    cam = (
        scene.add_camera(res=(1280, 720), pos=(0.5, -0.5, 0.55), lookat=(0, 0, 0.32), fov=45, GUI=False)
        if record
        else None
    )

    def soft_from_rigid(scene, part_rigid, material_soft, material_hybrid, surface):
        return scene.add_entity(
            material=material_soft,
            morph=gs.morphs.Mesh(file=str(OUT / "skin.obj"), pos=HAND_POS),
            surface=gs.surfaces.Default(vis_mode="visual"),
        )

    bone_to_joint = {b: j for j, b in joint_to_bone.items()}

    def association(part_rigid, part_soft):
        # mirror nimble_genesis.association: bind each particle to the bone whose
        # skinning weight dominates. skin_sw columns are indexed by JOINT id.
        from scipy.spatial import cKDTree

        by_name = {l.name: l for l in part_rigid.links if len(l.geoms) >= 1}
        order, li, gi, tr, qt = [], [], [], [], []
        for b in kept:
            link = by_name.get(f"bone_{b:02d}")
            if link is None:
                continue
            g = link.geoms[0]
            t, q = gu.transform_pos_quat_by_trans_quat(g.init_pos, g.init_quat, link.init_x_pos, link.init_x_quat)
            order.append(bone_to_joint[b]); li.append(link.idx); gi.append(g.idx); tr.append(t); qt.append(q)
        p = part_soft.init_particles
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        _, near = cKDTree(skin_v).query(p - np.asarray(HAND_POS), k=1)
        wsel = skin_sw[near][:, order]
        return wsel.argmax(1), li, gi, tr, qt

    hand = scene.add_entity(
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(gravity_compensation=1.0),
            material_soft=gs.materials.MPM.Muscle(E=3e4, nu=0.45, rho=1000.0, model="neohooken"),
            use_default_coupling=False,
            thickness=0.002,
            soft_dv_coef=0.05,
            damping=1000.0,
            func_instantiate_soft_from_rigid=soft_from_rigid,
            func_instantiate_rigid_soft_association=association,
        ),
        morph=gs.morphs.URDF(file=str(strip_visual_urdf(OUT / "nimble_hand.urdf")), fixed=False, pos=HAND_POS, convexify=False),
        surface=gs.surfaces.Default(),
    )

    scene.build(n_envs=0)
    rigid = hand.part_rigid
    name_to_dofs = {j.name: list(j.dofs_idx_local) for j in rigid.joints}
    # base = the free root joint's 6 dofs; fingers = everything else
    base_dofs = list(rigid.joints[0].dofs_idx_local)
    finger_dofs = [d for d in range(rigid.n_dofs) if d not in base_dofs]
    rigid.set_dofs_kp(np.full(len(finger_dofs), 14.0), finger_dofs)
    rigid.set_dofs_kv(np.full(len(finger_dofs), 1.4), finger_dofs)
    # base PD gains, scaled to the actual hand mass (~50 g of light bones): critical
    # damping at a few Hz. Stiff gains here produce a stiff ODE and NaN at this dt.
    m = float(sum(l.get_mass() for l in rigid.links))
    KP_LIN, KV_LIN = m * (2 * np.pi * 4.0) ** 2, 2 * np.sqrt(m * m * (2 * np.pi * 4.0) ** 2)
    KP_ANG, KV_ANG = 0.6, 0.06
    print(f"hand mass {m * 1e3:.1f} g -> KP_LIN {KP_LIN:.1f} KV_LIN {KV_LIN:.2f}")

    seq = kp_sequence_from_file(kp_file, steps, joints) if kp_file else make_kp_sequence(joints, flex, abd, steps)
    # start the free base at the trajectory's first pose so there is no large startup
    # transient (huge orientation error on the tiny rotational inertia -> NaN)
    rigid.set_pos(np.asarray(seq[0][1]))
    rigid.set_quat(np.asarray(seq[0][2]))
    fmap = {d: i for i, d in enumerate(finger_dofs)}

    def drive(frame):
        kp_local, tgt_pos, tgt_quat = frame
        # fingers: IK to keypoints, position-controlled on their own dofs only
        ang, _ = retarget(kp_local, joints, flex, abd)
        ftgt = np.zeros(len(finger_dofs))
        for jid, a in ang.items():
            b = joint_to_bone[jid]
            for nm, val in ((f"bone_{b:02d}_flex", a[0]), (f"bone_{b:02d}_abd", a[1] if len(a) > 1 else None)):
                if val is not None and nm in name_to_dofs:
                    ftgt[fmap[name_to_dofs[nm][0]]] = val
        rigid.control_dofs_position(ftgt, finger_dofs)
        # wrist: 6-DOF PD (force+torque) toward the tracked palm pose
        cur_p = np.asarray(rigid.get_pos().cpu())
        cur_q = np.asarray(rigid.get_quat().cpu())
        f = np.clip(KP_LIN * (np.asarray(tgt_pos) - cur_p) - KV_LIN * np.asarray(rigid.get_vel().cpu()), -10.0, 10.0)
        tau = np.clip(KP_ANG * quat_error_rotvec(tgt_quat, cur_q) - KV_ANG * np.asarray(rigid.get_ang().cpu()), -0.3, 0.3)
        rigid.control_dofs_force(np.concatenate([f, tau]), base_dofs)
        scene.step()
        return np.abs(np.asarray(rigid.get_dofs_control_force(finger_dofs).cpu())).sum(), np.linalg.norm(np.asarray(tgt_pos) - cur_p)

    def run_once():
        ts, trs = [], []
        for frame in seq:
            a, b = drive(frame)
            ts.append(a); trs.append(b)
            if cam:
                cam.render()
        return float(np.mean(ts)), float(np.mean(trs) * 1e3)

    if cam:
        cam.start_recording()
    if vis and not cam:
        # live viewer: loop the trajectory until the window is closed
        print("live viewer running -- close the window to stop")
        while scene.viewer.is_alive():
            mt, mtr = run_once()
            print(f"  cycle: mean |finger torque| {mt:.3f}   wrist err {mtr:.1f} mm")
    else:
        mt, mtr = run_once()
        if cam:
            out = Path(record).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            cam.stop_recording(save_to_filename=str(out), fps=40)
            print(f"wrote {out}")
        print(f"mean |finger control torque|: {mt:.4f}   mean wrist track err: {mtr:.1f} mm")


def _R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w + 1e-9)
    y = (R[0, 2] - R[2, 0]) / (4 * w + 1e-9)
    z = (R[1, 0] - R[0, 1]) / (4 * w + 1e-9)
    return np.array([w, x, y, z])


def quat_error_rotvec(q_target, q_current):
    """Axis-angle (world-frame) rotation taking current orientation to target: the
    angular error term of the wrist PD. Genesis quats are (w,x,y,z)."""
    wt, xt, yt, zt = q_target
    wc, xc, yc, zc = q_current
    # q_err = q_target * conj(q_current)
    xc, yc, zc = -xc, -yc, -zc
    w = wt * wc - xt * xc - yt * yc - zt * zc
    x = wt * xc + xt * wc + yt * zc - zt * yc
    y = wt * yc - xt * zc + yt * wc + zt * xc
    z = wt * zc + xt * yc - yt * xc + zt * wc
    v = np.array([x, y, z])
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    angle = 2 * np.arctan2(n, abs(w)) * (1 if w >= 0 else -1)
    return v / n * angle


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("demo")
    d.add_argument("--record", type=str, default=None)
    d.add_argument("--steps", type=int, default=200)
    d.add_argument("--vis", action="store_true")
    d.add_argument("--kp", type=str, default=None, help="(T,21,3) npz/npy of world-frame MANO keypoints; omit for synthetic")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        demo(a.record, a.steps, a.vis, a.kp)


if __name__ == "__main__":
    main()
