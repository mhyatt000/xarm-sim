"""Robosuite gripper ports: separate MJCF entities attached at the arm flange.

Morphs are the Genesis-patched XMLs in assets/grippers (regenerate with
scripts/port_grippers.py); ``mount_*`` carries the base-body transform the
patch zeroed out; ``tcp_*`` places the virtual TCP so the experts' top-down
grasp family closes the fingers across the cube faces. Grasp setpoints, hold
bands, and grasp_dz come from scripts/calibrate_gripper.py sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsim.suite.models.grippers.gripper_model import GripperModel

_GRIPPERS = Path(__file__).resolve().parents[5] / "assets/grippers"


@dataclass
class RethinkGripper(GripperModel):
    # two mirrored finger slides: l_finger_joint opens at +0.020833 and hard-
    # stops at -0.0115; r_finger is the negation. Scalars describe dof 0.
    name: str = "RethinkGripper"
    n_dofs: int = 2
    open_dof: float = 0.020833
    close_dof: float = -0.0115
    # pad separation at dof (0, 0) is ~the cube width (norm plateau barely
    # above rest); command ~8 mm past contact per finger so the position
    # servo actually squeezes — same idea as the xArm's 0.58/0.85 fraction
    grasp_dof: float = -0.008
    finger_link_names: tuple[str, str] = ("l_finger", "r_finger")
    kp: float = 1000.0
    kv: float = 60.0
    force_limit: float = 20.0
    # swept on 64-env Lift: -0.004/-0.006 -> 100%, +0.002 -> 91%, +0.006 -> 45%
    grasp_dz: float = -0.004
    max_open_width: float = 0.062  # measured finger-link separation at open
    held_radius: float = 0.035
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    # closed-on-air settles at norm ~0.11 (grasp_dofs vs open span); seated on
    # the 31.75 mm cube the plateau is 0.20-0.31 (measured, diag probe)
    hold_norm_lo: float = 0.15
    hold_norm_hi: float = 0.55
    morph_file: Path | None = _GRIPPERS / "rethink_gripper.xml"
    # robosuite's eef body sits at 0.109, but the pad grip center is at ~0.13
    # along the flange z (fingertip link origins project to 0.1195 + pad
    # geometry); 0.109 leaves the pads jamming the table at grasp height
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.0, 0.13)
    tcp_quat: tuple[float, float, float, float] | None = None  # keep base axes
    open_dofs: tuple[float, ...] = (0.020833, -0.020833)
    grasp_dofs: tuple[float, ...] = (-0.008, 0.008)


@dataclass
class PandaGripper(GripperModel):
    # mirrored finger slides: finger_joint1 opens at +0.04, finger_joint2 at
    # -0.04; both hard-stop at 0. Scalars describe dof 0.
    name: str = "PandaGripper"
    n_dofs: int = 2
    open_dof: float = 0.04
    close_dof: float = 0.0
    grasp_dof: float = 0.008
    finger_link_names: tuple[str, str] = ("leftfinger", "rightfinger")
    kp: float = 1000.0
    kv: float = 60.0
    force_limit: float = 20.0
    grasp_dz: float = -0.004
    max_open_width: float = 0.08
    held_radius: float = 0.035
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    hold_norm_lo: float = 0.15
    hold_norm_hi: float = 0.75
    morph_file: Path | None = _GRIPPERS / "panda_gripper.xml"
    # base body pose zeroed by the patch (see port_grippers.py output)
    mount_quat: tuple[float, float, float, float] = (0.707107, 0.0, 0.0, -0.707107)
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.0, 0.103)
    tcp_quat: tuple[float, float, float, float] | None = None
    open_dofs: tuple[float, ...] = (0.04, -0.04)
    grasp_dofs: tuple[float, ...] = (0.008, -0.008)


@dataclass
class BDGripper(GripperModel):
    # Spot arm jaw: one hinged jaw (mover) against the fixed stator, hinge
    # range -1.57 (open) .. 0 (closed); the jaw points along the base +x.
    # A straight-down bite is impossible — the stator reaches 56 mm below the
    # pinch and digs into the table — so tcp_quat pitches the bite only 80°
    # (10° shy of vertical) and the TCP sits at the measured bite pocket
    # 0.21 m out. Swept: tx 0.18 -> 0%, 0.20 -> 88%, 0.21 -> 100% (tilt
    # 75-85° all 100%; 65° -> 56%).
    name: str = "BDGripper"
    n_dofs: int = 1
    open_dof: float = -1.57
    close_dof: float = 0.0
    grasp_dof: float = -0.2
    finger_link_names: tuple[str, str] = ("arm_link_fngr", "right_gripper")
    kp: float = 300.0
    kv: float = 40.0
    force_limit: float = 50.0
    grasp_dz: float = 0.0
    max_open_width: float = 0.1
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    hold_norm_lo: float = 0.15
    hold_norm_hi: float = 0.85
    morph_file: Path | None = _GRIPPERS / "bd_gripper.xml"
    # Ry(80°); the cube's 4-fold yaw family absorbs the pinch-axis swap
    tcp_pos: tuple[float, float, float] | None = (0.21, 0.0, 0.015)
    tcp_quat: tuple[float, float, float, float] | None = (0.76604444, 0.0, 0.64278761, 0.0)


@dataclass
class Robotiq85Gripper(GripperModel):
    # 4-bar linkage per finger, dof order [l_outer(drive), l_inner_finger,
    # l_inner_knuckle, r_outer, r_inner_finger, r_inner_knuckle]. The passive
    # joints lost their spring tendons in the port; the per-dof setpoints
    # position-servo them along the linkage ratio (inner_finger counter-
    # rotates, inner_knuckle follows).
    name: str = "Robotiq85Gripper"
    n_dofs: int = 6
    open_dof: float = 0.0
    close_dof: float = 0.8
    # command the hard stop: contact stalls the drive on the cube, and the
    # closed-on-air norm bottoms at 0 so the holding band floor can sit below
    # any loaded-carry norm (a mid-range target let carry load walk the norm
    # under grip_lo -> phantom drops)
    grasp_dof: float = 0.8
    finger_link_names: tuple[str, str] = ("left_inner_finger", "right_inner_finger")
    kp: float = 200.0
    kv: float = 20.0
    force_limit: float = 20.0
    grasp_dz: float = 0.0
    max_open_width: float = 0.085
    held_radius: float = 0.04
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    # closed-on-air bottoms at 0; seated on the cube ~0.3-0.4 static,
    # creeping lower under carry load (soft followers give)
    hold_norm_lo: float = 0.08
    hold_norm_hi: float = 0.55
    morph_file: Path | None = _GRIPPERS / "robotiq_gripper_85.xml"
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.0, 0.145)
    tcp_quat: tuple[float, float, float, float] | None = None
    open_dofs: tuple[float, ...] = (0.0,) * 6
    # followers stay targeted at neutral; their kp is the spring — 80 is the
    # swept sweet spot (10 seats the grasp but creeps during carries -> 0%
    # stack; 200 forces the pad angle and ejects the cube at close; 80 -> 100%)
    grasp_dofs: tuple[float, ...] = (0.8, 0.0, 0.0, 0.8, 0.0, 0.0)
    kp_dofs: tuple[float, ...] = (200.0, 80.0, 80.0, 200.0, 80.0, 80.0)
    kv_dofs: tuple[float, ...] = (20.0, 16.0, 16.0, 20.0, 16.0, 16.0)


@dataclass
class JacoThreeFingerGripper(GripperModel):
    # three fingers (thumb opposing index+pinky across the palm y axis), dof
    # order [thumb, thumb_distal, index, index_distal, pinky, pinky_distal];
    # proximals 0.5 (settled open) -> 1.51 (closed), distal curls were spring
    # tendons -> soft-servoed
    # NOTE genesis shifts each joint's zero by the MJCF `ref` attribute
    # (proximals ref 1.1, distals ref -0.5), and jaco's convention is
    # inverted: LARGER proximal angle = more OPEN (robosuite's format_action
    # is sign-inverted for the same reason). Genesis frame: fully open 0.41,
    # fully closed -1.1.
    name: str = "JacoThreeFingerGripper"
    n_dofs: int = 6
    open_dof: float = 0.41
    close_dof: float = -1.1
    grasp_dof: float = -1.1
    finger_link_names: tuple[str, str] = ("thumb_distal", "index_distal")
    # force swept on 64-env Lift: 5 N -> 14%, 15 -> 63%, 35 -> 92%, 50 -> 86%
    kp: float = 160.0
    kv: float = 16.0
    force_limit: float = 35.0
    grasp_dz: float = 0.0
    max_open_width: float = 0.11
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    # thin margin: the tips only close a few mm past the cube width, so the
    # held band floor sits just above the closed-on-air norm
    hold_norm_lo: float = 0.03
    hold_norm_hi: float = 0.6
    morph_file: Path | None = _GRIPPERS / "jaco_three_finger_gripper.xml"
    # measured pinch center ~0.162 along the base z at grasp depth; the
    # palm/eef yaw offset is dropped (4-fold cube yaw family)
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.0, 0.162)
    tcp_quat: tuple[float, float, float, float] | None = None
    open_dofs: tuple[float, ...] = (0.41, 0.5, 0.41, 0.5, 0.41, 0.5)
    grasp_dofs: tuple[float, ...] = (-1.1, 1.2, -1.1, 1.2, -1.1, 1.2)
    kp_dofs: tuple[float, ...] = (160.0, 8.0, 160.0, 8.0, 160.0, 8.0)
    kv_dofs: tuple[float, ...] = (16.0, 1.5, 16.0, 1.5, 16.0, 1.5)


@dataclass
class RobotiqSGripper(GripperModel):
    # three-finger S model, dof order [palm_1, f1_j1..3, palm_2, f2_j1..3,
    # middle_j1..3]. The scissor (palm joints, genesis ref-shifted to
    # -0.58..0 / 0..0.58) stays pinned like robosuite does; proximal j1 of
    # each finger drives; j2/j3 were spring tendons -> soft-servoed curls.
    name: str = "RobotiqSGripper"
    n_dofs: int = 11
    open_dof: float = 0.0
    close_dof: float = 1.2217
    grasp_dof: float = 1.05
    drive_dof: int = 1  # finger_1_joint_1
    finger_link_names: tuple[str, str] = ("finger_1_link_3", "finger_middle_link_3")
    kp: float = 160.0
    kv: float = 16.0
    force_limit: float = 35.0
    grasp_dz: float = 0.0
    max_open_width: float = 0.15
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    hold_norm_lo: float = 0.05
    hold_norm_hi: float = 0.6
    morph_file: Path | None = _GRIPPERS / "robotiq_gripper_s.xml"
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.045)
    mount_quat: tuple[float, float, float, float] = (
        -0.49921826, -0.50133955, 0.50133955, 0.49921826,
    )
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.15, 0.0)
    tcp_quat: tuple[float, float, float, float] | None = (0.707105, -0.707105, 0.0, 0.0)
    open_dofs: tuple[float, ...] = (
        -0.58, 0.0, 0.0, 0.0, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    grasp_dofs: tuple[float, ...] = (
        -0.58, 1.05, 0.3, 0.2, 0.58, 1.05, 0.3, 0.2, 1.05, 0.3, 0.2,
    )
    kp_dofs: tuple[float, ...] = (
        100.0, 160.0, 8.0, 8.0, 100.0, 160.0, 8.0, 8.0, 160.0, 8.0, 8.0,
    )
    kv_dofs: tuple[float, ...] = (
        10.0, 16.0, 1.5, 1.5, 10.0, 16.0, 1.5, 1.5, 16.0, 1.5, 1.5,
    )


@dataclass
class Robotiq140Gripper(GripperModel):
    # same linkage family as the 85 with a 140 mm stroke; dof order matches.
    # open_dofs is robosuite's settled open pose; grasp adds the drive angle
    # with the followers holding the fingertip parallel.
    name: str = "Robotiq140Gripper"
    n_dofs: int = 6
    open_dof: float = 0.012
    close_dof: float = 0.7
    grasp_dof: float = 0.7  # hard stop; same rationale as the 85
    finger_link_names: tuple[str, str] = ("left_inner_finger", "right_inner_finger")
    kp: float = 200.0
    kv: float = 20.0
    force_limit: float = 20.0
    grasp_dz: float = 0.0
    max_open_width: float = 0.14
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 1 / 3
    hold_norm_lo: float = 0.08
    hold_norm_hi: float = 0.55
    morph_file: Path | None = _GRIPPERS / "robotiq_gripper_140.xml"
    mount_pos: tuple[float, float, float] = (0.0, 0.0, -0.0625)
    mount_quat: tuple[float, float, float, float] = (0.0, -0.707105, 0.707108, 0.0)
    tcp_pos: tuple[float, float, float] | None = (0.0, 0.0, -0.27)
    tcp_quat: tuple[float, float, float, float] | None = (0.0, 1.0, 0.0, 0.0)
    # the inner knuckle is the second bar of the parallelogram and must track
    # the drive (holding it at open jams the linkage at ~0.26); the inner
    # finger spring-holds the pad parallel. Lift works (~97%); StackRGY is
    # OPEN: the cube slides along the long fingers during transports
    # (TCP-to-cube offset creeps 6 -> 38 mm) and slips on long carries.
    open_dofs: tuple[float, ...] = (0.012, 0.065, 0.065, -0.012, 0.065, 0.065)
    grasp_dofs: tuple[float, ...] = (0.7, 0.065, -0.45, -0.7, 0.065, -0.45)
    kp_dofs: tuple[float, ...] = (200.0, 10.0, 10.0, 200.0, 10.0, 10.0)
    kv_dofs: tuple[float, ...] = (20.0, 2.0, 2.0, 20.0, 2.0, 2.0)
