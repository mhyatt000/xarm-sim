"""RUKA v2 tendon-driven hand, baked into the xArm7 URDF as trailing dofs.

The merged URDFs live in assets/ruka (regenerate with scripts/port_ruka.py);
``morph_file=None`` because the 21 hand joints trail the 7 arm dofs in the
robot's own morph. Genesis dof order (measured, depth-first — NOT the URDF's
BFS joint order): base_pitch, wrist_yaw, index[splay,mcp,pip,dip],
mid[mcp,pip,dip], ring[splay,mcp,pip,dip], pinky[splay,mcp,pip,dip],
thumb[cmc,mcp,ip], base_yaw (dead-end leaf joint, pinned 0).

Grasp family (tip-frame FK probe, link_eef frame, arm at home): the finger
flexion planes run along the eef X axis — index/middle tips close from
x -0.10 toward -0.05 while the thumb tip closes from -0.04 toward -0.07, so
the functional squeeze axis is +/-x_eef and tcp_quat stays None (cube faces
already meet the pads square; a pinch-vector yaw fit is a red herring — the
thumb->index vector mixes the closing x with a FIXED lateral y offset).
Contact pair = thumb (y ~0.010 at cmc 1.1) vs middle (y ~0.025); index
(y ~0.044) sweeps just past the cube's +y face. The thumb must close during
grasp (a pinned thumb is a static post: the curling fingers sweep the cube
sideways past it). Ring+pinky tuck at 1.5/1.3/0.5; palm dofs (base_pitch,
wrist_yaw) and splays stay 0 in both postures; index splay moves the tip
<1 mm (dead lever).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsim.suite.models.grippers.gripper_model import GripperModel


@dataclass
class RukaHand(GripperModel):
    # scalars describe index_mcp (dof 3). close_dof = the commanded grasp curl
    # (not the 1.92 mechanical stop) so closed-on-air gripper_norm ~= 0.
    name: str = "RukaHand"
    n_dofs: int = 21
    # scalars describe mid_mcp: a deep command lets the fingers wrap PAST the
    # cube (contacts migrate tip -> pip/mcp knuckles, pinch degrades to a
    # loose envelop and the drive reaches full command = phantom closed-on-air)
    # so the grasp stalls the pads ~0.25 rad past their face-contact angles
    open_dof: float = 0.4
    close_dof: float = 1.0
    grasp_dof: float = 1.0
    drive_dof: int = 6  # mid_mcp — the middle pad stalls hardest on the cube
    finger_link_names: tuple[str, str] = ("ruka_thumb___joint_3", "ruka_finger___joint_3_2")
    # capture-rate sweep (8 envs x 3 grasp cycles): kp 80 -> 0.58, 160 -> 0.33,
    # 40 -> 0.71, 25 -> 0.54 — soft fingers conform around the cube instead of
    # ejecting it. force 35 per the prior-hand pattern (Jaco/MHR).
    kp: float = 40.0
    kv: float = 8.0
    force_limit: float = 35.0
    # swept: 0 -> 0.58, -0.006 -> 0.71, -0.010 -> 0.50, +0.006 -> 0.12
    grasp_dz: float = -0.006
    max_open_width: float = 0.068  # thumb-middle x span at the open posture
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    # long shed dwell: the cube walks down out of the uncurling fingers along
    # the release_rise creep; short dwells retreat with it still in contact
    open_s: float = 2.0
    # the capture is a scoop-envelop (the squeeze seed-squirts the cube ~3 cm
    # up into the fist pocket, where it wedges); the drive reaches full command
    # either way, so norm can't separate held from closed-on-air — the band is
    # open at the bottom and held_radius does the discriminating (Inspire
    # precedent)
    hold_norm_lo: float = -1.0
    hold_norm_hi: float = 0.9
    # stack-release family swept @16 envs: the winning shed is a partial
    # uncurl held while the arm creeps upward — rel/clearance/dwell/creep
    # (0.8, 0.005, 2.0 s, 0.002) -> 0.69-0.75; creep 0 -> 0.0-0.19 (static
    # dwell then retreat-speed jerk drags the seated cube), creep 0.003 -> 0,
    # release 1.0 -> 0-0.25 (full-open extension sweeps the tower)
    release_a: float = 0.8
    place_clearance: float = 0.005
    release_rise: float = 0.002
    finger_friction: float | None = 2.0  # URDF has none; default drops carries
    morph_file: Path | None = None  # fingers baked into the robot URDF
    # squeeze pocket, link_eef frame: x -0.065 = where the closing thumb
    # (-0.042 -> -0.072) and middle (-0.083 -> -0.052) tips cross, with 7 mm
    # descent clearance between the open thumb and the cube's +x face;
    # y 0.018 = thumb/middle contact midline; z 0.212 = measured tip height
    # as each crosses its face plane (contacts land ~2-4 mm below cube center)
    tcp_pos: tuple[float, float, float] | None = (-0.065, 0.018, 0.212)
    tcp_quat: tuple[float, float, float, float] | None = None  # squeeze axis = x_eef
    # dof order: [base_pitch, wrist_yaw, index_splay, index_mcp, index_pip,
    # index_dip, mid_mcp, mid_pip, mid_dip, ring_splay, ring_mcp, ring_pip,
    # ring_dip, pinky_splay, pinky_mcp, pinky_pip, pinky_dip, thumb_cmc,
    # thumb_mcp, thumb_ip, base_yaw]
    open_dofs: tuple[float, ...] = (
        0.0, 0.0,
        0.0, 0.4, 0.28, 0.12,
        0.4, 0.2, 0.1,
        0.0, 1.5, 1.3, 0.5,
        0.0, 1.5, 1.3, 0.5,
        1.1, -0.5, 0.1,
        0.0,
    )
    grasp_dofs: tuple[float, ...] = (
        0.0, 0.0,
        0.0, 1.05, 0.6, 0.3,
        1.0, 0.5, 0.3,
        0.0, 1.5, 1.3, 0.5,
        0.0, 1.5, 1.3, 0.5,
        1.1, -0.4, 0.4,
        0.0,
    )


@dataclass
class RukaHandL(RukaHand):
    """Mirrored left hand (assets/ruka/xarm7_ruka_left.urdf; meshes mirror via
    scale="-1 1 1"). Same dof order and setpoints; tcp measured on the left."""

    name: str = "RukaHandL"
    tcp_pos: tuple[float, float, float] | None = (0.065, 0.018, 0.212)
