"""MANO hand as a gripper: the 20 finger dofs trailing the 6-dof floating base.

``morph_file=None`` — the fingers live in the robot's own (floating-base) URDF.
Genesis dof order (measured; equals URDF file order, depth-first per finger):
per finger [flex1, abd, flex2, flex3] for index, middle, pinky, ring, thumb.

Grasp family (hull-mesh distance probe, link_00 frame): fingers extend along
-x and curl toward +y (the palm normal); the thumb sits on the +z side. The
working pinch is thumb pad vs deep-curled index tip: the thumb CMC stays at
0.6 with abd -0.35 (an over-swept CMC crosses the palm and never meets the
fingertips — measured min gap 59 mm for the tip-to-tip wall-thumb family) and
only the thumb IP joints close, while index+middle wrap from (0.35,0.15,0.1)
to (1.5,1.1,0.8). Thumb-index hull gap closes 62 mm (open) -> 0.8 mm (full
command), crossing the 32 mm cube width at closure ~0.5 with pinch mid
(-0.072, 0.046, 0.059) and squeeze axis ~x_hand (0.97,-0.21,-0.06).
tcp_quat maps the TCP frame's z to +y_hand (palm normal), so the expert's
top-down quat family [0, cos h, sin h, 0] puts the palm face-down with the
squeeze axis horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsim.suite.models.grippers.gripper_model import GripperModel

_SQ2 = 0.7071067811865476


@dataclass
class ManoGrasp(GripperModel):
    name: str = "ManoGrasp"
    n_dofs: int = 20
    # scalars describe index flex1 (drive_dof): commanded curl, not the 1.6
    # mechanical stop, so closed-on-air gripper_norm ~= 0
    open_dof: float = 0.35
    close_dof: float = 1.5
    grasp_dof: float = 1.5
    drive_dof: int = 0  # index flex1
    finger_link_names: tuple[str, str] = ("link_15", "link_03")  # thumb tip, index tip
    kp: float = 40.0  # soft fingers conform around the cube (RUKA: 40 > 160)
    kv: float = 8.0
    force_limit: float = 35.0
    grasp_dz: float = -0.004
    max_open_width: float = 0.062  # thumb-index hull gap at the open posture
    held_radius: float = 0.05
    close_min_s: float = 0.4
    close_timeout_s: float = 1.0
    open_s: float = 2.0
    # the drive reaches full command held or not; the band is open at the
    # bottom and held_radius does the discriminating (Inspire/RUKA precedent)
    hold_norm_lo: float = -1.0
    hold_norm_hi: float = 0.9
    # stack-release: partial uncurl + upward creep during the OPEN dwell
    release_a: float = 0.8
    place_clearance: float = 0.005
    release_rise: float = 0.002
    finger_friction: float | None = 2.0  # URDF ships none; default drops carries
    morph_file: Path | None = None  # fingers baked into the robot URDF
    # thumb-index pinch mid at the 32 mm-gap closure, link_00 frame (measured)
    tcp_pos: tuple[float, float, float] | None = (-0.072, 0.046, 0.059)
    # Rx(-90): TCP z -> +y_hand (palm normal), TCP x -> x_hand (squeeze axis)
    tcp_quat: tuple[float, float, float, float] | None = (_SQ2, -_SQ2, 0.0, 0.0)
    # dof order: index[f1,abd,f2,f3] middle[...] pinky[...] ring[...] thumb[...]
    # ring+pinky tuck out of the way; thumb CMC (f1) and abd are pinned in both
    # postures — only the thumb IP pair closes (wall thumb, minimal ip close)
    open_dofs: tuple[float, ...] = (
        0.35, 0.0, 0.15, 0.1,
        0.35, 0.0, 0.15, 0.1,
        1.3, 0.0, 1.4, 0.9,
        1.3, 0.0, 1.4, 0.9,
        0.6, -0.35, 0.15, 0.1,
    )
    grasp_dofs: tuple[float, ...] = (
        1.5, 0.0, 1.1, 0.8,
        1.5, 0.0, 1.1, 0.8,
        1.3, 0.0, 1.4, 0.9,
        1.3, 0.0, 1.4, 0.9,
        0.6, -0.35, 0.8, 1.0,
    )


@dataclass
class ManoGraspL(ManoGrasp):
    """Mirrored left hand: same dof order and setpoints (the axis rule keeps
    joint angles chirality-consistent); the pinch pocket reflects across x=0."""

    name: str = "ManoGraspL"
    tcp_pos: tuple[float, float, float] | None = (0.072, 0.046, 0.059)
