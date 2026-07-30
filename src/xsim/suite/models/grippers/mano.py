"""MANO hand as a gripper: the 20 finger dofs trailing the 6-dof floating base.

``morph_file=None`` — the fingers live in the robot's own (floating-base) URDF.
Genesis dof order (measured; equals URDF file order, depth-first per finger):
per finger [flex1, abd, flex2, flex3] for index, middle, pinky, ring, thumb.

SIGN CONVENTION: the asset's flex axes point DORSAL, so anatomical (palmar)
curl is NEGATIVE flex, on the symmetric limits build_floating_urdf writes.
The grasp was first calibrated dorsal-side-down with positive angles (8/8 on
8-env Lift); these postures are that calibration reflected through the palm
plane, paired with the palm-down homes in robots/mano.py.

Grasp family: fingers extend along -x, thumb on the +z side, palmar curl
toward -y_hand. The grasp is a claw wrap onto a splayed-thumb wall:

  - open: index+middle hang claw-deep so the pads dangle at cube-face depth
    BESIDE the cube (a shallow open makes the closing arc land on the cube's
    top edge and bulldoze it — the dominant failure family).
  - thumb: CMC pre-swept with abd splayed so the thumb dangles as a
    near-table-height post opposing the finger pads; without the splay the
    thumb closes parallel to the fingers and nothing opposes (0/8).
  - close: index+middle wrap; their flat middle-phalanx pads press the cube
    onto the thumb post (flat-on-flat; curved-flank straddles slip-capped).
  - ring+pinky: partial curl, laterally clear of the cube; a full fist tuck
    works equally but reads wrong.

tcp_quat = Rx(+90) maps TCP z -> -y_hand (the anatomical palm normal), so the
expert's top-down quat family puts the palm face-down with the squeeze axis
(x_hand) horizontal. Friction 3.0 (5.0 makes the close drag the cube; capped
by Genesis at 5). ``ManoGrasp`` (right) holds the source-asset measurements;
``ManoGraspL`` reflects tcp x for the mirrored left hand.
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
    # scalars describe index flex1 (drive_dof): the commanded wrap, not the
    # mechanical stop, so closed-on-air gripper_norm ~= 0 (palmar = negative)
    open_dof: float = -1.0
    close_dof: float = -1.5
    grasp_dof: float = -1.5
    drive_dof: int = 0  # index flex1
    finger_link_names: tuple[str, str] = ("link_15", "link_03")  # thumb tip, index tip
    kp: float = 40.0  # soft fingers conform around the cube (RUKA: 40 > 160)
    kv: float = 8.0
    force_limit: float = 35.0
    # deep grasp: the palm caps the pocket (palm-on-cube) and the pinch line
    # lands at the cube's mid-face; -0.004 pinched the upper face and the
    # carry slip-capped at ee ~7 cm (max held-carry sweep)
    grasp_dz: float = -0.020
    max_open_width: float = 0.048  # thumb-post-to-index-pad mouth at open (measured)
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
    # the friction-pinch carry survives to ee ~0.10 (measured held-carry
    # ceiling at friction 3.0); 0.15 is unreachable, and 0.095 still clears
    # the mid-stack 2-cube tower during both StackRGY moves
    transport_height: float | None = 0.095
    # slow wrist re-alignment during the stack carry (the transport-entry
    # slerp at 0.15 grinds the pinched cube out within a few ticks)
    rot_frac: float | None = 0.05
    finger_friction: float | None = 3.0  # slip-limited carry; 2.0 caps lower, 5.0 drags
    morph_file: Path | None = None  # fingers baked into the robot URDF
    # pinch pocket in the link_00 frame: cube center at capture, measured by
    # FK probe at the palmar postures (thumb-post face -0.062 minus the cube
    # half-width; tips hang y ~-0.044, pocket z ~0 — NOT the dorsal
    # calibration's +0.040, the thumb opposes elsewhere palmarly)
    tcp_pos: tuple[float, float, float] | None = (-0.078, -0.045, 0.003)
    # Rx(+90): TCP z -> -y_hand (anatomical palm normal), TCP x -> x_hand
    # (squeeze axis)
    tcp_quat: tuple[float, float, float, float] | None = (_SQ2, _SQ2, 0.0, 0.0)
    # dof order: index[f1,abd,f2,f3] middle[...] pinky[...] ring[...] thumb[...]
    open_dofs: tuple[float, ...] = (
        -1.0, 0.0, -0.55, -0.35,
        -1.0, 0.0, -0.55, -0.35,
        -0.7, 0.0, -0.8, -0.5,
        -0.7, 0.0, -0.8, -0.5,
        -0.9, -0.35, -0.5, -0.3,
    )
    grasp_dofs: tuple[float, ...] = (
        -1.5, 0.0, -1.2, -0.9,
        -1.5, 0.0, -1.2, -0.9,
        -0.7, 0.0, -0.8, -0.5,
        -0.7, 0.0, -0.8, -0.5,
        -0.9, -0.35, -0.6, -0.4,
    )


@dataclass
class ManoGraspL(ManoGrasp):
    """Mirrored left hand: same dof order and setpoints (the axis rule keeps
    joint angles chirality-consistent); the pinch pocket reflects across x=0."""

    name: str = "ManoGraspL"
    tcp_pos: tuple[float, float, float] | None = (0.078, -0.045, 0.003)
