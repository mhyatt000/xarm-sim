"""dxarm7 gripper variants: the xarm7_variants leaves on the 4040 V-mount.

Pure composition — base pose + mount come from dxarm7 (L = +y = operator
left), arm + gripper calibration from the XArm7<X> parents. Handed pairs put
the LEFT hand on DXArm7L and the RIGHT hand on DXArm7R.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xsim.suite.models.mounts import Mount, VMount4040
from xsim.suite.models.robots.dxarm7 import _L_POS, _L_QUAT, _R_POS, _R_QUAT
from xsim.suite.models.robots.xarm7_variants import (
    XArm7BD,
    XArm7FourierL,
    XArm7FourierR,
    XArm7InspireL,
    XArm7InspireR,
    XArm7Jaco,
    XArm7Panda,
    XArm7Rethink,
    XArm7Robotiq85,
    XArm7Robotiq140,
    XArm7RobotiqS,
)


@dataclass
class DXArm7LBD(XArm7BD):
    name: str = "DXArm7LBD"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RBD(XArm7BD):
    name: str = "DXArm7RBD"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LFourier(XArm7FourierL):
    name: str = "DXArm7LFourier"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RFourier(XArm7FourierR):
    name: str = "DXArm7RFourier"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LInspire(XArm7InspireL):
    name: str = "DXArm7LInspire"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RInspire(XArm7InspireR):
    name: str = "DXArm7RInspire"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LJaco(XArm7Jaco):
    name: str = "DXArm7LJaco"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RJaco(XArm7Jaco):
    name: str = "DXArm7RJaco"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LPanda(XArm7Panda):
    name: str = "DXArm7LPanda"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RPanda(XArm7Panda):
    name: str = "DXArm7RPanda"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LRethink(XArm7Rethink):
    name: str = "DXArm7LRethink"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RRethink(XArm7Rethink):
    name: str = "DXArm7RRethink"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LRobotiq85(XArm7Robotiq85):
    name: str = "DXArm7LRobotiq85"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RRobotiq85(XArm7Robotiq85):
    name: str = "DXArm7RRobotiq85"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LRobotiq140(XArm7Robotiq140):
    name: str = "DXArm7LRobotiq140"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RRobotiq140(XArm7Robotiq140):
    name: str = "DXArm7RRobotiq140"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))


@dataclass
class DXArm7LRobotiqS(XArm7RobotiqS):
    name: str = "DXArm7LRobotiqS"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))


@dataclass
class DXArm7RRobotiqS(XArm7RobotiqS):
    name: str = "DXArm7RRobotiqS"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))
