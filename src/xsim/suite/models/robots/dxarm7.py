"""dxarm7: two xArm7s on the 45 deg 4040 V-mount.

Not one bimanual robot — each side is a standalone 7-dof robot model carrying
its half of the shared mount. Pass robots=["DXArm7L", "DXArm7R"] to compose
the full rig in any env.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xsim.suite.models.cameras import CameraSpec
from xsim.suite.models.mounts import Mount, VMount4040
from xsim.suite.models.robots.xarm7 import XArm7

_L_POS, _L_QUAT = VMount4040(side=-1).base_pose()
_R_POS, _R_QUAT = VMount4040(side=+1).base_pose()


@dataclass
class DXArm7L(XArm7):
    name: str = "DXArm7L"
    base_pos: tuple[float, float, float] = _L_POS
    base_quat: tuple[float, float, float, float] = _L_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=-1))
    # no calibrated wrist mount on the rig; inheriting XArm7's spec would give
    # both sides the same "wrist" camera name
    cameras: tuple[CameraSpec, ...] = ()


@dataclass
class DXArm7R(XArm7):
    name: str = "DXArm7R"
    base_pos: tuple[float, float, float] = _R_POS
    base_quat: tuple[float, float, float, float] = _R_QUAT
    mount: Mount | None = field(default_factory=lambda: VMount4040(side=+1))
    cameras: tuple[CameraSpec, ...] = ()
