"""robosuite-style layered environment suite on Genesis.

Layers, mirroring robosuite's seams:

- ``models``: :class:`Arena` + :class:`RobotModel` (with a :class:`GripperModel`
  from ``gripper_factory``) + free objects, composed by :class:`Task` into one
  world model. Genesis loads URDF/MJCF directly, so composition is entity-level
  — no XML merging.
- ``robots`` / ``controllers``: a runtime :class:`Robot` binds the entity and
  fans env actions out to part controllers (arm, gripper). Envs never touch
  actuators directly.
- ``environments``: ``GenesisEnv → RobotEnv → ManipulationEnv → Lift``, with
  robosuite's template hooks (``_load_model``, ``_setup_references``,
  ``_setup_observables``, ``_reset_internal``, ``reward``, ``_check_success``)
  on the gymnasium 5-tuple API.

New tasks subclass :class:`ManipulationEnv` and implement the hooks; they are
auto-registered and constructed via :func:`make`.
"""

from xsim.suite.environments import Lift, ManipulationEnv, RobotEnv
from xsim.suite.environments.base import REGISTERED_ENVS, GenesisEnv, make

__all__ = [
    "REGISTERED_ENVS",
    "GenesisEnv",
    "Lift",
    "ManipulationEnv",
    "RobotEnv",
    "make",
]
